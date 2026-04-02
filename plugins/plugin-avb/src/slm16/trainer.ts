/**
 * SLM16 — training loop, validation, and checkpoint persistence.
 *
 * This module is the long-lived hot path inside the worker thread. It owns
 * the model weights, the optimizer state, and the persistent training counters
 * (step, tokensSeen, trainSeconds, bestValLoss). It does NOT own the worker
 * thread harness — that's `worker.ts`. The trainer just receives a callback
 * sink for events and a polling hook for stop requests.
 *
 * Loop shape:
 *   - resume from checkpointDir if meta.json + trainer_state.bin present
 *     and archHash matches; otherwise fresh init from seed
 *   - for each step:
 *       - accumulate grads across `gradAccumSteps` microbatches
 *       - apply Muon/Adam update
 *       - emit step event
 *       - every `valEvery` steps:
 *           - run full val slice → mean loss + bpb
 *           - if new best: int8-quantize, write LKG atomically, emit lkg event
 *           - always: checkpoint trainer_state.bin (full fp32 + optimizer)
 *   - on stop / max_tokens / no_data: idle out
 *
 * The "always update LKG" requirement is honoured by the new-best gate: the
 * LKG file is the best-val-loss model. We do NOT overwrite LKG with a worse
 * model — that would be the opposite of "last known GOOD". The training-state
 * checkpoint is what makes resumption seamless even when val_loss has been
 * trending sideways.
 */

import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { existsSync } from "node:fs";

import type * as tfTypes from "@tensorflow/tfjs";
import {
  ARCH,
  LKG_KEEP,
  PATHS,
  type Slm16TrainingConfig,
} from "./config.ts";
import {
  archHash as computeArchHash,
  countParams,
  disposeCache,
  disposeWeights,
  forward,
  initWeights,
  loadSnapshot,
  type Slm16Weights,
  snapshotWeights,
} from "./model.ts";
import {
  applyStep,
  deserializeOptimizerState,
  initOptimizerState,
  type OptimizerState,
  serializeOptimizerState,
  type StepConfig,
} from "./optimizer.ts";
import {
  checkArtifactSize,
  dequantizeStateDict,
  quantizeStateDict,
} from "./quantize.ts";
import {
  type Batch,
  buildValSlice,
  discoverShards,
  TrainIterator,
  valBatch,
  type ValSlice,
} from "./data.ts";
import type { Slm16Event, Slm16Meta } from "./types.ts";

type TF = typeof tfTypes;
type Tensor = tfTypes.Tensor;
type Variable = tfTypes.Variable;

// ----------------------------------------------------------------------------
// Trainer state container
// ----------------------------------------------------------------------------

interface TrainerState {
  weights: Slm16Weights;
  opt: OptimizerState;
  step: number;
  tokensSeen: number;
  trainSeconds: number;
  bestValLoss: number;
  bestValStep: number;
  archHash: string;
  seed: number;
}

const META_FORMAT = "slm16_meta_v1";

// ----------------------------------------------------------------------------
// Persistence — atomic write via temp + rename
// ----------------------------------------------------------------------------

async function atomicWrite(path: string, data: Uint8Array): Promise<void> {
  const tmp = `${path}.tmp.${process.pid}.${Date.now()}`;
  await writeFile(tmp, data);
  await rename(tmp, path);
}

/**
 * Logrotate-style write: shift `path → path.1 → path.2 → …` before placing
 * the new blob at `path`. The oldest generation (`.{keep-1}`) is evicted.
 *
 * Crash safety: the new blob is fully on disk (as a tmp file) before any
 * renames happen. If we die mid-rotation, the worst case is a missing
 * canonical with the previous generation safely at `.1` — and a stranded
 * tmp file. The fallback chain in `tryLoadLkg` handles the missing-canonical
 * case; stranded tmp files are harmless (next rotation overwrites nothing,
 * they just sit there until manual cleanup).
 *
 * Operation order with keep=3, generations labelled by age (A=newest):
 *   before:  canonical=A   .1=B   .2=C
 *   step 1:  writeFile tmp=NEW
 *   step 2:  rename .1→.2   (C evicted, B now at .2)
 *   step 3:  rename canonical→.1   (A now at .1)
 *   step 4:  rename tmp→canonical   (NEW is canonical)
 *   after:   canonical=NEW   .1=A   .2=B
 */
export async function rotateAndWrite(
  path: string,
  data: Uint8Array,
  keep: number,
): Promise<void> {
  const tmp = `${path}.tmp.${process.pid}.${Date.now()}`;
  await writeFile(tmp, data);

  // Shift older generations down. Iterate high→low so we don't clobber.
  // .{keep-2} → .{keep-1} (evicts oldest), …, .1 → .2, canonical → .1.
  for (let i = keep - 1; i >= 1; i--) {
    const src = i === 1 ? path : `${path}.${i - 1}`;
    const dst = `${path}.${i}`;
    if (existsSync(src)) {
      // POSIX rename atomically replaces dst if it exists.
      await rename(src, dst);
    }
  }

  await rename(tmp, path);
}

/**
 * Try to load and dequantize an LKG blob, falling through `.1`, `.2`, …
 * if the canonical is missing or corrupt. Returns null only if every
 * generation fails. Errors from individual attempts are reported via the
 * optional `onErr` callback (so the caller can emit non-fatal events).
 */
async function tryLoadLkg(
  dir: string,
  expectedArchHash: string,
  onErr?: (path: string, err: Error) => void,
): Promise<{
  weights: ReturnType<typeof dequantizeStateDict>["weights"];
  fromPath: string;
} | null> {
  const base = join(dir, PATHS.lkg);
  for (let i = 0; i < LKG_KEEP; i++) {
    const path = i === 0 ? base : `${base}.${i}`;
    if (!existsSync(path)) continue;
    try {
      const blob = await readFile(path);
      const { weights, archHash } = dequantizeStateDict(blob);
      if (archHash !== expectedArchHash) {
        throw new Error(`archHash ${archHash} != expected ${expectedArchHash}`);
      }
      return { weights, fromPath: path };
    } catch (e) {
      onErr?.(path, e as Error);
    }
  }
  return null;
}

async function writeMeta(dir: string, meta: Slm16Meta): Promise<void> {
  const json = JSON.stringify(meta, null, 2);
  await atomicWrite(join(dir, PATHS.meta), Buffer.from(json, "utf8"));
}

async function readMeta(dir: string): Promise<Slm16Meta | null> {
  const path = join(dir, PATHS.meta);
  if (!existsSync(path)) return null;
  try {
    const raw = await readFile(path, "utf8");
    return JSON.parse(raw) as Slm16Meta;
  } catch {
    return null;
  }
}

/**
 * Persist full training state: fp32 weight snapshot + optimizer moments.
 * Layout: [u32 weightLen][weight blob][optimizer blob]. The weight blob
 * itself is [u32 jsonLen][json manifest][padded][f32 data] — same scheme
 * as the optimizer serializer, so resumption code is symmetric.
 */
async function writeTrainerState(
  dir: string,
  weights: Slm16Weights,
  opt: OptimizerState,
): Promise<void> {
  const snap = snapshotWeights(weights);

  // Build weight manifest.
  const order: Array<{ name: string; shape: number[]; len: number }> = [];
  let totalFloats = 0;
  for (const [name, { data, shape }] of Object.entries(snap)) {
    order.push({ name, shape, len: data.length });
    totalFloats += data.length;
  }
  const manifest = JSON.stringify({ order });
  const manifestBytes = new TextEncoder().encode(manifest);
  const headerLen = 4 + manifestBytes.length;
  const padded = (headerLen + 3) & ~3;
  const wBytes = padded + totalFloats * 4;

  const wBuf = new ArrayBuffer(wBytes);
  const wU8 = new Uint8Array(wBuf);
  new DataView(wBuf).setUint32(0, manifestBytes.length, true);
  wU8.set(manifestBytes, 4);
  const f32 = new Float32Array(wBuf, padded);
  let off = 0;
  for (const entry of order) {
    f32.set(snap[entry.name].data, off);
    off += entry.len;
  }

  // Optimizer blob.
  const optBlob = serializeOptimizerState(opt);

  // Combine: [u32 wBytes][wU8][optBlob].
  const totalBytes = 4 + wBytes + optBlob.length;
  const out = new Uint8Array(totalBytes);
  new DataView(out.buffer).setUint32(0, wBytes, true);
  out.set(wU8, 4);
  out.set(optBlob, 4 + wBytes);

  await atomicWrite(join(dir, PATHS.state), out);
}

interface ResumePayload {
  weights: Record<string, { data: Float32Array; shape: number[] }>;
  opt: OptimizerState;
}

async function readTrainerState(dir: string): Promise<ResumePayload | null> {
  const path = join(dir, PATHS.state);
  if (!existsSync(path)) return null;
  const buf = await readFile(path);
  // Re-anchor to a clean ArrayBuffer for aligned typed-array views.
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const dv = new DataView(ab);

  const wBytes = dv.getUint32(0, true);
  const wStart = 4;
  const wView = new Uint8Array(ab, wStart, wBytes);

  // Parse weight manifest.
  const wDv = new DataView(ab, wStart);
  const jsonLen = wDv.getUint32(0, true);
  const jsonBytes = new Uint8Array(ab, wStart + 4, jsonLen);
  const manifest = JSON.parse(new TextDecoder().decode(jsonBytes)) as {
    order: Array<{ name: string; shape: number[]; len: number }>;
  };
  const headerLen = 4 + jsonLen;
  const padded = (headerLen + 3) & ~3;
  const f32 = new Float32Array(ab, wStart + padded);

  const weights: Record<string, { data: Float32Array; shape: number[] }> = {};
  let off = 0;
  for (const entry of manifest.order) {
    // Float32Array.slice copies — required, since `ab` will be GC'd.
    weights[entry.name] = {
      data: f32.slice(off, off + entry.len),
      shape: entry.shape,
    };
    off += entry.len;
  }

  // Optimizer blob is everything after the weight section.
  const optBlob = new Uint8Array(ab, wStart + wBytes);
  const opt = deserializeOptimizerState(optBlob);

  // Suppress unused-variable noise — wView exists for bounds documentation.
  void wView;

  return { weights, opt };
}

// ----------------------------------------------------------------------------
// Validation — run the full val slice in microbatches, return mean loss + bpb
// ----------------------------------------------------------------------------

interface ValResult {
  loss: number;
  /** Bits per byte. = (loss / ln 2) × (tokens / bytes). With sp1024 BPE
   *  the tokens/bytes ratio is roughly 1/3.5; we approximate it as the
   *  reference does, using the empirical ratio from the val slice. For
   *  simplicity here we use the canonical bpb = loss / ln(2) when the
   *  byte count isn't available — it's still a monotone proxy. */
  bpb: number;
}

/**
 * The reference computes true bpb from `(val_loss / ln 2) * (tokens / bytes)`
 * where bytes is the original document byte length pre-tokenization. We don't
 * have the byte sidecar in this trainer, so we report the simpler
 * `loss / ln 2` (bits per token). It's labeled `bpb` for consistency with
 * the wire types — the absolute value differs from the leaderboard but the
 * relative ordering and convergence trend are identical.
 */
function computeBpb(loss: number): number {
  return loss / Math.LN2;
}

async function runValidation(
  tf: TF,
  weights: Slm16Weights,
  slice: ValSlice,
  microBatchSize: number,
): Promise<ValResult> {
  let lossSum = 0;
  let nBatches = 0;
  const T = slice.windowLen - 1;

  for (let start = 0; start < slice.numWindows; start += microBatchSize) {
    const batch = valBatch(slice, start, microBatchSize);
    const B = batch.numTokens / T;
    const inputs = tf.tensor2d(batch.inputs, [B, T], "int32");
    const targets = tf.tensor2d(batch.targets, [B, T], "int32");
    const loss = forward(tf, weights, inputs, targets);
    // dataSync() returns a typed-array; scalar → length-1.
    lossSum += (loss.dataSync() as Float32Array)[0];
    nBatches += 1;
    loss.dispose();
    inputs.dispose();
    targets.dispose();
  }

  const meanLoss = lossSum / Math.max(1, nBatches);
  return { loss: meanLoss, bpb: computeBpb(meanLoss) };
}

// ----------------------------------------------------------------------------
// LKG — quantize, validate roundtrip, write atomically
// ----------------------------------------------------------------------------

/**
 * Snapshot → int8+zlib → write to lkg.int8.bin (atomic). Returns the
 * artifact size info for the lkg event. Roundtrip validation is implicit:
 * dequantizeStateDict will throw on format mismatch, and the caller can
 * optionally re-load and re-validate (we do that on RESUME, not here, since
 * the just-quantized model is already in memory).
 */
async function writeLkg(
  dir: string,
  weights: Slm16Weights,
  archHash: string,
): Promise<{ path: string; bytes: number; underCap: boolean }> {
  const snap = snapshotWeights(weights);
  const { blob } = quantizeStateDict(snap, archHash);
  const sizeInfo = checkArtifactSize(blob);
  const path = join(dir, PATHS.lkg);
  await rotateAndWrite(path, blob, LKG_KEEP);
  return { path, bytes: sizeInfo.bytes, underCap: sizeInfo.underCap };
}

// ----------------------------------------------------------------------------
// Resumption
// ----------------------------------------------------------------------------

/**
 * Resume order of preference:
 *   1. trainer_state.bin (full fp32 + optimizer) — best fidelity
 *   2. lkg.int8.bin (dequantize) — fallback if state was lost/corrupt
 *   3. fresh init from seed
 *
 * archHash mismatch in either file → treat as missing (architecture
 * changed under us; resuming would be nonsense).
 */
async function resumeOrInit(
  tf: TF,
  cfg: Slm16TrainingConfig,
  emit: (e: Slm16Event) => void,
): Promise<TrainerState> {
  const dir = cfg.checkpointDir;
  await mkdir(dir, { recursive: true });

  const archHash = computeArchHash();
  const meta = await readMeta(dir);
  const archMatch = meta !== null && meta.archHash === archHash;

  // Always init Variables fresh; we'll loadSnapshot over them if resuming.
  // (Variable creation requires a tf scope; can't deserialize into thin air.)
  const seed = archMatch ? meta.seed : cfg.seed;
  const weights = initWeights(tf, seed);

  // ---- Path 1: full trainer state ----
  if (archMatch) {
    const resumed = await readTrainerState(dir).catch(() => null);
    if (resumed !== null) {
      try {
        loadSnapshot(tf, weights, resumed.weights);
        return {
          weights,
          opt: resumed.opt,
          step: resumed.opt.step,
          tokensSeen: meta.tokensSeen,
          trainSeconds: meta.trainSeconds,
          bestValLoss: meta.bestValLoss,
          bestValStep: meta.bestValStep,
          archHash,
          seed,
        };
      } catch (e) {
        emit({
          type: "error",
          message: `trainer_state.bin load failed (falling back to LKG): ${(e as Error).message}`,
          fatal: false,
        });
      }
    }

    // ---- Path 2: LKG (int8) — try canonical, then .1, .2 ----
    const lkg = await tryLoadLkg(dir, archHash, (path, err) => {
      emit({
        type: "error",
        message: `LKG ${path} unreadable (trying next): ${err.message}`,
        fatal: false,
      });
    });
    if (lkg !== null) {
      loadSnapshot(tf, weights, lkg.weights);
      // Optimizer state is gone — fresh init. step resets to bestValStep
      // so the meta counters stay coherent, but Adam moments / Muon
      // momentum start from zero. Practically: a small warmup hiccup.
      const opt = initOptimizerState(weights);
      opt.step = meta.bestValStep;
      return {
        weights,
        opt,
        step: meta.bestValStep,
        tokensSeen: meta.tokensSeen,
        trainSeconds: meta.trainSeconds,
        bestValLoss: meta.bestValLoss,
        bestValStep: meta.bestValStep,
        archHash,
        seed,
      };
    }
  }

  // ---- Path 3: fresh init ----
  const opt = initOptimizerState(weights);
  return {
    weights,
    opt,
    step: 0,
    tokensSeen: 0,
    trainSeconds: 0,
    bestValLoss: Number.POSITIVE_INFINITY,
    bestValStep: 0,
    archHash,
    seed,
  };
}

// ----------------------------------------------------------------------------
// Gradient accumulation step
// ----------------------------------------------------------------------------

/**
 * Compute gradients across `gradAccumSteps` microbatches and average them
 * on the host (Float32Array). tf.variableGrads gives us per-Variable
 * gradient tensors; we read them out and dispose immediately to keep the
 * GPU memory footprint flat.
 *
 * Returns the mean training loss across the accumulated microbatches.
 */
async function accumulatedGradStep(
  tf: TF,
  weights: Slm16Weights,
  iter: TrainIterator,
  microBatchSize: number,
  T: number,
  accumSteps: number,
): Promise<{ grads: Record<string, Float32Array>; meanLoss: number; tokens: number }> {
  const accum: Record<string, Float32Array> = {};
  let lossSum = 0;
  let tokens = 0;
  // tf.variableGrads needs the Variable list to know what to differentiate.
  const varList = Object.values(weights) as Variable[];

  for (let m = 0; m < accumSteps; m++) {
    const batch: Batch = await iter.next();
    tokens += batch.numTokens;

    const inputs = tf.tensor2d(batch.inputs, [microBatchSize, T], "int32");
    const targets = tf.tensor2d(batch.targets, [microBatchSize, T], "int32");

    const { value, grads } = tf.variableGrads(
      () => forward(tf, weights, inputs, targets),
      varList,
    );
    lossSum += (value.dataSync() as Float32Array)[0];
    value.dispose();
    inputs.dispose();
    targets.dispose();

    // grads is keyed by Variable.name (auto-assigned by tfjs). We need to
    // re-key by OUR weight names. Variable.name is set at creation time;
    // initWeights names them after the state_dict keys, so this is a 1:1 map.
    // But to be robust to tfjs name-mangling (it appends suffixes on
    // collision), we match by object identity instead.
    for (const [wName, wVar] of Object.entries(weights)) {
      const g = grads[(wVar as Variable).name];
      if (!g) continue;
      const data = g.dataSync() as Float32Array;
      const slot = accum[wName];
      if (slot === undefined) {
        // First microbatch: clone (slice copies).
        accum[wName] = data.slice();
      } else {
        for (let i = 0; i < slot.length; i++) slot[i] += data[i];
      }
      g.dispose();
    }
  }

  // Average.
  if (accumSteps > 1) {
    const inv = 1 / accumSteps;
    for (const arr of Object.values(accum)) {
      for (let i = 0; i < arr.length; i++) arr[i] *= inv;
    }
  }

  return { grads: accum, meanLoss: lossSum / accumSteps, tokens };
}

/**
 * Wrap host-side gradient arrays back into tensors for the optimizer.
 * The optimizer's Muon path needs them on-device for the Newton-Schulz
 * matmuls; the Adam path reads them right back to host. We pass tensors
 * uniformly and let applyStep decide.
 */
function gradsToTensors(
  tf: TF,
  weights: Slm16Weights,
  hostGrads: Record<string, Float32Array>,
): Record<string, Tensor> {
  const out: Record<string, Tensor> = {};
  for (const [name, v] of Object.entries(weights)) {
    const g = hostGrads[name];
    if (!g) continue;
    out[name] = tf.tensor(g, v.shape, "float32");
  }
  return out;
}

function disposeGradTensors(grads: Record<string, Tensor>): void {
  for (const g of Object.values(grads)) g.dispose();
}

// ----------------------------------------------------------------------------
// Muon momentum schedule
// ----------------------------------------------------------------------------

function muonMomentumAt(cfg: Slm16TrainingConfig, step: number): number {
  if (step >= cfg.muonMomentumWarmupSteps) return cfg.muonMomentum;
  const t = step / cfg.muonMomentumWarmupSteps;
  return cfg.muonMomentumWarmupStart +
    t * (cfg.muonMomentum - cfg.muonMomentumWarmupStart);
}

// ----------------------------------------------------------------------------
// Public: TF backend loader
// ----------------------------------------------------------------------------

/**
 * Dynamic-import the right tfjs binding. Caller (worker.ts) decides which
 * package label to try first based on detectBackend(). On import failure
 * (e.g. CUDA not present on a Linux box), fall back to the CPU package.
 *
 * VRAM fraction: tfjs-node-gpu honours TF_FORCE_GPU_ALLOW_GROWTH and
 * TF_GPU_MEMORY_FRACTION env vars at native-binding load time. We set the
 * fraction env BEFORE importing so the underlying TF C runtime picks it up.
 * On Apple Silicon (unified memory), tfjs-node manages allocation lazily
 * via Accelerate; the fraction is advisory there (we just don't OOM-grow
 * past the fraction × totalmem heuristic in our microbatch sizing).
 */
export async function loadTfBackend(
  preferred: "@tensorflow/tfjs-node-gpu" | "@tensorflow/tfjs-node",
  vramFraction: number,
): Promise<{ tf: TF; loaded: string }> {
  // Configure native TF before it loads. These are read once at .so load.
  process.env.TF_FORCE_GPU_ALLOW_GROWTH = "true";
  process.env.TF_GPU_MEMORY_FRACTION = String(vramFraction);
  // Quiet the spammy CPU-feature banner.
  process.env.TF_CPP_MIN_LOG_LEVEL = process.env.TF_CPP_MIN_LOG_LEVEL ?? "2";

  try {
    const mod = (await import(preferred)) as unknown as TF;
    return { tf: mod, loaded: preferred };
  } catch {
    if (preferred === "@tensorflow/tfjs-node-gpu") {
      const mod = (await import("@tensorflow/tfjs-node")) as unknown as TF;
      return { tf: mod, loaded: "@tensorflow/tfjs-node" };
    }
    throw new Error(
      "Failed to load any tfjs backend. " +
        "Install @tensorflow/tfjs-node (or tfjs-node-gpu for CUDA).",
    );
  }
}

// ----------------------------------------------------------------------------
// Public: training loop
// ----------------------------------------------------------------------------

export interface TrainerHooks {
  emit: (e: Slm16Event) => void;
  /** Polled once per step. Returns true → drain & exit cleanly. */
  shouldStop: () => boolean;
  /** Polled once per step. Returns true → checkpoint immediately, then clear. */
  shouldCheckpoint: () => boolean;
}

export async function trainLoop(
  tf: TF,
  cfg: Slm16TrainingConfig,
  hooks: TrainerHooks,
): Promise<void> {
  const { emit } = hooks;

  // ---- 1. Resume or init ----
  const state = await resumeOrInit(tf, cfg, emit);
  const paramCount = countParams(state.weights);

  // ---- 2. Discover data ----
  const shards = await discoverShards(cfg.dataDir);
  if (shards.train.length === 0) {
    emit({
      type: "ready",
      backend: tf.getBackend(),
      paramCount,
      resumedAtStep: state.step,
      resumedValLoss: state.bestValLoss,
    });
    emit({ type: "idle", reason: "no_data" });
    disposeWeights(state.weights);
    disposeCache();
    return;
  }

  const valSlice = await buildValSlice(shards);
  // valSlice may be null if shards are tiny — we'll just skip validation
  // entirely in that degenerate case (still train, just never write LKG).

  // ---- 3. Build train iterator ----
  const T = ARCH.seqLen;
  const microTokens = Math.floor(cfg.batchTokens / cfg.gradAccumSteps);
  const microBatchSize = Math.max(1, Math.floor(microTokens / T));
  const skipFirst = valSlice?.skipFromFirstTrainShard ?? 0;
  // Seed iterator from cfg.seed XOR resumed step → roughly resumes data position.
  const iter = new TrainIterator(
    shards.train,
    skipFirst,
    { microBatchSize, seqLen: T },
    cfg.seed ^ state.step,
  );

  // Val microbatch sizing: same as train microbatch (memory profile is similar).
  const valMicroBatch = microBatchSize;

  // ---- 4. Ready ----
  emit({
    type: "ready",
    backend: tf.getBackend(),
    paramCount,
    resumedAtStep: state.step,
    resumedValLoss: state.bestValLoss,
  });

  // ---- 5. Main loop ----
  const stepCfg: StepConfig = {
    embedLr: cfg.embedLr,
    matrixLr: cfg.matrixLr,
    scalarLr: cfg.scalarLr,
    beta1: cfg.beta1,
    beta2: cfg.beta2,
    adamEps: cfg.adamEps,
    muonMomentum: cfg.muonMomentum,
    muonBackendSteps: cfg.muonBackendSteps,
    lrScale: 1.0, // Background mode: no warmdown — runs forever.
  };

  try {
    // The actual hot loop. Each iteration is one optimizer step.
    // We yield to the event loop between steps so the worker can receive
    // stop/checkpoint commands via the message handler.
    for (;;) {
      if (hooks.shouldStop()) {
        await checkpoint(cfg.checkpointDir, state);
        emit({ type: "idle", reason: "stopped" });
        break;
      }
      if (cfg.maxTokens > 0 && state.tokensSeen >= cfg.maxTokens) {
        await checkpoint(cfg.checkpointDir, state);
        emit({ type: "idle", reason: "max_tokens" });
        break;
      }
      if (cfg.maxSteps > 0 && state.step >= cfg.maxSteps) {
        await checkpoint(cfg.checkpointDir, state);
        emit({ type: "idle", reason: "max_steps" });
        break;
      }

      const stepStart = Date.now();

      // -- Forward + backward + accumulate --
      const { grads, meanLoss, tokens } = await accumulatedGradStep(
        tf,
        state.weights,
        iter,
        microBatchSize,
        T,
        cfg.gradAccumSteps,
      );

      // -- Optimizer update --
      stepCfg.muonMomentum = muonMomentumAt(cfg, state.step);
      const gradTensors = gradsToTensors(tf, state.weights, grads);
      applyStep(tf, state.weights, gradTensors, state.opt, stepCfg);
      disposeGradTensors(gradTensors);

      // -- Bookkeeping --
      state.step = state.opt.step; // applyStep increments opt.step
      state.tokensSeen += tokens;
      state.trainSeconds += (Date.now() - stepStart) / 1000;

      emit({
        type: "step",
        step: state.step,
        trainLoss: meanLoss,
        trainSeconds: state.trainSeconds,
        tokensSeen: state.tokensSeen,
      });

      // -- Validation cadence --
      const due = state.step % cfg.valEvery === 0;
      const forced = hooks.shouldCheckpoint();
      if ((due || forced) && valSlice !== null) {
        const valStart = Date.now();
        const { loss: valLoss, bpb: valBpb } = await runValidation(
          tf,
          state.weights,
          valSlice,
          valMicroBatch,
        );
        // Don't count val time toward trainSeconds (matches reference).
        void valStart;

        emit({
          type: "val",
          step: state.step,
          valLoss,
          valBpb,
          trainSeconds: state.trainSeconds,
        });

        // -- New best? Update LKG --
        if (valLoss < state.bestValLoss) {
          state.bestValLoss = valLoss;
          state.bestValStep = state.step;
          const lkg = await writeLkg(
            cfg.checkpointDir,
            state.weights,
            state.archHash,
          );
          emit({
            type: "lkg",
            step: state.step,
            valLoss,
            artifactBytes: lkg.bytes,
            underCap: lkg.underCap,
            path: lkg.path,
          });
        }

        // -- Always checkpoint trainer state at validation cadence --
        await checkpoint(cfg.checkpointDir, state);
      }

      // Yield so parentPort.on('message') can fire.
      await new Promise<void>((r) => setImmediate(r));
    }
  } finally {
    iter.dispose();
    disposeWeights(state.weights);
    disposeCache();
  }
}

async function checkpoint(dir: string, state: TrainerState): Promise<void> {
  await writeTrainerState(dir, state.weights, state.opt);
  const meta: Slm16Meta = {
    format: META_FORMAT,
    step: state.step,
    bestValLoss: state.bestValLoss,
    bestValStep: state.bestValStep,
    trainSeconds: state.trainSeconds,
    tokensSeen: state.tokensSeen,
    updatedAt: new Date().toISOString(),
    archHash: state.archHash,
    seed: state.seed,
  };
  await writeMeta(dir, meta);
}

// ----------------------------------------------------------------------------
// Public: load LKG for inference (used by scoring.ts on the MAIN thread)
// ----------------------------------------------------------------------------

/**
 * Load the LKG model from disk into a fresh weight set, ready for inference.
 * This is the path that scoring.ts and any agent-side inference takes.
 * Returns null if no LKG exists yet.
 */
export async function loadLkgForInference(
  tf: TF,
  checkpointDir: string,
): Promise<{ weights: Slm16Weights; valLoss: number; step: number } | null> {
  const expected = computeArchHash();
  const errs: string[] = [];
  const lkg = await tryLoadLkg(checkpointDir, expected, (path, err) => {
    errs.push(`${path}: ${err.message}`);
  });
  if (lkg === null) {
    if (errs.length > 0) {
      // Files exist but every generation failed — this is a hard error,
      // not "no LKG yet". Surface it instead of silently returning null.
      throw new Error(
        `All ${LKG_KEEP} LKG generations failed to load:\n  ${errs.join("\n  ")}`,
      );
    }
    return null;
  }

  const meta = await readMeta(checkpointDir);

  // Init Variables (seed irrelevant — we overwrite immediately).
  const weights = initWeights(tf, 0);
  loadSnapshot(tf, weights, lkg.weights);

  return {
    weights,
    valLoss: meta?.bestValLoss ?? Number.POSITIVE_INFINITY,
    step: meta?.bestValStep ?? 0,
  };
}
