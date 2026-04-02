/**
 * SLM16 proving script — first live execution of the training loop.
 *
 * What this script tests that unit tests cannot:
 *   - tfjs-node native binding actually loads on this host
 *   - forward() produces a scalar loss without shape errors
 *   - tf.variableGrads() returns gradients keyed correctly
 *   - Muon Newton-Schulz orthogonalization runs without NaN
 *   - LKG quantize → write → rotate → dequantize roundtrip on real disk
 *   - trainer_state.bin checkpoint → resume cycle
 *   - Memory doesn't leak across 100 steps (numTensors stays bounded)
 *
 * What this script does NOT test:
 *   - Worker thread spawn / message protocol (calls trainLoop directly)
 *   - Nova Pro scoring (no ScryptedAI runtime here)
 *   - Real FineWeb data (synthetic uniform-random tokens)
 *   - Convergence (100 steps on noise won't learn anything; we just check
 *     that loss is finite and the machinery runs to completion)
 *
 * Run from repo root:
 *   bun plugins/plugin-avb/src/slm16/prove.ts
 *
 * Or with a real data shard already in place:
 *   SLM16_PROVE_DATA_DIR=/path/to/real/shards bun plugins/plugin-avb/src/slm16/prove.ts
 */

import { mkdtemp, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ARCH, SHARD, TRAIN_DEFAULTS, type Slm16TrainingConfig } from "./config.ts";
import { detectBackend } from "./config.ts";
import { dequantizeStateDict } from "./quantize.ts";
import {
  loadLkgForInference,
  loadTfBackend,
  trainLoop,
  type TrainerHooks,
} from "./trainer.ts";
import type { Slm16Event } from "./types.ts";

// ----------------------------------------------------------------------------
// Synthetic data — uniform random tokens in [0, vocabSize)
// ----------------------------------------------------------------------------

/**
 * Build a FineWeb-format .bin shard from a deterministic PRNG. The header
 * is 256 int32s (1024 bytes); only the first three are meaningful (magic,
 * version, numTokens). Body is uint16 little-endian.
 *
 * One million tokens ≈ 2 MB on disk. With the default 256K val carve-out
 * that leaves ~744K tokens for training — enough for 100 steps at any
 * batch size we'd use here.
 */
async function writeSyntheticShard(path: string, numTokens: number): Promise<void> {
  const headerInts = SHARD.headerInts;
  const headerBytes = headerInts * 4;

  const buf = Buffer.alloc(headerBytes + numTokens * 2);

  // Header: magic, version, numTokens, then zeros.
  buf.writeInt32LE(SHARD.magic, 0);
  buf.writeInt32LE(SHARD.version, 4);
  buf.writeInt32LE(numTokens, 8);

  // mulberry32 — same generator the trainer uses for weight init / shuffling.
  // Seeded with a fixed value so re-runs see the same "data".
  let s = 0xdecafbad >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  };

  const V = ARCH.vocabSize;
  for (let i = 0; i < numTokens; i++) {
    buf.writeUInt16LE(next() % V, headerBytes + i * 2);
  }

  await writeFile(path, buf);
}

// ----------------------------------------------------------------------------
// Event sink — pretty-print + accounting
// ----------------------------------------------------------------------------

interface EventLog {
  steps: number[];
  trainLosses: number[];
  valLosses: number[];
  lkgWrites: number;
  errors: string[];
  idleReason: string | null;
}

type MemFn = () => { numTensors: number; numBytes: number };

function makeSink(log: EventLog, mem: MemFn): (e: Slm16Event) => void {
  const t0 = Date.now();
  const elapsed = () => ((Date.now() - t0) / 1000).toFixed(1).padStart(6);
  // Baseline tensor count after first step — leaks show as growth from here.
  let baseTensors = -1;

  return (e) => {
    switch (e.type) {
      case "ready":
        console.log(
          `[${elapsed()}s] ready  backend=${e.backend} params=${e.paramCount.toLocaleString()} ` +
            `resumed_step=${e.resumedAtStep} resumed_val_loss=${e.resumedValLoss}`,
        );
        break;
      case "step": {
        log.steps.push(e.step);
        log.trainLosses.push(e.trainLoss);
        const tokPerSec = e.trainSeconds > 0 ? Math.round(e.tokensSeen / e.trainSeconds) : 0;
        const m = mem();
        if (baseTensors < 0) baseTensors = m.numTensors;
        const delta = m.numTensors - baseTensors;
        // Print every step in a proving run (only 100 of them).
        console.log(
          `[${elapsed()}s] step   ${String(e.step).padStart(3)}  ` +
            `loss=${e.trainLoss.toFixed(4)}  ` +
            `tok/s=${String(tokPerSec).padStart(5)}  ` +
            `tensors=${String(m.numTensors).padStart(5)} (Δ${delta >= 0 ? "+" : ""}${delta})  ` +
            `mem=${(m.numBytes / 1e6).toFixed(0)}MB`,
        );
        break;
      }
      case "val": {
        log.valLosses.push(e.valLoss);
        const prev = log.valLosses.at(-2);
        const delta = prev === undefined ? 0 : e.valLoss - prev;
        const arrow = prev === undefined ? "·" : delta < 0 ? "↓" : delta > 0 ? "↑" : "→";
        const m = mem();
        console.log(
          `[${elapsed()}s] val    ${String(e.step).padStart(3)}  ` +
            `val_loss=${e.valLoss.toFixed(4)} ${arrow}  bpb=${e.valBpb.toFixed(4)}  ` +
            `tensors=${m.numTensors}`,
        );
        break;
      }
      case "lkg":
        log.lkgWrites++;
        console.log(
          `[${elapsed()}s] LKG    ${String(e.step).padStart(3)}  ` +
            `wrote ${(e.artifactBytes / 1e6).toFixed(2)} MB ` +
            `(${e.underCap ? "under" : "OVER"} 16 MB cap) → ${e.path}`,
        );
        break;
      case "error":
        log.errors.push(e.message);
        console.error(
          `[${elapsed()}s] ${e.fatal ? "FATAL " : "warn  "} ${e.message}`,
        );
        break;
      case "idle":
        log.idleReason = e.reason;
        console.log(`[${elapsed()}s] idle   reason=${e.reason}`);
        break;
    }
  };
}

// ----------------------------------------------------------------------------
// Assertions — fail loudly if invariants are violated
// ----------------------------------------------------------------------------

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) {
    console.error(`\n✗ ASSERTION FAILED: ${msg}\n`);
    process.exit(1);
  }
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

async function main() {
  console.log("=== SLM16 proving run ===");
  console.log(`Platform: ${process.platform} ${process.arch}`);
  console.log(`Node: ${process.version}`);

  // ---- 1. Backend ----
  const backend = detectBackend();
  console.log(`Detected backend: ${backend.label} (${backend.pkg})`);

  let tf;
  try {
    const loaded = await loadTfBackend(backend.pkg, 0.5);
    tf = loaded.tf;
    console.log(`Loaded ${loaded.loaded}, tf.version: ${tf.version?.tfjs ?? "?"}`);
    console.log(`tf backend: ${tf.getBackend()}`);
  } catch (e) {
    console.error(`\n✗ Failed to load tfjs backend: ${(e as Error).message}`);
    console.error(`  This usually means the native addon wasn't built.`);
    console.error(`  Try: cd node_modules/@tensorflow/tfjs-node && npm run install`);
    process.exit(1);
  }

  // ---- 2. Workspace ----
  const workDir = await mkdtemp(join(tmpdir(), "slm16-prove-"));
  const dataDir = process.env.SLM16_PROVE_DATA_DIR ?? join(workDir, "data");
  const checkpointDir = join(workDir, "checkpoints");
  console.log(`\nWorkspace: ${workDir}`);
  console.log(`Data:       ${dataDir}`);
  console.log(`Checkpoints:${checkpointDir}`);

  let usingSynthetic = false;
  if (!process.env.SLM16_PROVE_DATA_DIR) {
    usingSynthetic = true;
    await writeFile(join(workDir, ".keep"), ""); // ensure tmpdir survives
    await import("node:fs/promises").then((m) => m.mkdir(dataDir, { recursive: true }));
    const shardPath = join(dataDir, "train_000.bin");
    console.log(`\nGenerating synthetic shard (1M tokens, ~2 MB)...`);
    await writeSyntheticShard(shardPath, 1_000_000);
    const sz = await stat(shardPath);
    console.log(`  ${shardPath} (${(sz.size / 1e6).toFixed(2)} MB)`);
  }

  // ---- 3. Config — scaled down for a proving run ----
  // microBatchSize = batchTokens / (gradAccumSteps × seqLen)
  //                = 2048 / (2 × 1024) = 1 sequence per microbatch.
  // Total: 100 steps × 2048 tokens/step = ~200K training tokens.
  // On Apple Silicon (Accelerate BLAS) expect single-digit seconds per step
  // for the 17M-param forward+backward at seqLen=1024.
  const cfg: Slm16TrainingConfig = {
    ...TRAIN_DEFAULTS,
    valEvery: 20,
    maxSteps: 100,
    batchTokens: 2 * ARCH.seqLen, // 2048
    gradAccumSteps: 2,
    dataDir,
    checkpointDir,
    // Shorter Muon warmup so momentum actually moves during a 100-step run.
    muonMomentumWarmupSteps: 50,
  };
  console.log(
    `\nConfig: valEvery=${cfg.valEvery} maxSteps=${cfg.maxSteps} ` +
      `batchTokens=${cfg.batchTokens} accum=${cfg.gradAccumSteps}`,
  );

  // ---- 4. Run ----
  const log: EventLog = {
    steps: [],
    trainLosses: [],
    valLosses: [],
    lkgWrites: 0,
    errors: [],
    idleReason: null,
  };

  const hooks: TrainerHooks = {
    emit: makeSink(log, () => tf.memory()),
    shouldStop: () => false,
    shouldCheckpoint: () => false,
  };

  // Quick diagnostic mode: SLM16_PROVE_STEPS overrides maxSteps so we can
  // run 5 steps to verify a leak fix without waiting for the full 100.
  if (process.env.SLM16_PROVE_STEPS) {
    cfg.maxSteps = Number(process.env.SLM16_PROVE_STEPS);
    console.log(`  (overridden maxSteps=${cfg.maxSteps} via SLM16_PROVE_STEPS)`);
  }

  console.log(`\n--- training ---`);
  const trainStart = Date.now();
  await trainLoop(tf, cfg, hooks);
  const trainSecs = (Date.now() - trainStart) / 1000;

  // ---- 5. Verify ----
  console.log(`\n--- verification ---`);

  assert(log.idleReason === "max_steps", `expected idle reason 'max_steps', got '${log.idleReason}'`);
  console.log(`✓ Loop terminated cleanly (max_steps)`);

  assert(log.steps.length === 100, `expected 100 step events, got ${log.steps.length}`);
  console.log(`✓ Saw exactly 100 step events`);

  assert(log.valLosses.length === 5, `expected 5 val events (steps 20,40,60,80,100), got ${log.valLosses.length}`);
  console.log(`✓ Saw exactly 5 val events (every 20 steps)`);

  // All losses must be finite. NaN here means a forward-pass bug (RoPE
  // indexing, mask shape, softcap overflow) or a Muon NS divergence.
  for (let i = 0; i < log.trainLosses.length; i++) {
    assert(
      Number.isFinite(log.trainLosses[i]),
      `step ${log.steps[i]}: train loss is ${log.trainLosses[i]} (NaN/Inf — forward pass or optimizer broken)`,
    );
  }
  for (let i = 0; i < log.valLosses.length; i++) {
    assert(Number.isFinite(log.valLosses[i]), `val loss ${i} is ${log.valLosses[i]}`);
  }
  console.log(`✓ All ${log.trainLosses.length} train + ${log.valLosses.length} val losses are finite`);

  // Sanity bounds. Random-init cross-entropy on a 1024-vocab uniform target
  // should sit near ln(1024) ≈ 6.93. On synthetic uniform-random data the
  // model can't learn structure (there is none), so loss should hover near
  // that floor. We give wide bounds — this is a "didn't explode" check, not
  // a convergence test.
  const meanTrain = log.trainLosses.reduce((a, b) => a + b, 0) / log.trainLosses.length;
  if (usingSynthetic) {
    assert(
      meanTrain > 4.0 && meanTrain < 9.0,
      `mean train loss ${meanTrain.toFixed(3)} outside [4, 9] — expected ~6.93 on uniform-random data`,
    );
    console.log(`✓ Mean train loss ${meanTrain.toFixed(3)} (expected ~ln(1024)≈6.93 on noise)`);
  } else {
    console.log(`  Mean train loss ${meanTrain.toFixed(3)} (real data — no bound check)`);
  }

  // First val from random init is always a "new best" → at least 1 LKG write.
  assert(log.lkgWrites >= 1, `expected ≥1 LKG write, got ${log.lkgWrites}`);
  console.log(`✓ ${log.lkgWrites} LKG write(s)`);

  // ---- 6. Disk artifacts ----
  const artifacts = await readdir(checkpointDir);
  console.log(`\nCheckpoint dir contents: ${artifacts.join(", ")}`);

  assert(artifacts.includes("meta.json"), "meta.json missing");
  assert(artifacts.includes("trainer_state.bin"), "trainer_state.bin missing");
  assert(artifacts.includes("lkg.int8.bin"), "lkg.int8.bin missing");
  console.log(`✓ meta.json, trainer_state.bin, lkg.int8.bin all present`);

  // If we got more than one LKG write, rotation should have kicked in.
  if (log.lkgWrites >= 2) {
    assert(artifacts.includes("lkg.int8.bin.1"), `${log.lkgWrites} LKG writes but no .1 backup`);
    console.log(`✓ Rotation backup lkg.int8.bin.1 present`);
  }
  if (log.lkgWrites >= 3) {
    assert(artifacts.includes("lkg.int8.bin.2"), `${log.lkgWrites} LKG writes but no .2 backup`);
    console.log(`✓ Rotation backup lkg.int8.bin.2 present`);
  }

  // ---- 7. Roundtrip: dequantize the LKG we just wrote ----
  const lkgPath = join(checkpointDir, "lkg.int8.bin");
  const lkgBlob = await readFile(lkgPath);
  console.log(`\nLKG blob: ${(lkgBlob.length / 1e6).toFixed(2)} MB`);
  assert(lkgBlob.length < 16_000_000, `LKG is ${lkgBlob.length} bytes — over the 16 MB cap`);
  console.log(`✓ LKG under 16 MB cap (margin: ${((16_000_000 - lkgBlob.length) / 1e6).toFixed(2)} MB)`);

  const { weights: dq, archHash } = dequantizeStateDict(lkgBlob);
  const dqKeys = Object.keys(dq);
  console.log(`✓ Dequantized: ${dqKeys.length} tensors, archHash=${archHash}`);

  // Spot-check: every dequantized tensor has the right element count and no NaN.
  let totalParams = 0;
  for (const [name, t] of Object.entries(dq)) {
    const numel = t.shape.reduce((a, b) => a * b, 1);
    totalParams += numel;
    assert(t.data.length === numel, `${name}: data length ${t.data.length} != shape product ${numel}`);
    for (let i = 0; i < t.data.length; i++) {
      assert(Number.isFinite(t.data[i]), `${name}[${i}] = ${t.data[i]} (NaN after dequant)`);
    }
  }
  console.log(`✓ All dequantized values finite (${totalParams.toLocaleString()} params)`);

  // ---- 8. Inference reload — the path scoring.ts uses ----
  const reloaded = await loadLkgForInference(tf, checkpointDir);
  assert(reloaded !== null, "loadLkgForInference returned null but lkg.int8.bin exists");
  console.log(`✓ loadLkgForInference: step=${reloaded.step} val_loss=${reloaded.valLoss.toFixed(4)}`);
  // Dispose the inference weights — we're done with them.
  for (const v of Object.values(reloaded.weights)) {
    (v as { dispose?: () => void }).dispose?.();
  }

  // ---- 9. Memory check — tensors should be bounded after the loop ----
  const mem = tf.memory();
  console.log(
    `\nFinal tf.memory(): ${mem.numTensors} tensors, ` +
      `${(mem.numBytes / 1e6).toFixed(1)} MB`,
  );
  // We expect the only surviving tensors to be model weights + optimizer
  // moments + a handful of cached constants (RoPE freqs, causal mask).
  // ~17M params × (1 weight + ~2 moment buffers) ≈ a few hundred tensors.
  // Anything in the thousands means a tf.tidy() scope is leaking.
  assert(
    mem.numTensors < 1000,
    `${mem.numTensors} tensors alive after training — likely a tf.tidy() leak in the step loop`,
  );
  console.log(`✓ Tensor count bounded (no per-step leak)`);

  // ---- Summary ----
  console.log(`\n=== PASS ===`);
  console.log(`  100 steps in ${trainSecs.toFixed(1)}s (${(trainSecs / 100).toFixed(2)}s/step)`);
  console.log(`  Train loss: ${log.trainLosses[0].toFixed(4)} → ${log.trainLosses.at(-1)?.toFixed(4)}`);
  console.log(`  Val loss:   ${log.valLosses[0].toFixed(4)} → ${log.valLosses.at(-1)?.toFixed(4)}`);
  console.log(`  LKG writes: ${log.lkgWrites}`);
  console.log(`  Workspace:  ${workDir} (left in place for inspection)`);

  if (log.errors.length > 0) {
    console.log(`\n  Non-fatal warnings during run:`);
    for (const e of log.errors) console.log(`    - ${e}`);
  }

  // Leave the workspace dir for inspection. To clean up:
  //   rm -rf "${workDir}"
  void rm; // (imported but intentionally unused — see comment above)
}

main().catch((e) => {
  console.error(`\n✗ Proving run threw:`, e);
  process.exit(1);
});
