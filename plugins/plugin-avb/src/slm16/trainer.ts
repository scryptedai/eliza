/**
 * SLM16 training loop.
 *
 * Glues model + optimizer + data together and runs the long-lived
 * train/validate/save-LKG cycle. Designed to be hosted inside a
 * `worker_threads.Worker` (see `worker.ts`) so the agent's main thread
 * stays responsive while the GPU spins.
 *
 * Behaviour mirrors openai/parameter-golf train_gpt.py at small scale:
 *   - tf.variableGrads over model.forward(x, y)
 *   - Muon+Adam step with linear-warmdown lrScale
 *   - validate every VAL_EVERY_STEPS over the held-out FineWeb split
 *   - on val_loss improvement: int8-quantize + zlib, assert <16 MiB,
 *     atomically replace the LKG checkpoint + meta JSON
 *   - resumable: on construct, if an LKG exists, load weights + step +
 *     loader cursor and carry on
 */

import * as fs from "node:fs";
import * as path from "node:path";

import type { TF } from "./backend.ts";
import {
  ARTIFACT_BYTE_LIMIT,
  DEFAULT_HPARAMS,
  LKG_FILENAME,
  LKG_KEEP_N,
  LKG_META_FILENAME,
  LKG_VERSIONED_RE,
  lkgVersionedFilename,
  lkgVersionedMetaFilename,
  REPORT_EVERY_STEPS,
  type Slm16Hyperparameters,
  TRAIN_GLOB,
  VAL_EVERY_STEPS,
  VAL_GLOB,
} from "./constants.ts";
import {
  type Batch,
  listShards,
  loadValidationTokens,
  TokenLoader,
} from "./data.ts";
import { Slm16Model } from "./model.ts";
import { Slm16Optimizer } from "./optimizer.ts";
import { deserializeInt8, type StateDict, serializeInt8 } from "./quantize.ts";

type Tensor = import("@tensorflow/tfjs").Tensor;
type Variable = import("@tensorflow/tfjs").Variable;

// ----------------------------------------------------------------------------
// Reporting
// ----------------------------------------------------------------------------

export type TrainerReport =
  | {
      type: "boot";
      step: number;
      backend: string;
      paramCount: number;
      resumed: boolean;
      bestValLoss: number;
    }
  | { type: "step"; step: number; trainLoss: number; lrScale: number }
  | {
      type: "validation";
      step: number;
      valLoss: number;
      bestValLoss: number;
      improved: boolean;
    }
  | {
      type: "checkpoint";
      step: number;
      valLoss: number;
      artifactBytes: number;
      paramCount: number;
      path: string;
    }
  | { type: "error"; step: number; message: string };

export type ReportFn = (r: TrainerReport) => void;

// ----------------------------------------------------------------------------
// LKG meta
// ----------------------------------------------------------------------------

export interface LkgHistoryEntry {
  step: number;
  valLoss: number;
  savedAt: string;
  file: string;
  metaFile: string;
}

export interface LkgMeta {
  step: number;
  valLoss: number;
  bestValLoss: number;
  hparams: Slm16Hyperparameters;
  loaderState: { fileIdx: number; pos: number };
  savedAt: string;
  artifactBytes: number;
  paramCount: number;
  /** Newest-first list of the versioned checkpoints currently on disk. */
  history: LkgHistoryEntry[];
}

export interface TrainerOptions {
  tf: TF;
  backendName: string;
  hp?: Slm16Hyperparameters;
  dataDir: string;
  ckptDir: string;
  valEvery?: number;
  reportEvery?: number;
  onReport?: ReportFn;
}

// ----------------------------------------------------------------------------

function atomicWrite(file: string, data: Buffer | string): void {
  const tmp = `${file}.tmp-${process.pid}-${Date.now()}`;
  fs.writeFileSync(tmp, data);
  fs.renameSync(tmp, file);
}

/** Return versioned LKG checkpoints in `dir`, sorted newest-step-first. */
export function listVersionedLkg(
  dir: string,
): Array<{ step: number; file: string; metaFile: string }> {
  if (!fs.existsSync(dir)) return [];
  const out: Array<{ step: number; file: string; metaFile: string }> = [];
  for (const f of fs.readdirSync(dir)) {
    const m = LKG_VERSIONED_RE.exec(f);
    if (!m) continue;
    const step = Number.parseInt(m[1], 10);
    out.push({
      step,
      file: path.join(dir, f),
      metaFile: path.join(dir, lkgVersionedMetaFilename(step)),
    });
  }
  return out.sort((a, b) => b.step - a.step);
}

function lrSchedule(step: number, warmdown: number): number {
  // Linear warmdown to 0 over the final `warmdown` iters; flat 1.0 before that.
  // We don't know total iters in an open-ended run, so we treat warmdown as a
  // soft floor: scale = max(0.05, 1 - step/(10*warmdown)). Keeps Muon stable
  // while still decaying as the run lengthens.
  const frac = step / (10 * Math.max(1, warmdown));
  return Math.max(0.05, 1 - frac);
}

// ----------------------------------------------------------------------------

export class Slm16Trainer {
  private readonly tf: TF;
  private readonly hp: Slm16Hyperparameters;
  private readonly model: Slm16Model;
  private readonly opt: Slm16Optimizer;
  private readonly loader: TokenLoader;
  private readonly valTokens: Int32Array;
  private readonly ckptDir: string;
  private readonly lkgPath: string;
  private readonly metaPath: string;
  private readonly valEvery: number;
  private readonly reportEvery: number;
  private readonly report: ReportFn;
  private readonly backendName: string;

  private bestValLoss = Number.POSITIVE_INFINITY;
  private stopRequested = false;

  constructor(opts: TrainerOptions) {
    this.tf = opts.tf;
    this.hp = opts.hp ?? DEFAULT_HPARAMS;
    this.backendName = opts.backendName;
    this.ckptDir = opts.ckptDir;
    this.lkgPath = path.join(this.ckptDir, LKG_FILENAME);
    this.metaPath = path.join(this.ckptDir, LKG_META_FILENAME);
    this.valEvery = opts.valEvery ?? VAL_EVERY_STEPS;
    this.reportEvery = opts.reportEvery ?? REPORT_EVERY_STEPS;
    this.report = opts.onReport ?? (() => {});

    fs.mkdirSync(this.ckptDir, { recursive: true });

    // Data
    const trainFiles = listShards(opts.dataDir, TRAIN_GLOB);
    const valFiles = listShards(opts.dataDir, VAL_GLOB);
    if (trainFiles.length === 0) {
      throw new Error(
        `SLM16: no training shards matching ${TRAIN_GLOB} in ${opts.dataDir}`,
      );
    }
    if (valFiles.length === 0) {
      throw new Error(
        `SLM16: no validation shards matching ${VAL_GLOB} in ${opts.dataDir}`,
      );
    }
    this.loader = new TokenLoader(trainFiles);
    this.valTokens = loadValidationTokens(valFiles, this.hp.trainSeqLen);

    // Model + optimizer
    this.model = new Slm16Model(this.tf, this.hp);
    this.opt = new Slm16Optimizer(this.tf, this.hp, this.model.paramGroups());

    // Resume
    const resumed = this.tryResume();

    this.report({
      type: "boot",
      step: this.opt.step_t,
      backend: this.backendName,
      paramCount: this.model.paramCount(),
      resumed,
      bestValLoss: this.bestValLoss,
    });
  }

  // --------------------------------------------------------------------------

  /**
   * Resume from disk. Tries the canonical pointer first; if that fails
   * (corruption / partial write / schema drift) walks the versioned
   * rotation set newest→oldest. Only returns false if every candidate
   * is unusable.
   */
  private tryResume(): boolean {
    const candidates: Array<{ blob: string; meta: string; label: string }> = [];
    if (fs.existsSync(this.lkgPath) && fs.existsSync(this.metaPath)) {
      candidates.push({
        blob: this.lkgPath,
        meta: this.metaPath,
        label: "canonical",
      });
    }
    for (const v of listVersionedLkg(this.ckptDir)) {
      if (fs.existsSync(v.metaFile)) {
        candidates.push({
          blob: v.file,
          meta: v.metaFile,
          label: `step ${v.step}`,
        });
      }
    }
    if (candidates.length === 0) return false;

    for (const c of candidates) {
      try {
        const meta = JSON.parse(fs.readFileSync(c.meta, "utf8")) as LkgMeta;
        const sd = deserializeInt8(fs.readFileSync(c.blob));
        this.model.loadStateDict(sd);
        this.opt.restoreStep(meta.step);
        this.loader.restore(meta.loaderState);
        this.bestValLoss = meta.bestValLoss;
        if (c.label !== "canonical") {
          this.report({
            type: "error",
            step: meta.step,
            message: `Canonical LKG unusable — recovered from rotation (${c.label})`,
          });
        }
        return true;
      } catch (err) {
        this.report({
          type: "error",
          step: 0,
          message: `Resume candidate '${c.label}' rejected: ${(err as Error).message}`,
        });
      }
    }
    return false;
  }

  // --------------------------------------------------------------------------

  private gradsFor(batch: Batch): {
    loss: number;
    grads: Map<Variable, Tensor>;
  } {
    const { tf } = this;
    const x = tf.tensor2d(batch.x, [batch.bsz, batch.seqLen], "int32");
    const y = tf.tensor2d(batch.y, [batch.bsz, batch.seqLen], "int32");
    const vars = Array.from(this.model.weights.values());
    const { value, grads } = tf.variableGrads(
      () => this.model.forward(x, y) as import("@tensorflow/tfjs").Scalar,
      vars,
    );
    const loss = (value.dataSync() as Float32Array)[0];
    value.dispose();
    x.dispose();
    y.dispose();
    // tf.variableGrads keys grads by variable.name; remap to Variable instances.
    const out = new Map<Variable, Tensor>();
    for (const v of vars) {
      const g = grads[v.name];
      if (g) out.set(v, g);
    }
    return { loss, grads: out };
  }

  private disposeGrads(grads: Map<Variable, Tensor>): void {
    for (const g of grads.values()) g.dispose();
  }

  // --------------------------------------------------------------------------

  private async validate(): Promise<number> {
    const { tf, hp } = this;
    const T = hp.trainSeqLen;
    const nSeq = Math.floor((this.valTokens.length - 1) / T);
    // Cap validation work so it doesn't dominate wall-clock on CPU backends.
    const maxSeq = Math.min(nSeq, 64);
    let total = 0;
    for (let i = 0; i < maxSeq; i++) {
      const off = i * T;
      const xs = this.valTokens.subarray(off, off + T);
      const ys = this.valTokens.subarray(off + 1, off + 1 + T);
      const x = tf.tensor2d(Int32Array.from(xs), [1, T], "int32");
      const y = tf.tensor2d(Int32Array.from(ys), [1, T], "int32");
      const loss = tf.tidy(() => this.model.forward(x, y));
      total += (await loss.data())[0];
      loss.dispose();
      x.dispose();
      y.dispose();
    }
    return total / Math.max(1, maxSeq);
  }

  // --------------------------------------------------------------------------

  /**
   * Persist a new LKG.
   *
   *   1. Write versioned blob+meta (`slm16.lkg.<step>.*`) atomically.
   *   2. Atomically copy versioned → canonical pointer (`slm16.lkg.*`).
   *   3. Prune the rotation set to the newest LKG_KEEP_N.
   *
   * Ordering matters: the versioned file lands first, so even if the
   * process dies mid-(2) the rotation set is intact and tryResume()
   * recovers from it.
   */
  private async saveLkg(step: number, valLoss: number): Promise<void> {
    const sd: StateDict = await this.model.stateDict();
    const { blob, stats } = serializeInt8(sd);
    if (stats.compressedBytes > ARTIFACT_BYTE_LIMIT) {
      this.report({
        type: "error",
        step,
        message: `LKG artifact ${stats.compressedBytes} bytes exceeds 16MiB ceiling — refusing to overwrite`,
      });
      return;
    }

    const savedAt = new Date().toISOString();
    const verBlob = path.join(this.ckptDir, lkgVersionedFilename(step));
    const verMeta = path.join(this.ckptDir, lkgVersionedMetaFilename(step));

    // Build history (this entry + whatever is already on disk), newest first.
    const prior = listVersionedLkg(this.ckptDir).filter((e) => e.step !== step);
    const history: LkgHistoryEntry[] = [
      {
        step,
        valLoss,
        savedAt,
        file: path.basename(verBlob),
        metaFile: path.basename(verMeta),
      },
      ...prior.map((e) => ({
        step: e.step,
        valLoss: Number.NaN, // filled from sidecar if a reader cares
        savedAt: "",
        file: path.basename(e.file),
        metaFile: path.basename(e.metaFile),
      })),
    ].slice(0, LKG_KEEP_N);

    const meta: LkgMeta = {
      step,
      valLoss,
      bestValLoss: this.bestValLoss,
      hparams: this.hp,
      loaderState: this.loader.state(),
      savedAt,
      artifactBytes: stats.compressedBytes,
      paramCount: stats.paramCount,
      history,
    };
    const metaJson = JSON.stringify(meta, null, 2);

    // (1) versioned
    atomicWrite(verBlob, blob);
    atomicWrite(verMeta, metaJson);
    // (2) canonical pointer
    atomicWrite(this.lkgPath, blob);
    atomicWrite(this.metaPath, metaJson);
    // (3) prune rotation
    for (const e of prior.slice(LKG_KEEP_N - 1)) {
      fs.rmSync(e.file, { force: true });
      fs.rmSync(e.metaFile, { force: true });
    }

    this.report({
      type: "checkpoint",
      step,
      valLoss,
      artifactBytes: stats.compressedBytes,
      paramCount: stats.paramCount,
      path: this.lkgPath,
    });
  }

  // --------------------------------------------------------------------------

  /** Run forever (or until `stop()` / maxSteps). Yields to the event loop
   *  between steps so worker messages can land. */
  async run(maxSteps = Number.POSITIVE_INFINITY): Promise<void> {
    const { hp } = this;
    while (!this.stopRequested && this.opt.step_t < maxSteps) {
      const step = this.opt.step_t + 1;
      const batch = this.loader.nextBatch(hp.trainBatchTokens, hp.trainSeqLen);
      const { loss, grads } = this.gradsFor(batch);
      const lrScale = lrSchedule(step, hp.warmdownIters);
      this.opt.step(grads, lrScale);
      this.disposeGrads(grads);

      if (step % this.reportEvery === 0) {
        this.report({ type: "step", step, trainLoss: loss, lrScale });
      }

      if (step % this.valEvery === 0) {
        const valLoss = await this.validate();
        const improved = valLoss < this.bestValLoss;
        if (improved) this.bestValLoss = valLoss;
        this.report({
          type: "validation",
          step,
          valLoss,
          bestValLoss: this.bestValLoss,
          improved,
        });
        if (improved) await this.saveLkg(step, valLoss);
      }

      // Yield so parentPort messages (stop) can be processed.
      await new Promise((r) => setImmediate(r));
    }
  }

  stop(): void {
    this.stopRequested = true;
  }

  dispose(): void {
    this.opt.dispose();
    this.model.dispose();
  }

  get lkgFile(): string {
    return this.lkgPath;
  }
}
