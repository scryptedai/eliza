/**
 * Slm16Service — owns the background SLM16 training thread and surfaces
 * its progress into the agent runtime.
 *
 * Responsibilities:
 *   - Spawn a `worker_threads.Worker` running `worker.ts` so the GPU/CPU
 *     heavy training loop never blocks the agent's main event loop.
 *   - Forward VRAM-fraction / backend / data-dir settings into the worker
 *     environment *before* the native tfjs binding loads.
 *   - Listen for TrainerReport messages and translate them into runtime
 *     settings (`slm16_val_loss`, `slm16_step`, `slm16_backend`,
 *     `slm16_intelligence`) that the personality provider reads.
 *   - On every new LKG checkpoint, lazily run the Nova-Pro evaluator
 *     (cosine-similarity against Amazon Nova Pro at T=0.1) and publish
 *     the resulting intelligence score.
 *   - Provide `ensureTraining()` so the AVB pipeline can autonomously
 *     restart training if the worker has died.
 */

import { Worker } from "node:worker_threads";

import { type IAgentRuntime, Service } from "@elizaos/core";
import { SCRYPTEDAI_SERVICE_TYPE } from "@elizaos/plugin-scryptedai";

/** TaskWorker name + tags for the recurring status sidejob. */
export const SLM16_OBSERVER_WORKER = "SLM16_STATUS_OBSERVER";
export const SLM16_OBSERVER_TAGS = ["queue", "repeat", "slm16"] as const;

import { loadBackend } from "./backend.ts";
import {
  DEFAULT_CKPT_DIR,
  DEFAULT_DATA_DIR,
  DEFAULT_TOKENIZER_PATH,
  DEFAULT_VRAM_FRACTION,
  ENV_SLM16_AUTOTRAIN,
  ENV_SLM16_BACKEND,
  ENV_SLM16_CKPT_DIR,
  ENV_SLM16_DATA_DIR,
  ENV_SLM16_EVAL_EVERY,
  ENV_SLM16_TOKENIZER_PATH,
  ENV_SLM16_TRAIN_SHARDS,
  ENV_SLM16_VRAM_FRACTION,
  SETTING_BACKEND,
  SETTING_INTELLIGENCE,
  SETTING_STEP,
  SETTING_VAL_LOSS,
  SLM16_SERVICE_TYPE,
  type Slm16BackendName,
} from "./constants.ts";
import { DEFAULT_TRAIN_SHARDS, ensureFinewebData } from "./data.ts";
import {
  type EvalRuntimeSurface,
  evaluateAgainstNovaPro,
  type IntelligenceReport,
} from "./evaluator.ts";
import { Slm16Inference } from "./inference.ts";
import type { Slm16WorkerData, Slm16WorkerMessage } from "./worker.ts";

// ----------------------------------------------------------------------------
// Runtime surface (structural cast — keeps tests light)
// ----------------------------------------------------------------------------

/** TaskWorker shape (subset of core's TaskWorker). */
interface Slm16TaskWorker {
  name: string;
  execute(
    runtime: unknown,
    options: Record<string, unknown>,
    task: { id?: string },
  ): Promise<void>;
}

export interface Slm16RuntimeSurface extends EvalRuntimeSurface {
  agentId: string;
  setSetting(key: string, value: unknown, secret?: boolean): void;
  getSetting(key: string): unknown;
  getServiceLoadPromise(type: string): Promise<unknown>;
  registerTaskWorker(worker: Slm16TaskWorker): void;
  createTask(task: {
    name: string;
    description: string;
    roomId?: string;
    tags: string[];
    metadata: Record<string, unknown>;
  }): Promise<string>;
  getTasksByName(name: string): Promise<Array<{ id?: string }>>;
  logger: {
    info: (m: string) => void;
    warn: (m: string) => void;
    error: (m: string) => void;
    debug: (m: string) => void;
  };
}

// ----------------------------------------------------------------------------
// Status snapshot (read by the provider + tests)
// ----------------------------------------------------------------------------

export interface Slm16Status {
  running: boolean;
  backend: string | null;
  step: number;
  trainLoss: number | null;
  valLoss: number | null;
  bestValLoss: number | null;
  intelligence: number | null;
  paramCount: number | null;
  artifactBytes: number | null;
  lastCheckpointAt: string | null;
  lastError: string | null;
}

// ----------------------------------------------------------------------------

export class Slm16Service extends Service {
  static serviceType = SLM16_SERVICE_TYPE;
  static serviceName = "SLM16";

  public capabilityDescription =
    "Trains a 16 MB Small Language Model (SLM16) in the background on local " +
    "GPU/CPU, validates every 100 steps, and benchmarks the LKG checkpoint " +
    "against Amazon Nova Pro via ScryptedAI to derive an intelligence score.";

  private rt!: Slm16RuntimeSurface;
  private worker: Worker | null = null;

  private readonly status: Slm16Status = {
    running: false,
    backend: null,
    step: 0,
    trainLoss: null,
    valLoss: null,
    bestValLoss: null,
    intelligence: null,
    paramCount: null,
    artifactBytes: null,
    lastCheckpointAt: null,
    lastError: null,
  };

  private ckptDir = DEFAULT_CKPT_DIR;
  private dataDir = DEFAULT_DATA_DIR;
  private tokenizerPath = DEFAULT_TOKENIZER_PATH;
  private trainShards = DEFAULT_TRAIN_SHARDS;
  private evalEvery = 1; // run Nova-Pro eval every N checkpoints
  private checkpointCount = 0;
  private evalInFlight = false;
  /** In-flight ensureFinewebData() promise (dedupes concurrent callers). */
  private bootstrap: Promise<void> | null = null;

  // --------------------------------------------------------------------------
  // Lifecycle
  // --------------------------------------------------------------------------

  static async start(runtime: IAgentRuntime): Promise<Slm16Service> {
    const svc = new Slm16Service(runtime);
    svc.rt = runtime as unknown as Slm16RuntimeSurface;

    svc.ckptDir =
      (svc.rt.getSetting(ENV_SLM16_CKPT_DIR) as string) ?? DEFAULT_CKPT_DIR;
    svc.dataDir =
      (svc.rt.getSetting(ENV_SLM16_DATA_DIR) as string) ?? DEFAULT_DATA_DIR;
    svc.tokenizerPath =
      (svc.rt.getSetting(ENV_SLM16_TOKENIZER_PATH) as string) ??
      DEFAULT_TOKENIZER_PATH;
    const ts = Number(svc.rt.getSetting(ENV_SLM16_TRAIN_SHARDS));
    if (Number.isFinite(ts) && ts > 0) svc.trainShards = Math.floor(ts);
    const ee = Number(svc.rt.getSetting(ENV_SLM16_EVAL_EVERY));
    if (Number.isFinite(ee) && ee > 0) svc.evalEvery = ee;

    // Wait for ScryptedAI so the evaluator works on first checkpoint.
    // Non-fatal if it never resolves.
    void svc.rt
      .getServiceLoadPromise(SCRYPTEDAI_SERVICE_TYPE)
      .catch(() => undefined);

    // Register the status-observer worker now so the AVB runner / pipeline
    // can spawn its task without racing service init.
    svc.rt.registerTaskWorker(svc.buildStatusObserverWorker());

    const auto = svc.rt.getSetting(ENV_SLM16_AUTOTRAIN);
    if (auto === undefined || auto === "true" || auto === true) {
      // The AVB spirit: the agent kicks off its own training without
      // being told. Bootstrap data first (fire-and-forget so service
      // load doesn't block on a multi-hundred-MB download), then spawn
      // the worker.
      void svc
        .ensureBootstrapped()
        .then(() => svc.ensureTraining())
        .catch((err) => {
          svc.status.lastError = `Bootstrap failed: ${(err as Error).message}`;
          svc.rt.logger.error(`[slm16] ${svc.status.lastError}`);
        });
    } else {
      svc.rt.logger.info("[slm16] Auto-training disabled via setting");
    }

    return svc;
  }

  async stop(): Promise<void> {
    if (this.worker) {
      this.worker.postMessage({ type: "stop" });
      // Give the trainer one event-loop turn to flush, then terminate.
      await new Promise((r) => setTimeout(r, 250));
      await this.worker.terminate().catch(() => undefined);
      this.worker = null;
      this.status.running = false;
    }
  }

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  isRunning(): boolean {
    return this.status.running;
  }

  /**
   * Ensure FineWeb shards + tokenizer exist on disk. Idempotent and
   * coalesced — concurrent callers (autotrain on boot, AvbService kick,
   * provider) all await the same download. Resolves immediately on
   * subsequent calls once data is in place.
   */
  ensureBootstrapped(): Promise<void> {
    if (this.bootstrap) return this.bootstrap;
    this.bootstrap = ensureFinewebData({
      dataDir: this.dataDir,
      tokenizerPath: this.tokenizerPath,
      trainShards: this.trainShards,
      onProgress: (m) => this.rt.logger.info(`[slm16:bootstrap] ${m}`),
    })
      .then((r) => {
        if (r.fetched.length > 0) {
          this.rt.logger.info(
            `[slm16:bootstrap] fetched ${r.fetched.length} file(s), ` +
              `${r.skipped.length} already present → ${r.dataDir}`,
          );
        } else {
          this.rt.logger.info(
            `[slm16:bootstrap] all ${r.skipped.length} file(s) already present`,
          );
        }
      })
      .catch((err) => {
        // Reset so a later kick can retry.
        this.bootstrap = null;
        throw err;
      });
    return this.bootstrap;
  }

  /**
   * Spawn the recurring SLM16_STATUS_OBSERVER task — the AVB-style
   * "sidejob" that surfaces training progress through the runtime's
   * TaskService instead of a blocking poll loop. Idempotent: returns
   * the existing task id if one already exists.
   */
  async startStatusObserver(intervalMs = 10_000): Promise<string> {
    const existing = await this.rt.getTasksByName(SLM16_OBSERVER_WORKER);
    if (existing.length > 0 && existing[0].id) return existing[0].id;
    const id = await this.rt.createTask({
      name: SLM16_OBSERVER_WORKER,
      description: "Periodic SLM16 training status observer",
      roomId: this.rt.agentId,
      tags: [...SLM16_OBSERVER_TAGS],
      metadata: { updatedAt: Date.now(), updateInterval: intervalMs },
    });
    this.rt.logger.info(
      `[slm16] Status observer task spawned (id=${id}, every ${intervalMs}ms)`,
    );
    return id;
  }

  // --------------------------------------------------------------------------
  // Status-observer TaskWorker (AVB async-tick pattern)
  // --------------------------------------------------------------------------

  private buildStatusObserverWorker(): Slm16TaskWorker {
    const svc = this;
    return {
      name: SLM16_OBSERVER_WORKER,
      async execute(): Promise<void> {
        const s = svc.status;
        // Self-heal: if the trainer died, kick it (data is already
        // bootstrapped or the in-flight bootstrap promise will gate it).
        if (!s.running && svc.bootstrap) {
          void svc.bootstrap
            .then(() => svc.ensureTraining())
            .catch(() => undefined);
        }
        const parts = [
          `running=${s.running}`,
          `backend=${s.backend ?? "?"}`,
          `step=${s.step}`,
        ];
        if (s.trainLoss != null)
          parts.push(`train_loss=${s.trainLoss.toFixed(4)}`);
        if (s.valLoss != null) parts.push(`val_loss=${s.valLoss.toFixed(4)}`);
        if (s.bestValLoss != null)
          parts.push(`best=${s.bestValLoss.toFixed(4)}`);
        if (s.intelligence != null) parts.push(`iq=${s.intelligence}/100`);
        if (s.artifactBytes != null)
          parts.push(`lkg=${(s.artifactBytes / 1024 / 1024).toFixed(2)}MiB`);
        if (s.lastError) parts.push(`err="${s.lastError}"`);
        svc.rt.logger.info(`[slm16:observer] ${parts.join(" ")}`);
      },
    };
  }

  getStatus(): Readonly<Slm16Status> {
    return this.status;
  }

  /**
   * Spawn the training worker if it isn't already running. Idempotent —
   * called by the AVB pipeline whenever it notices the trainer is down.
   */
  ensureTraining(): void {
    if (this.worker) return;

    const vram =
      Number(this.rt.getSetting(ENV_SLM16_VRAM_FRACTION)) ||
      DEFAULT_VRAM_FRACTION;
    const backend = this.rt.getSetting(ENV_SLM16_BACKEND) as
      | Slm16BackendName
      | undefined;

    const workerData: Slm16WorkerData = {
      dataDir: this.dataDir,
      ckptDir: this.ckptDir,
      backend,
    };

    try {
      this.worker = new Worker(new URL("./worker.ts", import.meta.url), {
        workerData,
        env: {
          ...process.env,
          [ENV_SLM16_VRAM_FRACTION]: String(vram),
          ...(backend ? { [ENV_SLM16_BACKEND]: backend } : {}),
        },
      });
    } catch (err) {
      this.status.lastError = `Failed to spawn SLM16 worker: ${(err as Error).message}`;
      this.rt.logger.error(`[slm16] ${this.status.lastError}`);
      return;
    }

    this.status.running = true;
    this.status.lastError = null;
    this.rt.logger.info(
      `[slm16] Training worker spawned (vram=${(vram * 100).toFixed(0)}%, data=${this.dataDir})`,
    );

    this.worker.on("message", (m: Slm16WorkerMessage) => this.onMessage(m));
    this.worker.on("error", (err: Error) => {
      this.status.lastError = err.message;
      this.rt.logger.error(`[slm16] Worker error: ${err.message}`);
    });
    this.worker.on("exit", (code) => {
      this.status.running = false;
      this.worker = null;
      this.rt.logger.warn(`[slm16] Worker exited (code=${code})`);
    });
  }

  /**
   * Generate text from the current LKG checkpoint. Loads a fresh
   * inference graph on the main-thread backend (CPU is fine for a 16 MB
   * model) so the training worker keeps its GPU memory.
   */
  async generate(
    prompt: string,
    opts?: { maxNewTokens?: number; temperature?: number },
  ): Promise<string> {
    const { tf } = await loadBackend("cpu");
    const inf = await Slm16Inference.load(tf, this.ckptDir);
    try {
      return await inf.generate(prompt, opts);
    } finally {
      inf.dispose();
    }
  }

  /** Force a Nova-Pro evaluation against the current LKG. */
  async evaluate(): Promise<IntelligenceReport | null> {
    return this.runEvaluation();
  }

  // --------------------------------------------------------------------------
  // Worker → main-thread message handling
  // --------------------------------------------------------------------------

  private onMessage(m: Slm16WorkerMessage): void {
    switch (m.type) {
      case "boot": {
        this.status.backend = m.backend;
        this.status.step = m.step;
        this.status.paramCount = m.paramCount;
        if (Number.isFinite(m.bestValLoss)) {
          this.status.bestValLoss = m.bestValLoss;
        }
        this.rt.setSetting(SETTING_BACKEND, m.backend);
        this.rt.logger.info(
          `[slm16] Trainer ready on ${m.backend} (params=${m.paramCount.toLocaleString()}, resumed=${m.resumed})`,
        );
        break;
      }
      case "step": {
        this.status.step = m.step;
        this.status.trainLoss = m.trainLoss;
        this.rt.setSetting(SETTING_STEP, m.step);
        this.rt.logger.debug(
          `[slm16] step ${m.step} loss=${m.trainLoss.toFixed(4)} lr×=${m.lrScale.toFixed(3)}`,
        );
        break;
      }
      case "validation": {
        this.status.valLoss = m.valLoss;
        this.status.bestValLoss = m.bestValLoss;
        this.rt.setSetting(SETTING_VAL_LOSS, m.valLoss);
        this.rt.logger.info(
          `[slm16] step ${m.step} val_loss=${m.valLoss.toFixed(4)} best=${m.bestValLoss.toFixed(4)}${m.improved ? " ★" : ""}`,
        );
        break;
      }
      case "checkpoint": {
        this.status.artifactBytes = m.artifactBytes;
        this.status.lastCheckpointAt = new Date().toISOString();
        this.checkpointCount++;
        this.rt.logger.info(
          `[slm16] LKG saved @ step ${m.step} (${(m.artifactBytes / 1024 / 1024).toFixed(2)} MiB / 16 MiB)`,
        );
        if (this.checkpointCount % this.evalEvery === 0) {
          void this.runEvaluation().catch((err) =>
            this.rt.logger.warn(
              `[slm16] Nova-Pro evaluation failed: ${(err as Error).message}`,
            ),
          );
        }
        break;
      }
      case "error": {
        this.status.lastError = m.message;
        this.rt.logger.error(`[slm16] ${m.message}`);
        break;
      }
      case "exit": {
        this.status.running = false;
        break;
      }
    }
  }

  // --------------------------------------------------------------------------

  private async runEvaluation(): Promise<IntelligenceReport | null> {
    if (this.evalInFlight) return null;
    this.evalInFlight = true;
    try {
      const { tf } = await loadBackend("cpu");
      const inf = await Slm16Inference.load(tf, this.ckptDir);
      try {
        const report = await evaluateAgainstNovaPro(this.rt, inf);
        if (report) {
          this.status.intelligence = report.score;
          this.rt.setSetting(SETTING_INTELLIGENCE, report.score);
        }
        return report;
      } finally {
        inf.dispose();
      }
    } finally {
      this.evalInFlight = false;
    }
  }
}
