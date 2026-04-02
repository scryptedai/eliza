/**
 * Slm16Service — main-thread orchestrator for the SLM16 trainer.
 *
 * Responsibilities:
 *   1. Spawn the worker thread on start() (unless SLM16_AUTOTRAIN=false).
 *      "The AVB should initiate this training if the training thread isn't running."
 *   2. Forward worker events into runtime settings, especially `slm16_val_loss`.
 *      "a setting for 'slm16_val_loss' will hint how strong the model is"
 *   3. On each new LKG event, schedule an intelligence-scoring round against
 *      Nova Pro and write `slm16_intelligence` once it completes.
 *   4. Expose `runScoring()` for ad-hoc evaluation and `getStatus()` for the
 *      personality to query.
 *   5. Clean shutdown: post 'stop' to the worker, wait for it to checkpoint
 *      and exit, then dispose.
 *
 * The Service does NOT load tfjs in the main thread for training — that
 * stays in the worker. It DOES load tfjs lazily for inference when scoring
 * is requested (the LKG model is loaded fresh each scoring round and
 * disposed immediately after, so the main thread's memory footprint stays
 * flat between scoring rounds).
 */

import { Worker } from "node:worker_threads";
import { type IAgentRuntime, Service } from "@elizaos/core";
import {
  SCRYPTEDAI_SERVICE_TYPE,
  type ScryptedAIService,
} from "@elizaos/plugin-scryptedai";
import {
  detectBackend,
  ENV,
  resolveTrainingConfig,
  SETTING,
  type Slm16TrainingConfig,
} from "./config.ts";
import { archHash, disposeCache, disposeWeights } from "./model.ts";
import { loadLkgForInference, loadTfBackend } from "./trainer.ts";
import { loadTokenizer, type Tokenizer } from "./tokenizer.ts";
import { scoreIntelligence } from "./scoring.ts";
import type {
  Slm16Event,
  Slm16ScoringResult,
  Slm16WorkerData,
} from "./types.ts";

// ----------------------------------------------------------------------------
// Runtime surface — structural subset we actually use
// ----------------------------------------------------------------------------

/**
 * The shared `AvbRuntimeSurface` in ../types.ts deliberately doesn't include
 * `setSetting` (the AVB pipeline never writes settings). We need it here, so
 * we declare a separate minimal surface rather than widening the shared type
 * and dragging every existing test along.
 *
 * `setSetting`'s signature on the real IAgentRuntime is:
 *   setSetting(key: string, value: string | boolean | null, secret = false): void
 */
interface Slm16RuntimeSurface {
  logger: {
    info: (msg: string) => void;
    warn: (msg: string) => void;
    error: (msg: string) => void;
    debug: (msg: string) => void;
  };
  getSetting(key: string): unknown;
  setSetting(key: string, value: string | boolean | null, secret?: boolean): void;
  getServiceLoadPromise(type: string): Promise<unknown>;
}

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------

export const SLM16_SERVICE_TYPE = "slm16" as const;

/** How long to wait for the worker to checkpoint+exit on stop(). */
const WORKER_SHUTDOWN_GRACE_MS = 30_000;

/** Throttle: don't log every step event (100/min would be noisy). */
const STEP_LOG_INTERVAL = 50;

// ----------------------------------------------------------------------------
// Service
// ----------------------------------------------------------------------------

export class Slm16Service extends Service {
  static serviceType = SLM16_SERVICE_TYPE;
  static serviceName = "SLM16";

  public capabilityDescription =
    "Background trainer for a 16MB language model (OpenAI Parameter Golf 9×512 U-Net GPT). " +
    "Trains continuously on FineWeb shards, validates every 100 steps, persists the " +
    "best int8-quantized checkpoint, and scores it against Nova Pro for an intelligence metric.";

  private rt!: Slm16RuntimeSurface;
  private worker: Worker | null = null;
  private cfg!: Slm16TrainingConfig;
  private tok: Tokenizer | null = null;

  /** Cached scoring result from the last round (for getStatus()). */
  private lastScoring: Slm16ScoringResult | null = null;
  /** Prevent overlapping scoring rounds. */
  private scoringInFlight = false;

  /** Last few events — exposed via getStatus() for diagnostics. */
  private recentEvents: Slm16Event[] = [];
  private static readonly EVENT_RING_SIZE = 20;

  // --------------------------------------------------------------------------
  // Lifecycle
  // --------------------------------------------------------------------------

  static async start(runtime: IAgentRuntime): Promise<Slm16Service> {
    const svc = new Slm16Service(runtime);
    svc.rt = runtime as unknown as Slm16RuntimeSurface;

    // Resolve config (env overrides merged in).
    svc.cfg = resolveTrainingConfig((k) => svc.rt.getSetting(k));

    // Tokenizer is cheap to load (sync, ~50KB protobuf parse). Load now so
    // scoring doesn't have to do it on the hot path. Falls back to byte-
    // identity if the .model file hasn't been downloaded yet.
    svc.tok = loadTokenizer(svc.cfg.checkpointDir);
    svc.rt.logger.info(
      `[slm16] tokenizer ${svc.tok.loaded ? "loaded" : "byte-fallback"} ` +
        `(vocab=${svc.tok.vocabSize}), checkpoint dir=${svc.cfg.checkpointDir}`,
    );

    // Initialize settings to "no model yet" sentinels so the personality
    // sees coherent values even before the worker has spun up.
    svc.rt.setSetting(SETTING.STATUS, "starting");
    svc.rt.setSetting(SETTING.VAL_LOSS, String(Number.POSITIVE_INFINITY));
    svc.rt.setSetting(SETTING.STEP, "0");
    svc.rt.setSetting(SETTING.TRAIN_TIME, "0");

    // Auto-spawn the trainer unless explicitly disabled.
    const autotrain = svc.rt.getSetting(ENV.AUTOTRAIN);
    const disabled =
      autotrain === false ||
      autotrain === "false" ||
      autotrain === "0";
    if (disabled) {
      svc.rt.logger.info(
        `[slm16] autotrain disabled (${ENV.AUTOTRAIN}=${String(autotrain)})`,
      );
      svc.rt.setSetting(SETTING.STATUS, "idle");
    } else {
      svc.spawnWorker();
    }

    return svc;
  }

  async stop(): Promise<void> {
    await this.stopWorker();
  }

  // --------------------------------------------------------------------------
  // Worker management
  // --------------------------------------------------------------------------

  /**
   * Spawn the trainer worker. Idempotent — if a worker is already alive,
   * this is a no-op (we never want two trainers fighting over the same
   * checkpoint directory).
   */
  spawnWorker(): void {
    if (this.worker !== null) {
      this.rt.logger.debug("[slm16] spawnWorker called but worker already running");
      return;
    }

    const wd: Slm16WorkerData = {
      config: this.cfg,
      archHash: archHash(),
    };

    // Bun resolves .ts worker entries directly. The URL form keeps the path
    // relative to THIS file regardless of where the agent is launched from.
    const workerUrl = new URL("./worker.ts", import.meta.url);

    let w: Worker;
    try {
      w = new Worker(workerUrl, { workerData: wd });
    } catch (e) {
      // Worker spawn can fail if the runtime doesn't support .ts entries
      // (e.g. plain Node without a loader). Surface as an error setting
      // rather than crashing the agent.
      this.rt.logger.error(
        `[slm16] failed to spawn worker: ${(e as Error).message}`,
      );
      this.rt.setSetting(SETTING.STATUS, "error");
      return;
    }

    w.on("message", (raw: unknown) => this.handleEvent(raw as Slm16Event));

    w.on("error", (err: Error) => {
      // 'error' fires for uncaught exceptions in the worker that bypassed
      // our try/catch. Treat as fatal.
      this.rt.logger.error(`[slm16] worker uncaught error: ${err.message}`);
      this.pushEvent({ type: "error", message: err.message, stack: err.stack, fatal: true });
      this.rt.setSetting(SETTING.STATUS, "error");
      this.worker = null;
    });

    w.on("exit", (code: number) => {
      if (code !== 0) {
        this.rt.logger.warn(`[slm16] worker exited with code ${code}`);
        this.rt.setSetting(SETTING.STATUS, "error");
      } else {
        this.rt.logger.info("[slm16] worker exited cleanly");
      }
      this.worker = null;
    });

    this.worker = w;
    this.rt.logger.info(
      `[slm16] worker spawned (data=${this.cfg.dataDir}, vram=${this.cfg.vramFraction})`,
    );
  }

  /**
   * Ask the worker to checkpoint and exit. Waits up to the grace period;
   * if it doesn't exit cleanly we terminate() it (which loses the
   * uncheckpointed step but preserves whatever was last written to disk).
   */
  private async stopWorker(): Promise<void> {
    const w = this.worker;
    if (w === null) return;

    return new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        this.rt.logger.warn(
          `[slm16] worker did not exit within ${WORKER_SHUTDOWN_GRACE_MS}ms; terminating`,
        );
        void w.terminate().finally(() => {
          this.worker = null;
          resolve();
        });
      }, WORKER_SHUTDOWN_GRACE_MS);

      w.once("exit", () => {
        clearTimeout(timer);
        this.worker = null;
        resolve();
      });

      // Politely ask first. The worker's main loop polls shouldStop()
      // each step, checkpoints, then process.exit(0).
      w.postMessage({ type: "stop" });
    });
  }

  /** Force a checkpoint write without stopping. Useful before scoring. */
  forceCheckpoint(): void {
    this.worker?.postMessage({ type: "checkpoint" });
  }

  /** Is the trainer currently alive? */
  isTraining(): boolean {
    return this.worker !== null;
  }

  // --------------------------------------------------------------------------
  // Event sink — forward worker progress into runtime settings
  // --------------------------------------------------------------------------

  private handleEvent(e: Slm16Event): void {
    this.pushEvent(e);

    switch (e.type) {
      case "ready": {
        this.rt.setSetting(SETTING.STATUS, "training");
        this.rt.setSetting(SETTING.STEP, String(e.resumedAtStep));
        if (Number.isFinite(e.resumedValLoss)) {
          this.rt.setSetting(SETTING.VAL_LOSS, e.resumedValLoss.toFixed(6));
        }
        this.rt.logger.info(
          `[slm16] ready: backend=${e.backend}, params=${e.paramCount.toLocaleString()}, ` +
            `resumed step=${e.resumedAtStep}, resumed val_loss=${
              Number.isFinite(e.resumedValLoss) ? e.resumedValLoss.toFixed(4) : "∞"
            }`,
        );
        break;
      }

      case "step": {
        this.rt.setSetting(SETTING.STEP, String(e.step));
        this.rt.setSetting(SETTING.TRAIN_TIME, e.trainSeconds.toFixed(1));
        if (e.step % STEP_LOG_INTERVAL === 0) {
          this.rt.logger.debug(
            `[slm16] step ${e.step}: train_loss=${e.trainLoss.toFixed(4)}, ` +
              `tokens=${e.tokensSeen.toLocaleString()}, time=${e.trainSeconds.toFixed(1)}s`,
          );
        }
        break;
      }

      case "val": {
        this.rt.logger.info(
          `[slm16] val @ step ${e.step}: loss=${e.valLoss.toFixed(4)}, bpb=${e.valBpb.toFixed(4)}`,
        );
        break;
      }

      case "lkg": {
        // ★ THIS is the key requirement: update slm16_val_loss whenever we
        //   get a new best model. The personality reads this to gauge strength.
        this.rt.setSetting(SETTING.VAL_LOSS, e.valLoss.toFixed(6));
        const sizeMb = (e.artifactBytes / 1_000_000).toFixed(2);
        const status = e.underCap ? "under cap" : "OVER 16MB CAP";
        this.rt.logger.info(
          `[slm16] new LKG @ step ${e.step}: val_loss=${e.valLoss.toFixed(4)}, ` +
            `artifact=${sizeMb}MB (${status}), path=${e.path}`,
        );
        // Kick off a scoring round (fire-and-forget). The result lands in
        // slm16_intelligence asynchronously.
        void this.runScoring().catch((err) => {
          this.rt.logger.warn(
            `[slm16] scoring after LKG failed: ${(err as Error).message}`,
          );
        });
        break;
      }

      case "error": {
        if (e.fatal) {
          this.rt.setSetting(SETTING.STATUS, "error");
          this.rt.logger.error(`[slm16] fatal: ${e.message}`);
        } else {
          this.rt.logger.warn(`[slm16] ${e.message}`);
        }
        break;
      }

      case "idle": {
        this.rt.setSetting(SETTING.STATUS, e.reason);
        this.rt.logger.info(`[slm16] idle: ${e.reason}`);
        break;
      }
    }
  }

  private pushEvent(e: Slm16Event): void {
    this.recentEvents.push(e);
    if (this.recentEvents.length > Slm16Service.EVENT_RING_SIZE) {
      this.recentEvents.shift();
    }
  }

  // --------------------------------------------------------------------------
  // Scoring — Nova Pro cosine similarity → intelligence
  // --------------------------------------------------------------------------

  /**
   * Run a full intelligence-scoring round. Loads the LKG from disk, runs
   * inference on the 8 scoring prompts at temp 0.1, fetches Nova Pro's
   * answers via ScryptedAI, computes trigram cosine, and writes the
   * result to `slm16_intelligence`.
   *
   * This loads tfjs into the MAIN thread for inference. The footprint is
   * one set of fp32 weights (~68MB) plus activation memory; we dispose
   * everything immediately after so it doesn't compound across rounds.
   *
   * Returns null if no LKG exists yet (model hasn't validated even once).
   */
  async runScoring(): Promise<Slm16ScoringResult | null> {
    if (this.scoringInFlight) {
      this.rt.logger.debug("[slm16] scoring already in flight; skipping");
      return this.lastScoring;
    }
    this.scoringInFlight = true;

    try {
      // 1. Get ScryptedAI (await load promise — service init order isn't guaranteed).
      const scrypted = (await this.rt.getServiceLoadPromise(
        SCRYPTEDAI_SERVICE_TYPE,
      )) as ScryptedAIService;

      // 2. Load tfjs backend on this thread.
      const backend = detectBackend();
      const { tf } = await loadTfBackend(backend.pkg, this.cfg.vramFraction);

      // 3. Load LKG from disk.
      const lkg = await loadLkgForInference(tf, this.cfg.checkpointDir);
      if (lkg === null) {
        this.rt.logger.info("[slm16] no LKG yet — skipping scoring round");
        return null;
      }

      const tok = this.tok ?? loadTokenizer(this.cfg.checkpointDir);

      // 4. Score.
      const result = await scoreIntelligence(
        tf,
        lkg.weights,
        tok,
        scrypted,
        lkg.valLoss,
        (msg) => this.rt.logger.debug(msg),
      );

      // 5. Publish.
      this.rt.setSetting(SETTING.INTELLIGENCE, result.intelligence.toFixed(6));
      this.lastScoring = result;
      this.rt.logger.info(
        `[slm16] intelligence score: ${result.intelligence.toFixed(4)} ` +
          `(mean cosine=${
            Number.isFinite(result.meanSimilarity)
              ? result.meanSimilarity.toFixed(4)
              : "n/a"
          }, val_loss=${result.valLoss.toFixed(4)})`,
      );

      // 6. Cleanup.
      disposeWeights(lkg.weights);
      disposeCache();

      return result;
    } finally {
      this.scoringInFlight = false;
    }
  }

  // --------------------------------------------------------------------------
  // Diagnostics
  // --------------------------------------------------------------------------

  getStatus(): {
    isTraining: boolean;
    config: Slm16TrainingConfig;
    recentEvents: Slm16Event[];
    lastScoring: Slm16ScoringResult | null;
    settings: Record<string, unknown>;
  } {
    return {
      isTraining: this.isTraining(),
      config: this.cfg,
      recentEvents: [...this.recentEvents],
      lastScoring: this.lastScoring,
      settings: {
        [SETTING.VAL_LOSS]: this.rt.getSetting(SETTING.VAL_LOSS),
        [SETTING.INTELLIGENCE]: this.rt.getSetting(SETTING.INTELLIGENCE),
        [SETTING.STEP]: this.rt.getSetting(SETTING.STEP),
        [SETTING.TRAIN_TIME]: this.rt.getSetting(SETTING.TRAIN_TIME),
        [SETTING.STATUS]: this.rt.getSetting(SETTING.STATUS),
      },
    };
  }
}
