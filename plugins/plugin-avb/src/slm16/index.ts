/**
 * SLM16 — Small Language Model under 16MB.
 *
 * TypeScript port of the OpenAI Parameter Golf "Naive Baseline" (9-layer,
 * 512-dim GPT with U-Net skip connections). Trains in the background on
 * FineWeb shards, validates every 100 steps, persists the best int8+zlib
 * checkpoint (<16MB), and scores it against Nova Pro for an intelligence
 * metric.
 *
 * Public surface: just the Service and the things you'd want to inspect
 * from outside the plugin (config constants, scoring result type). The
 * internals (model.ts, optimizer.ts, trainer.ts, etc.) are intentionally
 * not re-exported — they're worker-thread plumbing.
 */

export { Slm16Service, SLM16_SERVICE_TYPE } from "./service.ts";

export {
  ARCH,
  ENV as SLM16_ENV,
  PATHS as SLM16_PATHS,
  QUANT,
  SCORING,
  SETTING as SLM16_SETTING,
  type Slm16TrainingConfig,
  TRAIN_DEFAULTS,
} from "./config.ts";

export type {
  Slm16Event,
  Slm16GenerateOptions,
  Slm16GenerateResult,
  Slm16Meta,
  Slm16ScoringResult,
} from "./types.ts";
