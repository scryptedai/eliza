/**
 * SLM16 — Small Language Model, 16 MiB.
 *
 * A TypeScript port of the OpenAI Parameter-Golf naive baseline
 * (512-dim × 9-layer U-Net transformer, 1024-token BPE vocab, tied
 * embeddings, Muon+Adam) that trains in a background worker thread on
 * whatever GPU the host has — CUDA via tfjs-node-gpu, Apple Metal via
 * tfjs-node on darwin, or pure-JS CPU as a last resort.
 *
 * Public surface:
 *   - Slm16Service       background trainer + LKG inference + Nova-Pro eval
 *   - slm16StatusProvider personality provider with live progress
 *   - constants / types  for downstream consumers
 */

export { type BackendInfo, loadBackend, type TF } from "./backend.ts";
export * from "./constants.ts";
export {
  type Batch,
  type BootstrapOptions,
  type BootstrapResult,
  DEFAULT_TRAIN_SHARDS,
  ensureFinewebData,
  FINEWEB_HF_BASE,
  FINEWEB_HF_REPO,
  listShards,
  loadDataShard,
  loadValidationTokens,
  TokenLoader,
  TokenStream,
} from "./data.ts";
export {
  cosineSimilarity,
  type EvalRuntimeSurface,
  evaluateAgainstNovaPro,
  type IntelligenceReport,
  type JudgeVerdict,
  type PromptComparison,
  parseJudgeVerdict,
} from "./evaluator.ts";
export {
  loadTokenizer,
  Slm16Inference,
  type Tokenizer,
} from "./inference.ts";
export { type ParamGroups, Slm16Model } from "./model.ts";
export { newtonSchulz5, Slm16Optimizer } from "./optimizer.ts";
export { slm16StatusProvider } from "./provider.ts";
export {
  deserializeInt8,
  type QuantStats,
  type StateDict,
  serializeInt8,
} from "./quantize.ts";
export {
  SLM16_OBSERVER_TAGS,
  SLM16_OBSERVER_WORKER,
  type Slm16RuntimeSurface,
  Slm16Service,
  type Slm16Status,
} from "./service.ts";
export {
  type LkgMeta,
  Slm16Trainer,
  type TrainerOptions,
  type TrainerReport,
} from "./trainer.ts";
