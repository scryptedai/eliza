/**
 * SLM16 — wire types crossing the worker_threads boundary.
 *
 * Everything here must be structured-cloneable (no Tensor handles, no
 * closures). The trainer worker posts these to the parent thread; the
 * Slm16Service forwards them into runtime settings + logs.
 */

// ----------------------------------------------------------------------------
// Worker → main: progress events
// ----------------------------------------------------------------------------

export type Slm16Event =
  | Slm16ReadyEvent
  | Slm16StepEvent
  | Slm16ValEvent
  | Slm16LkgEvent
  | Slm16ErrorEvent
  | Slm16IdleEvent;

export interface Slm16ReadyEvent {
  type: "ready";
  /** "cuda" | "accelerate" | "cpu" — what the worker actually loaded. */
  backend: string;
  /** Total trainable parameter count. */
  paramCount: number;
  /** Resumed from a checkpoint at this step (0 = fresh init). */
  resumedAtStep: number;
  /** Best val_loss from the resumed checkpoint (Infinity = fresh). */
  resumedValLoss: number;
}

export interface Slm16StepEvent {
  type: "step";
  step: number;
  trainLoss: number;
  /** Cumulative wallclock seconds spent inside the training loop. */
  trainSeconds: number;
  /** Tokens consumed so far. */
  tokensSeen: number;
}

export interface Slm16ValEvent {
  type: "val";
  step: number;
  valLoss: number;
  /** Bits per byte (tokenizer-agnostic compression metric). */
  valBpb: number;
  trainSeconds: number;
}

export interface Slm16LkgEvent {
  type: "lkg";
  step: number;
  valLoss: number;
  /** Compressed int8+zlib artifact size in bytes. */
  artifactBytes: number;
  /** True if artifact ≤ 16,000,000 bytes. */
  underCap: boolean;
  /** Absolute path to the freshly written LKG file. */
  path: string;
}

export interface Slm16ErrorEvent {
  type: "error";
  message: string;
  /** Best-effort stack (string, not Error object — Error isn't cloneable). */
  stack?: string;
  /** If true, the worker is exiting and won't recover. */
  fatal: boolean;
}

export interface Slm16IdleEvent {
  type: "idle";
  reason: "no_data" | "stopped" | "max_tokens" | "max_steps";
}

// ----------------------------------------------------------------------------
// Main → worker: control commands
// ----------------------------------------------------------------------------

export type Slm16Command =
  | { type: "stop" }
  | { type: "checkpoint" }; // force-write trainer state immediately

// ----------------------------------------------------------------------------
// Persisted metadata (meta.json next to checkpoints)
// ----------------------------------------------------------------------------

export interface Slm16Meta {
  /** Format tag for forward compat. */
  format: string;
  /** Optimizer step at which trainer state was last saved. */
  step: number;
  /** Best val_loss observed (corresponds to lkg.int8.bin). */
  bestValLoss: number;
  /** Step at which bestValLoss was observed. */
  bestValStep: number;
  /** Cumulative training wallclock (seconds). */
  trainSeconds: number;
  /** Total tokens consumed across all sessions. */
  tokensSeen: number;
  /** ISO timestamp of last write. */
  updatedAt: string;
  /** Architecture digest (so we refuse to resume on shape mismatch). */
  archHash: string;
  /** PRNG seed used for the original init. */
  seed: number;
}

// ----------------------------------------------------------------------------
// Quantized state-dict on-disk format
// ----------------------------------------------------------------------------

/**
 * Tagged tensor entry inside the int8 blob. Either:
 *  - "q": int8 data + per-row (or per-tensor) fp16 scale → dequant to dtype
 *  - "f": fp16 passthrough (small tensors, control scalars)
 */
export type Slm16QuantEntry =
  | {
      kind: "q";
      shape: number[];
      /** Original dtype to restore after dequant ("float32"). */
      dtype: "float32";
      /** "row" → scale.length === shape[0]; "tensor" → scale.length === 1. */
      scaleAxis: "row" | "tensor";
    }
  | {
      kind: "f";
      shape: number[];
      dtype: "float32";
    };

/**
 * Header preceding the binary payload in lkg.int8.bin (before zlib).
 * Layout: [u32 jsonLen][json header][int8/fp16 tensor data, contiguous, in
 * the order listed under `entries`].
 */
export interface Slm16QuantHeader {
  format: string;
  archHash: string;
  /** Tensor name → entry metadata. Iteration order is the on-disk order. */
  entries: Record<string, Slm16QuantEntry>;
  /** Per-quantized-tensor scale offset (bytes from data start). */
  scaleOffsets: Record<string, number>;
  /** Per-tensor data offset (bytes from data start). */
  dataOffsets: Record<string, number>;
}

// ----------------------------------------------------------------------------
// Inference / scoring
// ----------------------------------------------------------------------------

export interface Slm16GenerateOptions {
  /** Token IDs to seed generation. */
  promptIds: number[];
  /** Max tokens to append. */
  maxNewTokens: number;
  /** Sampling temperature (0 = greedy). */
  temperature: number;
  /** Top-k filter (0 = disabled). */
  topK?: number;
}

export interface Slm16GenerateResult {
  /** promptIds + sampled completion. */
  tokenIds: number[];
  /** Just the sampled completion. */
  newTokenIds: number[];
  /** Decoded text (if tokenizer available). */
  text?: string;
}

export interface Slm16ScoringResult {
  /** Mean cosine similarity across all sampled prompt pairs [-1, 1]. */
  meanSimilarity: number;
  /** Per-prompt similarities. */
  similarities: number[];
  /**
   * Final intelligence score [0, 1]: linear remap of meanSimilarity from
   * [SCORING.similarityFloor, SCORING.similarityCeiling] → [0, 1], clamped.
   */
  intelligence: number;
  /** SLM16 completions (decoded text) for inspection. */
  slmCompletions: string[];
  /** Nova Pro completions for the same prompts. */
  novaCompletions: string[];
  /** val_loss of the LKG model used for inference. */
  valLoss: number;
}

// ----------------------------------------------------------------------------
// Worker constructor payload (workerData)
// ----------------------------------------------------------------------------

export interface Slm16WorkerData {
  /** Resolved training config (env already merged in). */
  config: import("./config.ts").Slm16TrainingConfig;
  /** Architecture hash (for resume validation). */
  archHash: string;
}
