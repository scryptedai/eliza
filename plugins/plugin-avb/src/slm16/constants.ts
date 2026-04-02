/**
 * SLM16 — Small Language Model under 16 MB.
 *
 * Hyperparameters and configuration ported from the OpenAI Parameter-Golf
 * Naive Baseline (`train_gpt.py`, repo: openai/parameter-golf @ main):
 *   - 9 transformer blocks at width 512
 *   - 8 attention heads / 4 KV heads (GQA), 2× MLP expansion
 *   - vocab 1024, seq 1024, tied embeddings
 *   - U-Net skip connections (encoder ↔ decoder)
 *   - int8 per-row PTQ + zlib for the <16 MB artifact
 *
 * Reference source files (see openai/parameter-golf @ main):
 *   - train_gpt.py                          → architecture, optimizer, int8 PTQ
 *   - data/cached_challenge_fineweb.py      → FineWeb shard download
 *   - data/README.md                        → shard layout, tokenizer specs
 *   - records/.../2026-03-17_NaiveBaseline  → reference 9×512 baseline
 *
 * The challenge metric is bits-per-byte on FineWeb val. SLM16 tracks plain
 * cross-entropy val_loss for the agent setting and additionally derives an
 * "intelligence score" by comparing generations against Nova Pro
 * (cosine similarity of text embeddings).
 */

import * as os from "node:os";
import * as path from "node:path";

// ----------------------------------------------------------------------------
// Service identity
// ----------------------------------------------------------------------------

export const SLM16_SERVICE_TYPE = "slm16" as const;

// ----------------------------------------------------------------------------
// Model shape (mirrors train_gpt.py Hyperparameters defaults)
// ----------------------------------------------------------------------------

export interface Slm16Hyperparameters {
  vocabSize: number;
  numLayers: number;
  modelDim: number;
  numHeads: number;
  numKvHeads: number;
  mlpMult: number;
  tieEmbeddings: boolean;
  ropeBase: number;
  logitSoftcap: number;
  qkGainInit: number;
  trainSeqLen: number;
  trainBatchTokens: number;
  // Optimizer
  embedLr: number;
  matrixLr: number;
  scalarLr: number;
  beta1: number;
  beta2: number;
  adamEps: number;
  muonMomentum: number;
  muonBackendSteps: number;
  warmdownIters: number;
}

export const DEFAULT_HPARAMS: Slm16Hyperparameters = {
  vocabSize: 1024,
  numLayers: 9,
  modelDim: 512,
  numHeads: 8,
  numKvHeads: 4,
  mlpMult: 2,
  tieEmbeddings: true,
  ropeBase: 10000.0,
  logitSoftcap: 30.0,
  qkGainInit: 1.5,
  trainSeqLen: 1024,
  trainBatchTokens: 8192, // single-process default (vs 524_288 on 8×H100)
  // Optimizer
  embedLr: 0.05, // tied_embed_lr
  matrixLr: 0.04,
  scalarLr: 0.04,
  beta1: 0.9,
  beta2: 0.95,
  adamEps: 1e-8,
  muonMomentum: 0.95,
  muonBackendSteps: 5,
  warmdownIters: 1200,
};

// ----------------------------------------------------------------------------
// Artifact / size constraints
// ----------------------------------------------------------------------------

/** Hard ceiling for the serialized LKG artifact (int8 + zlib). */
export const ARTIFACT_BYTE_LIMIT = 16 * 1024 * 1024;

/** int8 PTQ tunables (mirrors train_gpt.py). */
export const INT8_KEEP_FLOAT_MAX_NUMEL = 65_536;
export const INT8_CLIP_PERCENTILE = 99.99984;
export const INT8_QUANT_FORMAT = "int8_clean_per_row_v1";

/** Names of small control tensors that bypass int8 and stay fp32. */
export const CONTROL_TENSOR_NAME_PATTERNS = [
  "attn_scale",
  "mlp_scale",
  "resid_mix",
  "q_gain",
  "skip_weight",
] as const;

// ----------------------------------------------------------------------------
// Data / shard layout (mirrors train_gpt.py / data/README.md)
// ----------------------------------------------------------------------------

/** First int32 of every FineWeb .bin shard. */
export const SHARD_MAGIC = 20240520;
/** Number of int32s in the shard header. */
export const SHARD_HEADER_INTS = 256;

export const DEFAULT_DATA_DIR = path.join(
  os.homedir(),
  ".eliza",
  "slm16",
  "data",
  "fineweb10B_sp1024",
);
export const DEFAULT_TOKENIZER_PATH = path.join(
  os.homedir(),
  ".eliza",
  "slm16",
  "tokenizers",
  "fineweb_1024_bpe.model",
);
export const TRAIN_GLOB = "fineweb_train_*.bin";
export const VAL_GLOB = "fineweb_val_*.bin";

// ----------------------------------------------------------------------------
// Training cadence
// ----------------------------------------------------------------------------

/** Run validation (and consider saving a new LKG) every N optimizer steps. */
export const VAL_EVERY_STEPS = 100;
/** How often the worker posts a heartbeat status to the parent (steps). */
export const REPORT_EVERY_STEPS = 10;
/** Generation temperature for the SLM during evaluation/inference. */
export const INFERENCE_TEMPERATURE = 0.1;
/** Max tokens to generate per evaluation prompt. */
export const INFERENCE_MAX_NEW_TOKENS = 64;

// ----------------------------------------------------------------------------
// Checkpoint paths
// ----------------------------------------------------------------------------

export const DEFAULT_CKPT_DIR = path.join(os.homedir(), ".eliza", "slm16");
/** Last-Known-Good model artifact (int8 + zlib, <16 MB). This is the
 *  *canonical pointer* — always a byte-identical copy of the newest entry
 *  in the rotation set below. Used for both inference and resumption. */
export const LKG_FILENAME = "slm16.lkg.int8.ptz";
/** Sidecar JSON with step / val_loss / hparams / history for the LKG. */
export const LKG_META_FILENAME = "slm16.lkg.meta.json";
/** Versioned checkpoint pattern: `slm16.lkg.000123.int8.ptz`. The newest
 *  N of these are retained so a corrupted canonical file can be recovered
 *  by falling back through the rotation. */
export const LKG_VERSIONED_RE = /^slm16\.lkg\.(\d+)\.int8\.ptz$/;
export const lkgVersionedFilename = (step: number): string =>
  `slm16.lkg.${String(step).padStart(6, "0")}.int8.ptz`;
export const lkgVersionedMetaFilename = (step: number): string =>
  `slm16.lkg.${String(step).padStart(6, "0")}.meta.json`;
/** How many versioned LKG checkpoints to keep on disk. */
export const LKG_KEEP_N = 3;

// ----------------------------------------------------------------------------
// Backend / hardware
// ----------------------------------------------------------------------------

export type Slm16BackendName = "cuda" | "metal" | "cpu";

/** Default fraction of VRAM the trainer is allowed to occupy. */
export const DEFAULT_VRAM_FRACTION = 0.5;

// ----------------------------------------------------------------------------
// Runtime settings keys (read via runtime.getSetting / written via setSetting)
// ----------------------------------------------------------------------------

/** Validation cross-entropy of the current LKG model (lower = stronger). */
export const SETTING_VAL_LOSS = "slm16_val_loss";
/** Derived intelligence score (0–100; cosine-sim vs Nova Pro × 100). */
export const SETTING_INTELLIGENCE = "slm16_intelligence";
/** Current training step (monotonic across resumes). */
export const SETTING_STEP = "slm16_step";
/** Current backend in use ("cuda" | "metal" | "cpu"). */
export const SETTING_BACKEND = "slm16_backend";

// ----------------------------------------------------------------------------
// Environment variables (see .env.example)
// ----------------------------------------------------------------------------

/** "false"/"0" to disable autonomous background training on boot. */
export const ENV_SLM16_AUTOTRAIN = "SLM16_AUTOTRAIN";
/** Override checkpoint/artifact directory. */
export const ENV_SLM16_CKPT_DIR = "SLM16_CKPT_DIR";
/** Override training-data directory (FineWeb shards). */
export const ENV_SLM16_DATA_DIR = "SLM16_DATA_DIR";
/** Override SentencePiece tokenizer .model path. */
export const ENV_SLM16_TOKENIZER_PATH = "SLM16_TOKENIZER_PATH";
/** Number of FineWeb train shards to fetch on first boot (≈191 MB each). */
export const ENV_SLM16_TRAIN_SHARDS = "SLM16_TRAIN_SHARDS";
/** Fraction (0..1] of VRAM the trainer may use. */
export const ENV_SLM16_VRAM_FRACTION = "SLM16_VRAM_FRACTION";
/** Force a specific backend ("cuda" | "metal" | "cpu"). */
export const ENV_SLM16_BACKEND = "SLM16_BACKEND";
/** Run the Nova Pro intelligence evaluation every N validations. */
export const ENV_SLM16_EVAL_EVERY = "SLM16_EVAL_EVERY";

// ----------------------------------------------------------------------------
// Evaluation prompt set (used for SLM ↔ Nova Pro comparison)
// ----------------------------------------------------------------------------

export const EVAL_PROMPTS: readonly string[] = [
  "The capital of France is",
  "Water is composed of hydrogen and",
  "In computer science, a stack is a data structure that",
  "The mitochondria is often called the",
  "To sort a list of numbers in ascending order, one common algorithm is",
];
