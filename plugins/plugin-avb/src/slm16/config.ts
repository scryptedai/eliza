/**
 * SLM16 — Hyperparameters & runtime configuration.
 *
 * Direct port of OpenAI Parameter Golf "Naive Baseline" (2026-03-17):
 *   github.com/openai/parameter-golf/records/track_10min_16mb/2026-03-17_NaiveBaseline
 *
 * Architecture: 9-layer 512-dim GPT with U-Net skip connections, GQA(8/4),
 * RoPE, RMSNorm, relu² MLP, tied embeddings, logit softcap.
 *
 * Reference numbers (8×H100, 10min wallclock, ~13.8k steps):
 *   params:        ~17M
 *   val_loss:      2.0727 (post-int8-roundtrip)
 *   val_bpb:       1.2244
 *   artifact:      15,815,847 bytes (int8 + zlib9)
 *
 * Background-mode defaults below are scaled DOWN from the 8×H100 leaderboard
 * config — we run continuously on commodity hardware, not a 10-minute sprint.
 */

import { homedir } from "node:os";
import { join } from "node:path";

// ----------------------------------------------------------------------------
// Architecture (frozen — must match int8 quantizer assumptions)
// ----------------------------------------------------------------------------

export const ARCH = {
  /** SentencePiece sp1024 tokenizer vocab. */
  vocabSize: 1024,
  /** Total transformer blocks. Encoder = floor(L/2)=4, decoder = 5. */
  numLayers: 9,
  /** Residual stream width. */
  modelDim: 512,
  /** Total attention heads (Q). */
  numHeads: 8,
  /** KV heads for grouped-query attention. */
  numKvHeads: 4,
  /** MLP hidden = mlpMult × modelDim. */
  mlpMult: 2,
  /** Tie input embedding to output projection (logits = x @ embᵀ). */
  tieEmbeddings: true,
  /** RoPE base frequency. */
  ropeBase: 10000.0,
  /** logits = softcap × tanh(logits / softcap). */
  logitSoftcap: 30.0,
  /** Per-head Q gain (learnable scalar, init value). */
  qkGainInit: 1.5,
  /** Tied embedding init: N(0, std). */
  tiedEmbedInitStd: 0.005,
  /** Causal sequence length. */
  seqLen: 1024,
} as const;

/** Derived: per-head dimension. */
export const HEAD_DIM = ARCH.modelDim / ARCH.numHeads; // 64
/** Derived: KV projection dimension (GQA). */
export const KV_DIM = ARCH.numKvHeads * HEAD_DIM; // 256
/** Derived: U-Net split. */
export const NUM_ENCODER_LAYERS = Math.floor(ARCH.numLayers / 2); // 4
export const NUM_DECODER_LAYERS = ARCH.numLayers - NUM_ENCODER_LAYERS; // 5
/** Derived: skip-connection count (decoder pops encoder outputs LIFO). */
export const NUM_SKIP_WEIGHTS = Math.min(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS); // 4

// ----------------------------------------------------------------------------
// Quantization (int8 + zlib) — must produce <16MB artifact
// ----------------------------------------------------------------------------

export const QUANT = {
  /** 16,000,000 decimal bytes (NOT 16 MiB). Hard cap from challenge rules. */
  artifactCapBytes: 16_000_000,
  /** Tensors with ≤ this many elements stay in fp16 (passthrough). */
  keepFloatMaxNumel: 65_536,
  /** Per-row quantile clip before symmetric int8 scale derivation. */
  clipPercentile: 99.99984,
  /** zlib compression level for the final blob. */
  zlibLevel: 9,
  /** Format tag embedded in serialized header for forward compat. */
  formatTag: "slm16_int8_per_row_v1",
} as const;

// ----------------------------------------------------------------------------
// Training (background mode — continuous, resumable)
// ----------------------------------------------------------------------------

export interface Slm16TrainingConfig {
  /** PRNG seed for weight init + data shuffling. */
  seed: number;
  /** Validation cadence in optimizer steps. Task spec: every 100 steps. */
  valEvery: number;
  /** Tokens per gradient step (across all microbatches). */
  batchTokens: number;
  /** Microbatches per step (gradient accumulation). */
  gradAccumSteps: number;
  /** Adam: tied embedding LR. */
  embedLr: number;
  /** Muon: matrix (Linear weight) LR. */
  matrixLr: number;
  /** Adam: scalar/control parameter LR. */
  scalarLr: number;
  /** Muon momentum (final value after warmup). */
  muonMomentum: number;
  /** Muon momentum at step 0. */
  muonMomentumWarmupStart: number;
  /** Steps to ramp Muon momentum from start → final. */
  muonMomentumWarmupSteps: number;
  /** Newton-Schulz iterations for Muon orthogonalization. */
  muonBackendSteps: number;
  /** Adam β₁. */
  beta1: number;
  /** Adam β₂. */
  beta2: number;
  /** Adam ε. */
  adamEps: number;
  /** Fraction of available VRAM/unified memory to use [0,1]. */
  vramFraction: number;
  /** Where to read .bin token shards from. */
  dataDir: string;
  /** Where to persist checkpoints + LKG model. */
  checkpointDir: string;
  /** Hard cap on training tokens (0 = unlimited / run forever). */
  maxTokens: number;
  /** Hard cap on optimizer steps (0 = unlimited). For short proving runs. */
  maxSteps: number;
}

export const TRAIN_DEFAULTS: Slm16TrainingConfig = {
  seed: 1337,
  valEvery: 100,
  // Background mode: ~16K tokens/step is gentle. Leaderboard uses 524288.
  batchTokens: 16_384,
  gradAccumSteps: 4,
  embedLr: 0.05,
  matrixLr: 0.04,
  scalarLr: 0.04,
  muonMomentum: 0.95,
  muonMomentumWarmupStart: 0.85,
  muonMomentumWarmupSteps: 500,
  muonBackendSteps: 5,
  beta1: 0.9,
  beta2: 0.95,
  adamEps: 1e-8,
  vramFraction: 0.5,
  dataDir: join(homedir(), ".eliza", "slm16", "data"),
  checkpointDir: join(homedir(), ".eliza", "slm16", "checkpoints"),
  maxTokens: 0,
  maxSteps: 0,
};

// ----------------------------------------------------------------------------
// Environment variable keys (read via runtime.getSetting)
// ----------------------------------------------------------------------------

export const ENV = {
  /** Disable autonomous training spawn ("false"/"0"). */
  AUTOTRAIN: "SLM16_AUTOTRAIN",
  /** Override fraction of VRAM/unified memory [0,1]. */
  VRAM_FRACTION: "SLM16_VRAM_FRACTION",
  /** Override token-shard directory. */
  DATA_DIR: "SLM16_DATA_DIR",
  /** Override checkpoint directory. */
  CHECKPOINT_DIR: "SLM16_CHECKPOINT_DIR",
  /** Override validation cadence (steps). */
  VAL_EVERY: "SLM16_VAL_EVERY",
  /** Override batch tokens per step. */
  BATCH_TOKENS: "SLM16_BATCH_TOKENS",
  /** Override seed. */
  SEED: "SLM16_SEED",
} as const;

// ----------------------------------------------------------------------------
// Runtime setting keys (written via runtime.setSetting — visible to personality)
// ----------------------------------------------------------------------------

export const SETTING = {
  /**
   * Best (lowest) validation loss observed so far. Float, lower is better.
   * Reference: ~2.07 is the trained baseline. ~6.9 (= ln 1024) is random init.
   * The AVB personality reads this to gauge how "smart" its local model is.
   */
  VAL_LOSS: "slm16_val_loss",
  /**
   * Intelligence score [0,1] derived from cosine similarity vs Nova Pro.
   * 0 = orthogonal/random, 1 = perfect agreement at temp 0.1.
   */
  INTELLIGENCE: "slm16_intelligence",
  /** Current optimizer step (resumption point). */
  STEP: "slm16_step",
  /** Wallclock seconds spent training so far. */
  TRAIN_TIME: "slm16_train_seconds",
  /** "training" | "idle" | "error" | "no_data" */
  STATUS: "slm16_status",
} as const;

// ----------------------------------------------------------------------------
// Filesystem layout (under checkpointDir)
// ----------------------------------------------------------------------------

export const PATHS = {
  /** Last-Known-Good model: best val_loss seen, int8+zlib, <16MB. Used for inference + resumption. */
  lkg: "lkg.int8.bin",
  /** Latest training state (fp32 weights + optimizer moments + step). NOT size-capped. */
  state: "trainer_state.bin",
  /** JSON metadata (step, val_loss, timestamps). */
  meta: "meta.json",
  /** Tokenizer model (SentencePiece sp1024). */
  tokenizer: "fineweb_1024_bpe.model",
} as const;

/**
 * How many LKG generations to keep on disk. Canonical is `lkg.int8.bin`,
 * older copies are `lkg.int8.bin.1`, `.2`, etc. (logrotate-style). Resumption
 * and inference fall through the chain if the canonical fails to dequantize.
 */
export const LKG_KEEP = 3;

// ----------------------------------------------------------------------------
// Data shard format (FineWeb .bin)
// ----------------------------------------------------------------------------

export const SHARD = {
  /** Magic int32 at header[0]. */
  magic: 20240520,
  /** Version int32 at header[1]. */
  version: 1,
  /** Header length in int32 words. header[2] = num_tokens. */
  headerInts: 256,
  /** Token dtype on disk. */
  tokenBytes: 2, // uint16
} as const;

// ----------------------------------------------------------------------------
// Scoring (Nova Pro cosine similarity)
// ----------------------------------------------------------------------------

export const SCORING = {
  /** Sampling temperature for SLM16 inference during scoring. */
  temperature: 0.1,
  /** Number of prompt/completion pairs to evaluate per scoring round. */
  numSamples: 8,
  /** Max tokens to generate per sample. */
  maxNewTokens: 64,
  /**
   * Floor cosine similarity below which we treat as zero intelligence.
   * Random text vs coherent text typically lands ~0.3–0.5 with embedding models.
   */
  similarityFloor: 0.4,
  /** Ceiling cosine similarity treated as perfect score. */
  similarityCeiling: 0.95,
} as const;

/**
 * Built-in scoring prompts. Each is a (context, question) pair where Nova Pro
 * and SLM16 both complete the question; we embed both completions and measure
 * cosine similarity. Prompts are short, factual, and FineWeb-domain so a
 * 17M-param model has a fighting chance.
 */
export const SCORING_PROMPTS: ReadonlyArray<{
  context: string;
  question: string;
}> = [
  {
    context: "The capital of France is a major European city.",
    question: "What is the capital of France?",
  },
  {
    context: "Water boils at a specific temperature at sea level.",
    question: "At what temperature does water boil?",
  },
  {
    context: "The Earth orbits the Sun once per year.",
    question: "How long does it take Earth to orbit the Sun?",
  },
  {
    context: "Photosynthesis is how plants make food from sunlight.",
    question: "What process do plants use to convert sunlight to energy?",
  },
  {
    context: "The Pacific is the largest ocean on Earth.",
    question: "Which ocean is the largest?",
  },
  {
    context: "Shakespeare wrote many famous plays in English.",
    question: "Who wrote Romeo and Juliet?",
  },
  {
    context: "DNA carries genetic information in living things.",
    question: "What molecule carries genetic information?",
  },
  {
    context: "The speed of light is the fastest anything can travel.",
    question: "What is the speed of light approximately?",
  },
];

// ----------------------------------------------------------------------------
// Settings resolver (env → config with type coercion)
// ----------------------------------------------------------------------------

type GetSetting = (key: string) => unknown;

function num(g: GetSetting, key: string, fallback: number): number {
  const v = g(key);
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const n = Number.parseFloat(v);
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

function str(g: GetSetting, key: string, fallback: string): string {
  const v = g(key);
  return typeof v === "string" && v.length > 0 ? v : fallback;
}

export function resolveTrainingConfig(
  getSetting: GetSetting,
): Slm16TrainingConfig {
  const d = TRAIN_DEFAULTS;
  const vram = num(getSetting, ENV.VRAM_FRACTION, d.vramFraction);
  return {
    ...d,
    seed: num(getSetting, ENV.SEED, d.seed),
    valEvery: Math.max(1, Math.floor(num(getSetting, ENV.VAL_EVERY, d.valEvery))),
    batchTokens: Math.max(
      ARCH.seqLen,
      Math.floor(num(getSetting, ENV.BATCH_TOKENS, d.batchTokens)),
    ),
    vramFraction: Math.min(0.95, Math.max(0.05, vram)),
    dataDir: str(getSetting, ENV.DATA_DIR, d.dataDir),
    checkpointDir: str(getSetting, ENV.CHECKPOINT_DIR, d.checkpointDir),
  };
}

/**
 * Detect compute backend at runtime. Returns the backend name to load.
 * On macOS Apple Silicon, tfjs-node uses Apple's Accelerate framework
 * (vecLib BLAS + AMX coprocessor) — that is the practical "Metal-class"
 * path in Node. On Linux/Windows with NVIDIA, tfjs-node-gpu binds CUDA.
 */
export function detectBackend(): {
  pkg: "@tensorflow/tfjs-node-gpu" | "@tensorflow/tfjs-node";
  label: "cuda" | "accelerate" | "cpu";
} {
  if (process.platform === "darwin") {
    // tfjs-node-gpu does not exist for darwin; tfjs-node → Accelerate.
    return { pkg: "@tensorflow/tfjs-node", label: "accelerate" };
  }
  // Try GPU first; trainer will fall back to CPU pkg if import fails.
  return { pkg: "@tensorflow/tfjs-node-gpu", label: "cuda" };
}
