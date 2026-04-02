/**
 * SLM16 — GPT with U-Net skip connections.
 *
 * Faithful TypeScript port of the OpenAI Parameter Golf reference architecture
 * (`train_gpt.py`, Naive Baseline). All ops compose from tfjs primitives so
 * gradients are automatic — no tf.customGrad needed.
 *
 * Layer-by-layer correspondence (PyTorch → tfjs):
 *   nn.Embedding              → tf.gather on a [V,D] Variable
 *   F.rms_norm                → x * rsqrt(mean(x², -1) + eps)
 *   CastedLinear              → tf.matMul (we keep all weights fp32)
 *   Rotary                    → cached cos/sin tables, interleaved-half rotation
 *   F.scaled_dot_product_attention(is_causal, enable_gqa)
 *                             → manual QKᵀ + causal mask + softmax + V matmul
 *                               with K/V repeated to match Q heads
 *   relu² MLP                 → relu(fc(x))² then proj
 *   U-Net skip                → encoder pushes, decoder pops LIFO + learnable scale
 *   logit softcap             → softcap * tanh(logits / softcap)
 *   F.cross_entropy           → -log_softmax(logits)[target] mean
 *
 * The forward returns scalar loss; backprop comes from tf.variableGrads().
 *
 * Reference: github.com/openai/parameter-golf/blob/main/train_gpt.py L500–L724
 */

import type * as tfTypes from "@tensorflow/tfjs";
import {
  ARCH,
  HEAD_DIM,
  KV_DIM,
  NUM_DECODER_LAYERS,
  NUM_ENCODER_LAYERS,
  NUM_SKIP_WEIGHTS,
} from "./config.ts";

// We type against the pure-JS package but at runtime the worker injects
// tfjs-node or tfjs-node-gpu. Same surface, different binding.
type TF = typeof tfTypes;
type Tensor = tfTypes.Tensor;
type Variable = tfTypes.Variable;
type Scalar = tfTypes.Scalar;

/**
 * tfjs types `Tensor.shape` as `(number | undefined)[]` because symbolic
 * tensors can carry unknown dims. At runtime our shapes are always concrete
 * (we never use symbolic execution); this helper centralizes the assertion
 * so the rest of the file reads naturally.
 */
function dim(t: Tensor, axis: number): number {
  const d = t.shape[axis];
  if (d === undefined) {
    throw new Error(`tensor has symbolic dimension at axis ${axis}`);
  }
  return d;
}

/**
 * PyTorch-style linear: y = x @ Wᵀ where W is [out, in].
 *
 * Why this exists instead of inlining `tf.matMul(x, W, false, true)`:
 * tfjs's BatchMatMul gradient does NOT auto-reduce the batch dimension when
 * a 2D weight is broadcast against a 3D+ input. The forward pass implicitly
 * unsqueezes W to [1, out, in] for the batched op, and the backward pass
 * returns ∂L/∂W with that leading 1 still attached — failing tfjs's own
 * internal shape check before our code ever sees the gradient. This bit us
 * in the very first live training step (see prove.ts run 2026-04-02).
 *
 * The workaround is to flatten leading dims to 2D for the matmul (so there
 * IS no broadcast) and reshape back. Same numerics, autodiff-safe.
 *
 * x: [..., in], W: [out, in] → returns [..., out]
 */
function linear(tf: TF, x: Tensor, weight: Tensor): Tensor {
  const outDim = dim(weight, 0);
  const inDim = dim(weight, 1);
  const flat = tf.reshape(x, [-1, inDim]); // [N, in]
  const y = tf.matMul(flat, weight, false, true); // [N, out]
  // Reconstruct leading dims with the last axis swapped to outDim.
  // Cast: at runtime our shapes are always concrete (no tf.input()).
  const lead = x.shape.slice(0, -1) as number[];
  return tf.reshape(y, [...lead, outDim]);
}

const RMS_EPS = 1e-6;

// ----------------------------------------------------------------------------
// Parameter container — flat record so the optimizer can iterate easily
// ----------------------------------------------------------------------------

/**
 * All learnable weights, keyed by stable string names. The naming scheme
 * mirrors PyTorch's state_dict() so the int8 quantizer can use the same
 * pattern-matching rules (e.g. tensors containing "scale" or "skip" stay fp16).
 */
export interface Slm16Weights {
  // Embedding (also tied to output projection).
  "tok_emb.weight": Variable; // [V, D]

  // U-Net skip scales (one per decoder skip).
  skip_weights: Variable; // [numSkip, D]

  // Per-block weights, indexed by block number 0..L-1.
  // We flatten the block index into the key for a single record.
  // Example: "blocks.0.attn.c_q.weight"
  [key: string]: Variable;
}

interface BlockWeightSpec {
  cQ: string; // [D, D]
  cK: string; // [KV_DIM, D]
  cV: string; // [KV_DIM, D]
  proj: string; // [D, D]
  qGain: string; // [H]
  fc: string; // [D*mlpMult, D]
  mlpProj: string; // [D, D*mlpMult]
  attnScale: string; // [D]
  mlpScale: string; // [D]
  resMix: string; // [2, D]
}

function blockKeys(i: number): BlockWeightSpec {
  const b = `blocks.${i}`;
  return {
    cQ: `${b}.attn.c_q.weight`,
    cK: `${b}.attn.c_k.weight`,
    cV: `${b}.attn.c_v.weight`,
    proj: `${b}.attn.proj.weight`,
    qGain: `${b}.attn.q_gain`,
    fc: `${b}.mlp.fc.weight`,
    mlpProj: `${b}.mlp.proj.weight`,
    attnScale: `${b}.attn_scale`,
    mlpScale: `${b}.mlp_scale`,
    resMix: `${b}.resid_mix`,
  };
}

// ----------------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------------

/**
 * Build a freshly initialized weight set. Matches `_init_weights` in the
 * PyTorch reference: tied embedding ~ N(0, std), output projections zero-init,
 * everything else default (Glorot uniform for linears, ones/zeros for control).
 */
export function initWeights(tf: TF, seed: number): Slm16Weights {
  const D = ARCH.modelDim;
  const V = ARCH.vocabSize;
  const H = ARCH.numHeads;
  const hidden = D * ARCH.mlpMult;
  const w: Record<string, Variable> = {};

  // mulberry32 — small deterministic PRNG so the same seed gives the same
  // init across runs. tf.randomNormal accepts a seed but its sequence isn't
  // portable across backends; we generate on CPU and upload.
  let rng = seed >>> 0 || 1;
  const rand = (): number => {
    rng = (rng + 0x6d2b79f5) >>> 0;
    let t = Math.imul(rng ^ (rng >>> 15), 1 | rng);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  // Box-Muller normal sampler.
  const randn = (): number => {
    const u1 = Math.max(rand(), 1e-12);
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  const normal = (shape: number[], std: number): Float32Array => {
    const n = shape.reduce((a, b) => a * b, 1);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = randn() * std;
    return out;
  };
  const glorot = (fanOut: number, fanIn: number): Float32Array => {
    // Glorot/Xavier uniform: limit = sqrt(6 / (fanIn + fanOut))
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    const n = fanOut * fanIn;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = (rand() * 2 - 1) * limit;
    return out;
  };
  const mkVar = (
    name: string,
    data: Float32Array,
    shape: number[],
  ): Variable => {
    const v = tf.variable(tf.tensor(data, shape, "float32"), true, name);
    w[name] = v;
    return v;
  };
  const zeros = (shape: number[]) =>
    new Float32Array(shape.reduce((a, b) => a * b, 1));
  const ones = (shape: number[]) => {
    const a = new Float32Array(shape.reduce((a, b) => a * b, 1));
    a.fill(1);
    return a;
  };

  // Tied embedding: N(0, tiedEmbedInitStd) per reference.
  mkVar("tok_emb.weight", normal([V, D], ARCH.tiedEmbedInitStd), [V, D]);

  // Skip weights init to ones (per-channel scale).
  mkVar("skip_weights", ones([NUM_SKIP_WEIGHTS, D]), [NUM_SKIP_WEIGHTS, D]);

  for (let i = 0; i < ARCH.numLayers; i++) {
    const k = blockKeys(i);
    // Attention projections — Glorot for Q/K/V, zero for output proj.
    mkVar(k.cQ, glorot(D, D), [D, D]);
    mkVar(k.cK, glorot(KV_DIM, D), [KV_DIM, D]);
    mkVar(k.cV, glorot(KV_DIM, D), [KV_DIM, D]);
    mkVar(k.proj, zeros([D, D]), [D, D]); // _zero_init in reference
    // Per-head Q gain.
    mkVar(
      k.qGain,
      new Float32Array(H).fill(ARCH.qkGainInit),
      [H],
    );
    // MLP — Glorot for fc, zero for proj (output).
    mkVar(k.fc, glorot(hidden, D), [hidden, D]);
    mkVar(k.mlpProj, zeros([D, hidden]), [D, hidden]); // _zero_init
    // Control scalars.
    mkVar(k.attnScale, ones([D]), [D]);
    mkVar(k.mlpScale, ones([D]), [D]);
    // resid_mix: row 0 = ones (keep current), row 1 = zeros (no x0 mix).
    const mix = new Float32Array(2 * D);
    for (let j = 0; j < D; j++) mix[j] = 1; // row 0
    mkVar(k.resMix, mix, [2, D]);
  }

  return w as Slm16Weights;
}

/** Total trainable parameter count (sanity-check vs ~17M). */
export function countParams(w: Slm16Weights): number {
  let n = 0;
  for (const v of Object.values(w)) {
    n += v.shape.reduce((a, b) => a * b, 1);
  }
  return n;
}

/** Hash of architecture constants — used to refuse resuming a mismatched checkpoint. */
export function archHash(): string {
  const parts = [
    ARCH.vocabSize,
    ARCH.numLayers,
    ARCH.modelDim,
    ARCH.numHeads,
    ARCH.numKvHeads,
    ARCH.mlpMult,
    ARCH.tieEmbeddings ? 1 : 0,
    ARCH.seqLen,
  ];
  // FNV-1a 32-bit over the comma-joined string.
  let h = 0x811c9dc5;
  for (const c of parts.join(",")) {
    h ^= c.charCodeAt(0);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

// ----------------------------------------------------------------------------
// Forward pass
// ----------------------------------------------------------------------------

interface ForwardCache {
  /** Causal mask: [1, 1, T, T] with -inf above diagonal, 0 on/below. */
  causalMask: Tensor;
  /** RoPE cos: [1, 1, T, headDim/2]. */
  ropeCos: Tensor;
  /** RoPE sin: same shape. */
  ropeSin: Tensor;
  /** Cached for sequence length T. Rebuild if T changes. */
  seqLen: number;
}

let _cache: ForwardCache | null = null;

function buildCache(tf: TF, seqLen: number): ForwardCache {
  // Causal mask: lower-triangular keep, upper-triangular -inf.
  // tfjs doesn't have triu so we build it from index comparison.
  const idx = tf.range(0, seqLen, 1, "int32"); // [T]
  const row = idx.reshape([seqLen, 1]); // [T,1]
  const col = idx.reshape([1, seqLen]); // [1,T]
  const upper = col.greater(row); // [T,T] bool: true above diagonal
  const negInf = tf.fill([seqLen, seqLen], -1e9);
  const zerosMask = tf.zeros([seqLen, seqLen]);
  const mask = tf.where(upper, negInf, zerosMask).reshape([1, 1, seqLen, seqLen]);
  idx.dispose();
  row.dispose();
  col.dispose();
  upper.dispose();
  negInf.dispose();
  zerosMask.dispose();

  // RoPE tables.
  const halfDim = HEAD_DIM / 2;
  const invFreq = new Float32Array(halfDim);
  for (let i = 0; i < halfDim; i++) {
    invFreq[i] = 1.0 / ARCH.ropeBase ** ((2 * i) / HEAD_DIM);
  }
  const tArr = new Float32Array(seqLen);
  for (let i = 0; i < seqLen; i++) tArr[i] = i;
  // freqs[t, i] = t * invFreq[i]
  const freqs = new Float32Array(seqLen * halfDim);
  for (let t = 0; t < seqLen; t++) {
    for (let i = 0; i < halfDim; i++) {
      freqs[t * halfDim + i] = tArr[t] * invFreq[i];
    }
  }
  const cosArr = new Float32Array(seqLen * halfDim);
  const sinArr = new Float32Array(seqLen * halfDim);
  for (let i = 0; i < freqs.length; i++) {
    cosArr[i] = Math.cos(freqs[i]);
    sinArr[i] = Math.sin(freqs[i]);
  }
  const ropeCos = tf.tensor(cosArr, [1, 1, seqLen, halfDim], "float32");
  const ropeSin = tf.tensor(sinArr, [1, 1, seqLen, halfDim], "float32");

  // tf.keep() excludes these from the enclosing tidy/variableGrads scope's
  // cleanup list. Without it, the cache is built on the first forward()
  // call (inside a tidy), disposed when that tidy exits, and the second
  // forward hits a use-after-free at applyRope. We manage these tensors'
  // lifetime ourselves via disposeCache().
  return {
    causalMask: tf.keep(mask),
    ropeCos: tf.keep(ropeCos),
    ropeSin: tf.keep(ropeSin),
    seqLen,
  };
}

function getCache(tf: TF, seqLen: number): ForwardCache {
  if (_cache && _cache.seqLen === seqLen) return _cache;
  if (_cache) {
    _cache.causalMask.dispose();
    _cache.ropeCos.dispose();
    _cache.ropeSin.dispose();
  }
  _cache = buildCache(tf, seqLen);
  return _cache;
}

/** RMSNorm: x / sqrt(mean(x², -1) + eps). No learnable scale (matches reference). */
function rmsNorm(tf: TF, x: Tensor): Tensor {
  // mean over last dim, keepdims
  const sq = tf.square(x);
  const mean = tf.mean(sq, -1, true);
  const denom = tf.rsqrt(tf.add(mean, RMS_EPS));
  return tf.mul(x, denom);
}

/**
 * Apply RoPE to a [B, H, T, headDim] tensor. Reference splits the last dim
 * in half and rotates: out = [x1*cos + x2*sin, -x1*sin + x2*cos].
 */
function applyRope(
  tf: TF,
  x: Tensor,
  cos: Tensor,
  sin: Tensor,
): Tensor {
  const half = HEAD_DIM / 2;
  // Split last dim. Size -1 = "to end" along that axis.
  const x1 = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, half]);
  const x2 = tf.slice(x, [0, 0, 0, half], [-1, -1, -1, half]);
  // out1 = x1*cos + x2*sin ; out2 = x2*cos - x1*sin
  const out1 = tf.add(tf.mul(x1, cos), tf.mul(x2, sin));
  const out2 = tf.sub(tf.mul(x2, cos), tf.mul(x1, sin));
  return tf.concat([out1, out2], -1);
}

/**
 * Causal grouped-query attention. Q has H heads, K/V have Hkv heads.
 * Reference uses F.scaled_dot_product_attention(enable_gqa=True); we
 * manually repeat K/V groups to match Q heads (memory cost is small at d=512).
 */
function attention(
  tf: TF,
  x: Tensor, // [B, T, D]
  w: Slm16Weights,
  k: BlockWeightSpec,
  cache: ForwardCache,
): Tensor {
  const B = dim(x, 0);
  const T = dim(x, 1);
  const H = ARCH.numHeads;
  const Hkv = ARCH.numKvHeads;
  const groupSize = H / Hkv;

  // Linear projections. PyTorch Linear stores weight as [out, in] and
  // computes x @ Wᵀ; we keep the same convention so quantized weights
  // are interchangeable. The linear() helper handles the 3D→2D reshape
  // needed for autodiff (see its docstring).
  const q = linear(tf, x, w[k.cQ]); // [B,T,D]
  const kProj = linear(tf, x, w[k.cK]); // [B,T,KV_DIM]
  const v = linear(tf, x, w[k.cV]); // [B,T,KV_DIM]

  // Reshape to heads. Q: [B,T,H,d] → [B,H,T,d]; KV: [B,T,Hkv,d] → [B,Hkv,T,d].
  const qH = tf.transpose(q.reshape([B, T, H, HEAD_DIM]), [0, 2, 1, 3]);
  const kH = tf.transpose(kProj.reshape([B, T, Hkv, HEAD_DIM]), [0, 2, 1, 3]);
  const vH = tf.transpose(v.reshape([B, T, Hkv, HEAD_DIM]), [0, 2, 1, 3]);

  // QK-RMSNorm (per reference: normalize Q and K independently along headDim).
  const qN = rmsNorm(tf, qH);
  const kN = rmsNorm(tf, kH);

  // RoPE on Q and K.
  const qR = applyRope(tf, qN, cache.ropeCos, cache.ropeSin);
  const kR = applyRope(tf, kN, cache.ropeCos, cache.ropeSin);

  // Per-head learnable Q gain: [H] → [1,H,1,1].
  const gain = w[k.qGain].reshape([1, H, 1, 1]);
  const qG = tf.mul(qR, gain);

  // Repeat K/V along head axis to match Q (GQA expansion):
  // [B, Hkv, T, d] → [B, H, T, d] where each KV head is shared by `groupSize`
  // consecutive Q heads.
  //
  // The natural way is `tf.tile` over a dummy axis, but tfjs's tile gradient
  // is only implemented up to rank 4 — the rank-5 intermediate
  // `[B, Hkv, 1, T, d]` we'd need throws "not implemented for rank-5" during
  // backprop. We use `tf.gather` along the head axis instead: build an index
  // map [0,0,1,1,…,Hkv-1,Hkv-1] (each KV head repeated `groupSize` times) and
  // gather. The gather gradient is scatter-add, which sums grads from Q-heads
  // 0,1 back into KV-head 0 etc. — exactly the right reduction, and
  // rank-agnostic.
  //
  // The map is fixed by ARCH (8 ints for our 8/4 GQA). Computed inline; tfjs
  // will fold the small constant into the graph.
  const idxArr = new Int32Array(H);
  for (let i = 0; i < H; i++) idxArr[i] = (i / groupSize) | 0;
  const idx = tf.tensor1d(idxArr, "int32");
  const kRep = tf.gather(kR, idx, 1); // [B, H, T, d]
  const vRep = tf.gather(vH, idx, 1); // [B, H, T, d]

  // Attention scores: Q @ Kᵀ / sqrt(d)  → [B, H, T, T]
  const scale = 1.0 / Math.sqrt(HEAD_DIM);
  const scores = tf.mul(tf.matMul(qG, kRep, false, true), scale);
  const masked = tf.add(scores, cache.causalMask);
  const attn = tf.softmax(masked, -1);

  // Weighted values: attn @ V → [B, H, T, d]
  const out = tf.matMul(attn, vRep);
  // [B,H,T,d] → [B,T,H,d] → [B,T,D]
  const merged = tf.transpose(out, [0, 2, 1, 3]).reshape([B, T, ARCH.modelDim]);

  // Output projection (zero-init at start).
  return linear(tf, merged, w[k.proj]);
}

/** relu² MLP: proj(relu(fc(x))²). */
function mlp(tf: TF, x: Tensor, w: Slm16Weights, k: BlockWeightSpec): Tensor {
  const h = tf.relu(linear(tf, x, w[k.fc]));
  const h2 = tf.square(h);
  return linear(tf, h2, w[k.mlpProj]);
}

/**
 * Single transformer block. Per the reference:
 *   x = mix0*x + mix1*x0           (resid_mix to original embedding)
 *   x = x + attnScale * attn(rmsNorm(x))
 *   x = x + mlpScale  * mlp(rmsNorm(x))
 */
function block(
  tf: TF,
  x: Tensor,
  x0: Tensor,
  w: Slm16Weights,
  i: number,
  cache: ForwardCache,
): Tensor {
  const k = blockKeys(i);
  // resid_mix: [2, D] → mix0 [1,1,D], mix1 [1,1,D]
  const mix = w[k.resMix];
  const D = ARCH.modelDim;
  const mix0 = tf.slice(mix, [0, 0], [1, D]).reshape([1, 1, D]);
  const mix1 = tf.slice(mix, [1, 0], [1, D]).reshape([1, 1, D]);
  let xMixed = tf.add(tf.mul(mix0, x), tf.mul(mix1, x0));

  const aOut = attention(tf, rmsNorm(tf, xMixed), w, k, cache);
  const aScale = w[k.attnScale].reshape([1, 1, D]);
  xMixed = tf.add(xMixed, tf.mul(aScale, aOut));

  const mOut = mlp(tf, rmsNorm(tf, xMixed), w, k);
  const mScale = w[k.mlpScale].reshape([1, 1, D]);
  return tf.add(xMixed, tf.mul(mScale, mOut));
}

/**
 * Full forward pass returning scalar cross-entropy loss.
 *
 * U-Net pattern (reference L700–L713):
 *   - Encoder layers 0..3: x = block_i(x, x0); push x to skip stack.
 *   - Decoder layers 4..8: pop skip, x = x + skipWeight * skip; x = block_i(x, x0).
 *     (Layer 8 has no skip to pop since numEncoder=4 < numDecoder=5.)
 *
 * Logits use the tied embedding (x @ embᵀ) with softcap, then mean CE.
 */
export function forward(
  tf: TF,
  w: Slm16Weights,
  inputIds: Tensor, // [B, T] int32
  targetIds: Tensor, // [B, T] int32
): Scalar {
  return tf.tidy(() => {
    const T = dim(inputIds, 1);
    const D = ARCH.modelDim;
    const cache = getCache(tf, T);

    // Embedding lookup + initial RMSNorm. tf.gather on a Variable returns
    // a Tensor at runtime but the @tensorflow/tfjs typing infers Variable
    // (it preserves the input's nominal type). Widen to Tensor explicitly
    // so subsequent reassignments (block, add) typecheck.
    let x: Tensor = tf.gather(w["tok_emb.weight"], inputIds); // [B,T,D]
    x = rmsNorm(tf, x);
    const x0 = x; // frozen initial embedding for resid_mix

    // Encoder half: push skips.
    const skips: Tensor[] = [];
    for (let i = 0; i < NUM_ENCODER_LAYERS; i++) {
      x = block(tf, x, x0, w, i, cache);
      skips.push(x);
    }

    // Decoder half: pop skips LIFO with learnable per-channel scale.
    for (let j = 0; j < NUM_DECODER_LAYERS; j++) {
      if (skips.length > 0) {
        const skip = skips.pop() as Tensor;
        // skip_weights[j] : [D] → [1,1,D]
        const sw = tf.slice(w.skip_weights, [j, 0], [1, D]).reshape([1, 1, D]);
        x = tf.add(x, tf.mul(sw, skip));
      }
      x = block(tf, x, x0, w, NUM_ENCODER_LAYERS + j, cache);
    }

    // Final norm + tied logits.
    const xN = rmsNorm(tf, x); // [B,T,D]
    const flat = xN.reshape([-1, D]); // [B*T, D]
    const logitsRaw = tf.matMul(flat, w["tok_emb.weight"], false, true); // [B*T, V]

    // Logit softcap: softcap * tanh(logits / softcap).
    const sc = ARCH.logitSoftcap;
    const logits = tf.mul(tf.tanh(tf.div(logitsRaw, sc)), sc);

    // Cross-entropy: -log_softmax(logits)[target], mean.
    const logProbs = tf.logSoftmax(logits, -1); // [B*T, V]
    const targetsFlat = targetIds.reshape([-1]); // [B*T] int32
    const targetsOh = tf.oneHot(targetsFlat, ARCH.vocabSize); // [B*T, V]
    const nll = tf.neg(tf.sum(tf.mul(targetsOh, logProbs), -1)); // [B*T]
    return tf.mean(nll) as Scalar;
  });
}

/**
 * Inference-only forward returning logits for the LAST token position.
 * Used by the autoregressive sampler. No gradient tracking.
 */
export function forwardLogits(
  tf: TF,
  w: Slm16Weights,
  inputIds: Tensor, // [B, T] int32
): Tensor {
  return tf.tidy(() => {
    const T = dim(inputIds, 1);
    const D = ARCH.modelDim;
    const cache = getCache(tf, T);

    let x: Tensor = tf.gather(w["tok_emb.weight"], inputIds);
    x = rmsNorm(tf, x);
    const x0 = x;

    const skips: Tensor[] = [];
    for (let i = 0; i < NUM_ENCODER_LAYERS; i++) {
      x = block(tf, x, x0, w, i, cache);
      skips.push(x);
    }
    for (let j = 0; j < NUM_DECODER_LAYERS; j++) {
      if (skips.length > 0) {
        const skip = skips.pop() as Tensor;
        const sw = tf.slice(w.skip_weights, [j, 0], [1, D]).reshape([1, 1, D]);
        x = tf.add(x, tf.mul(sw, skip));
      }
      x = block(tf, x, x0, w, NUM_ENCODER_LAYERS + j, cache);
    }

    const xN = rmsNorm(tf, x);
    // Slice last position only: [B, 1, D] → [B, D]
    const last = tf.slice(xN, [0, T - 1, 0], [-1, 1, -1]).reshape([-1, D]);
    const logitsRaw = tf.matMul(last, w["tok_emb.weight"], false, true);
    const sc = ARCH.logitSoftcap;
    return tf.mul(tf.tanh(tf.div(logitsRaw, sc)), sc); // [B, V]
  });
}

/** Free all weight tensors. */
export function disposeWeights(w: Slm16Weights): void {
  for (const v of Object.values(w)) v.dispose();
}

/** Free the global RoPE/mask cache (call on worker shutdown). */
export function disposeCache(): void {
  if (_cache) {
    _cache.causalMask.dispose();
    _cache.ropeCos.dispose();
    _cache.ropeSin.dispose();
    _cache = null;
  }
}

/**
 * Snapshot all weights to plain Float32Arrays (for checkpointing/quantization).
 * Synchronous — runs on the worker thread, not the main agent.
 */
export function snapshotWeights(
  w: Slm16Weights,
): Record<string, { data: Float32Array; shape: number[] }> {
  const out: Record<string, { data: Float32Array; shape: number[] }> = {};
  for (const [name, v] of Object.entries(w)) {
    out[name] = { data: v.dataSync() as Float32Array, shape: [...v.shape] };
  }
  return out;
}

/**
 * Load a snapshot back into existing Variables (in-place assign).
 * Shapes must match — caller is responsible for arch-hash validation.
 */
export function loadSnapshot(
  tf: TF,
  w: Slm16Weights,
  snap: Record<string, { data: Float32Array; shape: number[] }>,
): void {
  for (const [name, v] of Object.entries(w)) {
    const s = snap[name];
    if (!s) throw new Error(`loadSnapshot: missing tensor ${name}`);
    const t = tf.tensor(s.data, s.shape, "float32");
    v.assign(t);
    t.dispose();
  }
}
