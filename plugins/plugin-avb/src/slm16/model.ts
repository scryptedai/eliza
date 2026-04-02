/**
 * SLM16 model — TypeScript port of the Parameter-Golf Naive-Baseline GPT.
 *
 * Architecture (mirrors `class GPT` in openai/parameter-golf train_gpt.py):
 *   - Token embedding (tied to lm_head)
 *   - 9 transformer blocks at d=512:
 *       • RMSNorm → GQA causal self-attention (RoPE, q_gain, qk-norm)
 *       • RMSNorm → ReLU² MLP (2× expansion)
 *       • per-block resid_mix(x, x0), attn_scale, mlp_scale
 *   - U-Net skip connections: encoder half pushes hidden states,
 *     decoder half pops + adds with learned skip_weight
 *   - Final RMSNorm → tied lm_head → logit softcap → cross-entropy
 *
 * Weights are kept in a flat name→tf.Variable map whose keys match the
 * PyTorch `state_dict()` names so the int8 quantizer (quantize.ts) and
 * checkpoint format are interchangeable with the reference implementation.
 */

import type { TF } from "./backend.ts";
import {
  CONTROL_TENSOR_NAME_PATTERNS,
  type Slm16Hyperparameters,
} from "./constants.ts";

type Tensor = import("@tensorflow/tfjs").Tensor;
type Scalar = import("@tensorflow/tfjs").Scalar;
type Variable = import("@tensorflow/tfjs").Variable;

export interface ParamGroups {
  embed: Variable[];
  matrix: Variable[];
  scalar: Variable[];
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

function isControlTensor(name: string): boolean {
  return CONTROL_TENSOR_NAME_PATTERNS.some((p) => name.includes(p));
}

// ----------------------------------------------------------------------------
// Slm16Model
// ----------------------------------------------------------------------------

export class Slm16Model {
  readonly hp: Slm16Hyperparameters;
  readonly weights: Map<string, Variable> = new Map();

  private readonly tf: TF;
  private readonly headDim: number;
  private readonly numEnc: number;
  private readonly numDec: number;
  private readonly numSkip: number;
  /** Cached RoPE cos/sin tables (recomputed if seqLen changes). */
  private rotaryCache?: { seqLen: number; cos: Tensor; sin: Tensor };
  /** Cached additive causal mask, shape [1, 1, T, T]. */
  private maskCache?: { seqLen: number; mask: Tensor };

  constructor(tf: TF, hp: Slm16Hyperparameters) {
    if (hp.modelDim % hp.numHeads !== 0) {
      throw new Error("modelDim must be divisible by numHeads");
    }
    if (hp.numHeads % hp.numKvHeads !== 0) {
      throw new Error("numHeads must be divisible by numKvHeads");
    }
    this.tf = tf;
    this.hp = hp;
    this.headDim = hp.modelDim / hp.numHeads;
    if (this.headDim % 2 !== 0) {
      throw new Error("headDim must be even for RoPE");
    }
    this.numEnc = Math.floor(hp.numLayers / 2);
    this.numDec = hp.numLayers - this.numEnc;
    this.numSkip = Math.min(this.numEnc, this.numDec);
    this.initWeights();
  }

  // --------------------------------------------------------------------------
  // Weight initialization (mirrors GPT._init_weights)
  // --------------------------------------------------------------------------

  private addVar(name: string, init: Tensor): Variable {
    const v = this.tf.variable(init, true, name);
    this.weights.set(name, v);
    return v;
  }

  private initWeights(): void {
    const { tf, hp, headDim } = this;
    const kvDim = hp.numKvHeads * headDim;
    const hidden = hp.mlpMult * hp.modelDim;

    // Default Linear init: U(-1/sqrt(fanIn), 1/sqrt(fanIn)) (PyTorch default).
    const linear = (out: number, inn: number): Tensor => {
      const k = 1 / Math.sqrt(inn);
      return tf.randomUniform([out, inn], -k, k, "float32");
    };

    // Tied embedding: N(0, tied_embed_init_std).
    this.addVar(
      "tok_emb.weight",
      tf.randomNormal([hp.vocabSize, hp.modelDim], 0, 0.005, "float32"),
    );

    // U-Net skip weights: ones [numSkip, dim].
    this.addVar(
      "skip_weights",
      tf.ones([Math.max(this.numSkip, 1), hp.modelDim], "float32"),
    );

    for (let i = 0; i < hp.numLayers; i++) {
      const b = `blocks.${i}`;
      // Attention
      this.addVar(`${b}.attn.c_q.weight`, linear(hp.modelDim, hp.modelDim));
      this.addVar(`${b}.attn.c_k.weight`, linear(kvDim, hp.modelDim));
      this.addVar(`${b}.attn.c_v.weight`, linear(kvDim, hp.modelDim));
      // proj is zero-init
      this.addVar(
        `${b}.attn.proj.weight`,
        tf.zeros([hp.modelDim, hp.modelDim], "float32"),
      );
      this.addVar(
        `${b}.attn.q_gain`,
        tf.fill([hp.numHeads], hp.qkGainInit, "float32"),
      );
      // MLP
      this.addVar(`${b}.mlp.fc.weight`, linear(hidden, hp.modelDim));
      // proj is zero-init
      this.addVar(
        `${b}.mlp.proj.weight`,
        tf.zeros([hp.modelDim, hidden], "float32"),
      );
      // Block scalars
      this.addVar(`${b}.attn_scale`, tf.ones([hp.modelDim], "float32"));
      this.addVar(`${b}.mlp_scale`, tf.ones([hp.modelDim], "float32"));
      // resid_mix: stack(ones, zeros)
      this.addVar(
        `${b}.resid_mix`,
        tf.stack([tf.ones([hp.modelDim]), tf.zeros([hp.modelDim])]),
      );
    }
  }

  // --------------------------------------------------------------------------
  // Building blocks
  // --------------------------------------------------------------------------

  /** RMSNorm over the last dim with eps matching torch default. */
  private rmsNorm(x: Tensor): Tensor {
    const { tf } = this;
    const ms = tf.mean(tf.square(x), -1, true);
    return tf.div(x, tf.sqrt(tf.add(ms, tf.scalar(1e-6))));
  }

  /**
   * F.linear(x, W) with W shape [out, in].
   *
   * Flattens leading dims to a single batch dim so the matmul is strictly
   * 2-D × 2-D. This is required because tfjs's BatchMatMul gradient does
   * NOT sum over broadcast batch dimensions — passing a 3-D `x` against a
   * 2-D `W` would broadcast W and then return a 3-D gradient for W,
   * triggering "gradient of input 'b' has shape [B,out,in]" errors.
   */
  private linear(x: Tensor, name: string): Tensor {
    const w = this.weights.get(name);
    if (!w) throw new Error(`missing weight: ${name}`);
    const { tf } = this;
    const inDim = w.shape[1] as number;
    const outDim = w.shape[0] as number;
    const lead = x.shape.slice(0, -1);
    const x2d = tf.reshape(x, [-1, inDim]);
    const y2d = tf.matMul(x2d, w, false, true);
    return tf.reshape(y2d, [...lead, outDim]);
  }

  private rotary(seqLen: number): { cos: Tensor; sin: Tensor } {
    if (this.rotaryCache?.seqLen === seqLen) {
      return this.rotaryCache;
    }
    const { tf, headDim, hp } = this;
    const half = headDim / 2;
    const idx = tf.range(0, half, 1, "float32");
    const invFreq = tf.div(
      tf.scalar(1),
      tf.pow(tf.scalar(hp.ropeBase), tf.div(tf.mul(idx, 2), headDim)),
    );
    const t = tf.range(0, seqLen, 1, "float32");
    const freqs = tf.outerProduct(t.as1D(), invFreq.as1D()); // [T, half]
    // shape [1, 1, T, half] for broadcast over [B, H, T, half]
    const cos = tf.keep(tf.reshape(tf.cos(freqs), [1, 1, seqLen, half]));
    const sin = tf.keep(tf.reshape(tf.sin(freqs), [1, 1, seqLen, half]));
    this.rotaryCache?.cos.dispose();
    this.rotaryCache?.sin.dispose();
    this.rotaryCache = { seqLen, cos, sin };
    return this.rotaryCache;
  }

  /** apply_rotary_emb on [B, H, T, D]. */
  private applyRope(x: Tensor, cos: Tensor, sin: Tensor): Tensor {
    const { tf } = this;
    const half = this.headDim / 2;
    const [x1, x2] = tf.split(x, 2, -1);
    // [x1*cos + x2*sin, -x1*sin + x2*cos]
    const r1 = tf.add(tf.mul(x1, cos), tf.mul(x2, sin));
    const r2 = tf.add(tf.mul(tf.neg(x1), sin), tf.mul(x2, cos));
    void half;
    return tf.concat([r1, r2], -1);
  }

  private causalMask(seqLen: number): Tensor {
    if (this.maskCache?.seqLen === seqLen) return this.maskCache.mask;
    const { tf } = this;
    // Upper-triangular (j>i) → -inf additive mask. Shape [1,1,T,T].
    const ones = tf.ones([seqLen, seqLen], "float32");
    const upper = tf.sub(
      ones,
      tf.linalg.bandPart(ones, -1, 0), // lower-tri incl. diag
    );
    const mask = tf.keep(
      tf.reshape(tf.mul(upper, tf.scalar(-1e9)), [1, 1, seqLen, seqLen]),
    );
    this.maskCache?.mask.dispose();
    this.maskCache = { seqLen, mask };
    return mask;
  }

  /** GQA causal self-attention for block `i`. Input/output [B, T, D]. */
  private attention(x: Tensor, i: number): Tensor {
    const { tf, hp, headDim } = this;
    const b = `blocks.${i}.attn`;
    const [B, T] = x.shape as [number, number, number];
    const H = hp.numHeads;
    const Hk = hp.numKvHeads;
    const repeat = H / Hk;
    const { cos, sin } = this.rotary(T);

    // Project + reshape to [B, H, T, headDim]
    const toHeads = (t: Tensor, h: number): Tensor =>
      tf.transpose(tf.reshape(t, [B, T, h, headDim]), [0, 2, 1, 3]);

    let q = toHeads(this.linear(x, `${b}.c_q.weight`), H);
    let k = toHeads(this.linear(x, `${b}.c_k.weight`), Hk);
    let v = toHeads(this.linear(x, `${b}.c_v.weight`), Hk);

    // qk-norm (RMSNorm over headDim)
    q = this.rmsNorm(q);
    k = this.rmsNorm(k);
    // RoPE
    q = this.applyRope(q, cos, sin);
    k = this.applyRope(k, cos, sin);
    // q_gain per head
    const qGain = tf.reshape(this.weights.get(`${b}.q_gain`) as Variable, [
      1,
      H,
      1,
      1,
    ]);
    q = tf.mul(q, qGain);

    // Repeat KV heads for GQA. Implemented via gather (not tile) because
    // tfjs has no gradient for `tile` above rank 4, and the obvious
    // reshape→tile→reshape path needs a rank-5 intermediate.
    if (repeat > 1) {
      const idx = tf.tensor1d(
        Int32Array.from({ length: H }, (_, i) => Math.floor(i / repeat)),
        "int32",
      );
      k = tf.gather(k, idx, 1);
      v = tf.gather(v, idx, 1);
    }

    // Scaled dot-product attention
    const scale = 1 / Math.sqrt(headDim);
    let attn = tf.mul(tf.matMul(q, k, false, true), tf.scalar(scale)); // [B,H,T,T]
    attn = tf.add(attn, this.causalMask(T));
    attn = tf.softmax(attn, -1);
    let y = tf.matMul(attn, v); // [B,H,T,headDim]
    y = tf.reshape(tf.transpose(y, [0, 2, 1, 3]), [B, T, hp.modelDim]);
    return this.linear(y, `${b}.proj.weight`);
  }

  /** ReLU² MLP for block `i`. */
  private mlp(x: Tensor, i: number): Tensor {
    const { tf } = this;
    const b = `blocks.${i}.mlp`;
    const h = tf.relu(this.linear(x, `${b}.fc.weight`));
    return this.linear(tf.square(h), `${b}.proj.weight`);
  }

  /** Single transformer block with resid_mix(x, x0). */
  private block(x: Tensor, x0: Tensor, i: number): Tensor {
    const { tf } = this;
    const b = `blocks.${i}`;
    const mix = this.weights.get(`${b}.resid_mix`) as Variable; // [2, D]
    const [m0, m1] = tf.split(mix, 2, 0);
    let h = tf.add(
      tf.mul(x, tf.reshape(m0, [1, 1, -1])),
      tf.mul(x0, tf.reshape(m1, [1, 1, -1])),
    );
    const aScale = tf.reshape(
      this.weights.get(`${b}.attn_scale`) as Variable,
      [1, 1, -1],
    );
    h = tf.add(h, tf.mul(aScale, this.attention(this.rmsNorm(h), i)));
    const mScale = tf.reshape(
      this.weights.get(`${b}.mlp_scale`) as Variable,
      [1, 1, -1],
    );
    h = tf.add(h, tf.mul(mScale, this.mlp(this.rmsNorm(h), i)));
    return h;
  }

  // --------------------------------------------------------------------------
  // Forward
  // --------------------------------------------------------------------------

  /** Trunk: tokens → final-norm hidden states [B, T, D]. */
  private trunk(inputIds: Tensor): Tensor {
    const { tf } = this;
    const emb = this.weights.get("tok_emb.weight") as Variable;
    let x: Tensor = tf.gather(emb, tf.cast(inputIds, "int32"));
    x = this.rmsNorm(x);
    const x0 = x;
    const skips: Tensor[] = [];

    for (let i = 0; i < this.numEnc; i++) {
      x = this.block(x, x0, i);
      skips.push(x);
    }
    const skipW = this.weights.get("skip_weights") as Variable; // [numSkip, D]
    for (let i = 0; i < this.numDec; i++) {
      if (skips.length > 0 && i < this.numSkip) {
        const s = skips.pop() as Tensor;
        const w = tf.reshape(
          tf.slice(skipW, [i, 0], [1, this.hp.modelDim]),
          [1, 1, -1],
        );
        x = tf.add(x, tf.mul(w, s));
      }
      x = this.block(x, x0, this.numEnc + i);
    }
    return this.rmsNorm(x);
  }

  /** Pre-softcap logits [B, T, V]. */
  private projLogits(h: Tensor): Tensor {
    const { tf, hp } = this;
    const emb = this.weights.get("tok_emb.weight") as Variable;
    // 2-D matmul for the same BatchMatMul-gradient reason as `linear()`.
    const lead = h.shape.slice(0, -1);
    const h2d = tf.reshape(h, [-1, hp.modelDim]);
    const raw = tf.reshape(tf.matMul(h2d, emb, false, true), [
      ...lead,
      hp.vocabSize,
    ]);
    return tf.mul(
      tf.tanh(tf.div(raw, tf.scalar(hp.logitSoftcap))),
      tf.scalar(hp.logitSoftcap),
    );
  }

  /** Compute logits for inference (no targets). */
  logits(inputIds: Tensor): Tensor {
    return this.tf.tidy(() => this.projLogits(this.trunk(inputIds)));
  }

  /**
   * Training/eval forward: cross-entropy loss (mean over all positions).
   * Mirrors `GPT.forward(input_ids, target_ids)`.
   */
  forward(inputIds: Tensor, targetIds: Tensor): Scalar {
    const { tf, hp } = this;
    return tf.tidy(() => {
      const h = this.trunk(inputIds);
      const logits = this.projLogits(h); // [B,T,V]
      const flatLogits = tf.reshape(logits, [-1, hp.vocabSize]);
      const flatTargets = tf.reshape(tf.cast(targetIds, "int32"), [-1]);
      const oneHot = tf.oneHot(flatTargets, hp.vocabSize);
      const ce = tf.losses.softmaxCrossEntropy(oneHot, flatLogits);
      return ce.asScalar();
    });
  }

  // --------------------------------------------------------------------------
  // State / parameter access
  // --------------------------------------------------------------------------

  /** Snapshot weights to plain Float32Array (CPU) keyed by name. */
  async stateDict(): Promise<
    Map<string, { shape: number[]; data: Float32Array }>
  > {
    const out = new Map<string, { shape: number[]; data: Float32Array }>();
    for (const [name, v] of this.weights) {
      const data = (await v.data()) as Float32Array;
      out.set(name, { shape: v.shape.slice(), data: new Float32Array(data) });
    }
    return out;
  }

  /** Load weights from a plain state dict (strict: shapes must match). */
  loadStateDict(
    sd: Map<string, { shape: number[]; data: Float32Array }>,
  ): void {
    const { tf } = this;
    for (const [name, v] of this.weights) {
      const entry = sd.get(name);
      if (!entry) throw new Error(`loadStateDict: missing ${name}`);
      const want = v.shape.join(",");
      const got = entry.shape.join(",");
      if (want !== got) {
        throw new Error(
          `loadStateDict: shape mismatch for ${name}: ${want} vs ${got}`,
        );
      }
      v.assign(tf.tensor(entry.data, entry.shape, "float32"));
    }
  }

  /** Total trainable parameter count. */
  paramCount(): number {
    let n = 0;
    for (const v of this.weights.values()) n += v.size;
    return n;
  }

  /**
   * Split parameters into optimizer groups (mirrors train_gpt.py):
   *   - embed:  tok_emb.weight
   *   - matrix: 2-D weights inside blocks (Muon)
   *   - scalar: <2-D + control tensors + skip_weights (Adam)
   */
  paramGroups(): ParamGroups {
    const embed: Variable[] = [];
    const matrix: Variable[] = [];
    const scalar: Variable[] = [];
    for (const [name, v] of this.weights) {
      if (name === "tok_emb.weight") {
        embed.push(v);
      } else if (name === "skip_weights" || isControlTensor(name)) {
        scalar.push(v);
      } else if (v.shape.length === 2) {
        matrix.push(v);
      } else {
        scalar.push(v);
      }
    }
    return { embed, matrix, scalar };
  }

  dispose(): void {
    for (const v of this.weights.values()) v.dispose();
    this.weights.clear();
    this.rotaryCache?.cos.dispose();
    this.rotaryCache?.sin.dispose();
    this.maskCache?.mask.dispose();
  }
}
