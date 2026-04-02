/**
 * SLM16 — Muon + Adam optimizer split.
 *
 * Faithful port of the optimizer scheme in OpenAI Parameter Golf:
 *   - Muon for 2D matrix weights (Linear layers)
 *   - Adam for embeddings and scalar/control parameters
 *
 * Muon (Modular Unconstrained Newton): orthogonalizes the gradient via a
 * fast Newton-Schulz iteration before applying it. The intuition is that the
 * SVD of the update should be approximately identity (all singular values ≈1),
 * which is what zero-power orthogonalization gives you. This decouples the
 * update direction from the gradient's spectral norm, dramatically improving
 * conditioning for matrix-shaped parameters in deep nets.
 *
 * Reference: github.com/openai/parameter-golf/blob/main/train_gpt.py L96–L168
 *            kellerjordan.github.io/posts/muon/
 *
 * The Newton-Schulz constants (3.4445, -4.7750, 2.0315) are tuned for fast
 * convergence to the orthogonal polar factor — verbatim from the reference.
 */

import type * as tfTypes from "@tensorflow/tfjs";
import type { Slm16Weights } from "./model.ts";

type TF = typeof tfTypes;
type Tensor = tfTypes.Tensor;
type Variable = tfTypes.Variable;

// ----------------------------------------------------------------------------
// Newton-Schulz orthogonalization (Muon backend)
// ----------------------------------------------------------------------------

const NS_A = 3.4445;
const NS_B = -4.7750;
const NS_C = 2.0315;
const NS_EPS = 1e-7;

/**
 * Orthogonalize a 2D update matrix. After `steps` iterations, the result has
 * approximately unit singular values (orthogonal polar factor of G).
 *
 * X ← G / (||G|| + ε)
 * if rows > cols: transpose first (operate on the wide form)
 * for each step:
 *   A = X Xᵀ
 *   B = b·A + c·A²
 *   X = a·X + B·X
 * un-transpose if needed.
 *
 * Reference uses bf16 for the inner loop; we stay in fp32 since tfjs-node's
 * bf16 support is incomplete and the matrix sizes here (≤512×1024) are tiny.
 */
function newtonSchulz5(tf: TF, G: Tensor, steps: number): Tensor {
  return tf.tidy(() => {
    const norm = tf.sqrt(tf.sum(tf.square(G)));
    let X = tf.div(G, tf.add(norm, NS_EPS));
    // shape is concrete here (G is always a 2D weight gradient).
    const [rows, cols] = G.shape as [number, number];
    const transposed = rows > cols;
    if (transposed) X = tf.transpose(X);

    for (let i = 0; i < steps; i++) {
      const A = tf.matMul(X, X, false, true); // X @ Xᵀ
      const A2 = tf.matMul(A, A);
      const B = tf.add(tf.mul(A, NS_B), tf.mul(A2, NS_C));
      X = tf.add(tf.mul(X, NS_A), tf.matMul(B, X));
    }

    return transposed ? tf.transpose(X) : X;
  });
}

// ----------------------------------------------------------------------------
// Parameter classification — same rules as the reference
// ----------------------------------------------------------------------------

/**
 * Control-tensor name patterns: these stay fp32 in quantization AND go to
 * the Adam scalar group (not Muon). Mirrors CONTROL_TENSOR_NAME_PATTERNS
 * in train_gpt.py.
 */
const CONTROL_PATTERNS = [
  "attn_scale",
  "mlp_scale",
  "resid_mix",
  "q_gain",
  "skip_weight",
] as const;

function isControlTensor(name: string): boolean {
  return CONTROL_PATTERNS.some((p) => name.includes(p));
}

export type ParamGroup = "embed" | "matrix" | "scalar";

/**
 * Classify a weight by name + shape:
 *   embed  → tok_emb.weight (Adam, special LR)
 *   matrix → 2D non-control tensors in blocks (Muon)
 *   scalar → everything else (Adam)
 */
export function classifyParam(name: string, shape: number[]): ParamGroup {
  if (name === "tok_emb.weight") return "embed";
  if (shape.length === 2 && !isControlTensor(name)) return "matrix";
  return "scalar";
}

// ----------------------------------------------------------------------------
// Optimizer state
// ----------------------------------------------------------------------------

/** Per-parameter Adam moment buffers (m, v). Always fp32. */
interface AdamState {
  m: Float32Array;
  v: Float32Array;
}

/** Per-parameter Muon momentum buffer. */
interface MuonState {
  momentum: Float32Array;
}

export interface OptimizerState {
  /** Optimizer step counter (for Adam bias correction). */
  step: number;
  adam: Map<string, AdamState>;
  muon: Map<string, MuonState>;
}

export function initOptimizerState(w: Slm16Weights): OptimizerState {
  const adam = new Map<string, AdamState>();
  const muon = new Map<string, MuonState>();
  for (const [name, v] of Object.entries(w)) {
    const numel = v.shape.reduce((a, b) => a * b, 1);
    const group = classifyParam(name, v.shape);
    if (group === "matrix") {
      muon.set(name, { momentum: new Float32Array(numel) });
    } else {
      adam.set(name, {
        m: new Float32Array(numel),
        v: new Float32Array(numel),
      });
    }
  }
  return { step: 0, adam, muon };
}

// ----------------------------------------------------------------------------
// Apply step
// ----------------------------------------------------------------------------

export interface StepConfig {
  embedLr: number;
  matrixLr: number;
  scalarLr: number;
  beta1: number;
  beta2: number;
  adamEps: number;
  /** Current Muon momentum (caller handles warmup ramp). */
  muonMomentum: number;
  muonBackendSteps: number;
  /** LR scale (warmdown / schedule). 1.0 = full. */
  lrScale: number;
}

/**
 * Single optimizer step. Mutates `w` Variables in-place and updates `state`.
 *
 * Muon path (matrix params):
 *   buf ← momentum * buf + grad
 *   g   ← grad + momentum * buf       (Nesterov)
 *   g   ← NewtonSchulz(g)
 *   g   ← g * sqrt(max(1, rows/cols))   (scale correction from Muon ref)
 *   p   ← p - lr * g
 *
 * Adam path (embed + scalar):
 *   m ← β₁m + (1-β₁)g
 *   v ← β₂v + (1-β₂)g²
 *   m̂ = m / (1-β₁ᵗ)
 *   v̂ = v / (1-β₂ᵗ)
 *   p ← p - lr * m̂ / (√v̂ + ε)
 *
 * Gradients are read synchronously per parameter (dataSync) — fine for ~17M
 * params on the trainer worker thread. Updates go back via tf.tensor + assign.
 */
export function applyStep(
  tf: TF,
  w: Slm16Weights,
  grads: Record<string, Tensor>,
  state: OptimizerState,
  cfg: StepConfig,
): void {
  state.step += 1;
  const t = state.step;
  const bc1 = 1 - cfg.beta1 ** t;
  const bc2 = 1 - cfg.beta2 ** t;

  for (const [name, param] of Object.entries(w)) {
    const grad = grads[name];
    if (!grad) continue; // No gradient flowed to this param this step.
    const group = classifyParam(name, param.shape);
    const numel = param.shape.reduce((a, b) => a * b, 1);

    if (group === "matrix") {
      // ---- Muon ----
      const ms = state.muon.get(name);
      if (!ms) continue;
      const mom = cfg.muonMomentum;

      // CPU-side momentum update (Nesterov).
      const g = grad.dataSync() as Float32Array;
      const buf = ms.momentum;
      for (let i = 0; i < numel; i++) {
        buf[i] = mom * buf[i] + g[i];
        g[i] = g[i] + mom * buf[i]; // Nesterov: g + mom*buf
      }

      // Newton-Schulz on device (matmuls benefit from BLAS/CUDA).
      const gT = tf.tensor(g, param.shape, "float32");
      const ortho = newtonSchulz5(tf, gT, cfg.muonBackendSteps);
      gT.dispose();

      // Scale correction: sqrt(max(1, rows/cols)).
      const [rows, cols] = param.shape;
      const scaleCorr = Math.sqrt(Math.max(1, rows / cols));
      const lr = cfg.matrixLr * cfg.lrScale * scaleCorr;

      // p ← p - lr * ortho. tf.sub is differentiable but we don't need grads
      // through the optimizer — wrap in tidy and assign.
      tf.tidy(() => {
        const update = tf.mul(ortho, lr);
        const newP = tf.sub(param, update);
        param.assign(newP);
      });
      ortho.dispose();
    } else {
      // ---- Adam ----
      const as = state.adam.get(name);
      if (!as) continue;
      const lrBase = group === "embed" ? cfg.embedLr : cfg.scalarLr;
      const lr = lrBase * cfg.lrScale;

      const g = grad.dataSync() as Float32Array;
      const p = param.dataSync() as Float32Array;
      const { m, v } = as;
      const b1 = cfg.beta1;
      const b2 = cfg.beta2;
      const eps = cfg.adamEps;

      for (let i = 0; i < numel; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        p[i] = p[i] - (lr * mHat) / (Math.sqrt(vHat) + eps);
      }

      const newP = tf.tensor(p, param.shape, "float32");
      param.assign(newP);
      newP.dispose();
    }
  }
}

// ----------------------------------------------------------------------------
// Serialization (for resumable training)
// ----------------------------------------------------------------------------

/**
 * Pack optimizer state into a single contiguous Float32Array preceded by
 * a JSON manifest. Layout:
 *   [u32 jsonLen][json: {step, order: [{name, kind, len}]}][f32 data...]
 *
 * The manifest carries the iteration order so we can robustly slice the
 * payload back even if Map iteration order changed between Node versions.
 */
export function serializeOptimizerState(state: OptimizerState): Uint8Array {
  const order: Array<{ name: string; kind: "adam" | "muon"; len: number }> = [];
  let totalFloats = 0;

  for (const [name, s] of state.adam) {
    const len = s.m.length;
    order.push({ name, kind: "adam", len });
    totalFloats += len * 2; // m + v
  }
  for (const [name, s] of state.muon) {
    const len = s.momentum.length;
    order.push({ name, kind: "muon", len });
    totalFloats += len;
  }

  const manifest = JSON.stringify({ step: state.step, order });
  const manifestBytes = new TextEncoder().encode(manifest);
  const headerLen = 4 + manifestBytes.length;
  // Align data start to 4 bytes for clean Float32Array views.
  const padded = (headerLen + 3) & ~3;
  const totalBytes = padded + totalFloats * 4;
  const buf = new ArrayBuffer(totalBytes);
  const u8 = new Uint8Array(buf);

  new DataView(buf).setUint32(0, manifestBytes.length, true);
  u8.set(manifestBytes, 4);

  const f32 = new Float32Array(buf, padded);
  let off = 0;
  for (const entry of order) {
    if (entry.kind === "adam") {
      const s = state.adam.get(entry.name);
      if (!s) continue;
      f32.set(s.m, off);
      off += entry.len;
      f32.set(s.v, off);
      off += entry.len;
    } else {
      const s = state.muon.get(entry.name);
      if (!s) continue;
      f32.set(s.momentum, off);
      off += entry.len;
    }
  }

  return u8;
}

export function deserializeOptimizerState(blob: Uint8Array): OptimizerState {
  // Re-anchor the view: blob may be a slice with non-zero byteOffset.
  const ab = blob.buffer.slice(
    blob.byteOffset,
    blob.byteOffset + blob.byteLength,
  );
  const dv = new DataView(ab);
  const jsonLen = dv.getUint32(0, true);
  const jsonBytes = new Uint8Array(ab, 4, jsonLen);
  const manifest = JSON.parse(new TextDecoder().decode(jsonBytes)) as {
    step: number;
    order: Array<{ name: string; kind: "adam" | "muon"; len: number }>;
  };

  const headerLen = 4 + jsonLen;
  const padded = (headerLen + 3) & ~3;
  const f32 = new Float32Array(ab, padded);

  const adam = new Map<string, AdamState>();
  const muon = new Map<string, MuonState>();
  let off = 0;
  for (const entry of manifest.order) {
    if (entry.kind === "adam") {
      const m = f32.slice(off, off + entry.len);
      off += entry.len;
      const v = f32.slice(off, off + entry.len);
      off += entry.len;
      adam.set(entry.name, { m, v });
    } else {
      const momentum = f32.slice(off, off + entry.len);
      off += entry.len;
      muon.set(entry.name, { momentum });
    }
  }

  return { step: manifest.step, adam, muon };
}
