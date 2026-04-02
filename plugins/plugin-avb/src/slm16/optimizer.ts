/**
 * SLM16 optimizer split — port of the Muon + Adam combination from
 * openai/parameter-golf train_gpt.py.
 *
 * - Embedding + scalar params: Adam (β1=0.9, β2=0.95) at their own LRs.
 * - 2-D matrix params inside blocks: Muon — momentum SGD where the
 *   update is orthogonalized via a 5-step Newton–Schulz iteration before
 *   being applied (scale-corrected by √(rows/cols)).
 *
 * tfjs has no built-in NS5, so we hand-roll the matrix-orthogonalization
 * step on the same backend the model lives on.
 */

import type { TF } from "./backend.ts";
import type { Slm16Hyperparameters } from "./constants.ts";
import type { ParamGroups } from "./model.ts";

type Tensor = import("@tensorflow/tfjs").Tensor;
type Variable = import("@tensorflow/tfjs").Variable;

// ----------------------------------------------------------------------------
// Newton–Schulz orthogonalization (zeropower_via_newtonschulz5)
// ----------------------------------------------------------------------------

const NS5 = { a: 3.4445, b: -4.775, c: 2.0315 } as const;

export function newtonSchulz5(tf: TF, G: Tensor, steps: number): Tensor {
  return tf.tidy(() => {
    let X = tf.div(G, tf.add(tf.norm(G), tf.scalar(1e-7)));
    const [rows, cols] = G.shape as [number, number];
    const transposed = rows > cols;
    if (transposed) X = tf.transpose(X);
    for (let i = 0; i < steps; i++) {
      const A = tf.matMul(X, X, false, true);
      const B = tf.add(
        tf.mul(A, tf.scalar(NS5.b)),
        tf.mul(tf.matMul(A, A), tf.scalar(NS5.c)),
      );
      X = tf.add(tf.mul(X, tf.scalar(NS5.a)), tf.matMul(B, X));
    }
    return transposed ? tf.transpose(X) : X;
  });
}

// ----------------------------------------------------------------------------
// Slm16Optimizer
// ----------------------------------------------------------------------------

export class Slm16Optimizer {
  private readonly tf: TF;
  private readonly hp: Slm16Hyperparameters;
  private readonly groups: ParamGroups;

  // Adam state for embed + scalar
  private readonly m = new Map<Variable, Variable>();
  private readonly v = new Map<Variable, Variable>();
  // Muon momentum buffers for matrix
  private readonly mom = new Map<Variable, Variable>();
  private t = 0;

  constructor(tf: TF, hp: Slm16Hyperparameters, groups: ParamGroups) {
    this.tf = tf;
    this.hp = hp;
    this.groups = groups;
    for (const p of [...groups.embed, ...groups.scalar]) {
      this.m.set(p, tf.variable(tf.zerosLike(p), false));
      this.v.set(p, tf.variable(tf.zerosLike(p), false));
    }
    for (const p of groups.matrix) {
      this.mom.set(p, tf.variable(tf.zerosLike(p), false));
    }
  }

  /**
   * Apply one optimizer step given a name→grad map (output of
   * `tf.variableGrads`). `lrScale` implements the warmdown schedule.
   */
  step(grads: Map<Variable, Tensor>, lrScale: number): void {
    const { tf, hp } = this;
    this.t++;
    const bc1 = 1 - hp.beta1 ** this.t;
    const bc2 = 1 - hp.beta2 ** this.t;

    const adam = (params: Variable[], baseLr: number): void => {
      const lr = baseLr * lrScale;
      for (const p of params) {
        const g = grads.get(p);
        if (!g) continue;
        const m = this.m.get(p) as Variable;
        const v = this.v.get(p) as Variable;
        tf.tidy(() => {
          m.assign(tf.add(tf.mul(m, hp.beta1), tf.mul(g, 1 - hp.beta1)));
          v.assign(
            tf.add(tf.mul(v, hp.beta2), tf.mul(tf.square(g), 1 - hp.beta2)),
          );
          const mHat = tf.div(m, bc1);
          const vHat = tf.div(v, bc2);
          const upd = tf.div(mHat, tf.add(tf.sqrt(vHat), hp.adamEps));
          p.assign(tf.sub(p, tf.mul(upd, lr)));
        });
      }
    };

    adam(this.groups.embed, hp.embedLr);
    adam(this.groups.scalar, hp.scalarLr);

    // Muon
    const muonLr = hp.matrixLr * lrScale;
    for (const p of this.groups.matrix) {
      const g = grads.get(p);
      if (!g) continue;
      const buf = this.mom.get(p) as Variable;
      tf.tidy(() => {
        buf.assign(tf.add(tf.mul(buf, hp.muonMomentum), g));
        // Nesterov: g + momentum * buf
        const gN = tf.add(g, tf.mul(buf, hp.muonMomentum));
        let u = newtonSchulz5(tf, gN, hp.muonBackendSteps);
        const [rows, cols] = p.shape as [number, number];
        const scale = Math.sqrt(Math.max(1, rows / cols));
        u = tf.mul(u, scale);
        p.assign(tf.sub(p, tf.mul(u, muonLr)));
      });
    }
  }

  get step_t(): number {
    return this.t;
  }

  restoreStep(t: number): void {
    this.t = t;
  }

  dispose(): void {
    for (const v of this.m.values()) v.dispose();
    for (const v of this.v.values()) v.dispose();
    for (const v of this.mom.values()) v.dispose();
  }
}
