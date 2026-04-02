/**
 * SLM16 inference — load the LKG int8 checkpoint and run autoregressive
 * generation at low temperature for the Nova-Pro comparison.
 *
 * Tokenizer: the FineWeb sp1024 variant ships a SentencePiece BPE model
 * (`fineweb_1024_bpe.model`). We load it via `sentencepiece-js`, which
 * wraps the upstream C++ library through wasm and works on every Node
 * platform without a native build step.
 */

import * as fs from "node:fs";
import * as path from "node:path";

import type { TF } from "./backend.ts";
import {
  DEFAULT_HPARAMS,
  DEFAULT_TOKENIZER_PATH,
  ENV_SLM16_TOKENIZER_PATH,
  INFERENCE_MAX_NEW_TOKENS,
  INFERENCE_TEMPERATURE,
  LKG_FILENAME,
  LKG_META_FILENAME,
  type Slm16Hyperparameters,
} from "./constants.ts";
import { Slm16Model } from "./model.ts";
import { deserializeInt8 } from "./quantize.ts";
import type { LkgMeta } from "./trainer.ts";

type Tensor = import("@tensorflow/tfjs").Tensor;

// ----------------------------------------------------------------------------
// Tokenizer (SentencePiece via wasm)
// ----------------------------------------------------------------------------

export interface Tokenizer {
  encode(text: string): number[];
  decode(ids: number[]): string;
}

export async function loadTokenizer(modelPath?: string): Promise<Tokenizer> {
  const file =
    modelPath ??
    process.env[ENV_SLM16_TOKENIZER_PATH] ??
    DEFAULT_TOKENIZER_PATH;
  if (!fs.existsSync(file)) {
    throw new Error(
      `SLM16: SentencePiece model not found at ${file} — run the parameter-golf data script (cached_challenge_fineweb.py --variant sp1024) and copy fineweb_1024_bpe.model there, or set ${ENV_SLM16_TOKENIZER_PATH}`,
    );
  }
  // Dynamic so the wasm blob is only paged in when inference is requested.
  const spec = "sentencepiece-js";
  const mod = await import(/* @vite-ignore */ spec);
  const sp = new mod.SentencePieceProcessor();
  await sp.load(file);
  return {
    encode: (text: string): number[] =>
      Array.from(sp.encodeIds(text) as Iterable<number>),
    decode: (ids: number[]): string => sp.decodeIds(ids),
  };
}

// ----------------------------------------------------------------------------
// Sampling
// ----------------------------------------------------------------------------

function sampleFrom(probs: Float32Array): number {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

// ----------------------------------------------------------------------------
// Slm16Inference
// ----------------------------------------------------------------------------

export class Slm16Inference {
  private readonly tf: TF;
  private readonly hp: Slm16Hyperparameters;
  private readonly model: Slm16Model;
  private readonly tokenizer: Tokenizer;
  private readonly meta: LkgMeta | null;

  private constructor(
    tf: TF,
    hp: Slm16Hyperparameters,
    model: Slm16Model,
    tokenizer: Tokenizer,
    meta: LkgMeta | null,
  ) {
    this.tf = tf;
    this.hp = hp;
    this.model = model;
    this.tokenizer = tokenizer;
    this.meta = meta;
  }

  /** Load the LKG checkpoint from `ckptDir` (falls back to fresh init). */
  static async load(
    tf: TF,
    ckptDir: string,
    tokenizerPath?: string,
  ): Promise<Slm16Inference> {
    const lkg = path.join(ckptDir, LKG_FILENAME);
    const metaPath = path.join(ckptDir, LKG_META_FILENAME);
    let hp = DEFAULT_HPARAMS;
    let meta: LkgMeta | null = null;
    if (fs.existsSync(metaPath)) {
      meta = JSON.parse(fs.readFileSync(metaPath, "utf8")) as LkgMeta;
      hp = meta.hparams;
    }
    const model = new Slm16Model(tf, hp);
    if (fs.existsSync(lkg)) {
      model.loadStateDict(deserializeInt8(fs.readFileSync(lkg)));
    }
    const tok = await loadTokenizer(tokenizerPath);
    return new Slm16Inference(tf, hp, model, tok, meta);
  }

  get lkgMeta(): LkgMeta | null {
    return this.meta;
  }

  /**
   * Greedy-ish autoregressive generation. `temperature` defaults to 0.1
   * (near-deterministic) so cosine-similarity against Nova Pro is stable.
   * Returns only the *continuation*, not the prompt.
   */
  async generate(
    prompt: string,
    opts: { maxNewTokens?: number; temperature?: number } = {},
  ): Promise<string> {
    const { tf, hp } = this;
    const maxNew = opts.maxNewTokens ?? INFERENCE_MAX_NEW_TOKENS;
    const temp = Math.max(1e-3, opts.temperature ?? INFERENCE_TEMPERATURE);

    let ids = this.tokenizer.encode(prompt);
    if (ids.length === 0) ids = [0];
    const generated: number[] = [];

    for (let n = 0; n < maxNew; n++) {
      // Truncate context to model window.
      const ctx = ids.slice(-hp.trainSeqLen);
      const x = tf.tensor2d(Int32Array.from(ctx), [1, ctx.length], "int32");
      const probs = tf.tidy(() => {
        const logits = this.model.logits(x); // [1, T, V]
        const last = tf.slice(
          logits,
          [0, ctx.length - 1, 0],
          [1, 1, hp.vocabSize],
        );
        return tf.softmax(tf.div(tf.reshape(last, [hp.vocabSize]), temp));
      }) as Tensor;
      const p = (await probs.data()) as Float32Array;
      probs.dispose();
      x.dispose();
      const next = temp < 0.05 ? p.indexOf(Math.max(...p)) : sampleFrom(p);
      ids.push(next);
      generated.push(next);
    }

    return this.tokenizer.decode(generated);
  }

  dispose(): void {
    this.model.dispose();
  }
}
