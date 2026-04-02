/**
 * SLM16 — autoregressive sampling for inference.
 *
 * The model has no KV cache (the reference doesn't have one either — it's
 * trained for next-token loss, not optimized for generation). So each step
 * recomputes attention over the full prefix. For maxNewTokens=64 with
 * seqLen=1024, that's 64 forwards over a sequence growing from |prompt| to
 * |prompt|+64. On commodity hardware this is a few seconds per sample — fine
 * for the 8-prompt scoring round, not fine for interactive chat.
 *
 * Sampling: temperature → top-k → multinomial. At temp=0.1 (the scoring
 * config) this is nearly greedy, which is what we want for reproducible
 * comparisons against Nova Pro.
 */

import type * as tfTypes from "@tensorflow/tfjs";
import { ARCH } from "./config.ts";
import { forwardLogits, type Slm16Weights } from "./model.ts";
import type { Slm16GenerateOptions, Slm16GenerateResult } from "./types.ts";
import type { Tokenizer } from "./tokenizer.ts";

type TF = typeof tfTypes;

/**
 * Sample one token from a logit vector.
 * Pure-CPU implementation: we read the [V]-sized logit array out once per
 * step (V=1024 → trivially cheap), apply temperature + top-k on the host,
 * then multinomial sample. Avoids tf.multinomial which has had backend
 * inconsistencies.
 */
function sampleToken(
  logits: Float32Array,
  temperature: number,
  topK: number,
  rng: () => number,
): number {
  const V = logits.length;

  // Greedy fast path.
  if (temperature <= 0) {
    let best = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < V; i++) {
      if (logits[i] > bestVal) {
        bestVal = logits[i];
        best = i;
      }
    }
    return best;
  }

  // Temperature scaling + top-k filtering + softmax + sample.
  // We work in a scratch array so we don't mutate the input.
  const scaled = new Float32Array(V);
  let maxL = -Infinity;
  for (let i = 0; i < V; i++) {
    const v = logits[i] / temperature;
    scaled[i] = v;
    if (v > maxL) maxL = v;
  }

  // Top-k: find the (V-k)th largest value as a threshold, mask the rest.
  if (topK > 0 && topK < V) {
    // Partial sort via copy+sort — V=1024, sorting is microseconds.
    const sorted = Float32Array.from(scaled).sort();
    const thresh = sorted[V - topK];
    for (let i = 0; i < V; i++) {
      if (scaled[i] < thresh) scaled[i] = -Infinity;
    }
    // Recompute max after masking (it might have been below thresh in
    // a degenerate flat distribution).
    maxL = -Infinity;
    for (let i = 0; i < V; i++) {
      if (scaled[i] > maxL) maxL = scaled[i];
    }
  }

  // Softmax (numerically stable: subtract max).
  let sumExp = 0;
  for (let i = 0; i < V; i++) {
    const e = Math.exp(scaled[i] - maxL);
    scaled[i] = e;
    sumExp += e;
  }

  // Multinomial draw (inverse CDF).
  const r = rng() * sumExp;
  let cum = 0;
  for (let i = 0; i < V; i++) {
    cum += scaled[i];
    if (r < cum) return i;
  }
  return V - 1; // numerical edge: r ≈ sumExp
}

/**
 * Generate `maxNewTokens` continuation tokens from a prompt.
 * The prompt is left-truncated to seqLen-1 if it's too long (we always
 * leave room for at least one generated token in the context window).
 */
export function generate(
  tf: TF,
  weights: Slm16Weights,
  opts: Slm16GenerateOptions,
  tok?: Tokenizer,
): Slm16GenerateResult {
  const maxCtx = ARCH.seqLen;
  // Per-call PRNG: seed from prompt hash so the same prompt at the same
  // temperature gives the same output (reproducible scoring) without
  // a global RNG state we'd have to thread through.
  let seed = 0x9e3779b9;
  for (const id of opts.promptIds) seed = (seed ^ id) >>> 0;
  let s = seed;
  const rng = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  // Working sequence: starts with the (possibly truncated) prompt.
  const seq: number[] =
    opts.promptIds.length >= maxCtx
      ? opts.promptIds.slice(opts.promptIds.length - (maxCtx - 1))
      : [...opts.promptIds];
  const newTokens: number[] = [];

  for (let n = 0; n < opts.maxNewTokens; n++) {
    // Build [1, T] int32 input from the current sequence (right-justified
    // to the most recent maxCtx tokens).
    const ctx = seq.length > maxCtx ? seq.slice(seq.length - maxCtx) : seq;
    const T = ctx.length;
    const input = tf.tensor2d(new Int32Array(ctx), [1, T], "int32");

    // forwardLogits returns [B, V] for the LAST position; B=1 here.
    const logitsT = forwardLogits(tf, weights, input);
    const logits = logitsT.dataSync() as Float32Array; // length V
    logitsT.dispose();
    input.dispose();

    const next = sampleToken(
      logits,
      opts.temperature,
      opts.topK ?? 0,
      rng,
    );
    seq.push(next);
    newTokens.push(next);
  }

  return {
    tokenIds: seq,
    newTokenIds: newTokens,
    text: tok ? tok.decode(newTokens) : undefined,
  };
}
