/**
 * SLM16 — intelligence scoring via Nova Pro cosine similarity.
 *
 * The metric: how similar is SLM16's continuation of a prompt to Nova Pro's
 * continuation of the same prompt? Both run at temperature 0.1 (near-greedy)
 * so the comparison is stable.
 *
 * Cosine similarity is computed over CHARACTER TRIGRAM frequency vectors.
 * This is a deliberate choice over neural embeddings:
 *   - No external embedding service dependency
 *   - Tokenizer-agnostic (Nova uses its own tokenizer; SLM16 uses sp1024)
 *   - Captures lexical agreement, which is what matters at this scale —
 *     a 17M-param model isn't going to nail semantics, but if it produces
 *     the same WORDS as Nova that's a strong signal it's learning English
 *   - The trigram space (~40k common trigrams in English) is high-dim
 *     enough that cosine is meaningful
 *
 * Score remap: meanCosine ∈ [-1, 1] → intelligence ∈ [0, 1] via linear
 * remap from [floor=0.4, ceiling=0.95]. The floor is roughly where two
 * unrelated English passages land (shared "the", "ing", " a "); the ceiling
 * is where paraphrases land. Random byte noise vs Nova → near 0.0 cosine,
 * which clamps to intelligence=0.
 */

import type * as tfTypes from "@tensorflow/tfjs";
import type { ScryptedAIService } from "@elizaos/plugin-scryptedai";
import { SCORING, SCORING_PROMPTS } from "./config.ts";
import { generate } from "./inference.ts";
import type { Slm16Weights } from "./model.ts";
import type { Tokenizer } from "./tokenizer.ts";
import type { Slm16ScoringResult } from "./types.ts";

type TF = typeof tfTypes;

// ----------------------------------------------------------------------------
// Character trigram cosine
// ----------------------------------------------------------------------------

/**
 * Build a normalized character-trigram frequency vector.
 *
 * We lowercase and pad with spaces so leading/trailing bigrams get captured.
 * The vector is a sparse Map<trigram, freq>; cosine works on the
 * intersection so we never materialize the full ~40k-dim dense vector.
 */
function trigramVector(text: string): Map<string, number> {
  const padded = `  ${text.toLowerCase()}  `;
  const counts = new Map<string, number>();
  let total = 0;
  for (let i = 0; i + 3 <= padded.length; i++) {
    const tri = padded.slice(i, i + 3);
    counts.set(tri, (counts.get(tri) ?? 0) + 1);
    total += 1;
  }
  // Normalize to unit L2 norm (so cosine = dot product).
  let sumSq = 0;
  for (const v of counts.values()) sumSq += v * v;
  const norm = Math.sqrt(sumSq) || 1;
  for (const [k, v] of counts) counts.set(k, v / norm);
  void total;
  return counts;
}

/**
 * Cosine similarity between two unit-normalized sparse vectors.
 * Iterates the smaller map for efficiency (O(min(|a|, |b|))).
 */
function cosineSimilarity(a: Map<string, number>, b: Map<string, number>): number {
  const [smaller, larger] = a.size <= b.size ? [a, b] : [b, a];
  let dot = 0;
  for (const [k, va] of smaller) {
    const vb = larger.get(k);
    if (vb !== undefined) dot += va * vb;
  }
  // Vectors are pre-normalized → dot IS the cosine.
  return dot;
}

// ----------------------------------------------------------------------------
// Nova Pro completion via ScryptedAI
// ----------------------------------------------------------------------------

const NOVA_MODEL_ID = "amazon.nova-pro-v1:0";
const NOVA_TIMEOUT_MS = 90_000;

/**
 * Get Nova Pro's completion for one prompt. We feed the context as a system
 * envelope and the question as the user prompt — same shape as introspect.ts
 * uses. Temperature 0.1, max_tokens matching SLM16's generation budget so
 * the comparison is fair (neither side gets to ramble longer).
 */
async function novaComplete(
  scrypted: ScryptedAIService,
  context: string,
  question: string,
): Promise<string> {
  const { jobId } = await scrypted.startTextGeneration({
    model_id: NOVA_MODEL_ID,
    system_prompt:
      `You are completing a short factual prompt. Context: ${context}\n` +
      `Answer briefly in one or two sentences.`,
    user_prompt: question,
    temperature: SCORING.temperature,
    max_tokens: SCORING.maxNewTokens,
  });
  const result = await scrypted.awaitJob(jobId, NOVA_TIMEOUT_MS);
  if (result.status !== "completed" || !result.text) {
    throw new Error(
      `Nova completion failed (status=${result.status}, error=${result.error ?? "none"})`,
    );
  }
  return result.text;
}

// ----------------------------------------------------------------------------
// Public: full scoring round
// ----------------------------------------------------------------------------

/**
 * Run the full scoring round: 8 prompts × {SLM16, Nova} → 8 cosines → mean
 * → linear remap → intelligence ∈ [0, 1].
 *
 * Failure handling: if Nova times out on a prompt, that pair contributes
 * similarity=NaN and is excluded from the mean. If ALL pairs fail (Nova
 * is unreachable), intelligence falls back to a val_loss-derived proxy
 * so the agent still gets a number rather than nothing.
 *
 * `valLoss` is the LKG model's validation loss — passed in from the caller
 * who already has it from meta.json. It's included in the result for
 * downstream correlation, and used as the fallback signal.
 */
export async function scoreIntelligence(
  tf: TF,
  weights: Slm16Weights,
  tok: Tokenizer,
  scrypted: ScryptedAIService,
  valLoss: number,
  log: (msg: string) => void = () => {},
): Promise<Slm16ScoringResult> {
  const slmCompletions: string[] = [];
  const novaCompletions: string[] = [];
  const similarities: number[] = [];

  for (let i = 0; i < SCORING_PROMPTS.length; i++) {
    const { context, question } = SCORING_PROMPTS[i];
    const fullPrompt = `${context} ${question}`;

    // ---- SLM16 side ----
    let slmText: string;
    try {
      const promptIds = tok.encode(fullPrompt);
      const gen = generate(
        tf,
        weights,
        {
          promptIds,
          maxNewTokens: SCORING.maxNewTokens,
          temperature: SCORING.temperature,
          topK: 50, // mild safety net against degenerate repeats
        },
        tok,
      );
      slmText = gen.text ?? "";
    } catch (e) {
      log(`[slm16-scoring] SLM inference failed on prompt ${i}: ${(e as Error).message}`);
      slmText = "";
    }
    slmCompletions.push(slmText);

    // ---- Nova side ----
    let novaText: string;
    try {
      novaText = await novaComplete(scrypted, context, question);
    } catch (e) {
      log(`[slm16-scoring] Nova failed on prompt ${i}: ${(e as Error).message}`);
      novaText = "";
    }
    novaCompletions.push(novaText);

    // ---- Cosine ----
    if (slmText.length === 0 || novaText.length === 0) {
      similarities.push(Number.NaN);
      continue;
    }
    const sim = cosineSimilarity(trigramVector(slmText), trigramVector(novaText));
    similarities.push(sim);
    log(
      `[slm16-scoring] prompt ${i}: cosine=${sim.toFixed(4)}\n` +
        `  slm:  ${slmText.slice(0, 80)}\n` +
        `  nova: ${novaText.slice(0, 80)}`,
    );
  }

  // ---- Aggregate ----
  const valid = similarities.filter((s) => Number.isFinite(s));
  let intelligence: number;
  let meanSimilarity: number;

  if (valid.length === 0) {
    // No Nova comparisons succeeded. Fall back to a val_loss-derived
    // proxy: ln(1024)≈6.93 (random init) → 0, val_loss=2.0 (trained
    // baseline) → ~0.85. Linear interpolation, clamped.
    meanSimilarity = Number.NaN;
    const randomLoss = Math.log(1024); // ≈ 6.93 — uniform-over-vocab cross-entropy
    const trainedLoss = 2.0; // reference baseline
    const t = (randomLoss - valLoss) / (randomLoss - trainedLoss);
    intelligence = Math.max(0, Math.min(1, t)) * 0.85; // cap at 0.85 — proxy is less trustworthy
    log(
      `[slm16-scoring] all Nova comparisons failed; ` +
        `falling back to val_loss proxy (val_loss=${valLoss.toFixed(4)}, intelligence=${intelligence.toFixed(4)})`,
    );
  } else {
    meanSimilarity = valid.reduce((a, b) => a + b, 0) / valid.length;
    // Linear remap [floor, ceiling] → [0, 1], clamped.
    const t = (meanSimilarity - SCORING.similarityFloor) /
      (SCORING.similarityCeiling - SCORING.similarityFloor);
    intelligence = Math.max(0, Math.min(1, t));
  }

  return {
    meanSimilarity,
    similarities,
    intelligence,
    slmCompletions,
    novaCompletions,
    valLoss,
  };
}
