/**
 * SLM16 evaluator — "intelligence score" against Amazon Nova Pro.
 *
 * For each prompt in EVAL_PROMPTS:
 *   1. Ask Nova Pro (via ScryptedAI) for a continuation at temperature 0.1
 *   2. Ask the local SLM16 LKG model for a continuation at temperature 0.1
 *   3. Send BOTH continuations back to Nova Pro with a judge prompt that
 *      asks it to rate similarity-of-meaning on [0,1] and explain why,
 *      returning strict JSON: { "similarity": 0.65, "reasoning": "…" }.
 *
 * The intelligence score is the mean similarity × 100, clamped to
 * [0, 100]. Using the LLM as judge avoids needing a TEXT_EMBEDDING model
 * registered in the runtime — the only external dependency is the
 * ScryptedAI Nova-Pro endpoint that is already required for the
 * reference completions.
 */

import {
  SCRYPTEDAI_SERVICE_TYPE,
  type ScryptedAIService,
} from "@elizaos/plugin-scryptedai";

import {
  EVAL_PROMPTS,
  INFERENCE_MAX_NEW_TOKENS,
  INFERENCE_TEMPERATURE,
} from "./constants.ts";
import type { Slm16Inference } from "./inference.ts";

// ----------------------------------------------------------------------------
// Runtime surface (structural)
// ----------------------------------------------------------------------------

export interface EvalRuntimeSurface {
  getService<T = unknown>(type: string): T | undefined;
  logger: { info: (m: string) => void; warn: (m: string) => void };
}

// ----------------------------------------------------------------------------
// Result
// ----------------------------------------------------------------------------

/** Raw shape Nova Pro is asked to return when judging. */
export interface JudgeVerdict {
  /** Similarity of the candidate to the reference, 0..1. */
  similarity: number;
  /** Free-text rationale (kept for debugging / log inspection). */
  reasoning: string;
}

export interface PromptComparison {
  prompt: string;
  slm16: string;
  novaPro: string;
  similarity: number;
  reasoning: string;
}

export interface IntelligenceReport {
  /** Mean judge similarity × 100, clamped [0,100]. */
  score: number;
  /** Per-prompt breakdown. */
  comparisons: PromptComparison[];
  /** ISO timestamp. */
  evaluatedAt: string;
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

const NOVA_PRO_TIMEOUT_MS = 120_000;

const JUDGE_SYSTEM_PROMPT = `You are an evaluation judge comparing the output of a small \
student language model against a reference completion from a strong model. \
Both were given the same prompt and asked to continue it. \
Score how well the CANDIDATE matches the REFERENCE in meaning, factual \
correctness, and task fit — NOT surface wording. \
Respond with ONLY a single JSON object on one line, no prose, no code \
fences, exactly: {"similarity": <float 0..1>, "reasoning": "<short string>"}. \
0 = unrelated/incorrect, 0.5 = partially correct, 1 = equivalent.`;

function buildJudgeUserPrompt(
  prompt: string,
  reference: string,
  candidate: string,
): string {
  return [
    `PROMPT:\n${prompt}`,
    `REFERENCE (Nova Pro):\n${reference}`,
    `CANDIDATE (SLM16):\n${candidate}`,
    "Return only the JSON object.",
  ].join("\n\n");
}

/**
 * Cosine similarity. No longer used by the evaluator (replaced by the
 * Nova-Pro LLM-as-judge path) but kept exported because it's a generic
 * vector utility and the unit tests cover its identities.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 0 ? dot / denom : 0;
}

/**
 * Pull the first `{ … }` JSON object out of an LLM response and coerce
 * it into a JudgeVerdict. Tolerant of leading/trailing prose and code
 * fences; throws if no parseable object with a numeric `similarity`
 * is found.
 */
export function parseJudgeVerdict(raw: string): JudgeVerdict {
  // Strip code fences if the model wrapped its answer.
  const stripped = raw.replace(/```(?:json)?/gi, "").trim();
  const start = stripped.indexOf("{");
  const end = stripped.lastIndexOf("}");
  if (start < 0 || end <= start) {
    throw new Error(`No JSON object in judge response: ${raw.slice(0, 120)}`);
  }
  const obj = JSON.parse(stripped.slice(start, end + 1)) as Partial<
    JudgeVerdict & { score: number }
  >;
  // Accept "score" as an alias in case the model drifts.
  const sim = typeof obj.similarity === "number" ? obj.similarity : obj.score;
  if (typeof sim !== "number" || !Number.isFinite(sim)) {
    throw new Error(
      `Judge response missing numeric "similarity": ${stripped.slice(start, end + 1)}`,
    );
  }
  return {
    similarity: Math.max(0, Math.min(1, sim)),
    reasoning:
      typeof obj.reasoning === "string" ? obj.reasoning : "(no reasoning)",
  };
}

async function novaProComplete(
  scrypted: ScryptedAIService,
  prompt: string,
): Promise<string> {
  const { jobId } = await scrypted.startTextGeneration({
    user_prompt: prompt,
    system_prompt:
      "Continue the user's text naturally with a single short sentence. Output only the continuation.",
    temperature: INFERENCE_TEMPERATURE,
    max_tokens: INFERENCE_MAX_NEW_TOKENS,
  });
  const result = await scrypted.awaitJob(jobId, NOVA_PRO_TIMEOUT_MS);
  if (result.status !== "completed" || !result.text) {
    throw new Error(
      `Nova Pro generation failed (jobId=${jobId}, status=${result.status})`,
    );
  }
  return result.text.trim();
}

async function novaProJudge(
  scrypted: ScryptedAIService,
  prompt: string,
  reference: string,
  candidate: string,
): Promise<JudgeVerdict> {
  const { jobId } = await scrypted.startTextGeneration({
    system_prompt: JUDGE_SYSTEM_PROMPT,
    user_prompt: buildJudgeUserPrompt(prompt, reference, candidate),
    // Judging should be deterministic; T=0.1 to match the rest of SLM16.
    temperature: INFERENCE_TEMPERATURE,
    max_tokens: 256,
  });
  const result = await scrypted.awaitJob(jobId, NOVA_PRO_TIMEOUT_MS);
  if (result.status !== "completed" || !result.text) {
    throw new Error(
      `Nova Pro judge failed (jobId=${jobId}, status=${result.status})`,
    );
  }
  return parseJudgeVerdict(result.text);
}

// ----------------------------------------------------------------------------
// Public
// ----------------------------------------------------------------------------

/**
 * Run the full Nova-Pro comparison suite. Returns null if ScryptedAI
 * isn't loaded — the caller should treat that as "score unavailable",
 * not an error.
 */
export async function evaluateAgainstNovaPro(
  runtime: EvalRuntimeSurface,
  slm16: Slm16Inference,
  prompts: readonly string[] = EVAL_PROMPTS,
): Promise<IntelligenceReport | null> {
  const scrypted = runtime.getService<ScryptedAIService>(
    SCRYPTEDAI_SERVICE_TYPE,
  );
  if (!scrypted) {
    runtime.logger.warn(
      "[slm16] ScryptedAI service unavailable — skipping Nova Pro eval",
    );
    return null;
  }

  const comparisons: PromptComparison[] = [];

  for (const prompt of prompts) {
    try {
      const [candidate, reference] = await Promise.all([
        slm16.generate(prompt, { temperature: INFERENCE_TEMPERATURE }),
        novaProComplete(scrypted, prompt),
      ]);
      const verdict = await novaProJudge(
        scrypted,
        prompt,
        reference,
        candidate,
      );
      comparisons.push({
        prompt,
        slm16: candidate,
        novaPro: reference,
        similarity: verdict.similarity,
        reasoning: verdict.reasoning,
      });
      runtime.logger.info(
        `[slm16:judge] "${prompt.slice(0, 32)}…" sim=${verdict.similarity.toFixed(2)} — ${verdict.reasoning}`,
      );
    } catch (err) {
      runtime.logger.warn(
        `[slm16] Eval prompt failed (${prompt.slice(0, 30)}…): ${(err as Error).message}`,
      );
    }
  }

  if (comparisons.length === 0) return null;

  const mean =
    comparisons.reduce((s, c) => s + c.similarity, 0) / comparisons.length;
  const score = Math.max(0, Math.min(100, Math.round(mean * 100)));

  runtime.logger.info(
    `[slm16] Intelligence score: ${score}/100 (mean similarity=${mean.toFixed(3)} over ${comparisons.length} prompts, Nova-Pro-judged)`,
  );

  return { score, comparisons, evaluatedAt: new Date().toISOString() };
}
