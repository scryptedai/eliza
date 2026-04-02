/**
 * Prompt sets and JSON parsers for FFM personality expansion.
 *
 * Two sequential ScryptedAI text-gen calls:
 *   1. narrativePromptSet — backstory, motivation, beliefs, adjectives, topics
 *   2. voicePromptSet     — dialogue, posts, style (consumes #1's output)
 *
 * The split exists because a single LLM call asked to produce all of these
 * at once spreads its token budget thin and yields shallow output. Two
 * focused calls — where the second reads the first — let the model build
 * a coherent voice on top of an established backstory.
 */

import { PromptSet, type RenderedPromptSet } from "@elizaos/core";
import { FFM_TEXT_MODEL, TRAIT_PARAMS } from "./constants.ts";
import type {
  FfmProfile,
  MessageTurn,
  NarrativeExpansion,
  VoiceExpansion,
} from "./types.ts";

// ============================================================================
// Trait-to-behaviour translation
//
// The system prompts don't just tell the LLM "Openness 0.71" — they explain
// what each score MEANS BEHAVIOURALLY so the model doesn't fall back on
// surface word-association ("high openness → artist"). Each trait gets a
// gradient of three descriptors (low / mid / high) and the closest one is
// selected by score.
// ============================================================================

interface TraitGradient {
  lo: string;
  mid: string;
  hi: string;
}

const BEHAVIOURAL: Record<keyof typeof TRAIT_PARAMS, TraitGradient> = {
  O: {
    lo: "prefers the familiar; finds comfort in routine; abstract ideas feel like a detour from the real thing",
    mid: "open to new ideas when they're useful; not seeking novelty for its own sake",
    hi: "drawn to ideas, aesthetics, what-ifs; gets bored with repetition; the unfamiliar is the interesting part",
  },
  C: {
    lo: "improvises more than plans; deadlines are suggestions; loses things, finds them, loses them again",
    mid: "organized when it matters, relaxed when it doesn't; reliable on the things they care about",
    hi: "plans ahead; finishes what they start; physically uncomfortable leaving something half-done",
  },
  E: {
    lo: "recharges alone; one-on-one over groups; the party is fine but leaving the party is better",
    mid: "comfortable in groups but doesn't seek them out; energy is steady rather than spiked by company",
    hi: "energized by people; talks to think; silence in a room is a problem to be solved",
  },
  A: {
    lo: "skeptical of others' motives; says the unpopular thing; cooperation is a choice, not a default",
    mid: "cooperative without being a pushover; gives benefit of the doubt but not infinitely",
    hi: "assumes good intent; conflict-averse; would rather absorb a cost than impose one",
  },
  N: {
    lo: "shrugs off setbacks; rarely catastrophizes; may underestimate real risks because nothing feels that urgent",
    mid: "worries proportionately; bad days are bad days, not patterns",
    hi: "anticipates what could go wrong; setbacks linger; the body knows before the mind does",
  },
};

/** Pick the gradient band for a [0,1] score. Thirds: <0.4, 0.4–0.6, >0.6. */
function band(score: number, g: TraitGradient): string {
  if (score < 0.4) return g.lo;
  if (score > 0.6) return g.hi;
  return g.mid;
}

/**
 * Render the five-trait behavioural digest that gets spliced into both
 * system prompts. Deterministic; same profile → same digest.
 */
export function renderTraitDigest(profile: FfmProfile): string {
  const t = profile.traits;
  const lines: string[] = [];
  for (const [key, name] of [
    ["O", "Openness"],
    ["C", "Conscientiousness"],
    ["E", "Extraversion"],
    ["A", "Agreeableness"],
    ["N", "Neuroticism"],
  ] as const) {
    const score = t[key];
    const median = TRAIT_PARAMS[key].mean;
    const pole = score >= median ? "high" : "low";
    const behaviour = band(score, BEHAVIOURAL[key]);
    lines.push(
      `- ${name}: ${score.toFixed(2)} (${pole}). Behavioural read: ${behaviour}.`,
    );
  }
  return lines.join("\n");
}

// ============================================================================
// Call #1 — Narrative expansion
// ============================================================================

const NARRATIVE_SYSTEM = `You are a character writer translating a Five Factor Model personality profile into a coherent fictional person.

You will receive five trait scores (0.00–1.00) with behavioural readings, plus an archetype label. Your job is to invent a SPECIFIC person who would produce those scores — not describe the scores. The traits should be VISIBLE in how this person talks about their past, what they want, what they avoid. Never name a trait directly ("she is highly conscientious"). Show it.

Output format: one JSON object with exactly these four keys, no other text before or after.

{
  "bio": [
    /* 8–10 strings. Structure them like this: */
    /* [0–2] WHO THEY ARE NOW — present tense, concrete, what they do and how */
    /* [3–5] FORMATIVE BACKSTORY — what shaped them; one specific event or period, not a CV */
    /* [6–8] CURRENT MOTIVATION — what they're chasing, what they're avoiding, why */
    /* [9]   CONTRADICTION — one line where two of their traits visibly pull against each other */
  ],
  "beliefs": [
    /* 4–6 worldview statements IN THEIR OWN VOICE. First person. */
    /* These are things they'd say if you got them talking late at night. */
    /* High-N example: "I think most plans fall apart. The trick is falling apart slower than the plan." */
    /* Low-A example: "People say they want honesty but they actually want agreement." */
  ],
  "adjectives": [
    /* 10–14 single words. Weight by trait dominance — if Openness is the */
    /* highest score, openness-flavoured words should dominate. */
    /* Avoid generic positives ("smart", "kind"). Be specific. */
  ],
  "topics": [
    /* 8–12 things this person would actually talk about unprompted. */
    /* Not "philosophy" — "whether free will is just a useful fiction". */
    /* Not "music" — "why the bridge is the best part of any song". */
  ]
}

Constraints:
- Each bio line: one sentence, 12–25 words. No bullet markers inside the strings.
- The backstory event must be SPECIFIC (a place, an age, a moment) not generic ("had a difficult childhood").
- The contradiction line must name the tension implicitly through behaviour, not by saying "contradiction:".
- Beliefs must sound spoken, not written. Contractions are fine.
- Output ONLY the JSON object. No markdown fences, no preamble.`;

const NARRATIVE_USER = `Trait profile:
{{TRAIT_DIGEST}}

Archetype: {{ARCHETYPE_LABEL}} ({{ARCHETYPE_SLOAN}})
{{ARCHETYPE_SUMMARY}}

Subject name: {{AGENT_NAME}}

Write the JSON object now.`;

export const narrativePromptSet = new PromptSet({
  system: NARRATIVE_SYSTEM,
  user: NARRATIVE_USER,
  model: FFM_TEXT_MODEL,
});

/** Render the narrative prompt for a given profile + agent name. */
export function renderNarrativePrompt(
  profile: FfmProfile,
  agentName: string,
): RenderedPromptSet {
  return narrativePromptSet.render({
    TRAIT_DIGEST: renderTraitDigest(profile),
    ARCHETYPE_LABEL: profile.archetype.label,
    ARCHETYPE_SLOAN: profile.archetype.sloan,
    ARCHETYPE_SUMMARY: profile.archetype.summary,
    AGENT_NAME: agentName,
  });
}

// ============================================================================
// Call #2 — Voice expansion (consumes call #1's output)
// ============================================================================

const VOICE_SYSTEM = `You are writing dialogue and posts for an established character. You will be given their bio, their beliefs, and their Five Factor trait scores. Your job is to write SIX short conversation exchanges and several standalone posts that sound like ONE consistent person talking — not a personality profile reading itself aloud.

The six exchanges are scenario-targeted. Each tests a different trait. The character's reply should DEMONSTRATE the trait through word choice, sentence rhythm, what they focus on, what they leave unsaid — not state it.

Scenario → trait mapping (this is for YOU, do not include this in output):
  1. Someone asks for help → Agreeableness (do they help readily? grudgingly? with conditions?)
  2. Someone disagrees with them → Agreeableness + Neuroticism (do they push back? fold? get defensive?)
  3. Casual small talk → Extraversion (do they expand the conversation or close it down?)
  4. Someone shares bad news → Neuroticism + Agreeableness (do they catastrophize? reassure? deflect?)
  5. An abstract or weird question → Openness (do they engage with the strange or redirect to the practical?)
  6. Asked to commit to a plan → Conscientiousness (do they commit firmly? hedge? overpromise?)

Output format: one JSON object, no other text.

{
  "messageExamples": [
    [ {"name": "user", "content": {"text": "<scenario 1 prompt>"}},
      {"name": "{{AGENT_NAME}}", "content": {"text": "<reply, 2–4 sentences>"}} ],
    [ {"name": "user", "content": {"text": "<scenario 2 prompt>"}},
      {"name": "{{AGENT_NAME}}", "content": {"text": "<reply>"}} ],
    /* ... six total, in scenario order 1–6 */
  ],
  "postExamples": [
    /* 6–8 standalone posts. The voice in MONOLOGUE — no interlocutor. */
    /* This is where idiolect shows: sentence length, punctuation habits, */
    /* what they find funny, what they notice. 1–3 sentences each. */
  ],
  "style": {
    "all":  [ /* 6–8 rules DERIVED from the dialogue you just wrote */ ],
    "chat": [ /* 4–5 chat-specific rules (how they reply to people) */ ],
    "post": [ /* 3–4 post-specific rules (how they write unprompted) */ ]
  }
}

Critical constraints:
- The user prompts in messageExamples must be NATURAL — things a real person would say, not "demonstrate your Agreeableness".
- The agent name in every agent turn must be exactly "{{AGENT_NAME}}".
- Replies must vary in length and structure. If all six replies are three medium sentences with the same rhythm, you have failed.
- Style rules must be DESCRIPTIVE of the dialogue you wrote ("uses fragments when excited", "asks a question back instead of answering directly") not generic advice ("be helpful", "be authentic").
- postExamples must NOT be addressed to anyone. No "you", no @-mentions.
- Output ONLY the JSON object. No markdown fences.`;

const VOICE_USER = `Character: {{AGENT_NAME}}

Trait profile:
{{TRAIT_DIGEST}}

Bio (who they are, backstory, motivation):
{{BIO}}

Beliefs (things they'd say unprompted):
{{BELIEFS}}

Archetype: {{ARCHETYPE_LABEL}} — {{ARCHETYPE_SUMMARY}}

Write the JSON object now. Six exchanges in scenario order, then posts, then style.`;

export const voicePromptSet = new PromptSet({
  system: VOICE_SYSTEM,
  user: VOICE_USER,
  model: FFM_TEXT_MODEL,
});

/** Render the voice prompt. Requires the narrative output from call #1. */
export function renderVoicePrompt(
  profile: FfmProfile,
  narrative: NarrativeExpansion,
  agentName: string,
): RenderedPromptSet {
  return voicePromptSet.render({
    AGENT_NAME: agentName,
    TRAIT_DIGEST: renderTraitDigest(profile),
    BIO: narrative.bio.map((l) => `- ${l}`).join("\n"),
    BELIEFS: narrative.beliefs.map((b) => `- "${b}"`).join("\n"),
    ARCHETYPE_LABEL: profile.archetype.label,
    ARCHETYPE_SUMMARY: profile.archetype.summary,
  });
}

// ============================================================================
// JSON parsing — robust against LLM output noise
// ============================================================================

/**
 * Extract the first balanced `{...}` block from text. Handles markdown
 * fences, leading prose, trailing noise. Returns null if no balanced
 * object found.
 */
export function extractJsonBlock(text: string): string | null {
  // Strip ```json ... ``` and ``` ... ``` fences first.
  let s = text.trim();
  const fence = /```(?:json)?\s*([\s\S]*?)```/i.exec(s);
  if (fence) s = fence[1].trim();

  // Find first '{' and balance to its matching '}'.
  const start = s.indexOf("{");
  if (start === -1) return null;

  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < s.length; i++) {
    const c = s[i];
    if (esc) {
      esc = false;
      continue;
    }
    if (c === "\\") {
      esc = true;
      continue;
    }
    if (c === '"') {
      inStr = !inStr;
      continue;
    }
    if (inStr) continue;
    if (c === "{") depth++;
    else if (c === "}") {
      depth--;
      if (depth === 0) return s.slice(start, i + 1);
    }
  }
  return null;
}

/** Typed parse error so callers can distinguish from other failures. */
export class FfmParseError extends Error {
  constructor(
    message: string,
    public readonly rawText: string,
  ) {
    super(`[ffm] ${message}`);
    this.name = "FfmParseError";
  }
}

// --- Narrative parsing -------------------------------------------------------

function asStringArray(v: unknown, name: string, raw: string): string[] {
  if (!Array.isArray(v)) {
    throw new FfmParseError(`${name} is not an array`, raw);
  }
  const out: string[] = [];
  for (const item of v) {
    if (typeof item === "string" && item.trim()) out.push(item.trim());
  }
  if (out.length === 0) {
    throw new FfmParseError(`${name} has no usable entries`, raw);
  }
  return out;
}

/** Parse and structurally validate the output of narrativePromptSet. */
export function parseNarrative(rawText: string): NarrativeExpansion {
  const block = extractJsonBlock(rawText);
  if (!block) {
    throw new FfmParseError(
      "no JSON object found in narrative output",
      rawText,
    );
  }
  let obj: unknown;
  try {
    obj = JSON.parse(block);
  } catch (e) {
    throw new FfmParseError(
      `JSON.parse failed: ${e instanceof Error ? e.message : String(e)}`,
      rawText,
    );
  }
  if (typeof obj !== "object" || obj === null) {
    throw new FfmParseError("narrative root is not an object", rawText);
  }
  const o = obj as Record<string, unknown>;
  return {
    bio: asStringArray(o.bio, "bio", rawText),
    beliefs: asStringArray(o.beliefs, "beliefs", rawText),
    adjectives: asStringArray(o.adjectives, "adjectives", rawText),
    topics: asStringArray(o.topics, "topics", rawText),
  };
}

// --- Voice parsing -----------------------------------------------------------

function asMessageTurn(v: unknown, ctx: string, raw: string): MessageTurn {
  if (typeof v !== "object" || v === null) {
    throw new FfmParseError(`${ctx}: turn is not an object`, raw);
  }
  const o = v as Record<string, unknown>;
  const name = typeof o.name === "string" ? o.name : "";
  if (!name) {
    throw new FfmParseError(`${ctx}: turn missing name`, raw);
  }
  const content = o.content as Record<string, unknown> | undefined;
  const text =
    content && typeof content.text === "string" ? content.text.trim() : "";
  if (!text) {
    throw new FfmParseError(`${ctx}: turn missing content.text`, raw);
  }
  return { name, content: { text } };
}

/** Parse and structurally validate the output of voicePromptSet. */
export function parseVoice(rawText: string): VoiceExpansion {
  const block = extractJsonBlock(rawText);
  if (!block) {
    throw new FfmParseError("no JSON object found in voice output", rawText);
  }
  let obj: unknown;
  try {
    obj = JSON.parse(block);
  } catch (e) {
    throw new FfmParseError(
      `JSON.parse failed: ${e instanceof Error ? e.message : String(e)}`,
      rawText,
    );
  }
  if (typeof obj !== "object" || obj === null) {
    throw new FfmParseError("voice root is not an object", rawText);
  }
  const o = obj as Record<string, unknown>;

  // messageExamples: Array<Array<MessageTurn>>, each inner length === 2
  if (!Array.isArray(o.messageExamples)) {
    throw new FfmParseError("messageExamples is not an array", rawText);
  }
  const messageExamples: MessageTurn[][] = [];
  for (let i = 0; i < o.messageExamples.length; i++) {
    const ex = o.messageExamples[i];
    if (!Array.isArray(ex) || ex.length < 2) {
      throw new FfmParseError(
        `messageExamples[${i}] is not a 2-turn exchange`,
        rawText,
      );
    }
    messageExamples.push([
      asMessageTurn(ex[0], `messageExamples[${i}][0]`, rawText),
      asMessageTurn(ex[1], `messageExamples[${i}][1]`, rawText),
    ]);
  }
  if (messageExamples.length === 0) {
    throw new FfmParseError("messageExamples is empty", rawText);
  }

  // style: { all, chat, post }
  const styleRaw = o.style as Record<string, unknown> | undefined;
  if (!styleRaw || typeof styleRaw !== "object") {
    throw new FfmParseError("style is not an object", rawText);
  }
  const style = {
    all: asStringArray(styleRaw.all, "style.all", rawText),
    chat: asStringArray(styleRaw.chat, "style.chat", rawText),
    post: asStringArray(styleRaw.post, "style.post", rawText),
  };

  return {
    messageExamples,
    postExamples: asStringArray(o.postExamples, "postExamples", rawText),
    style,
  };
}

// ============================================================================
// Quality gates (used by the demo script and AVB integration logging)
// ============================================================================

/** Tokenize on whitespace + punctuation, lowercase, drop empties. */
function tokenize(s: string): Set<string> {
  return new Set(
    s
      .toLowerCase()
      .split(/[\s.,!?;:'"()[\]{}—–-]+/)
      .filter((t) => t.length > 0),
  );
}

/**
 * Compute pairwise Jaccard token overlap between agent replies. Returns
 * the maximum overlap found across all pairs (0 = totally distinct,
 * 1 = identical token sets). Used to detect LLM repetition collapse.
 */
export function maxReplyOverlap(messageExamples: MessageTurn[][]): number {
  const replies = messageExamples
    .map((ex) => ex[1]?.content?.text ?? "")
    .filter((t) => t.length > 0)
    .map(tokenize);
  let maxOverlap = 0;
  for (let i = 0; i < replies.length; i++) {
    for (let j = i + 1; j < replies.length; j++) {
      const a = replies[i];
      const b = replies[j];
      let inter = 0;
      for (const t of a) if (b.has(t)) inter++;
      const union = a.size + b.size - inter;
      if (union > 0) maxOverlap = Math.max(maxOverlap, inter / union);
    }
  }
  return maxOverlap;
}
