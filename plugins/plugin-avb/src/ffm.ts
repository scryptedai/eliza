/**
 * FFM — Five-Factor Model (OCEAN) personality engine for AVBs.
 *
 * A self-contained, deterministic module that turns a 256-bit seed into
 * a Big-Five personality profile, classifies it into one of 32 binary
 * archetypes, and (optionally) expands it into a full ElizaOS Character
 * via a ScryptedAI text-generation PromptSet.
 *
 * Pipeline:
 *   createFfmSeed()                 — 32 random bytes → 64-char hex
 *     └─ rollFfmTraits(seed)        — seeded PRNG → 5 trait scores ∈ [0,1]
 *          └─ classifyArchetype()   — high/low per trait → 1-of-32 archetype
 *               └─ ffmPromptSet     — render LLM prompt from scores+archetype
 *                    └─ expandFfmToCharacterFields()  — call ScryptedAI, parse JSON
 *                         └─ buildFfmCharacter() / ensureFfmCharacterFile()
 *
 * Determinism guarantee: every step downstream of the seed is a pure
 * function of the seed. Persist the seed (e.g. in character.settings)
 * and the entire personality is reproducible.
 *
 * Distribution rationale (req. 4):
 *   Population-level Big-Five scores are approximately bell-shaped on a
 *   bounded scale (NEO-PI-R T-scores; BFI-2 norms — Soto & John 2017;
 *   cross-cultural — Schmitt et al. 2007). Traits are NOT uniform:
 *   Agreeableness/Conscientiousness skew high in self-report; Neuroticism
 *   skews low (most agents should NOT be highly neurotic). We model each
 *   trait with a Beta(α,β) distribution — naturally bounded on [0,1], no
 *   truncation/clamping artefacts — with per-trait (α,β) solved from
 *   target (μ,σ) via the method-of-moments. Sampling uses Cheng's BB
 *   rejection method (uniform-only; no Box-Muller / normal variates).
 *   See FFM_DISTRIBUTIONS below.
 */

import { randomBytes } from "node:crypto";
import fs from "node:fs";
import {
  type Character,
  PromptSet,
  saveCharacter,
  toScryptedPayload,
} from "@elizaos/core";
import {
  SCRYPTEDAI_SERVICE_TYPE,
  type ScryptedAIService,
} from "@elizaos/plugin-scryptedai";

import { IMAGE_PROMPT_MODEL } from "./introspect.ts";

// ----------------------------------------------------------------------------
// Trait keys & types
// ----------------------------------------------------------------------------

/** Canonical OCEAN ordering. Index order matters (archetype bit-encoding). */
export const FFM_TRAITS = [
  "openness",
  "conscientiousness",
  "extraversion",
  "agreeableness",
  "neuroticism",
] as const;

export type FfmTrait = (typeof FFM_TRAITS)[number];

/** Five-Factor scores, each ∈ [0, 1]. */
export type FfmScores = Readonly<Record<FfmTrait, number>>;

export interface FfmProfile {
  /** 64-char lowercase hex (256 bits). */
  seed: string;
  /** Continuous scores ∈ [0,1] per trait. */
  scores: FfmScores;
  /** Nearest of the 32 binary archetypes. */
  archetype: FfmArchetype;
}

// ----------------------------------------------------------------------------
// Seed (req. 1)
//
// 256 bits = 32 bytes = 64 hex chars. Chosen to be crypto-native: the same
// seed can later be HKDF-stretched into ERC-8004 (Trustless Agents) -compatible
// secrets / addresses without re-rolling the personality. Matches core's
// encryption-salt size; ~10^77 distinct seeds. Hex is JSON-safe and embeds
// cleanly into character.settings.
// ----------------------------------------------------------------------------

export const FFM_SEED_BYTES = 32;
export const FFM_SEED_HEX_LEN = FFM_SEED_BYTES * 2;

/** Generate a fresh 256-bit seed as lowercase hex. */
export function createFfmSeed(): string {
  return randomBytes(FFM_SEED_BYTES).toString("hex");
}

/** Validate + canonicalize a seed string. Throws on invalid input. */
export function normalizeFfmSeed(seed: string): string {
  const s = String(seed).trim().toLowerCase();
  if (s.length !== FFM_SEED_HEX_LEN || !/^[0-9a-f]+$/.test(s)) {
    throw new Error(
      `FFM seed must be ${FFM_SEED_HEX_LEN} hex chars (got ${s.length})`,
    );
  }
  return s;
}

// ----------------------------------------------------------------------------
// Seeded PRNG
//
// sfc32 (Small Fast Counter) — public domain, 128-bit state, period ≈ 2^128.
// Passes PractRand to 32 TB. State is the first 128 bits of the seed
// (4×uint32 words from hex chars 0..31), so the PRNG stream is a pure
// function of the seed.
// ----------------------------------------------------------------------------

type Rng = () => number;

function sfc32(a: number, b: number, c: number, d: number): Rng {
  return () => {
    a |= 0;
    b |= 0;
    c |= 0;
    d |= 0;
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
}

/** Build a deterministic PRNG from a 256-bit hex seed. */
export function rngFromSeed(seed: string): Rng {
  const s = normalizeFfmSeed(seed);
  // Words are taken from hex offsets 0,8,16,24 (first 128 bits).
  // Remaining 128 bits are reserved for future derivations (e.g. visual
  // genome) without disturbing trait determinism.
  const w = (off: number) => parseInt(s.slice(off, off + 8), 16) >>> 0;
  const rng = sfc32(w(0), w(8), w(16), w(24));
  // Warm up: sfc32 needs a few rounds to diffuse a low-entropy seed.
  for (let i = 0; i < 12; i++) rng();
  return rng;
}

// ----------------------------------------------------------------------------
// Trait distributions (req. 4)
//
// Each trait is sampled from a Beta(α,β) on its native support [0,1].
// Beta is the textbook "naturally bounded" choice here: no clamping, no
// truncation bias, smooth density that can be symmetric or skewed by
// shape parameters alone.
//
// Per-trait (α,β) are solved from target (μ,σ) via the method of moments:
//   ν = μ(1−μ)/σ² − 1,   α = μν,   β = (1−μ)ν.
// Targets are chosen so the implied population matches large normative
// Big-Five samples after rescaling raw scores to [0,1]:
//
//   - O, E:   centred near 0.50–0.55, σ ≈ 0.20–0.22 (broad spread)
//   - C, A:   shifted high (μ ≈ 0.56–0.58) — self-report positivity bias
//   - N:      shifted LOW  (μ = 0.40) — most agents are NOT highly neurotic;
//             P(N ≥ 0.5) under Beta(1.58, 2.38) ≈ 0.30, matching the spec.
//
// Sampling: Cheng's BB rejection method (Cheng 1978, CACM 21(4)). Uses
// only uniform draws + log/exp — no Box-Muller / normal variates — and
// is valid whenever min(α,β) > 1, which holds for every trait below.
//
// References (normative means, rescaled):
//   Costa & McCrae (1992) NEO-PI-R manual; Soto & John (2017) BFI-2 norms;
//   Schmitt, Allik, McCrae & Benet-Martínez (2007) 56-nation BFI.
// ----------------------------------------------------------------------------

export interface TraitDistribution {
  /** Target population mean on [0,1] (for reference / tests). */
  mean: number;
  /** Target population SD on [0,1] (for reference / tests). */
  sd: number;
  /** Beta shape α (derived from mean,sd via method of moments). */
  alpha: number;
  /** Beta shape β. */
  beta: number;
}

/** Solve Beta(α,β) shape parameters from target (μ,σ). */
function betaFromMoments(mean: number, sd: number): TraitDistribution {
  const v = sd * sd;
  const nu = (mean * (1 - mean)) / v - 1;
  return { mean, sd, alpha: mean * nu, beta: (1 - mean) * nu };
}

export const FFM_DISTRIBUTIONS: Readonly<Record<FfmTrait, TraitDistribution>> =
  Object.freeze({
    openness: betaFromMoments(0.55, 0.2), //  α≈2.85 β≈2.33
    conscientiousness: betaFromMoments(0.56, 0.18), //  α≈3.70 β≈2.91
    extraversion: betaFromMoments(0.5, 0.22), //  α≈2.08 β≈2.08
    agreeableness: betaFromMoments(0.58, 0.18), //  α≈3.78 β≈2.74
    neuroticism: betaFromMoments(0.4, 0.22), //  α≈1.58 β≈2.38
  });

const LN4 = Math.log(4);
const LN5_PLUS_1 = 1 + Math.log(5);

/**
 * Sample X ~ Beta(α,β) using Cheng's BB algorithm.
 * Requires min(α,β) > 1. Average ≤ ~1.4 iterations.
 */
function sampleBeta(rng: Rng, alpha: number, beta: number): number {
  // Order so a = min, b = max; remember whether we swapped.
  const swapped = alpha > beta;
  const a = swapped ? beta : alpha;
  const b = swapped ? alpha : beta;
  const sum = a + b;
  const lambda = Math.sqrt((sum - 2) / (2 * a * b - sum));
  const c = a + 1 / lambda;

  // Rejection loop. Bounded in expectation; hard cap defends against a
  // pathological RNG returning 0/1 repeatedly.
  for (let i = 0; i < 1000; i++) {
    const u1 = rng();
    const u2 = rng();
    // Logit transform; guard endpoints (sfc32 can emit 0).
    const uu = Math.min(Math.max(u1, Number.EPSILON), 1 - Number.EPSILON);
    const v = lambda * Math.log(uu / (1 - uu));
    const w = a * Math.exp(v);
    const z = u1 * u1 * u2;
    const r = c * v - LN4;
    const s = a + r - w;
    let accept = s + LN5_PLUS_1 >= 5 * z;
    if (!accept) {
      const t = Math.log(Math.max(z, Number.EPSILON));
      accept = s >= t || r + sum * Math.log(sum / (b + w)) >= t;
    }
    if (accept) {
      // X' ~ Beta(a,b) = w/(b+w). If we swapped, want Beta(b,a) = 1 − X'.
      return swapped ? b / (b + w) : w / (b + w);
    }
  }
  // Unreachable in practice; fall back to mean.
  return alpha / (alpha + beta);
}

/** Sample one trait from its configured Beta distribution. */
function sampleTrait(rng: Rng, dist: TraitDistribution): number {
  return sampleBeta(rng, dist.alpha, dist.beta);
}

// ----------------------------------------------------------------------------
// Roll (req. 2)
// ----------------------------------------------------------------------------

/**
 * Deterministically derive Five-Factor scores from a seed.
 * Same seed → same scores, always.
 */
export function rollFfmTraits(seed: string): FfmScores {
  const rng = rngFromSeed(seed);
  const out: Record<FfmTrait, number> = {} as Record<FfmTrait, number>;
  for (const trait of FFM_TRAITS) {
    out[trait] = sampleTrait(rng, FFM_DISTRIBUTIONS[trait]);
  }
  return Object.freeze(out);
}

// ----------------------------------------------------------------------------
// Archetypes (req. 3)
//
// 2^5 = 32 binary combinations of high (≥0.5) / low (<0.5) on each trait.
// Encoded as a 5-bit integer with bit i = 1 ⇔ FFM_TRAITS[i] is HIGH.
//   bit 0 (LSB) = O, bit 1 = C, bit 2 = E, bit 3 = A, bit 4 = N.
// Code string is "O±C±E±A±N±" for human readability.
// ----------------------------------------------------------------------------

export const FFM_ARCHETYPE_THRESHOLD = 0.5;

export interface FfmArchetype {
  /** 0..31 — stable bit-encoded identifier. */
  id: number;
  /** e.g. "O+C-E+A+N-" */
  code: string;
  /** Short evocative label. */
  name: string;
  /** One-line behavioural sketch. */
  description: string;
}

/** Per-trait pole adjectives used to synthesize archetype descriptions. */
const POLE_WORDS: Readonly<
  Record<FfmTrait, { high: string; low: string }>
> = {
  openness: { high: "imaginative", low: "grounded" },
  conscientiousness: { high: "disciplined", low: "spontaneous" },
  extraversion: { high: "outgoing", low: "reserved" },
  agreeableness: { high: "warm", low: "blunt" },
  neuroticism: { high: "volatile", low: "steady" },
};

/**
 * The 32 archetypes. Indexed by bit-encoded id.
 *
 * Names are original to this module (not a licensed inventory). Each is
 * chosen to evoke the dominant behavioural flavour of its high/low combo;
 * the `description` is generated from POLE_WORDS so it stays faithful to
 * the underlying bits even if a name is later edited.
 */
export const FFM_ARCHETYPES: ReadonlyArray<FfmArchetype> = Object.freeze(
  buildArchetypeTable([
    /* 00000 O-C-E-A-N- */ "The Hermit",
    /* 00001 O+C-E-A-N- */ "The Wanderer",
    /* 00010 O-C+E-A-N- */ "The Inspector",
    /* 00011 O+C+E-A-N- */ "The Architect",
    /* 00100 O-C-E+A-N- */ "The Hustler",
    /* 00101 O+C-E+A-N- */ "The Maverick",
    /* 00110 O-C+E+A-N- */ "The Commander",
    /* 00111 O+C+E+A-N- */ "The Pioneer",
    /* 01000 O-C-E-A+N- */ "The Caretaker",
    /* 01001 O+C-E-A+N- */ "The Dreamer",
    /* 01010 O-C+E-A+N- */ "The Steward",
    /* 01011 O+C+E-A+N- */ "The Sage",
    /* 01100 O-C-E+A+N- */ "The Companion",
    /* 01101 O+C-E+A+N- */ "The Entertainer",
    /* 01110 O-C+E+A+N- */ "The Captain",
    /* 01111 O+C+E+A+N- */ "The Luminary",
    /* 10000 O-C-E-A-N+ */ "The Recluse",
    /* 10001 O+C-E-A-N+ */ "The Brooder",
    /* 10010 O-C+E-A-N+ */ "The Sentinel",
    /* 10011 O+C+E-A-N+ */ "The Perfectionist",
    /* 10100 O-C-E+A-N+ */ "The Firebrand",
    /* 10101 O+C-E+A-N+ */ "The Provocateur",
    /* 10110 O-C+E+A-N+ */ "The Driver",
    /* 10111 O+C+E+A-N+ */ "The Crusader",
    /* 11000 O-C-E-A+N+ */ "The Worrier",
    /* 11001 O+C-E-A+N+ */ "The Romantic",
    /* 11010 O-C+E-A+N+ */ "The Guardian",
    /* 11011 O+C+E-A+N+ */ "The Idealist",
    /* 11100 O-C-E+A+N+ */ "The Empath",
    /* 11101 O+C-E+A+N+ */ "The Performer",
    /* 11110 O-C+E+A+N+ */ "The Advocate",
    /* 11111 O+C+E+A+N+ */ "The Tempest",
  ]),
);

function buildArchetypeTable(names: readonly string[]): FfmArchetype[] {
  if (names.length !== 32) {
    throw new Error(`FFM archetype table must have exactly 32 entries`);
  }
  return names.map((name, id) => {
    const codeParts: string[] = [];
    const descParts: string[] = [];
    for (let i = 0; i < FFM_TRAITS.length; i++) {
      const trait = FFM_TRAITS[i];
      const high = (id >> i) & 1;
      codeParts.push(`${trait[0].toUpperCase()}${high ? "+" : "-"}`);
      descParts.push(high ? POLE_WORDS[trait].high : POLE_WORDS[trait].low);
    }
    return {
      id,
      code: codeParts.join(""),
      name,
      description: `${descParts.slice(0, 4).join(", ")}, and ${descParts[4]}.`,
    };
  });
}

/** Map continuous scores → nearest of the 32 binary archetypes. */
export function classifyArchetype(scores: FfmScores): FfmArchetype {
  let id = 0;
  for (let i = 0; i < FFM_TRAITS.length; i++) {
    if (scores[FFM_TRAITS[i]] >= FFM_ARCHETYPE_THRESHOLD) id |= 1 << i;
  }
  return FFM_ARCHETYPES[id];
}

// ----------------------------------------------------------------------------
// One-shot: seed → full profile
// ----------------------------------------------------------------------------

/** Derive the complete deterministic profile (scores + archetype) from a seed. */
export function deriveFfmProfile(seed: string): FfmProfile {
  const s = normalizeFfmSeed(seed);
  const scores = rollFfmTraits(s);
  return Object.freeze({
    seed: s,
    scores,
    archetype: classifyArchetype(scores),
  });
}

/** Human-readable one-liner, e.g. "O:0.71 C:0.44 E:0.58 A:0.62 N:0.21". */
export function formatFfmScores(scores: FfmScores): string {
  return FFM_TRAITS.map(
    (t) => `${t[0].toUpperCase()}:${scores[t].toFixed(2)}`,
  ).join(" ");
}

// ----------------------------------------------------------------------------
// PromptSet — FFM → ElizaOS character fields (req. 5)
//
// System prompt is a stable instruction envelope; user prompt embeds the
// concrete scores + archetype. Output contract is a strict JSON object so
// it can be parsed and merged straight into a Character.
// ----------------------------------------------------------------------------

/** Model used for personality expansion (same backend as introspect). */
export const FFM_TEXT_MODEL = IMAGE_PROMPT_MODEL;

export const ffmPromptSet = new PromptSet({
  system:
    "You are a personality writer for autonomous AI agents. Given a " +
    "Five-Factor (OCEAN) personality profile — five scores in [0,1] plus " +
    "a named archetype — produce a vivid, internally-consistent character " +
    "sheet as STRICT JSON. Scores near 0 mean very LOW on that trait; near " +
    "1 mean very HIGH. Let every field reflect the scores: a low-extraversion " +
    "agent should sound reserved; a high-neuroticism agent should sound " +
    "anxious or intense. Output ONLY the JSON object — no prose, no markdown " +
    "fences, no trailing commentary. Required schema:\n" +
    '{ "bio": string[] (3-5 first-person sentences),\n' +
    '  "adjectives": string[] (6-10 single words),\n' +
    '  "topics": string[] (5-8 interests),\n' +
    '  "style": { "all": string[] (4-6 rules), "chat": string[] (3-5), "post": string[] (3-5) },\n' +
    '  "postExamples": string[] (3-5 short sample posts in this voice),\n' +
    '  "beliefs": string[] (3-5 core convictions) }',
  user:
    "Agent name: {{NAME}}\n" +
    "Archetype: {{ARCHETYPE_NAME}} ({{ARCHETYPE_CODE}}) — {{ARCHETYPE_DESC}}\n" +
    "OCEAN scores (0=low, 1=high):\n" +
    "  Openness:          {{O}}\n" +
    "  Conscientiousness: {{C}}\n" +
    "  Extraversion:      {{E}}\n" +
    "  Agreeableness:     {{A}}\n" +
    "  Neuroticism:       {{N}}\n\n" +
    "Write the JSON character sheet now.",
  model: FFM_TEXT_MODEL,
});

/** Render the personality prompt for a given profile. */
export function renderFfmPrompt(
  profile: FfmProfile,
  name = "The agent",
): ReturnType<PromptSet["render"]> {
  const s = profile.scores;
  return ffmPromptSet.render({
    NAME: name,
    ARCHETYPE_NAME: profile.archetype.name,
    ARCHETYPE_CODE: profile.archetype.code,
    ARCHETYPE_DESC: profile.archetype.description,
    O: s.openness.toFixed(2),
    C: s.conscientiousness.toFixed(2),
    E: s.extraversion.toFixed(2),
    A: s.agreeableness.toFixed(2),
    N: s.neuroticism.toFixed(2),
  });
}

// ----------------------------------------------------------------------------
// LLM output parsing
// ----------------------------------------------------------------------------

/** Shape of the JSON the LLM is instructed to emit. All fields optional. */
export interface FfmCharacterFields {
  bio?: string[];
  adjectives?: string[];
  topics?: string[];
  style?: { all?: string[]; chat?: string[]; post?: string[] };
  postExamples?: string[];
  beliefs?: string[];
}

const strArr = (v: unknown): string[] | undefined =>
  Array.isArray(v)
    ? v.filter((x): x is string => typeof x === "string" && x.trim() !== "")
    : undefined;

/**
 * Defensively extract the JSON object from an LLM response.
 * Handles ```json fences and leading/trailing prose by slicing from the
 * first `{` to the last `}`.
 */
export function parseFfmCharacterFields(raw: string): FfmCharacterFields {
  let body = raw.trim();
  // Strip ``` fences if present.
  body = body
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();
  const start = body.indexOf("{");
  const end = body.lastIndexOf("}");
  if (start < 0 || end <= start) {
    throw new Error("FFM: model output contained no JSON object");
  }
  const obj = JSON.parse(body.slice(start, end + 1)) as Record<
    string,
    unknown
  >;
  const style = (obj.style ?? {}) as Record<string, unknown>;
  return {
    bio: strArr(obj.bio),
    adjectives: strArr(obj.adjectives),
    topics: strArr(obj.topics),
    style: {
      all: strArr(style.all),
      chat: strArr(style.chat),
      post: strArr(style.post),
    },
    postExamples: strArr(obj.postExamples),
    beliefs: strArr(obj.beliefs),
  };
}

// ----------------------------------------------------------------------------
// ScryptedAI expansion (req. 5)
// ----------------------------------------------------------------------------

/** Minimal runtime surface needed to reach the ScryptedAI service. */
export interface FfmRuntimeSurface {
  getService<T = unknown>(type: string): T | undefined;
}

/**
 * Submit the FFM prompt to ScryptedAI, await the terminal result, and
 * parse the returned JSON into character fields.
 *
 * Blocks until the text job is terminal (or `timeoutMs` elapses). For
 * the db-persisted/non-blocking variant, wire a TEXT_PHASE-style worker
 * instead — this helper is for boot-time / script usage.
 */
export async function expandFfmToCharacterFields(
  runtime: FfmRuntimeSurface,
  profile: FfmProfile,
  opts: { name?: string; timeoutMs?: number } = {},
): Promise<FfmCharacterFields> {
  const scrypted = runtime.getService<ScryptedAIService>(
    SCRYPTEDAI_SERVICE_TYPE,
  );
  if (!scrypted) {
    throw new Error(
      "FFM: ScryptedAI service not available (is plugin-scryptedai loaded?)",
    );
  }

  const rendered = renderFfmPrompt(profile, opts.name);
  const { jobId } = await scrypted.startTextGeneration(
    toScryptedPayload(rendered),
  );
  const result = await scrypted.awaitJob(jobId, opts.timeoutMs ?? 90_000);

  if (result.status !== "completed" || !result.text) {
    throw new Error(
      `FFM: text generation ${result.status}${result.error ? ` — ${result.error}` : ""}`,
    );
  }
  return parseFfmCharacterFields(result.text);
}

// ----------------------------------------------------------------------------
// Character assembly + persistence
// ----------------------------------------------------------------------------

/** Settings key under which the seed is persisted on the Character. */
export const FFM_SEED_SETTING = "AVB_FFM_SEED";

/**
 * Return a structural clone of `character` with `secrets` and
 * `settings.secrets` removed.
 *
 * Core's `loadCharacter()` injects process-env secrets (API keys, encryption
 * salt) into the in-memory Character. Core's `saveCharacter()` writes the
 * object verbatim (only locking file mode to 0600). Any code that persists a
 * Character derived from the runtime — as AVB does in `bootstrapFfmPersonality`
 * and `ensureFfmCharacterFile` — MUST pass the result of this function to
 * `saveCharacter`, never the live runtime character, or credentials will land
 * on disk in plaintext.
 */
export function stripCharacterSecrets(character: Character): Character {
  const { secrets: _s, ...rest } = character as Character & {
    secrets?: unknown;
  };
  const settings = { ...((rest.settings ?? {}) as Record<string, unknown>) };
  delete settings.secrets;
  return { ...rest, settings } as Character;
}

/**
 * Merge an FFM profile + expanded fields into a Character object.
 * The seed and raw scores are stored under `settings` so the personality
 * can be re-derived later without the LLM.
 */
export function buildFfmCharacter(
  base: Partial<Character>,
  profile: FfmProfile,
  fields: FfmCharacterFields,
): Character {
  const name = base.name ?? profile.archetype.name.replace(/^The\s+/i, "");
  const baseSettings = (base.settings ?? {}) as Record<string, unknown>;
  return {
    ...base,
    name,
    bio: fields.bio ?? base.bio ?? [profile.archetype.description],
    adjectives: fields.adjectives ?? base.adjectives,
    topics: fields.topics ?? base.topics,
    postExamples: fields.postExamples ?? base.postExamples,
    style: {
      ...(base.style ?? {}),
      all: fields.style?.all ?? base.style?.all,
      chat: fields.style?.chat ?? base.style?.chat,
      post: fields.style?.post ?? base.style?.post,
    },
    settings: {
      ...baseSettings,
      [FFM_SEED_SETTING]: profile.seed,
      AVB_FFM_SCORES: { ...profile.scores },
      AVB_FFM_ARCHETYPE: {
        id: profile.archetype.id,
        code: profile.archetype.code,
        name: profile.archetype.name,
      },
      ...(fields.beliefs ? { AVB_FFM_BELIEFS: fields.beliefs } : {}),
    },
  } as Character;
}

/**
 * Auto-generate an FFM-driven character.json if one does not already exist.
 *
 * - If `filePath` exists → no-op (returns the existing path).
 * - Otherwise: roll a seed (or use the supplied one), expand via ScryptedAI,
 *   build a Character, and persist it with core's `saveCharacter()`.
 *
 * The seed is the single source of truth: persisting it means the entire
 * personality (scores, archetype, prompt) can be reconstructed at any time.
 */
export async function ensureFfmCharacterFile(
  runtime: FfmRuntimeSurface,
  opts: {
    filePath?: string;
    seed?: string;
    name?: string;
    base?: Partial<Character>;
    timeoutMs?: number;
  } = {},
): Promise<{ filePath: string; created: boolean; profile: FfmProfile }> {
  const filePath = opts.filePath ?? "character.json";

  const seed = opts.seed ? normalizeFfmSeed(opts.seed) : createFfmSeed();
  const profile = deriveFfmProfile(seed);

  if (fs.existsSync(filePath)) {
    return { filePath, created: false, profile };
  }

  const fields = await expandFfmToCharacterFields(runtime, profile, {
    name: opts.name,
    timeoutMs: opts.timeoutMs,
  });
  const character = buildFfmCharacter(
    { ...(opts.base ?? {}), name: opts.name ?? opts.base?.name },
    profile,
    fields,
  );

  await saveCharacter(stripCharacterSecrets(character), filePath);
  return { filePath, created: true, profile };
}
