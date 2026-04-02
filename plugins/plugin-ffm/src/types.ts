/**
 * Public types for @elizaos/plugin-ffm.
 *
 * Two layers:
 *  - Pure derivation (FfmTraits, FfmArchetype, FfmProfile) — instant, deterministic
 *  - LLM expansion (NarrativeExpansion, VoiceExpansion) — slow, ScryptedAI text gen
 */

import type { NormalizedJobResult } from "@elizaos/plugin-scryptedai";

// ----------------------------------------------------------------------------
// Trait keys (canonical OCEAN order)
// ----------------------------------------------------------------------------

export type TraitKey = "O" | "C" | "E" | "A" | "N";
export const TRAIT_KEYS = ["O", "C", "E", "A", "N"] as const;

// ----------------------------------------------------------------------------
// FfmTraits — five values in [0,1], deterministically derived from a seed
// ----------------------------------------------------------------------------

export interface FfmTraits {
  /** Openness to experience: inventive/curious ↔ consistent/cautious */
  readonly O: number;
  /** Conscientiousness: efficient/organized ↔ easy-going/careless */
  readonly C: number;
  /** Extraversion: outgoing/energetic ↔ solitary/reserved */
  readonly E: number;
  /** Agreeableness: friendly/compassionate ↔ challenging/detached */
  readonly A: number;
  /** Neuroticism: sensitive/nervous ↔ resilient/confident */
  readonly N: number;
}

// ----------------------------------------------------------------------------
// FfmArchetype — one of 32 binary trait combinations
// ----------------------------------------------------------------------------

export interface FfmArchetype {
  /** 5-bit code, OCEAN bit order MSB→LSB. 0b11110 = O+C+E+A+N− = 30. */
  readonly code: number;
  /** SLOAN-style 5-letter name (e.g. "IOSAC"). */
  readonly sloan: string;
  /** Human-readable label (e.g. "The Diplomat"). */
  readonly label: string;
  /** One-sentence summary of the type's behavioural signature. */
  readonly summary: string;
}

// ----------------------------------------------------------------------------
// FfmProfile — full deterministic derivation result
// ----------------------------------------------------------------------------

export interface FfmProfile {
  /** 256-bit seed as a 64-character lowercase hex string. */
  readonly seed: string;
  /** Five trait scores in [0,1]. */
  readonly traits: FfmTraits;
  /** The nearest of 32 archetypes (binarized at distribution median). */
  readonly archetype: FfmArchetype;
}

// ----------------------------------------------------------------------------
// LLM expansion outputs
//
// Two sequential ScryptedAI text-gen calls. NarrativeExpansion is call #1
// (backstory, motivation, beliefs). VoiceExpansion is call #2 (dialogue,
// posts, style) and consumes #1's output.
// ----------------------------------------------------------------------------

/** Output of narrativePromptSet (call #1). */
export interface NarrativeExpansion {
  /**
   * 8–10 lines structured as:
   *   [0–2] who they are now
   *   [3–5] formative backstory (what shaped them)
   *   [6–8] current motivation (what they want, what they avoid)
   *   [9]   contradiction line (visible trait tension)
   */
  bio: string[];
  /** 4–6 worldview statements in their own voice. Folded into bio on merge. */
  beliefs: string[];
  /** 10–14 trait-weighted adjectives. */
  adjectives: string[];
  /** 8–12 things this being would actually orbit conversationally. */
  topics: string[];
}

/** A single conversation turn in a messageExamples exchange. */
export interface MessageTurn {
  name: string;
  content: { text: string };
}

/** Output of voicePromptSet (call #2). */
export interface VoiceExpansion {
  /**
   * 6 user→agent exchanges, each scenario-targeted to test a trait:
   *   [0] help request → Agreeableness
   *   [1] disagreement → Agreeableness + Neuroticism
   *   [2] small talk → Extraversion
   *   [3] bad news → Neuroticism + Agreeableness
   *   [4] weird question → Openness
   *   [5] commitment ask → Conscientiousness
   * Each exchange is exactly two turns: [user, agent].
   */
  messageExamples: MessageTurn[][];
  /** 6–8 standalone monologue posts (the voice without an interlocutor). */
  postExamples: string[];
  /** Style rules derived from the dialogue, not invented separately. */
  style: {
    all: string[];
    chat: string[];
    post: string[];
  };
}

/** Discriminated union: either kind of expansion result. */
export type Expansion =
  | ({ kind: "narrative" } & NarrativeExpansion)
  | ({ kind: "voice" } & VoiceExpansion);

// ----------------------------------------------------------------------------
// Service-internal job tracking
// ----------------------------------------------------------------------------

export type ExpansionKind = "narrative" | "voice";

export interface ExpansionRecord {
  jobId: string;
  kind: ExpansionKind;
  profile: FfmProfile;
  status: "pending" | "completed" | "failed";
  result?: Expansion;
  error?: string;
  /** Raw LLM text (kept for debugging when parse fails). */
  rawText?: string;
  createdAt: number;
  updatedAt: number;
}

export type ExpansionTerminalListener = (record: ExpansionRecord) => void;

// ----------------------------------------------------------------------------
// FfmRuntimeSurface — structural subset of IAgentRuntime that this plugin uses
//
// Mirrors AvbRuntimeSurface: declare exactly what we touch so the cast in
// service.ts is documented and tests can supply a minimal fake.
// ----------------------------------------------------------------------------

export interface FfmRuntimeSurface {
  agentId: string;
  logger: {
    info(msg: string, ...args: unknown[]): void;
    warn(msg: string, ...args: unknown[]): void;
    error(msg: string, ...args: unknown[]): void;
    debug(msg: string, ...args: unknown[]): void;
  };
  /** Resolves once the named service has finished start(). */
  getServiceLoadPromise<T = unknown>(serviceType: string): Promise<T>;
}

// ----------------------------------------------------------------------------
// Surface FfmService needs from ScryptedAIService (subset, for testability)
// ----------------------------------------------------------------------------

export interface ScryptedAILike {
  startTextGeneration(
    inputData: Record<string, unknown>,
    opts?: { pollFallback?: boolean; metadata?: Record<string, unknown> },
  ): Promise<{ jobId: string }>;
  onTerminal(listener: (result: NormalizedJobResult) => void): () => void;
}

// ----------------------------------------------------------------------------
// Character merge target — structural subset of @elizaos/core's Character
// ----------------------------------------------------------------------------

/** The character.json fields that mergePersonality reads/writes. */
export interface MergeableCharacter {
  name: string;
  bio?: string[] | string;
  adjectives?: string[];
  topics?: string[];
  messageExamples?: MessageTurn[][];
  postExamples?: string[];
  style?: {
    all?: string[];
    chat?: string[];
    post?: string[];
  };
  settings?: Record<string, unknown> & {
    ffm?: FfmSettingsBlock;
  };
  [key: string]: unknown;
}

/** What gets written to character.settings.ffm — the deterministic core. */
export interface FfmSettingsBlock {
  seed: string;
  traits: FfmTraits;
  archetype: { code: number; sloan: string; label: string; summary: string };
  generatedAt: string;
  version: number;
}
