/**
 * @elizaos/plugin-ffm
 *
 * Five Factor Model (OCEAN) personality engine for elizaOS.
 *
 * Two layers:
 *   1. Pure derivation — 256-bit seed → 5 trait scores in [0,1] →
 *      one of 32 archetypes. Deterministic, instant, no I/O. Trait
 *      distributions match empirical research (NEO-PI-R / IPIP-NEO):
 *      Neuroticism is right-skewed with mean ≈ 0.42; Agreeableness
 *      is left-skewed with mean ≈ 0.58.
 *
 *   2. LLM expansion — two sequential ScryptedAI text-gen calls
 *      (narrative → voice) that turn the trait profile into a full
 *      character.json: 8–10 bio lines with backstory and motivation,
 *      6 dialogue exchanges scenario-targeted to test each trait,
 *      6–8 standalone posts, and derived style rules.
 *
 * Service surface (via runtime.getService("ffm")):
 *   ffm.deriveProfile(seed?)              → FfmProfile (instant)
 *   ffm.startNarrativeExpansion(p, name)  → {jobId} (returns immediately)
 *   ffm.startVoiceExpansion(p, n, name)   → {jobId} (returns immediately)
 *   ffm.awaitExpansion(jobId, timeoutMs?) → Expansion (optional blocking)
 *   ffm.onExpansionTerminal(listener)     → unsubscribe
 *
 * Dependencies: @elizaos/plugin-scryptedai (for the LLM expansion layer
 * only — pure derivation has no dependencies).
 */

import type { Plugin } from "@elizaos/core";
import { FfmService } from "./service.ts";

// ----------------------------------------------------------------------------
// Plugin definition
// ----------------------------------------------------------------------------

export const ffmPlugin: Plugin = {
  name: "ffm",
  description:
    "Five Factor Model (OCEAN) personality engine — seeded trait derivation " +
    "with empirical distributions, plus LLM-driven character expansion. " +
    "Same seed → same personality, every time.",
  dependencies: ["scryptedai"],
  services: [FfmService],
};

export default ffmPlugin;

// ----------------------------------------------------------------------------
// Public API — flat re-exports
// ----------------------------------------------------------------------------

export {
  archetypeForCode,
  binarize,
  classifyTraits,
  traitsToCode,
  traitsToSloan,
} from "./archetype.ts";
// Constants
export {
  ARCHETYPES,
  FFM_SERVICE_TYPE,
  FFM_SETTINGS_VERSION,
  FFM_TEXT_MODEL,
  MAX_REPLY_TOKEN_OVERLAP,
  MIN_BIO_LINES,
  MIN_MESSAGE_EXAMPLES,
  MIN_POST_EXAMPLES,
  SEED_BYTES,
  SEED_HEX_LEN,
  SEED_LANES,
  SLOAN_LETTERS,
  TRAIT_BIT,
  TRAIT_LANE,
  TRAIT_PARAMS,
  type TraitParams,
} from "./constants.ts";
// Character merge
export { hasFfmSeed, mergePersonality } from "./merge.ts";
// Pure derivation
export {
  applySkew,
  boxMuller,
  deriveTraits,
  generateSeed,
  laneToUniforms,
  normalizeSeed,
  sampleTrait,
  seedToLanes,
} from "./prng.ts";
// Prompt sets + parsing
export {
  extractJsonBlock,
  FfmParseError,
  maxReplyOverlap,
  narrativePromptSet,
  parseNarrative,
  parseVoice,
  renderNarrativePrompt,
  renderTraitDigest,
  renderVoicePrompt,
  voicePromptSet,
} from "./promptset.ts";
// Service
export { FfmService } from "./service.ts";

// Types
export type {
  Expansion,
  ExpansionKind,
  ExpansionRecord,
  ExpansionTerminalListener,
  FfmArchetype,
  FfmProfile,
  FfmRuntimeSurface,
  FfmSettingsBlock,
  FfmTraits,
  MergeableCharacter,
  MessageTurn,
  NarrativeExpansion,
  ScryptedAILike,
  TraitKey,
  VoiceExpansion,
} from "./types.ts";
export { TRAIT_KEYS } from "./types.ts";
