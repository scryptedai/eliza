/**
 * Character merge: fold an FFM profile + expansions into a Character object.
 *
 * Policy:
 *   - Only fill EMPTY/MISSING fields. Hand-authored content wins.
 *   - settings.ffm is ALWAYS written (it's the seed of record).
 *   - bio: array form preferred. Beliefs are appended after bio lines.
 *   - Pure function: returns a new object, never mutates the input.
 *
 * FFM is a provider — it doesn't decide WHEN to merge or WHEN to persist.
 * AVB (the orchestrator) calls this, then calls saveCharacter() from core.
 */

import { FFM_SETTINGS_VERSION, SEED_HEX_LEN } from "./constants.ts";
import type {
  FfmProfile,
  FfmSettingsBlock,
  MergeableCharacter,
  NarrativeExpansion,
  VoiceExpansion,
} from "./types.ts";

// ----------------------------------------------------------------------------
// Sufficiency check
// ----------------------------------------------------------------------------

const HEX64 = /^[0-9a-f]{64}$/;

/**
 * A character is "FFM-sufficient" iff it has a valid 64-hex seed in
 * settings.ffm.seed. If the seed is present, the entire deterministic
 * profile is re-derivable, so we trust the rest of the sheet.
 */
export function hasFfmSeed(character: MergeableCharacter): boolean {
  const seed = character.settings?.ffm?.seed;
  return (
    typeof seed === "string" && seed.length === SEED_HEX_LEN && HEX64.test(seed)
  );
}

// ----------------------------------------------------------------------------
// Merge helpers
// ----------------------------------------------------------------------------

/** True if a string-array field is missing or empty after trimming. */
function isEmptyStrArr(v: unknown): boolean {
  if (!Array.isArray(v)) return true;
  return v.filter((x) => typeof x === "string" && x.trim()).length === 0;
}

/**
 * True if an array-of-anything field is missing or has zero length.
 * Use this for messageExamples (MessageTurn[][]) — its elements aren't
 * strings, so isEmptyStrArr would treat any value as empty.
 */
function isEmptyArr(v: unknown): boolean {
  return !Array.isArray(v) || v.length === 0;
}

/** True if a bio (string | string[]) is missing or effectively empty. */
function isEmptyBio(v: unknown): boolean {
  if (typeof v === "string") return v.trim().length === 0;
  return isEmptyStrArr(v);
}

// ----------------------------------------------------------------------------
// Settings block builder
// ----------------------------------------------------------------------------

function buildFfmSettings(profile: FfmProfile): FfmSettingsBlock {
  return {
    seed: profile.seed,
    traits: { ...profile.traits },
    archetype: {
      code: profile.archetype.code,
      sloan: profile.archetype.sloan,
      label: profile.archetype.label,
      summary: profile.archetype.summary,
    },
    generatedAt: new Date().toISOString(),
    version: FFM_SETTINGS_VERSION,
  };
}

// ----------------------------------------------------------------------------
// Main merge
// ----------------------------------------------------------------------------

/**
 * Merge an FFM profile + optional expansions into a character.
 *
 * Returns a NEW object (shallow-cloned at top level + settings + style).
 * The deterministic settings.ffm block is always written. Generated content
 * (bio, dialogue, etc.) only fills empty slots.
 *
 * Both expansions are optional so callers can merge incrementally
 * (e.g. write the seed immediately, then merge narrative later, then voice).
 */
export function mergePersonality(
  character: MergeableCharacter,
  profile: FfmProfile,
  narrative?: NarrativeExpansion,
  voice?: VoiceExpansion,
): MergeableCharacter {
  const out: MergeableCharacter = { ...character };

  // ----- settings.ffm — ALWAYS written ------------------------------------
  out.settings = {
    ...(character.settings ?? {}),
    ffm: buildFfmSettings(profile),
  };

  // ----- narrative expansion → bio, adjectives, topics --------------------
  if (narrative) {
    if (isEmptyBio(out.bio)) {
      // Fold beliefs into bio (Character has no separate beliefs field).
      // Bio first, then a blank-ish separator isn't representable in
      // string[], so we just append beliefs as additional bio lines.
      out.bio = [...narrative.bio, ...narrative.beliefs];
    }
    if (isEmptyStrArr(out.adjectives)) {
      out.adjectives = [...narrative.adjectives];
    }
    if (isEmptyStrArr(out.topics)) {
      out.topics = [...narrative.topics];
    }
  }

  // ----- voice expansion → messageExamples, postExamples, style ----------
  if (voice) {
    if (isEmptyArr(out.messageExamples)) {
      out.messageExamples = voice.messageExamples.map((ex) =>
        ex.map((turn) => ({
          name: turn.name,
          content: { text: turn.content.text },
        })),
      );
    }
    if (isEmptyStrArr(out.postExamples)) {
      out.postExamples = [...voice.postExamples];
    }

    // Style is a sub-object — merge per-key, not all-or-nothing.
    const existingStyle = out.style ?? {};
    out.style = {
      all: isEmptyStrArr(existingStyle.all)
        ? [...voice.style.all]
        : existingStyle.all,
      chat: isEmptyStrArr(existingStyle.chat)
        ? [...voice.style.chat]
        : existingStyle.chat,
      post: isEmptyStrArr(existingStyle.post)
        ? [...voice.style.post]
        : existingStyle.post,
    };
  }

  return out;
}
