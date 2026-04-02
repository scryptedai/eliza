/**
 * Trait binarization and archetype lookup.
 *
 * Each trait is binarized at its distribution MEDIAN (not 0.5). Because
 * the sinh-arcsinh transform in prng.ts is recentered so median = mean,
 * the threshold is simply TRAIT_PARAMS[k].mean for every trait.
 *
 * Five high/low bits → 32 archetypes. All pure functions.
 */

import {
  ARCHETYPES,
  SLOAN_LETTERS,
  TRAIT_BIT,
  TRAIT_PARAMS,
} from "./constants.ts";
import type { FfmArchetype, FfmTraits, TraitKey } from "./types.ts";
import { TRAIT_KEYS } from "./types.ts";

// ----------------------------------------------------------------------------
// Binarization
// ----------------------------------------------------------------------------

/**
 * High/low classification per trait. `true` means at-or-above the
 * distribution median (high pole); `false` means below (low pole).
 *
 * Threshold is the median, which equals the mean by construction (see
 * prng.ts applySkew). Tie goes to high — at-or-above is high.
 */
export function binarize(traits: FfmTraits): Record<TraitKey, boolean> {
  const out = {} as Record<TraitKey, boolean>;
  for (const k of TRAIT_KEYS) {
    out[k] = traits[k] >= TRAIT_PARAMS[k].mean;
  }
  return out;
}

/**
 * Pack five high/low bits into a 5-bit code (O is MSB, N is LSB).
 * 0b11110 = O+ C+ E+ A+ N− = 30.
 */
export function traitsToCode(traits: FfmTraits): number {
  const hi = binarize(traits);
  let code = 0;
  for (const k of TRAIT_KEYS) {
    if (hi[k]) code |= TRAIT_BIT[k];
  }
  return code;
}

// ----------------------------------------------------------------------------
// SLOAN string
// ----------------------------------------------------------------------------

/**
 * Build the SLOAN-style 5-letter code (e.g. "IOSAC") from trait bits.
 * Used for human inspection — the archetype table already has these baked
 * in, but this lets you compute the string for arbitrary trait sets.
 */
export function traitsToSloan(traits: FfmTraits): string {
  const hi = binarize(traits);
  return TRAIT_KEYS.map((k) =>
    hi[k] ? SLOAN_LETTERS[k].hi : SLOAN_LETTERS[k].lo,
  ).join("");
}

// ----------------------------------------------------------------------------
// Lookup
// ----------------------------------------------------------------------------

/**
 * Look up the archetype record for a 5-bit code. Throws on out-of-range
 * (caller bug — codes from traitsToCode are always 0–31).
 */
export function archetypeForCode(code: number): FfmArchetype {
  if (!Number.isInteger(code) || code < 0 || code > 31) {
    throw new Error(`[ffm] Archetype code out of range: ${code}`);
  }
  return ARCHETYPES[code];
}

/** Convenience: traits → archetype in one call. */
export function classifyTraits(traits: FfmTraits): FfmArchetype {
  return archetypeForCode(traitsToCode(traits));
}
