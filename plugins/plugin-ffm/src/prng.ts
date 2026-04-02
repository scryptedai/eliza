/**
 * Seeded deterministic RNG and skew-normal sampling.
 *
 * All functions are PURE: same input → same output, no I/O, no global state.
 * The whole derivation chain (seed → traits) runs in one synchronous tick.
 *
 * Pipeline:
 *   hex seed (64 chars) → 32 bytes → 8 × uint32 lanes
 *   lane[k] → split into two 16-bit halves → two uniforms in (0,1)
 *   two uniforms → Box-Muller → one standard normal z
 *   z → sinh-arcsinh skew → recenter → scale by σ, shift by μ → clamp [0,1]
 */

import { randomBytes } from "node:crypto";
import {
  SEED_BYTES,
  SEED_HEX_LEN,
  SEED_LANES,
  TRAIT_LANE,
  TRAIT_PARAMS,
} from "./constants.ts";
import type { FfmTraits, TraitKey } from "./types.ts";

// ----------------------------------------------------------------------------
// Seed generation & validation
// ----------------------------------------------------------------------------

/** Generate a fresh 256-bit random seed as a 64-char lowercase hex string. */
export function generateSeed(): string {
  return randomBytes(SEED_BYTES).toString("hex");
}

const HEX_RE = /^[0-9a-f]{64}$/;

/**
 * Normalize a seed string: trim, lowercase, validate. Throws on bad input
 * (wrong length or non-hex). Returns the canonical 64-char lowercase form.
 */
export function normalizeSeed(seed: string): string {
  const s = seed.trim().toLowerCase();
  if (s.length !== SEED_HEX_LEN || !HEX_RE.test(s)) {
    throw new Error(
      `[ffm] Invalid seed: expected ${SEED_HEX_LEN} hex chars, got ${seed.length} chars`,
    );
  }
  return s;
}

// ----------------------------------------------------------------------------
// Lane decomposition
// ----------------------------------------------------------------------------

/**
 * Split a 256-bit seed into 8 × 32-bit lanes (big-endian).
 * Lane 0 is the first 4 bytes; lane 7 is the last 4.
 */
export function seedToLanes(seedHex: string): Uint32Array {
  const canonical = normalizeSeed(seedHex);
  const bytes = Buffer.from(canonical, "hex");
  const lanes = new Uint32Array(SEED_LANES);
  for (let i = 0; i < SEED_LANES; i++) {
    lanes[i] = bytes.readUInt32BE(i * 4);
  }
  return lanes;
}

// ----------------------------------------------------------------------------
// Box-Muller: two uniforms → one standard normal
// ----------------------------------------------------------------------------

const TWO_PI = 2 * Math.PI;

/**
 * Box-Muller transform. Takes two uniforms in (0,1), returns one standard
 * normal deviate. (The transform produces two; we only need one per trait
 * and have plenty of seed entropy, so the second is discarded.)
 *
 * u1 must be strictly positive (log(0) = -∞).
 */
export function boxMuller(u1: number, u2: number): number {
  const r = Math.sqrt(-2 * Math.log(u1));
  return r * Math.cos(TWO_PI * u2);
}

/**
 * Convert a 32-bit lane into two open-interval uniforms.
 * High 16 bits → u1, low 16 bits → u2.
 * Mapping (k + 0.5)/65536 puts both in the open interval (0, 1) — never
 * exactly 0 or 1, so log(u1) is always finite.
 */
export function laneToUniforms(lane: number): [number, number] {
  const hi = (lane >>> 16) & 0xffff;
  const lo = lane & 0xffff;
  return [(hi + 0.5) / 65536, (lo + 0.5) / 65536];
}

// ----------------------------------------------------------------------------
// Sinh-arcsinh skew transform (Jones & Pewsey 2009, simplified δ=1)
// ----------------------------------------------------------------------------

/**
 * Apply skew to a standard normal deviate.
 *
 *   skew > 0  → positive (right) skew: long right tail, mass concentrated left
 *   skew < 0  → negative (left) skew: long left tail, mass concentrated right
 *   skew = 0  → identity (sinh∘asinh = id)
 *
 * The raw transform sinh(asinh(z) + ε) shifts the median to sinh(ε); we
 * subtract that so the output median is exactly 0. This means that after
 * scaling by σ and shifting by μ, the distribution median equals μ —
 * which makes μ the correct binarization threshold for archetype assignment.
 *
 * Variance increases by ~cosh(ε)² for small ε; we accept this as the SD
 * parameters in TRAIT_PARAMS are population approximations, not exact.
 */
export function applySkew(z: number, skew: number): number {
  if (skew === 0) return z;
  return Math.sinh(Math.asinh(z) + skew) - Math.sinh(skew);
}

// ----------------------------------------------------------------------------
// Full sample: lane → trait value in [0,1]
// ----------------------------------------------------------------------------

function clamp01(x: number): number {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

/**
 * Draw one trait value from a 32-bit lane.
 * Pure: same (lane, mean, sd, skew) → same value, always.
 */
export function sampleTrait(
  lane: number,
  mean: number,
  sd: number,
  skew: number,
): number {
  const [u1, u2] = laneToUniforms(lane);
  const z = boxMuller(u1, u2);
  const zSkewed = applySkew(z, skew);
  return clamp01(mean + sd * zSkewed);
}

// ----------------------------------------------------------------------------
// Top-level: seed → five trait scores
// ----------------------------------------------------------------------------

/**
 * Derive all five OCEAN trait scores from a seed.
 * Deterministic: same seed → same traits, every call, every machine.
 */
export function deriveTraits(seedHex: string): FfmTraits {
  const lanes = seedToLanes(seedHex);
  const out = {} as { -readonly [K in TraitKey]: number };
  for (const key of ["O", "C", "E", "A", "N"] as const) {
    const p = TRAIT_PARAMS[key];
    out[key] = sampleTrait(lanes[TRAIT_LANE[key]], p.mean, p.sd, p.skew);
  }
  return Object.freeze(out);
}
