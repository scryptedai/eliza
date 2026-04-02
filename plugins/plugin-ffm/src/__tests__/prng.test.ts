/**
 * Tests for the seeded PRNG and skew-normal sampler.
 *
 * Three concerns:
 *   1. DETERMINISM — same seed → identical output, every call, every machine
 *   2. DISTRIBUTION SHAPE — empirical mean/SD/skew over 10k rolls match the
 *      target parameters in TRAIT_PARAMS within tolerance
 *   3. EDGE CASES — seed validation, lane extraction, clamping
 */

import { describe, expect, it } from "vitest";
import { TRAIT_PARAMS } from "../constants.ts";
import {
  applySkew,
  boxMuller,
  deriveTraits,
  generateSeed,
  laneToUniforms,
  normalizeSeed,
  sampleTrait,
  seedToLanes,
} from "../prng.ts";

// ----------------------------------------------------------------------------
// Seed validation
// ----------------------------------------------------------------------------

describe("normalizeSeed", () => {
  it("accepts a valid 64-char lowercase hex seed", () => {
    const s = "a".repeat(64);
    expect(normalizeSeed(s)).toBe(s);
  });

  it("lowercases and trims", () => {
    const s = "  " + "ABCDEF0123456789".repeat(4) + "  ";
    expect(normalizeSeed(s)).toBe("abcdef0123456789".repeat(4));
  });

  it("rejects wrong length", () => {
    expect(() => normalizeSeed("abc")).toThrow(/64 hex chars/);
    expect(() => normalizeSeed("a".repeat(63))).toThrow();
    expect(() => normalizeSeed("a".repeat(65))).toThrow();
  });

  it("rejects non-hex characters", () => {
    expect(() => normalizeSeed("g".repeat(64))).toThrow();
    expect(() => normalizeSeed("z" + "a".repeat(63))).toThrow();
  });
});

describe("generateSeed", () => {
  it("produces 64 lowercase hex chars", () => {
    const s = generateSeed();
    expect(s).toMatch(/^[0-9a-f]{64}$/);
  });

  it("produces different seeds on successive calls", () => {
    // Not a guarantee in theory, but 256 bits of entropy → P(collision) ≈ 0
    const seen = new Set<string>();
    for (let i = 0; i < 50; i++) seen.add(generateSeed());
    expect(seen.size).toBe(50);
  });
});

// ----------------------------------------------------------------------------
// Lane decomposition
// ----------------------------------------------------------------------------

describe("seedToLanes", () => {
  it("splits a known seed into the expected big-endian uint32 lanes", () => {
    // 8 lanes of 0x01020304 = bytes 01 02 03 04 repeated 8x
    const seed = "01020304".repeat(8);
    const lanes = seedToLanes(seed);
    expect(lanes.length).toBe(8);
    for (let i = 0; i < 8; i++) {
      expect(lanes[i]).toBe(0x01020304);
    }
  });

  it("is deterministic", () => {
    const seed = "deadbeef".repeat(8);
    const a = seedToLanes(seed);
    const b = seedToLanes(seed);
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it("first 4 bytes become lane 0 (big-endian)", () => {
    const seed = "ff000000" + "00".repeat(28);
    const lanes = seedToLanes(seed);
    expect(lanes[0]).toBe(0xff000000);
    expect(lanes[1]).toBe(0);
  });
});

describe("laneToUniforms", () => {
  it("maps to the open interval (0,1) — never exactly 0 or 1", () => {
    const [u1a, u2a] = laneToUniforms(0x00000000); // both halves = 0
    expect(u1a).toBeGreaterThan(0);
    expect(u2a).toBeGreaterThan(0);

    const [u1b, u2b] = laneToUniforms(0xffffffff); // both halves = 65535
    expect(u1b).toBeLessThan(1);
    expect(u2b).toBeLessThan(1);
  });

  it("high 16 bits → u1, low 16 bits → u2", () => {
    const [u1, u2] = laneToUniforms(0xffff0000);
    expect(u1).toBeCloseTo(65535.5 / 65536, 6);
    expect(u2).toBeCloseTo(0.5 / 65536, 6);
  });
});

// ----------------------------------------------------------------------------
// Box-Muller
// ----------------------------------------------------------------------------

describe("boxMuller", () => {
  it("is deterministic for fixed inputs", () => {
    expect(boxMuller(0.5, 0.5)).toBe(boxMuller(0.5, 0.5));
  });

  it("returns finite values for valid open-interval inputs", () => {
    // Smallest possible u1 from a 16-bit lane: (0+0.5)/65536
    const z = boxMuller(0.5 / 65536, 0.5);
    expect(Number.isFinite(z)).toBe(true);
    // sqrt(-2*log(7.6e-6)) ≈ 4.8 — extreme but finite
    expect(Math.abs(z)).toBeLessThan(10);
  });

  it("u2=0.25 → cos(π/2)=0 → z=0", () => {
    expect(boxMuller(0.5, 0.25)).toBeCloseTo(0, 10);
  });
});

// ----------------------------------------------------------------------------
// Sinh-arcsinh skew transform
// ----------------------------------------------------------------------------

describe("applySkew", () => {
  it("is the identity when skew=0", () => {
    for (const z of [-3, -1, -0.5, 0, 0.5, 1, 3]) {
      expect(applySkew(z, 0)).toBe(z);
    }
  });

  it("preserves z=0 (median stays at 0 after recentering)", () => {
    // applySkew(0, ε) = sinh(asinh(0) + ε) - sinh(ε) = sinh(ε) - sinh(ε) = 0
    expect(applySkew(0, 0.4)).toBeCloseTo(0, 12);
    expect(applySkew(0, -0.4)).toBeCloseTo(0, 12);
  });

  it("positive skew stretches the right tail and compresses the left", () => {
    const skew = 0.4;
    // Right tail: |output| > |input|
    expect(applySkew(2, skew)).toBeGreaterThan(2);
    // Left tail: |output| < |input|
    expect(Math.abs(applySkew(-2, skew))).toBeLessThan(2);
  });

  it("negative skew stretches the left tail and compresses the right", () => {
    const skew = -0.4;
    expect(Math.abs(applySkew(-2, skew))).toBeGreaterThan(2);
    expect(applySkew(2, skew)).toBeLessThan(2);
  });
});

// ----------------------------------------------------------------------------
// sampleTrait — single draw, deterministic, clamped
// ----------------------------------------------------------------------------

describe("sampleTrait", () => {
  it("is deterministic for fixed (lane, params)", () => {
    const a = sampleTrait(0x12345678, 0.5, 0.15, 0);
    const b = sampleTrait(0x12345678, 0.5, 0.15, 0);
    expect(a).toBe(b);
  });

  it("output is always in [0,1]", () => {
    // Try a bunch of lane values including extremes
    for (const lane of [0, 1, 0xffff, 0xffff0000, 0xffffffff, 0x80008000]) {
      const v = sampleTrait(lane, 0.5, 0.15, 0);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("clamps extreme draws — large SD doesn't escape [0,1]", () => {
    // sd=2 with mean=0.5 would give values way outside [0,1] without clamping
    for (let lane = 0; lane < 1000; lane++) {
      const v = sampleTrait(lane * 0x10001, 0.5, 2.0, 0);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

// ----------------------------------------------------------------------------
// deriveTraits — top-level chain, full determinism
// ----------------------------------------------------------------------------

describe("deriveTraits — determinism", () => {
  const SEED = "deadbeef".repeat(8);

  it("same seed → identical output across 100 calls", () => {
    const ref = deriveTraits(SEED);
    for (let i = 0; i < 100; i++) {
      const t = deriveTraits(SEED);
      expect(t.O).toBe(ref.O);
      expect(t.C).toBe(ref.C);
      expect(t.E).toBe(ref.E);
      expect(t.A).toBe(ref.A);
      expect(t.N).toBe(ref.N);
    }
  });

  it("different seeds → different outputs", () => {
    const a = deriveTraits("a".repeat(64));
    const b = deriveTraits("b".repeat(64));
    // At least one trait must differ (P(all same) ≈ 0)
    const same =
      a.O === b.O && a.C === b.C && a.E === b.E && a.A === b.A && a.N === b.N;
    expect(same).toBe(false);
  });

  it("returns frozen output", () => {
    const t = deriveTraits(SEED);
    expect(Object.isFrozen(t)).toBe(true);
  });

  it("each trait is independently determined by its lane", () => {
    // Two seeds that share lane 0 (Openness) but differ elsewhere should
    // produce identical O scores. lane 0 = first 8 hex chars.
    const seedA = "12345678" + "a".repeat(56);
    const seedB = "12345678" + "b".repeat(56);
    expect(deriveTraits(seedA).O).toBe(deriveTraits(seedB).O);
    expect(deriveTraits(seedA).C).not.toBe(deriveTraits(seedB).C);
  });
});

// ----------------------------------------------------------------------------
// Distribution shape over 10k rolls
//
// This is the empirical-research validation: assert that a population of
// rolled traits matches the target distribution parameters. Tolerances are
// set to ~3σ of the sampling distribution at N=10k so the test is stable
// across CI runs but still catches real regressions.
// ----------------------------------------------------------------------------

describe("deriveTraits — distribution shape over 10k rolls", () => {
  const N = 10_000;

  // Roll N seeds. Using deterministic seeds (counter-derived) so the test
  // itself is reproducible — but the seeds cover the full lane space.
  const samples: { O: number; C: number; E: number; A: number; N: number }[] =
    [];
  for (let i = 0; i < N; i++) {
    // Spread the counter across the seed so all lanes get good coverage.
    // Each lane gets a different transform of i.
    const hex = (n: number) => (n >>> 0).toString(16).padStart(8, "0");
    const seed =
      hex(i * 2654435761) + // golden ratio hash, lane 0
      hex(i * 2246822519) + // lane 1
      hex(i * 3266489917) + // lane 2
      hex(i * 668265263) + // lane 3
      hex(i * 374761393) + // lane 4
      hex(i) +
      hex(i ^ 0xaaaaaaaa) +
      hex(i ^ 0x55555555);
    samples.push(deriveTraits(seed));
  }

  function stats(arr: number[]) {
    const n = arr.length;
    const mean = arr.reduce((s, x) => s + x, 0) / n;
    const variance = arr.reduce((s, x) => s + (x - mean) ** 2, 0) / n;
    const sd = Math.sqrt(variance);
    // Pearson moment skewness
    const m3 = arr.reduce((s, x) => s + (x - mean) ** 3, 0) / n;
    const skew = m3 / sd ** 3;
    // Sample median
    const sorted = [...arr].sort((a, b) => a - b);
    const median = sorted[Math.floor(n / 2)];
    return { mean, sd, skew, median };
  }

  for (const key of ["O", "C", "E", "A", "N"] as const) {
    const target = TRAIT_PARAMS[key];
    const observed = stats(samples.map((s) => s[key]));

    // The sinh-arcsinh transform recenters so MEDIAN(z') = 0, which means
    // MEDIAN(trait) = target.mean exactly. For symmetric distributions
    // (skew=0) the sample mean ALSO lands at target.mean. For skewed
    // distributions, the long tail pulls the mean away from the median
    // (positive skew → mean > median; negative → mean < median). So:
    //
    //   - the MEDIAN is the construction guarantee → tight assertion
    //   - the MEAN drifts predictably → just check the sign of the drift
    //
    // This is the correct read of the math: TRAIT_PARAMS[k].mean is the
    // 50th percentile, the threshold for high/low binarization.

    it(`${key}: empirical median ≈ ${target.mean} (the construction guarantee)`, () => {
      // Median has SE ≈ 1.25·σ/√N ≈ 0.0019 at N=10k. ±0.02 is huge headroom.
      expect(Math.abs(observed.median - target.mean)).toBeLessThan(0.02);
    });

    if (target.skew === 0) {
      it(`${key}: empirical mean ≈ ${target.mean} (symmetric, mean = median)`, () => {
        expect(Math.abs(observed.mean - target.mean)).toBeLessThan(0.02);
      });
    } else if (target.skew > 0) {
      it(`${key}: empirical mean > median (right tail pulls mean up)`, () => {
        expect(observed.mean).toBeGreaterThan(observed.median);
      });
    } else {
      it(`${key}: empirical mean < median (left tail pulls mean down)`, () => {
        expect(observed.mean).toBeLessThan(observed.median);
      });
    }

    it(`${key}: empirical SD is in a sane range around ${target.sd}`, () => {
      // The skew transform inflates variance by ~cosh(ε)² ≈ 1.16 for ε=0.4,
      // and clamping pulls it back slightly. Loose bounds — we care about
      // shape, not exact match.
      expect(observed.sd).toBeGreaterThan(target.sd * 0.7);
      expect(observed.sd).toBeLessThan(target.sd * 1.5);
    });

    if (target.skew !== 0) {
      const direction = target.skew > 0 ? "positive" : "negative";
      it(`${key}: empirical skew is ${direction} (target ${target.skew})`, () => {
        // Sign must match. Magnitude is approximate (the ε parameter
        // doesn't equal Pearson skewness exactly).
        if (target.skew > 0) {
          expect(observed.skew).toBeGreaterThan(0.05);
        } else {
          expect(observed.skew).toBeLessThan(-0.05);
        }
      });
    } else {
      it(`${key}: empirical skew ≈ 0 (symmetric)`, () => {
        expect(Math.abs(observed.skew)).toBeLessThan(0.15);
      });
    }
  }

  // The headline requirement: Neuroticism is right-skewed, mean ≈ 0.42.
  // "You're not as likely to be neurotic as non-neurotic."
  it("Neuroticism: more samples below 0.5 than above (right-skewed, low mean)", () => {
    const nVals = samples.map((s) => s.N);
    const below = nVals.filter((v) => v < 0.5).length;
    const above = nVals.filter((v) => v >= 0.5).length;
    expect(below).toBeGreaterThan(above);
    // With mean=0.42 and right-skew, expect ~60–65% below 0.5
    expect(below / N).toBeGreaterThan(0.55);
  });

  // Agreeableness: opposite — most people self-report as agreeable.
  it("Agreeableness: more samples above 0.5 than below (left-skewed, high mean)", () => {
    const aVals = samples.map((s) => s.A);
    const above = aVals.filter((v) => v >= 0.5).length;
    expect(above / N).toBeGreaterThan(0.55);
  });
});
