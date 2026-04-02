/**
 * Archetype binarization & lookup tests.
 *
 * The 32-archetype table is static data; what we test is the *mapping*
 * from continuous trait scores → 5-bit code → table entry. Coverage:
 *   - binarize() thresholds at distribution median (= mean by construction)
 *   - traitsToCode() bit-packing (O is MSB, N is LSB)
 *   - traitsToSloan() agrees with the static table for every code
 *   - all 32 codes are reachable from some trait set
 *   - the static table itself is well-formed (no duplicates, codes 0–31)
 */

import { describe, expect, it } from "vitest";
import {
  archetypeForCode,
  binarize,
  classifyTraits,
  traitsToCode,
  traitsToSloan,
} from "../archetype.ts";
import {
  ARCHETYPES,
  SLOAN_LETTERS,
  TRAIT_BIT,
  TRAIT_PARAMS,
} from "../constants.ts";
import type { FfmTraits, TraitKey } from "../types.ts";
import { TRAIT_KEYS } from "../types.ts";

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/** Build a trait set from a 5-bit code: high traits → mean+0.1, low → mean−0.1. */
function traitsFor(code: number): FfmTraits {
  const out = {} as Record<TraitKey, number>;
  for (const k of TRAIT_KEYS) {
    const hi = (code & TRAIT_BIT[k]) !== 0;
    out[k] = TRAIT_PARAMS[k].mean + (hi ? 0.1 : -0.1);
  }
  return out as FfmTraits;
}

/** Compute the SLOAN string for a code from first principles. */
function sloanFor(code: number): string {
  return TRAIT_KEYS.map((k) =>
    (code & TRAIT_BIT[k]) !== 0 ? SLOAN_LETTERS[k].hi : SLOAN_LETTERS[k].lo,
  ).join("");
}

const EPS = 1e-9;

// ----------------------------------------------------------------------------
// binarize() — threshold semantics
// ----------------------------------------------------------------------------

describe("binarize", () => {
  it("classifies high pole at-or-above mean (tie goes to high)", () => {
    for (const k of TRAIT_KEYS) {
      const m = TRAIT_PARAMS[k].mean;
      const traits = { O: 0, C: 0, E: 0, A: 0, N: 0, [k]: m } as FfmTraits;
      expect(binarize(traits)[k]).toBe(true);
    }
  });

  it("classifies low pole strictly below mean", () => {
    for (const k of TRAIT_KEYS) {
      const m = TRAIT_PARAMS[k].mean;
      const traits = {
        O: 0,
        C: 0,
        E: 0,
        A: 0,
        N: 0,
        [k]: m - EPS,
      } as FfmTraits;
      expect(binarize(traits)[k]).toBe(false);
    }
  });

  it("uses per-trait thresholds (Neuroticism median ≈ 0.42, not 0.5)", () => {
    // 0.45 is below 0.5 but above N's median of 0.42 → HIGH neuroticism.
    const traits: FfmTraits = { O: 0.45, C: 0.45, E: 0.45, A: 0.45, N: 0.45 };
    const hi = binarize(traits);
    expect(hi.N).toBe(true); // 0.45 >= 0.42
    expect(hi.O).toBe(false); // 0.45 <  0.50
    expect(hi.E).toBe(false); // 0.45 <  0.50
    // And 0.55 is above 0.5 but below A's median of 0.58 → LOW agreeableness.
    const traits2: FfmTraits = { O: 0.55, C: 0.55, E: 0.55, A: 0.55, N: 0.55 };
    const hi2 = binarize(traits2);
    expect(hi2.A).toBe(false); // 0.55 < 0.58
    expect(hi2.O).toBe(true); // 0.55 >= 0.50
  });
});

// ----------------------------------------------------------------------------
// traitsToCode() — bit packing
// ----------------------------------------------------------------------------

describe("traitsToCode", () => {
  it("all-low traits → code 0", () => {
    expect(traitsToCode(traitsFor(0))).toBe(0);
  });

  it("all-high traits → code 31", () => {
    expect(traitsToCode(traitsFor(31))).toBe(31);
  });

  it("O is MSB (bit 16): O+ alone → code 16", () => {
    expect(traitsToCode(traitsFor(0b10000))).toBe(16);
  });

  it("N is LSB (bit 1): N+ alone → code 1", () => {
    expect(traitsToCode(traitsFor(0b00001))).toBe(1);
  });

  it("0b11110 → 30 (the canonical IOSAC example)", () => {
    expect(traitsToCode(traitsFor(0b11110))).toBe(30);
  });

  it("round-trips for all 32 codes", () => {
    for (let code = 0; code < 32; code++) {
      expect(traitsToCode(traitsFor(code))).toBe(code);
    }
  });
});

// ----------------------------------------------------------------------------
// traitsToSloan() — letter mapping
// ----------------------------------------------------------------------------

describe("traitsToSloan", () => {
  it("all-low → NUREC", () => {
    expect(traitsToSloan(traitsFor(0))).toBe("NUREC");
  });

  it("all-high → IOSAL", () => {
    expect(traitsToSloan(traitsFor(31))).toBe("IOSAL");
  });

  it("0b11110 → IOSAC (The Diplomat)", () => {
    expect(traitsToSloan(traitsFor(30))).toBe("IOSAC");
  });

  it("computed SLOAN matches static table for every code", () => {
    for (let code = 0; code < 32; code++) {
      const computed = traitsToSloan(traitsFor(code));
      const tabled = ARCHETYPES[code].sloan;
      expect(computed).toBe(tabled);
      expect(computed).toBe(sloanFor(code));
    }
  });
});

// ----------------------------------------------------------------------------
// archetypeForCode() — lookup
// ----------------------------------------------------------------------------

describe("archetypeForCode", () => {
  it("returns the correct entry for each code", () => {
    for (let code = 0; code < 32; code++) {
      const a = archetypeForCode(code);
      expect(a.code).toBe(code);
      expect(a.sloan).toBe(ARCHETYPES[code].sloan);
    }
  });

  it("throws on out-of-range codes", () => {
    expect(() => archetypeForCode(-1)).toThrow(/out of range/);
    expect(() => archetypeForCode(32)).toThrow(/out of range/);
    expect(() => archetypeForCode(0.5)).toThrow(/out of range/);
    expect(() => archetypeForCode(NaN)).toThrow(/out of range/);
  });

  it("returns frozen objects", () => {
    expect(Object.isFrozen(archetypeForCode(0))).toBe(true);
  });
});

// ----------------------------------------------------------------------------
// classifyTraits() — full chain
// ----------------------------------------------------------------------------

describe("classifyTraits", () => {
  it("traits → code → archetype, all 32 reachable", () => {
    const seen = new Set<number>();
    for (let code = 0; code < 32; code++) {
      const a = classifyTraits(traitsFor(code));
      expect(a.code).toBe(code);
      seen.add(a.code);
    }
    expect(seen.size).toBe(32);
  });

  it("known case: high-O low-C high-E high-A low-N → IUSAC The Free Spirit (22)", () => {
    const traits: FfmTraits = { O: 0.71, C: 0.4, E: 0.62, A: 0.65, N: 0.3 };
    const a = classifyTraits(traits);
    expect(a.code).toBe(22);
    expect(a.sloan).toBe("IUSAC");
    expect(a.label).toBe("The Free Spirit");
  });
});

// ----------------------------------------------------------------------------
// ARCHETYPES table — static-data integrity
// ----------------------------------------------------------------------------

describe("ARCHETYPES table", () => {
  it("has exactly 32 entries", () => {
    expect(ARCHETYPES.length).toBe(32);
  });

  it("array index equals code for every entry", () => {
    for (let i = 0; i < 32; i++) {
      expect(ARCHETYPES[i].code).toBe(i);
    }
  });

  it("SLOAN strings are unique and 5 characters", () => {
    const sloans = new Set(ARCHETYPES.map((a) => a.sloan));
    expect(sloans.size).toBe(32);
    for (const a of ARCHETYPES) {
      expect(a.sloan).toMatch(/^[A-Z]{5}$/);
    }
  });

  it("labels are unique and non-empty", () => {
    const labels = new Set(ARCHETYPES.map((a) => a.label));
    expect(labels.size).toBe(32);
    for (const a of ARCHETYPES) {
      expect(a.label.length).toBeGreaterThan(0);
      expect(a.summary.length).toBeGreaterThan(0);
    }
  });

  it("the whole table is frozen", () => {
    expect(Object.isFrozen(ARCHETYPES)).toBe(true);
    for (const a of ARCHETYPES) {
      expect(Object.isFrozen(a)).toBe(true);
    }
  });
});
