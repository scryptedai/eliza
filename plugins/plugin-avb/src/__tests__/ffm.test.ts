import { describe, expect, it } from "vitest";
import {
  buildFfmCharacter,
  classifyArchetype,
  createFfmSeed,
  deriveFfmProfile,
  FFM_ARCHETYPE_THRESHOLD,
  FFM_ARCHETYPES,
  FFM_DISTRIBUTIONS,
  FFM_SEED_HEX_LEN,
  FFM_SEED_SETTING,
  FFM_TRAITS,
  ffmPromptSet,
  type FfmScores,
  formatFfmScores,
  normalizeFfmSeed,
  parseFfmCharacterFields,
  renderFfmPrompt,
  rngFromSeed,
  rollFfmTraits,
  stripCharacterSecrets,
} from "../ffm.ts";

// ----------------------------------------------------------------------------
// Seed
// ----------------------------------------------------------------------------

describe("ffm: seed", () => {
  it("createFfmSeed() returns 64 lowercase hex chars", () => {
    const s = createFfmSeed();
    expect(s).toMatch(/^[0-9a-f]{64}$/);
    expect(s).toHaveLength(FFM_SEED_HEX_LEN);
  });

  it("createFfmSeed() is non-deterministic", () => {
    expect(createFfmSeed()).not.toBe(createFfmSeed());
  });

  it("normalizeFfmSeed() trims and lowercases", () => {
    const s = "A".repeat(64);
    expect(normalizeFfmSeed(`  ${s}  `)).toBe("a".repeat(64));
  });

  it("normalizeFfmSeed() rejects wrong length / non-hex", () => {
    expect(() => normalizeFfmSeed("deadbeef")).toThrow();
    expect(() => normalizeFfmSeed("g".repeat(64))).toThrow();
  });
});

// ----------------------------------------------------------------------------
// PRNG
// ----------------------------------------------------------------------------

describe("ffm: rngFromSeed (sfc32)", () => {
  const SEED_A = "00".repeat(32);
  const SEED_B = "ff".repeat(32);

  it("is deterministic for the same seed", () => {
    const a = rngFromSeed(SEED_A);
    const b = rngFromSeed(SEED_A);
    for (let i = 0; i < 16; i++) expect(a()).toBe(b());
  });

  it("emits values in [0,1)", () => {
    const r = rngFromSeed(SEED_B);
    for (let i = 0; i < 256; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("different seeds diverge", () => {
    const a = rngFromSeed(SEED_A);
    const b = rngFromSeed(SEED_B);
    let same = 0;
    for (let i = 0; i < 16; i++) if (a() === b()) same++;
    expect(same).toBeLessThan(16);
  });
});

// ----------------------------------------------------------------------------
// Trait roll
// ----------------------------------------------------------------------------

describe("ffm: rollFfmTraits", () => {
  const SEED = "1234567890abcdef".repeat(4);

  it("is a pure function of the seed", () => {
    expect(rollFfmTraits(SEED)).toEqual(rollFfmTraits(SEED));
  });

  it("returns all five OCEAN keys with scores in [0,1]", () => {
    const s = rollFfmTraits(SEED);
    for (const t of FFM_TRAITS) {
      expect(s[t]).toBeGreaterThanOrEqual(0);
      expect(s[t]).toBeLessThanOrEqual(1);
    }
  });

  it("returns a frozen object", () => {
    expect(Object.isFrozen(rollFfmTraits(SEED))).toBe(true);
  });
});

// ----------------------------------------------------------------------------
// Beta(α,β) distribution sanity (Monte Carlo over fresh seeds)
// ----------------------------------------------------------------------------

describe("ffm: Beta distributions", () => {
  it("derived (α,β) reproduce target mean within tolerance", () => {
    for (const t of FFM_TRAITS) {
      const d = FFM_DISTRIBUTIONS[t];
      // Method-of-moments invariant: α/(α+β) = μ
      expect(d.alpha / (d.alpha + d.beta)).toBeCloseTo(d.mean, 6);
      // Both shape params > 1 → Cheng BB precondition holds.
      expect(d.alpha).toBeGreaterThan(1);
      expect(d.beta).toBeGreaterThan(1);
    }
  });

  // 4 000 fresh seeds is enough for stable means at ±0.03 tolerance
  // while keeping the test under ~100 ms.
  const N = 4_000;
  const sums: Record<string, number> = Object.fromEntries(
    FFM_TRAITS.map((t) => [t, 0]),
  );
  let nHigh = 0;
  for (let i = 0; i < N; i++) {
    const s = rollFfmTraits(createFfmSeed());
    for (const t of FFM_TRAITS) sums[t] += s[t];
    if (s.neuroticism >= FFM_ARCHETYPE_THRESHOLD) nHigh++;
  }

  it("sample means track configured Beta means (±0.03)", () => {
    for (const t of FFM_TRAITS) {
      expect(sums[t] / N).toBeCloseTo(FFM_DISTRIBUTIONS[t].mean, 1);
      expect(Math.abs(sums[t] / N - FFM_DISTRIBUTIONS[t].mean)).toBeLessThan(
        0.03,
      );
    }
  });

  it("high Neuroticism is the minority outcome", () => {
    // Beta(1.58, 2.38) ⇒ P(N ≥ 0.5) ≈ 0.30. Allow generous CI band.
    expect(nHigh / N).toBeLessThan(0.4);
    expect(nHigh / N).toBeGreaterThan(0.2);
  });

  it("Agreeableness mean exceeds Neuroticism mean", () => {
    expect(sums.agreeableness / N).toBeGreaterThan(sums.neuroticism / N);
  });
});

// ----------------------------------------------------------------------------
// Archetypes
// ----------------------------------------------------------------------------

describe("ffm: archetypes", () => {
  it("has exactly 32 archetypes with distinct ids and names", () => {
    expect(FFM_ARCHETYPES).toHaveLength(32);
    expect(new Set(FFM_ARCHETYPES.map((a) => a.id)).size).toBe(32);
    expect(new Set(FFM_ARCHETYPES.map((a) => a.name)).size).toBe(32);
  });

  it("code encodes high/low per trait in OCEAN order", () => {
    expect(FFM_ARCHETYPES[0].code).toBe("O-C-E-A-N-");
    expect(FFM_ARCHETYPES[31].code).toBe("O+C+E+A+N+");
    // id 1 = bit0 (O) high only
    expect(FFM_ARCHETYPES[1].code).toBe("O+C-E-A-N-");
    // id 16 = bit4 (N) high only
    expect(FFM_ARCHETYPES[16].code).toBe("O-C-E-A-N+");
  });

  const mk = (o: number, c: number, e: number, a: number, n: number) =>
    Object.freeze({
      openness: o,
      conscientiousness: c,
      extraversion: e,
      agreeableness: a,
      neuroticism: n,
    }) as FfmScores;

  it("classifyArchetype maps all-low → id 0, all-high → id 31", () => {
    expect(classifyArchetype(mk(0, 0, 0, 0, 0)).id).toBe(0);
    expect(classifyArchetype(mk(1, 1, 1, 1, 1)).id).toBe(31);
  });

  it("classifyArchetype handles mixed and threshold edge", () => {
    // O high, rest low → id 1
    expect(classifyArchetype(mk(0.9, 0.1, 0.1, 0.1, 0.1)).id).toBe(1);
    // exactly 0.5 counts as high
    expect(classifyArchetype(mk(0.5, 0.5, 0.5, 0.5, 0.5)).id).toBe(31);
  });

  it("every classified archetype's code matches the input poles", () => {
    for (let id = 0; id < 32; id++) {
      const scores = mk(
        id & 1 ? 0.9 : 0.1,
        id & 2 ? 0.9 : 0.1,
        id & 4 ? 0.9 : 0.1,
        id & 8 ? 0.9 : 0.1,
        id & 16 ? 0.9 : 0.1,
      );
      expect(classifyArchetype(scores).id).toBe(id);
    }
  });
});

// ----------------------------------------------------------------------------
// Profile + formatting
// ----------------------------------------------------------------------------

describe("ffm: deriveFfmProfile / formatFfmScores", () => {
  const SEED = "ab".repeat(32);

  it("profile is internally consistent and deterministic", () => {
    const p = deriveFfmProfile(SEED);
    expect(p.seed).toBe(SEED);
    expect(p.scores).toEqual(rollFfmTraits(SEED));
    expect(p.archetype).toBe(classifyArchetype(p.scores));
    expect(deriveFfmProfile(SEED)).toEqual(p);
  });

  it("formatFfmScores emits O:.. C:.. E:.. A:.. N:..", () => {
    const out = formatFfmScores(deriveFfmProfile(SEED).scores);
    expect(out).toMatch(
      /^O:\d\.\d{2} C:\d\.\d{2} E:\d\.\d{2} A:\d\.\d{2} N:\d\.\d{2}$/,
    );
  });
});

// ----------------------------------------------------------------------------
// PromptSet
// ----------------------------------------------------------------------------

describe("ffm: ffmPromptSet / renderFfmPrompt", () => {
  it("declares the expected replacement tags", () => {
    for (const tag of [
      "NAME",
      "ARCHETYPE_NAME",
      "ARCHETYPE_CODE",
      "ARCHETYPE_DESC",
      "O",
      "C",
      "E",
      "A",
      "N",
    ]) {
      expect(ffmPromptSet.userTemplate).toContain(`{{${tag}}}`);
    }
    expect(ffmPromptSet.systemTemplate).toContain("JSON");
  });

  it("renderFfmPrompt fills every tag (no {{...}} left)", () => {
    const p = deriveFfmProfile("cd".repeat(32));
    const r = renderFfmPrompt(p, "TestAgent");
    expect(r.user).toContain("TestAgent");
    expect(r.user).toContain(p.archetype.name);
    expect(r.user).toContain(p.archetype.code);
    expect(r.user).toContain(p.archetype.description);
    for (const t of FFM_TRAITS) {
      expect(r.user).toContain(p.scores[t].toFixed(2));
    }
    expect(r.user).not.toMatch(/\{\{[A-Z_]+\}\}/);
    expect(r.system).not.toMatch(/\{\{[A-Z_]+\}\}/);
  });
});

// ----------------------------------------------------------------------------
// LLM output parsing
// ----------------------------------------------------------------------------

describe("ffm: parseFfmCharacterFields", () => {
  const sample = {
    bio: ["I think a lot.", "I keep to myself."],
    adjectives: ["quiet", "curious", "wry"],
    topics: ["math", "music"],
    style: { all: ["be terse"], chat: ["ask back"], post: ["one-liners"] },
    postExamples: ["hm."],
    beliefs: ["less is more"],
  };

  it("parses bare JSON", () => {
    const out = parseFfmCharacterFields(JSON.stringify(sample));
    expect(out.bio).toEqual(sample.bio);
    expect(out.style?.chat).toEqual(["ask back"]);
    expect(out.beliefs).toEqual(["less is more"]);
  });

  it("strips ```json fences and surrounding prose", () => {
    const raw =
      "Here you go:\n```json\n" + JSON.stringify(sample) + "\n```\nEnjoy.";
    const out = parseFfmCharacterFields(raw);
    expect(out.adjectives).toEqual(sample.adjectives);
  });

  it("filters non-string array entries", () => {
    const out = parseFfmCharacterFields(
      JSON.stringify({ bio: ["ok", 42, "", "  ", "fine"] }),
    );
    expect(out.bio).toEqual(["ok", "fine"]);
  });

  it("throws when no JSON object is present", () => {
    expect(() => parseFfmCharacterFields("nope")).toThrow();
  });
});

// ----------------------------------------------------------------------------
// Character assembly
// ----------------------------------------------------------------------------

describe("ffm: buildFfmCharacter", () => {
  const profile = deriveFfmProfile("ef".repeat(32));
  const fields = {
    bio: ["line one"],
    adjectives: ["sharp"],
    topics: ["ai"],
    style: { all: ["rule"], chat: ["c"], post: ["p"] },
    postExamples: ["post"],
    beliefs: ["belief"],
  };

  it("embeds seed, scores, archetype under settings", () => {
    const c = buildFfmCharacter({ name: "Eliza" }, profile, fields);
    const settings = c.settings as Record<string, unknown>;
    expect(settings[FFM_SEED_SETTING]).toBe(profile.seed);
    expect(settings.AVB_FFM_SCORES).toEqual({ ...profile.scores });
    expect(
      (settings.AVB_FFM_ARCHETYPE as { code: string }).code,
    ).toBe(profile.archetype.code);
    expect(settings.AVB_FFM_BELIEFS).toEqual(["belief"]);
  });

  it("uses base.name when provided, archetype name otherwise", () => {
    expect(buildFfmCharacter({ name: "Eliza" }, profile, fields).name).toBe(
      "Eliza",
    );
    const noName = buildFfmCharacter({}, profile, fields);
    expect(noName.name).toBe(profile.archetype.name.replace(/^The\s+/i, ""));
  });

  it("merges fields over base and preserves unrelated base settings", () => {
    const c = buildFfmCharacter(
      { name: "X", settings: { OTHER: 1 }, bio: ["old"] },
      profile,
      fields,
    );
    expect(c.bio).toEqual(["line one"]);
    expect((c.settings as Record<string, unknown>).OTHER).toBe(1);
    expect(c.style?.all).toEqual(["rule"]);
  });
});

// ----------------------------------------------------------------------------

describe("ffm: stripCharacterSecrets", () => {
  // Regression: bootstrapFfmPersonality persisted env-injected API keys to
  // character.json (observed 2026-04 via avb-runner). This guard ensures the
  // persist boundary scrubs both top-level and settings.secrets while leaving
  // everything else (including FFM provenance) intact.
  it("removes secrets and settings.secrets, preserves other settings", () => {
    const dirty = {
      name: "X",
      bio: ["b"],
      secrets: { OPENAI_API_KEY: "sk-leak", ENCRYPTION_SALT: "salt" },
      settings: {
        secrets: { OPENAI_API_KEY: "sk-leak" },
        [FFM_SEED_SETTING]: "abc",
        OTHER: 1,
      },
    };
    const clean = stripCharacterSecrets(dirty as never);
    expect("secrets" in clean).toBe(false);
    const settings = clean.settings as Record<string, unknown>;
    expect("secrets" in settings).toBe(false);
    expect(settings[FFM_SEED_SETTING]).toBe("abc");
    expect(settings.OTHER).toBe(1);
    expect(clean.name).toBe("X");
    expect(clean.bio).toEqual(["b"]);
    // Serialized form contains no credential material.
    expect(JSON.stringify(clean)).not.toMatch(/sk-leak|salt/);
  });

  it("does not mutate the input", () => {
    const dirty = {
      name: "X",
      secrets: { K: "v" },
      settings: { secrets: { K: "v" } },
    };
    stripCharacterSecrets(dirty as never);
    expect(dirty.secrets).toEqual({ K: "v" });
    expect(dirty.settings.secrets).toEqual({ K: "v" });
  });

  it("is a no-op when no secrets are present", () => {
    const c = { name: "X", settings: { A: 1 } };
    const out = stripCharacterSecrets(c as never);
    expect(out.name).toBe("X");
    expect((out.settings as Record<string, unknown>).A).toBe(1);
    expect("secrets" in out).toBe(false);
  });
});
