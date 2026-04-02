/**
 * Character merge tests.
 *
 * Critical invariants:
 *   - hand-authored fields are NEVER clobbered
 *   - settings.ffm is ALWAYS written (it's the seed of record)
 *   - input character is never mutated (pure function)
 *   - per-key style merge (style.all can be filled while style.chat is preserved)
 */

import { describe, expect, it } from "vitest";
import { hasFfmSeed, mergePersonality } from "../merge.ts";
import type {
  FfmProfile,
  MergeableCharacter,
  NarrativeExpansion,
  VoiceExpansion,
} from "../types.ts";

// ----------------------------------------------------------------------------
// Fixtures
// ----------------------------------------------------------------------------

const SEED = "deadbeef".repeat(8);

const PROFILE: FfmProfile = {
  seed: SEED,
  traits: { O: 0.71, C: 0.4, E: 0.62, A: 0.65, N: 0.3 },
  archetype: {
    code: 22,
    sloan: "IUSAC",
    label: "The Free Spirit",
    summary: "Curious, spontaneous, drawn to people.",
  },
};

const NARRATIVE: NarrativeExpansion = {
  bio: ["bio1", "bio2", "bio3", "bio4", "bio5", "bio6", "bio7", "bio8"],
  beliefs: ["belief1", "belief2"],
  adjectives: ["adj1", "adj2", "adj3"],
  topics: ["topic1", "topic2"],
};

const VOICE: VoiceExpansion = {
  messageExamples: [
    [
      { name: "user", content: { text: "hi" } },
      { name: "Eliza", content: { text: "hello" } },
    ],
    [
      { name: "user", content: { text: "help" } },
      { name: "Eliza", content: { text: "sure" } },
    ],
  ],
  postExamples: ["post1", "post2", "post3"],
  style: {
    all: ["all1", "all2"],
    chat: ["chat1"],
    post: ["postStyle1"],
  },
};

function blank(): MergeableCharacter {
  return { name: "Eliza" };
}

// ----------------------------------------------------------------------------
// hasFfmSeed
// ----------------------------------------------------------------------------

describe("hasFfmSeed", () => {
  it("true for valid 64-hex seed", () => {
    expect(
      hasFfmSeed({ name: "x", settings: { ffm: { seed: SEED } as never } }),
    ).toBe(true);
  });

  it("false for missing settings", () => {
    expect(hasFfmSeed({ name: "x" })).toBe(false);
  });

  it("false for missing ffm block", () => {
    expect(hasFfmSeed({ name: "x", settings: {} })).toBe(false);
  });

  it("false for non-string seed", () => {
    expect(
      hasFfmSeed({ name: "x", settings: { ffm: { seed: 123 } as never } }),
    ).toBe(false);
  });

  it("false for wrong-length seed", () => {
    expect(
      hasFfmSeed({ name: "x", settings: { ffm: { seed: "abc" } as never } }),
    ).toBe(false);
  });

  it("false for non-hex characters", () => {
    expect(
      hasFfmSeed({
        name: "x",
        settings: { ffm: { seed: "g".repeat(64) } as never },
      }),
    ).toBe(false);
  });

  it("false for uppercase hex (we normalize to lowercase, stored seed must match)", () => {
    expect(
      hasFfmSeed({
        name: "x",
        settings: { ffm: { seed: "DEADBEEF".repeat(8) } as never },
      }),
    ).toBe(false);
  });
});

// ----------------------------------------------------------------------------
// mergePersonality — settings.ffm always written
// ----------------------------------------------------------------------------

describe("mergePersonality — settings.ffm", () => {
  it("always writes settings.ffm with full profile", () => {
    const out = mergePersonality(blank(), PROFILE);
    expect(out.settings?.ffm?.seed).toBe(SEED);
    expect(out.settings?.ffm?.traits).toEqual(PROFILE.traits);
    expect(out.settings?.ffm?.archetype.code).toBe(22);
    expect(out.settings?.ffm?.archetype.sloan).toBe("IUSAC");
    expect(out.settings?.ffm?.version).toBe(1);
    expect(out.settings?.ffm?.generatedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
  });

  it("merged character passes hasFfmSeed", () => {
    const out = mergePersonality(blank(), PROFILE);
    expect(hasFfmSeed(out)).toBe(true);
  });

  it("preserves other settings keys", () => {
    const char: MergeableCharacter = {
      name: "Eliza",
      settings: { someKey: "someValue", another: 42 },
    };
    const out = mergePersonality(char, PROFILE);
    expect(out.settings?.someKey).toBe("someValue");
    expect(out.settings?.another).toBe(42);
    expect(out.settings?.ffm?.seed).toBe(SEED);
  });

  it("overwrites a stale settings.ffm (seed is always source of truth)", () => {
    const char: MergeableCharacter = {
      name: "Eliza",
      settings: { ffm: { seed: "oldoldold" } as never },
    };
    const out = mergePersonality(char, PROFILE);
    expect(out.settings?.ffm?.seed).toBe(SEED);
  });
});

// ----------------------------------------------------------------------------
// mergePersonality — never clobber hand-authored fields
// ----------------------------------------------------------------------------

describe("mergePersonality — preserves hand-authored content", () => {
  it("preserves existing bio (array form)", () => {
    const char: MergeableCharacter = {
      name: "Eliza",
      bio: ["I am hand-written", "do not replace me"],
    };
    const out = mergePersonality(char, PROFILE, NARRATIVE);
    expect(out.bio).toEqual(["I am hand-written", "do not replace me"]);
  });

  it("preserves existing bio (string form)", () => {
    const char: MergeableCharacter = {
      name: "Eliza",
      bio: "Hand-written single-string bio.",
    };
    const out = mergePersonality(char, PROFILE, NARRATIVE);
    expect(out.bio).toBe("Hand-written single-string bio.");
  });

  it("preserves existing adjectives", () => {
    const char: MergeableCharacter = {
      name: "Eliza",
      adjectives: ["custom1", "custom2"],
    };
    const out = mergePersonality(char, PROFILE, NARRATIVE);
    expect(out.adjectives).toEqual(["custom1", "custom2"]);
  });

  it("preserves existing topics", () => {
    const char: MergeableCharacter = {
      name: "Eliza",
      topics: ["my-topic"],
    };
    const out = mergePersonality(char, PROFILE, NARRATIVE);
    expect(out.topics).toEqual(["my-topic"]);
  });

  it("preserves existing messageExamples", () => {
    const existing = [
      [
        { name: "user", content: { text: "original" } },
        { name: "Eliza", content: { text: "original reply" } },
      ],
    ];
    const char: MergeableCharacter = {
      name: "Eliza",
      messageExamples: existing,
    };
    const out = mergePersonality(char, PROFILE, NARRATIVE, VOICE);
    expect(out.messageExamples).toBe(existing);
  });

  it("preserves existing postExamples", () => {
    const char: MergeableCharacter = {
      name: "Eliza",
      postExamples: ["my post"],
    };
    const out = mergePersonality(char, PROFILE, NARRATIVE, VOICE);
    expect(out.postExamples).toEqual(["my post"]);
  });

  it("style merge is per-key: existing style.all preserved, missing style.chat filled", () => {
    const char: MergeableCharacter = {
      name: "Eliza",
      style: { all: ["my-style-rule"] }, // chat and post missing
    };
    const out = mergePersonality(char, PROFILE, NARRATIVE, VOICE);
    expect(out.style?.all).toEqual(["my-style-rule"]); // preserved
    expect(out.style?.chat).toEqual(["chat1"]); // filled from voice
    expect(out.style?.post).toEqual(["postStyle1"]); // filled from voice
  });
});

// ----------------------------------------------------------------------------
// mergePersonality — fills empty fields
// ----------------------------------------------------------------------------

describe("mergePersonality — fills empty fields", () => {
  it("treats empty array as empty (fills it)", () => {
    const char: MergeableCharacter = { name: "Eliza", bio: [] };
    const out = mergePersonality(char, PROFILE, NARRATIVE);
    // bio[8] + beliefs[2] = 10
    expect(out.bio).toHaveLength(10);
    expect((out.bio as string[])[0]).toBe("bio1");
    expect((out.bio as string[])[8]).toBe("belief1");
  });

  it("treats whitespace-only string bio as empty (fills it)", () => {
    const char: MergeableCharacter = { name: "Eliza", bio: "   " };
    const out = mergePersonality(char, PROFILE, NARRATIVE);
    expect(Array.isArray(out.bio)).toBe(true);
    expect(out.bio).toHaveLength(10);
  });

  it("treats array of empty strings as empty (fills it)", () => {
    const char: MergeableCharacter = { name: "Eliza", adjectives: ["", "  "] };
    const out = mergePersonality(char, PROFILE, NARRATIVE);
    expect(out.adjectives).toEqual(["adj1", "adj2", "adj3"]);
  });

  it("fills messageExamples with deep-copied turns", () => {
    const out = mergePersonality(blank(), PROFILE, NARRATIVE, VOICE);
    expect(out.messageExamples).toHaveLength(2);
    expect(out.messageExamples?.[0][0].content.text).toBe("hi");
    // Verify deep copy: mutating the output doesn't touch VOICE
    out.messageExamples![0][0].content.text = "MUTATED";
    expect(VOICE.messageExamples[0][0].content.text).toBe("hi");
  });

  it("fills all voice fields when blank", () => {
    const out = mergePersonality(blank(), PROFILE, NARRATIVE, VOICE);
    expect(out.postExamples).toEqual(["post1", "post2", "post3"]);
    expect(out.style?.all).toEqual(["all1", "all2"]);
    expect(out.style?.chat).toEqual(["chat1"]);
    expect(out.style?.post).toEqual(["postStyle1"]);
  });

  it("works with profile only (no expansions)", () => {
    const out = mergePersonality(blank(), PROFILE);
    expect(out.settings?.ffm?.seed).toBe(SEED);
    expect(out.bio).toBeUndefined();
    expect(out.messageExamples).toBeUndefined();
  });

  it("works with narrative only (no voice)", () => {
    const out = mergePersonality(blank(), PROFILE, NARRATIVE);
    expect(out.bio).toHaveLength(10);
    expect(out.adjectives).toHaveLength(3);
    expect(out.messageExamples).toBeUndefined();
  });
});

// ----------------------------------------------------------------------------
// mergePersonality — purity
// ----------------------------------------------------------------------------

describe("mergePersonality — purity", () => {
  it("never mutates the input character", () => {
    const char: MergeableCharacter = {
      name: "Eliza",
      settings: { existingKey: "existingValue" },
    };
    const before = JSON.stringify(char);
    mergePersonality(char, PROFILE, NARRATIVE, VOICE);
    expect(JSON.stringify(char)).toBe(before);
  });

  it("returns a new top-level object", () => {
    const char = blank();
    const out = mergePersonality(char, PROFILE);
    expect(out).not.toBe(char);
  });

  it("returns a new settings object (not aliased to input)", () => {
    const char: MergeableCharacter = {
      name: "Eliza",
      settings: { foo: "bar" },
    };
    const out = mergePersonality(char, PROFILE);
    expect(out.settings).not.toBe(char.settings);
  });

  it("preserves unrecognized top-level fields", () => {
    const char: MergeableCharacter = {
      name: "Eliza",
      lore: ["some lore"],
      customField: { nested: true },
    };
    const out = mergePersonality(char, PROFILE, NARRATIVE, VOICE);
    expect(out.lore).toEqual(["some lore"]);
    expect(out.customField).toEqual({ nested: true });
  });
});
