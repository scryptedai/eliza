/**
 * PromptSet rendering & parser tests.
 *
 * The prompts themselves are tested by *running* them (ffm-demo.ts).
 * Here we test the supporting machinery:
 *   - extractJsonBlock(): markdown-fence stripping, brace balancing,
 *     string-escape handling
 *   - parseNarrative()/parseVoice(): structural validation & FfmParseError
 *   - maxReplyOverlap(): Jaccard token similarity
 *   - renderTraitDigest(): deterministic, all five traits present
 *   - render*Prompt(): tags substituted, no `{{` leftovers
 */

import { describe, expect, it } from "vitest";
import {
  extractJsonBlock,
  FfmParseError,
  maxReplyOverlap,
  parseNarrative,
  parseVoice,
  renderNarrativePrompt,
  renderTraitDigest,
  renderVoicePrompt,
} from "../promptset.ts";
import type { FfmProfile, MessageTurn, NarrativeExpansion } from "../types.ts";

// ----------------------------------------------------------------------------
// Fixtures
// ----------------------------------------------------------------------------

const PROFILE: FfmProfile = {
  seed: "deadbeef".repeat(8),
  traits: { O: 0.71, C: 0.4, E: 0.62, A: 0.65, N: 0.3 },
  archetype: {
    code: 22,
    sloan: "IUSAC",
    label: "The Free Spirit",
    summary: "Curious, spontaneous, drawn to people.",
  },
};

const NARRATIVE: NarrativeExpansion = {
  bio: ["line one", "line two", "line three"],
  beliefs: ["I think things work out", "Plans are suggestions"],
  adjectives: ["curious", "warm"],
  topics: ["music", "travel"],
};

function turn(name: string, text: string): MessageTurn {
  return { name, content: { text } };
}

// ----------------------------------------------------------------------------
// extractJsonBlock
// ----------------------------------------------------------------------------

describe("extractJsonBlock", () => {
  it("extracts a bare JSON object", () => {
    expect(extractJsonBlock('{"a":1}')).toBe('{"a":1}');
  });

  it("strips ```json fences", () => {
    const text = '```json\n{"a":1}\n```';
    expect(extractJsonBlock(text)).toBe('{"a":1}');
  });

  it("strips ``` fences without language tag", () => {
    const text = '```\n{"a":1}\n```';
    expect(extractJsonBlock(text)).toBe('{"a":1}');
  });

  it("ignores prose before and after the JSON", () => {
    const text = 'Sure, here is the output:\n{"a":1}\nHope that helps!';
    expect(extractJsonBlock(text)).toBe('{"a":1}');
  });

  it("balances nested braces", () => {
    const text = '{"a":{"b":{"c":1}}}';
    expect(extractJsonBlock(text)).toBe('{"a":{"b":{"c":1}}}');
  });

  it("ignores braces inside string literals", () => {
    const text = '{"msg":"this } is not the end"}';
    expect(extractJsonBlock(text)).toBe('{"msg":"this } is not the end"}');
  });

  it("handles escaped quotes inside strings", () => {
    const text = '{"msg":"she said \\"hello}\\""}';
    expect(extractJsonBlock(text)).toBe('{"msg":"she said \\"hello}\\""}');
  });

  it("returns null when no opening brace exists", () => {
    expect(extractJsonBlock("just some text")).toBeNull();
  });

  it("returns null when braces never balance", () => {
    expect(extractJsonBlock('{"a":1')).toBeNull();
  });

  it("returns the FIRST balanced object when there are several", () => {
    const text = '{"first":1} and then {"second":2}';
    expect(extractJsonBlock(text)).toBe('{"first":1}');
  });
});

// ----------------------------------------------------------------------------
// parseNarrative
// ----------------------------------------------------------------------------

describe("parseNarrative", () => {
  const VALID = JSON.stringify({
    bio: ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"],
    beliefs: ["belief one", "belief two", "belief three", "belief four"],
    adjectives: ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"],
    topics: ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"],
  });

  it("parses valid narrative JSON", () => {
    const out = parseNarrative(VALID);
    expect(out.bio).toHaveLength(8);
    expect(out.beliefs).toHaveLength(4);
    expect(out.adjectives).toHaveLength(10);
    expect(out.topics).toHaveLength(8);
  });

  it("parses valid narrative wrapped in markdown fences", () => {
    const out = parseNarrative("```json\n" + VALID + "\n```");
    expect(out.bio).toHaveLength(8);
  });

  it("trims string entries", () => {
    const json = JSON.stringify({
      bio: ["  spaced  "],
      beliefs: ["x"],
      adjectives: ["x"],
      topics: ["x"],
    });
    expect(parseNarrative(json).bio[0]).toBe("spaced");
  });

  it("filters non-string and empty-string entries", () => {
    const json = JSON.stringify({
      bio: ["keep", 42, null, "", "   ", "also keep"],
      beliefs: ["x"],
      adjectives: ["x"],
      topics: ["x"],
    });
    expect(parseNarrative(json).bio).toEqual(["keep", "also keep"]);
  });

  it("throws FfmParseError when no JSON is found", () => {
    expect(() => parseNarrative("hello")).toThrow(FfmParseError);
  });

  it("throws FfmParseError on malformed JSON", () => {
    expect(() => parseNarrative('{"bio": [trailing,]}')).toThrow(FfmParseError);
  });

  it("throws FfmParseError when bio is not an array", () => {
    const json = JSON.stringify({
      bio: "not an array",
      beliefs: ["x"],
      adjectives: ["x"],
      topics: ["x"],
    });
    expect(() => parseNarrative(json)).toThrow(/bio is not an array/);
  });

  it("throws FfmParseError when an array has no usable entries", () => {
    const json = JSON.stringify({
      bio: ["", "   ", null],
      beliefs: ["x"],
      adjectives: ["x"],
      topics: ["x"],
    });
    expect(() => parseNarrative(json)).toThrow(/no usable entries/);
  });

  it("FfmParseError carries rawText for debugging", () => {
    try {
      parseNarrative("garbage with no json");
      expect.fail("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(FfmParseError);
      expect((e as FfmParseError).rawText).toBe("garbage with no json");
    }
  });
});

// ----------------------------------------------------------------------------
// parseVoice
// ----------------------------------------------------------------------------

describe("parseVoice", () => {
  function buildVoice(over: Partial<Record<string, unknown>> = {}): string {
    return JSON.stringify({
      messageExamples: [
        [turn("user", "hi"), turn("Eliza", "hello")],
        [turn("user", "help me"), turn("Eliza", "sure, what's up?")],
      ],
      postExamples: ["post one", "post two", "post three"],
      style: { all: ["s1"], chat: ["c1"], post: ["p1"] },
      ...over,
    });
  }

  it("parses valid voice JSON", () => {
    const out = parseVoice(buildVoice());
    expect(out.messageExamples).toHaveLength(2);
    expect(out.messageExamples[0]).toHaveLength(2);
    expect(out.messageExamples[0][0].name).toBe("user");
    expect(out.messageExamples[0][1].content.text).toBe("hello");
    expect(out.postExamples).toHaveLength(3);
    expect(out.style.all).toEqual(["s1"]);
  });

  it("throws when messageExamples is not an array", () => {
    const bad = buildVoice({ messageExamples: "nope" });
    expect(() => parseVoice(bad)).toThrow(/messageExamples is not an array/);
  });

  it("throws when an exchange has fewer than 2 turns", () => {
    const bad = buildVoice({ messageExamples: [[turn("user", "lonely")]] });
    expect(() => parseVoice(bad)).toThrow(/not a 2-turn exchange/);
  });

  it("throws when a turn is missing name", () => {
    const bad = buildVoice({
      messageExamples: [[{ content: { text: "x" } }, turn("Eliza", "y")]],
    });
    expect(() => parseVoice(bad)).toThrow(/turn missing name/);
  });

  it("throws when a turn is missing content.text", () => {
    const bad = buildVoice({
      messageExamples: [[{ name: "user", content: {} }, turn("Eliza", "y")]],
    });
    expect(() => parseVoice(bad)).toThrow(/turn missing content.text/);
  });

  it("throws when style is not an object", () => {
    const bad = buildVoice({ style: null });
    expect(() => parseVoice(bad)).toThrow(/style is not an object/);
  });

  it("throws when style.all is missing", () => {
    const bad = buildVoice({ style: { chat: ["c"], post: ["p"] } });
    expect(() => parseVoice(bad)).toThrow(/style.all is not an array/);
  });

  it("trims turn text", () => {
    const json = buildVoice({
      messageExamples: [[turn("user", "  hi  "), turn("Eliza", "  hey  ")]],
    });
    const out = parseVoice(json);
    expect(out.messageExamples[0][1].content.text).toBe("hey");
  });

  it("takes only the first two turns of an over-long exchange", () => {
    // The schema asks for 2-turn exchanges; if the LLM gives more, we keep
    // [user, agent] and drop the rest. Verifies it doesn't blow up.
    const json = buildVoice({
      messageExamples: [
        [turn("user", "a"), turn("Eliza", "b"), turn("user", "extra")],
      ],
    });
    const out = parseVoice(json);
    expect(out.messageExamples[0]).toHaveLength(2);
  });
});

// ----------------------------------------------------------------------------
// maxReplyOverlap
// ----------------------------------------------------------------------------

describe("maxReplyOverlap", () => {
  it("returns 0 for completely distinct replies", () => {
    const ex: MessageTurn[][] = [
      [turn("user", "x"), turn("Eliza", "alpha bravo charlie")],
      [turn("user", "x"), turn("Eliza", "delta echo foxtrot")],
    ];
    expect(maxReplyOverlap(ex)).toBe(0);
  });

  it("returns 1 for identical replies", () => {
    const ex: MessageTurn[][] = [
      [turn("user", "x"), turn("Eliza", "alpha bravo charlie")],
      [turn("user", "y"), turn("Eliza", "alpha bravo charlie")],
    ];
    expect(maxReplyOverlap(ex)).toBe(1);
  });

  it("computes Jaccard correctly for partial overlap", () => {
    // {alpha,bravo,charlie} ∩ {alpha,bravo,delta} = {alpha,bravo} → |2|
    // union = {alpha,bravo,charlie,delta} → |4|
    // jaccard = 0.5
    const ex: MessageTurn[][] = [
      [turn("user", "x"), turn("Eliza", "alpha bravo charlie")],
      [turn("user", "x"), turn("Eliza", "alpha bravo delta")],
    ];
    expect(maxReplyOverlap(ex)).toBeCloseTo(0.5, 5);
  });

  it("returns the MAX pairwise overlap, not the average", () => {
    const ex: MessageTurn[][] = [
      [turn("user", "x"), turn("Eliza", "alpha bravo")],
      [turn("user", "x"), turn("Eliza", "alpha bravo")], // dup with [0] → 1.0
      [turn("user", "x"), turn("Eliza", "completely different words here")],
    ];
    expect(maxReplyOverlap(ex)).toBe(1);
  });

  it("ignores case and punctuation", () => {
    const ex: MessageTurn[][] = [
      [turn("user", "x"), turn("Eliza", "Alpha, BRAVO!")],
      [turn("user", "x"), turn("Eliza", "alpha bravo")],
    ];
    expect(maxReplyOverlap(ex)).toBe(1);
  });

  it("returns 0 for fewer than 2 exchanges", () => {
    expect(maxReplyOverlap([])).toBe(0);
    expect(maxReplyOverlap([[turn("user", "x"), turn("Eliza", "y")]])).toBe(0);
  });
});

// ----------------------------------------------------------------------------
// renderTraitDigest
// ----------------------------------------------------------------------------

describe("renderTraitDigest", () => {
  it("produces five lines, one per trait, in OCEAN order", () => {
    const out = renderTraitDigest(PROFILE);
    const lines = out.split("\n");
    expect(lines).toHaveLength(5);
    expect(lines[0]).toMatch(/Openness/);
    expect(lines[1]).toMatch(/Conscientiousness/);
    expect(lines[2]).toMatch(/Extraversion/);
    expect(lines[3]).toMatch(/Agreeableness/);
    expect(lines[4]).toMatch(/Neuroticism/);
  });

  it("includes the score and high/low pole call", () => {
    const out = renderTraitDigest(PROFILE);
    expect(out).toMatch(/Openness: 0\.71 \(high\)/);
    expect(out).toMatch(/Conscientiousness: 0\.40 \(low\)/);
    expect(out).toMatch(/Neuroticism: 0\.30 \(low\)/);
  });

  it("is deterministic", () => {
    expect(renderTraitDigest(PROFILE)).toBe(renderTraitDigest(PROFILE));
  });
});

// ----------------------------------------------------------------------------
// render*Prompt — tag substitution
// ----------------------------------------------------------------------------

describe("renderNarrativePrompt", () => {
  it("substitutes all tags (no `{{` leftovers)", () => {
    const r = renderNarrativePrompt(PROFILE, "Eliza");
    expect(r.system).not.toMatch(/\{\{/);
    expect(r.user).not.toMatch(/\{\{/);
  });

  it("includes the agent name", () => {
    const r = renderNarrativePrompt(PROFILE, "TestAgent");
    const combined = r.system + r.user;
    expect(combined).toContain("TestAgent");
  });

  it("includes the trait digest", () => {
    const r = renderNarrativePrompt(PROFILE, "Eliza");
    const combined = r.system + r.user;
    expect(combined).toContain("Openness: 0.71");
    expect(combined).toContain("IUSAC");
  });
});

describe("renderVoicePrompt", () => {
  it("substitutes all tags including narrative-derived ones", () => {
    const r = renderVoicePrompt(PROFILE, NARRATIVE, "Eliza");
    expect(r.system).not.toMatch(/\{\{/);
    expect(r.user).not.toMatch(/\{\{/);
  });

  it("splices the narrative bio into the prompt", () => {
    const r = renderVoicePrompt(PROFILE, NARRATIVE, "Eliza");
    const combined = r.system + r.user;
    expect(combined).toContain("line one");
    expect(combined).toContain("I think things work out");
  });
});
