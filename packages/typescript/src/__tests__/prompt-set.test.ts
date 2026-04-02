import { describe, expect, it } from "vitest";
import {
  DEFAULT_MODEL_ID,
  getModelEntry,
  getModelLimits,
  hasModel,
  listModels,
} from "../model-registry";
import {
  DEFAULT_MODEL_LIMITS,
  NOVA_PRO_LIMITS,
  PromptSet,
  replaceTags,
  toScryptedPayload,
} from "../prompt-set";
import { estimateTokens } from "../utils";

// ============================================================================
// 1. Tag replacement
// ============================================================================

describe("PromptSet: tag replacement", () => {
  it("replaces {{TAG}} placeholders with values", () => {
    const out = replaceTags("Hello {{NAME}}, you are {{AGE}}.", {
      NAME: "Alice",
      AGE: 30,
    });
    expect(out).toBe("Hello Alice, you are 30.");
  });

  it("missing tags resolve to empty string", () => {
    const out = replaceTags("{{A}}-{{B}}-{{C}}", { A: "x" });
    expect(out).toBe("x--");
  });

  it("trims whitespace inside braces before lookup", () => {
    const out = replaceTags("{{  NAME  }}", { NAME: "Bob" });
    expect(out).toBe("Bob");
  });

  it("is case-sensitive", () => {
    const out = replaceTags("{{Name}}/{{NAME}}", {
      NAME: "upper",
      Name: "mixed",
    });
    expect(out).toBe("mixed/upper");
  });

  it("null and undefined values replace to empty", () => {
    const out = replaceTags("a{{X}}b{{Y}}c", {
      X: null,
      Y: undefined,
    });
    expect(out).toBe("abc");
  });

  it("stringifies non-string values", () => {
    const out = replaceTags("{{N}}/{{B}}/{{A}}", {
      N: 42,
      B: true,
      A: [1, 2, 3],
    });
    expect(out).toBe("42/true/1,2,3");
  });

  it("does NOT re-expand tags inside replacement values (single-pass)", () => {
    const out = replaceTags("{{A}}", { A: "{{B}}", B: "EVIL" });
    expect(out).toBe("{{B}}");
  });

  it("leaves non-identifier tag content unreplaced", () => {
    const out = replaceTags("{{#if x}}{{NAME}}{{/if}}", { NAME: "ok" });
    expect(out).toContain("{{#if x}}");
    expect(out).toContain("{{/if}}");
    expect(out).toContain("ok");
  });

  it("handles adjacent and nested-looking braces", () => {
    const out = replaceTags("{{{A}}} {{A}} { {A} }", { A: "X" });
    // Triple braces contain a valid {{A}} inside → becomes {X}
    expect(out).toBe("{X} X { {A} }");
  });

  it("handles empty template", () => {
    expect(replaceTags("", { A: "x" })).toBe("");
  });

  it("handles template with no tags", () => {
    const tpl = "plain text, no placeholders here.";
    expect(replaceTags(tpl, { A: "x" })).toBe(tpl);
  });
});

// ============================================================================
// 2. Model registry
// ============================================================================

describe("Model registry", () => {
  it("default model ID is present in the registry", () => {
    expect(listModels()).toContain(DEFAULT_MODEL_ID);
    expect(hasModel(DEFAULT_MODEL_ID)).toBe(true);
    expect(getModelEntry(DEFAULT_MODEL_ID)).toBeDefined();
  });

  it("every registered model has positive integer limits", () => {
    for (const id of listModels()) {
      const entry = getModelEntry(id);
      expect(entry).toBeDefined();
      if (!entry) continue;
      expect(Number.isInteger(entry.maxInputTokens)).toBe(true);
      expect(Number.isInteger(entry.maxOutputTokens)).toBe(true);
      expect(entry.maxInputTokens).toBeGreaterThan(0);
      expect(entry.maxOutputTokens).toBeGreaterThan(0);
    }
  });

  it("unknown model: hasModel/getModelEntry say no, getModelLimits falls back", () => {
    expect(hasModel("does-not-exist-xyz")).toBe(false);
    expect(getModelEntry("does-not-exist-xyz")).toBeUndefined();
    // getModelLimits soft-fails to the default so prompt pipelines don't break
    expect(getModelLimits("does-not-exist-xyz")).toEqual(DEFAULT_MODEL_LIMITS);
  });

  it("getModelLimits() with no arg returns default", () => {
    expect(getModelLimits()).toEqual(DEFAULT_MODEL_LIMITS);
    expect(getModelLimits()).toEqual(getModelLimits(DEFAULT_MODEL_ID));
  });

  it("lookup is exact-match — near-misses fall through to default", () => {
    // Short names that an alias system might have accepted must NOT match.
    // This guards against the namespace-pollution risk with versioned IDs.
    for (const near of ["nova-pro", "nova", "amazon.nova-pro", "nova-pro-v1"]) {
      expect(hasModel(near)).toBe(false);
      expect(getModelEntry(near)).toBeUndefined();
    }
  });

  it("prototype pollution: dangerous keys do not leak through lookup", () => {
    for (const bad of ["__proto__", "constructor", "hasOwnProperty"]) {
      expect(hasModel(bad)).toBe(false);
      expect(getModelEntry(bad)).toBeUndefined();
      expect(getModelLimits(bad)).toEqual(DEFAULT_MODEL_LIMITS);
    }
  });
});

// ============================================================================
// 3. PromptSet construction & render
// ============================================================================

describe("PromptSet: construction", () => {
  it("defaults to the registry's default model limits", () => {
    const ps = new PromptSet({ system: "s", user: "u" });
    // Assert against the registry, not hardcoded numbers — if the registry
    // changes, this test stays valid without a code edit.
    expect(ps.limits).toEqual(getModelLimits(DEFAULT_MODEL_ID));
    expect(ps.limits).toEqual(DEFAULT_MODEL_LIMITS);
  });

  it("NOVA_PRO_LIMITS (deprecated shim) matches the registry entry", () => {
    expect(NOVA_PRO_LIMITS).toEqual(getModelLimits("amazon.nova-pro-v1:0"));
  });

  it("model option resolves limits from the registry (exact ID)", () => {
    const ps = new PromptSet({
      system: "s",
      user: "u",
      model: "amazon.nova-pro-v1:0",
    });
    expect(ps.limits).toEqual(getModelLimits("amazon.nova-pro-v1:0"));
  });

  it("explicit limits override model option", () => {
    const ps = new PromptSet({
      system: "s",
      user: "u",
      model: "amazon.nova-pro-v1:0",
      limits: { maxInputTokens: 42, maxOutputTokens: 7 },
    });
    expect(ps.limits.maxInputTokens).toBe(42);
    expect(ps.limits.maxOutputTokens).toBe(7);
  });

  it("unknown model falls back to registry default (soft fail)", () => {
    const ps = new PromptSet({
      system: "s",
      user: "u",
      model: "totally-nonexistent-model-9000",
    });
    expect(ps.limits).toEqual(DEFAULT_MODEL_LIMITS);
  });

  it("custom limits are respected", () => {
    const ps = new PromptSet({
      system: "s",
      user: "u",
      limits: { maxInputTokens: 1000, maxOutputTokens: 200 },
    });
    expect(ps.limits.maxInputTokens).toBe(1000);
    expect(ps.limits.maxOutputTokens).toBe(200);
  });

  it("clamps bad limit values to safe floor", () => {
    const ps = new PromptSet({
      system: "s",
      user: "u",
      limits: {
        maxInputTokens: -5,
        maxOutputTokens: NaN,
      },
    });
    expect(ps.limits.maxInputTokens).toBe(1);
    expect(ps.limits.maxOutputTokens).toBe(1);
  });

  it("stores templates verbatim", () => {
    const ps = new PromptSet({
      system: "You are {{ROLE}}.",
      user: "Task: {{TASK}}",
    });
    expect(ps.systemTemplate).toBe("You are {{ROLE}}.");
    expect(ps.userTemplate).toBe("Task: {{TASK}}");
  });
});

describe("PromptSet: render", () => {
  it("replaces tags in both system and user", () => {
    const ps = new PromptSet({
      system: "You are {{ROLE}}.",
      user: "Do {{TASK}}.",
    });
    const r = ps.render({ ROLE: "assistant", TASK: "X" });
    expect(r.system).toBe("You are assistant.");
    expect(r.user).toBe("Do X.");
    expect(r.truncated).toBe(false);
  });

  it("render() with no args = empty tag values", () => {
    const ps = new PromptSet({
      system: "prefix {{X}} suffix",
      user: "user {{Y}}",
    });
    const r = ps.render();
    expect(r.system).toBe("prefix  suffix");
    expect(r.user).toBe("user ");
  });

  it("render is deterministic (same input → same output)", () => {
    const ps = new PromptSet({
      system: "{{A}} / {{B}}",
      user: "{{C}}",
    });
    const vals = { A: "1", B: "2", C: "3" };
    const r1 = ps.render(vals);
    const r2 = ps.render(vals);
    expect(r1.system).toBe(r2.system);
    expect(r1.user).toBe(r2.user);
    expect(r1.truncated).toBe(r2.truncated);
  });

  it("result is frozen", () => {
    const ps = new PromptSet({ system: "s", user: "u" });
    const r = ps.render();
    expect(Object.isFrozen(r)).toBe(true);
  });

  it("limits are forwarded to the result", () => {
    const custom = { maxInputTokens: 999, maxOutputTokens: 42 };
    const ps = new PromptSet({ system: "s", user: "u", limits: custom });
    const r = ps.render();
    expect(r.limits.maxInputTokens).toBe(999);
    expect(r.limits.maxOutputTokens).toBe(42);
  });
});

// ============================================================================
// 3. Length sanitization
// ============================================================================

describe("PromptSet: length sanitization", () => {
  const tight = { maxInputTokens: 50, maxOutputTokens: 10 };

  it("does not truncate when under budget", () => {
    const ps = new PromptSet({
      system: "short system",
      user: "short user",
      limits: tight,
    });
    const r = ps.render();
    expect(r.truncated).toBe(false);
    expect(r.system).toBe("short system");
    expect(r.user).toBe("short user");
  });

  it("truncates when combined prompts exceed budget", () => {
    const ps = new PromptSet({
      system: "x".repeat(100),
      user: "y".repeat(400),
      limits: tight,
    });
    const r = ps.render();
    expect(r.truncated).toBe(true);
    expect(
      estimateTokens(r.system) + estimateTokens(r.user),
    ).toBeLessThanOrEqual(tight.maxInputTokens);
  });

  it("truncated text ends with ellipsis", () => {
    const ps = new PromptSet({
      system: "a",
      user: "y".repeat(400),
      limits: tight,
    });
    const r = ps.render();
    expect(r.truncated).toBe(true);
    expect(r.user.endsWith("…")).toBe(true);
  });

  it("preserves system prompt when only user overflows", () => {
    const ps = new PromptSet({
      system: "short instructions",
      user: "y".repeat(10_000),
      limits: tight,
    });
    const r = ps.render();
    expect(r.system).toBe("short instructions");
    expect(r.truncated).toBe(true);
  });

  it("respects systemReserveFraction when both overflow", () => {
    const ps = new PromptSet({
      system: "s".repeat(10_000),
      user: "u".repeat(10_000),
      limits: { maxInputTokens: 100, maxOutputTokens: 10 },
      systemReserveFraction: 0.4,
    });
    const r = ps.render();
    const sysTokens = estimateTokens(r.system);
    const usrTokens = estimateTokens(r.user);
    // system should get ~40% of budget
    expect(sysTokens).toBeLessThanOrEqual(40);
    // user gets the rest
    expect(usrTokens).toBeLessThanOrEqual(60);
    expect(sysTokens + usrTokens).toBeLessThanOrEqual(100);
  });

  it("sanitize() works on pre-built strings (no templates)", () => {
    const ps = new PromptSet({ system: "", user: "", limits: tight });
    const r = ps.sanitize("inline sys", "x".repeat(500));
    expect(r.system).toBe("inline sys");
    expect(r.truncated).toBe(true);
    expect(estimateTokens(r.user)).toBeLessThanOrEqual(tight.maxInputTokens);
  });

  it("Nova Pro limits comfortably fit normal prompts", () => {
    const ps = new PromptSet({
      system: "You are a helpful assistant. ".repeat(50),
      user: "Here is a long document: ".repeat(1000),
    });
    const r = ps.render();
    expect(r.truncated).toBe(false);
  });
});

// ============================================================================
// 4. Utility methods
// ============================================================================

describe("PromptSet: getTags", () => {
  it("lists unique tag names from both templates", () => {
    const ps = new PromptSet({
      system: "{{ROLE}} and {{ROLE}}",
      user: "{{TASK}} for {{ROLE}}",
    });
    const tags = ps.getTags().sort();
    expect(tags).toEqual(["ROLE", "TASK"]);
  });

  it("returns empty array when no tags present", () => {
    const ps = new PromptSet({ system: "plain", user: "text" });
    expect(ps.getTags()).toEqual([]);
  });

  it("ignores non-identifier placeholders", () => {
    const ps = new PromptSet({
      system: "{{#if x}}{{OK}}{{/if}}",
      user: "",
    });
    expect(ps.getTags()).toEqual(["OK"]);
  });
});

describe("PromptSet: withLimits / withModel", () => {
  it("withLimits returns a new instance with swapped limits", () => {
    const ps = new PromptSet({
      system: "{{A}}",
      user: "{{B}}",
    });
    const ps2 = ps.withLimits({ maxInputTokens: 500, maxOutputTokens: 100 });
    expect(ps2).not.toBe(ps);
    expect(ps2.limits.maxInputTokens).toBe(500);
    expect(ps2.systemTemplate).toBe(ps.systemTemplate);
    expect(ps2.userTemplate).toBe(ps.userTemplate);
  });

  it("withModel swaps limits via registry lookup", () => {
    const ps = new PromptSet({
      system: "{{A}}",
      user: "{{B}}",
      limits: { maxInputTokens: 1, maxOutputTokens: 1 },
    });
    const ps2 = ps.withModel("amazon.nova-pro-v1:0");
    expect(ps2).not.toBe(ps);
    expect(ps2.limits).toEqual(getModelLimits("amazon.nova-pro-v1:0"));
    expect(ps2.systemTemplate).toBe(ps.systemTemplate);
  });
});

describe("PromptSet: toScryptedPayload", () => {
  it("converts to snake_case wire shape and disables server auto-calc", () => {
    const ps = new PromptSet({
      system: "sys",
      user: "usr {{X}}",
      limits: { maxInputTokens: 1000, maxOutputTokens: 256 },
    });
    const payload = toScryptedPayload(ps.render({ X: "data" }));
    expect(payload).toEqual({
      system_prompt: "sys",
      user_prompt: "usr data",
      max_tokens: 256,
      auto_calculate_tokens: false,
    });
  });
});

// ============================================================================
// 5. Fuzz tests — adversarial / edge-case input should never crash and
//    invariants must always hold.
// ============================================================================

describe("PromptSet: fuzz", () => {
  const adversarialStrings = [
    "",
    "   \t\n  ",
    "\0\0\0",
    "a".repeat(100_000),
    "🚀🌙💎🙌",
    "你好世界 こんにちは 안녕하세요 مرحبا",
    "{{",
    "}}",
    "{{}}",
    "{{ }}",
    "{{{{A}}}}",
    "{{A{{B}}C}}",
    "{{A}}".repeat(10_000),
    "<script>alert('xss')</script>",
    "SELECT * FROM users; DROP TABLE users;--",
    "{{__proto__}}{{constructor}}{{prototype}}",
    "{{A}}" + "\0" + "{{B}}",
    String.fromCharCode(0xd800), // lone high surrogate
    "normal {{TAG}} normal",
    "line1\nline2\r\nline3\rline4",
    `{{TAG}}${"x".repeat(2_000_000)}`, // pushes way past Nova Pro limit
  ];

  const adversarialValues: Record<string, unknown>[] = [
    {},
    { A: "" },
    { A: null, B: undefined, TAG: 0 },
    { A: "{{B}}", B: "{{A}}" }, // recursive attempt
    { A: "a".repeat(1_000_000) },
    { __proto__: "polluted" } as Record<string, unknown>,
    { constructor: "c", prototype: "p" },
    { TAG: "<img onerror=alert(1)>" },
    { TAG: "\0\0\0" },
    { A: 123, B: true, C: [1, 2], D: { nested: "x" } },
  ];

  it("replaceTags never throws and returns a string", () => {
    for (const tpl of adversarialStrings) {
      for (const vals of adversarialValues) {
        const out = replaceTags(tpl, vals);
        expect(typeof out).toBe("string");
      }
    }
  });

  it("PromptSet.render never throws and output is always within budget", () => {
    const tightLimits = { maxInputTokens: 100, maxOutputTokens: 50 };
    for (const sys of adversarialStrings) {
      for (const usr of adversarialStrings) {
        const ps = new PromptSet({
          system: sys,
          user: usr,
          limits: tightLimits,
        });
        for (const vals of adversarialValues) {
          const r = ps.render(vals);
          // Invariants:
          expect(typeof r.system).toBe("string");
          expect(typeof r.user).toBe("string");
          expect(typeof r.truncated).toBe("boolean");
          const total = estimateTokens(r.system) + estimateTokens(r.user);
          expect(total).toBeLessThanOrEqual(tightLimits.maxInputTokens);
        }
      }
    }
  });

  it("render is idempotent under any input", () => {
    for (const tpl of adversarialStrings) {
      const ps = new PromptSet({ system: tpl, user: tpl });
      const r1 = ps.render({ TAG: tpl, A: "x" });
      const r2 = ps.render({ TAG: tpl, A: "x" });
      expect(r1.system).toBe(r2.system);
      expect(r1.user).toBe(r2.user);
    }
  });

  it("getTags never throws on adversarial templates", () => {
    for (const tpl of adversarialStrings) {
      const ps = new PromptSet({ system: tpl, user: tpl });
      const tags = ps.getTags();
      expect(Array.isArray(tags)).toBe(true);
      for (const t of tags) {
        expect(typeof t).toBe("string");
        // All reported tags must be safe identifiers
        expect(t).toMatch(/^[A-Za-z0-9_.-]+$/);
      }
    }
  });

  it("withLimits never corrupts templates", () => {
    for (const tpl of adversarialStrings) {
      const ps = new PromptSet({ system: tpl, user: tpl });
      const ps2 = ps.withLimits({
        maxInputTokens: 50,
        maxOutputTokens: 10,
      });
      expect(ps2.systemTemplate).toBe(tpl);
      expect(ps2.userTemplate).toBe(tpl);
    }
  });

  it("extreme limit values do not crash", () => {
    const wildLimits = [
      { maxInputTokens: 0, maxOutputTokens: 0 },
      { maxInputTokens: -1, maxOutputTokens: -999 },
      { maxInputTokens: Infinity, maxOutputTokens: Infinity },
      { maxInputTokens: NaN, maxOutputTokens: NaN },
      { maxInputTokens: 1e308, maxOutputTokens: 1 },
      { maxInputTokens: Number.MAX_SAFE_INTEGER, maxOutputTokens: 1 },
    ];
    for (const limits of wildLimits) {
      const ps = new PromptSet({ system: "x".repeat(1000), user: "y", limits });
      const r = ps.render();
      expect(typeof r.system).toBe("string");
      expect(typeof r.user).toBe("string");
      // Output budget is at least 1 after clamping
      expect(r.limits.maxInputTokens).toBeGreaterThanOrEqual(1);
      expect(r.limits.maxOutputTokens).toBeGreaterThanOrEqual(1);
    }
  });

  it("recursive tag injection does not cause infinite expansion", () => {
    const ps = new PromptSet({
      system: "{{A}}",
      user: "{{B}}",
    });
    const r = ps.render({
      A: "{{B}}",
      B: "{{A}}",
    });
    // Single-pass: values containing {{...}} are preserved literally
    expect(r.system).toBe("{{B}}");
    expect(r.user).toBe("{{A}}");
  });
});
