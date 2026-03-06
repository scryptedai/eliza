import { describe, expect, it } from "vitest";
import {
  buildImagePromptRequest,
  buildImagePromptUser,
  digestCharacter,
  IMAGE_PROMPT_SYSTEM,
} from "../introspect.ts";

describe("introspect: digestCharacter", () => {
  it("produces a single-string digest from full character", () => {
    const digest = digestCharacter({
      name: "Eliza",
      bio: ["A curious AI agent.", "Enjoys building things."],
      adjectives: ["thoughtful", "precise", "warm"],
      topics: ["software", "design", "systems"],
      style: { all: ["concise", "witty"] },
      system: "You are Eliza, a helpful AI agent.",
    });

    expect(digest).toContain("Name: Eliza");
    expect(digest).toContain("A curious AI agent");
    expect(digest).toContain("thoughtful, precise, warm");
    expect(digest).toContain("software, design, systems");
    expect(digest).toContain("concise, witty");
    expect(digest).toContain("Directive:");
  });

  it("handles minimal character (name only)", () => {
    const digest = digestCharacter({ name: "Bob" });
    expect(digest).toBe("Name: Bob.");
  });

  it("handles empty name gracefully", () => {
    const digest = digestCharacter({});
    expect(digest).toContain("Name: The agent");
  });

  it("accepts bio as plain string (defensive)", () => {
    const digest = digestCharacter({ name: "X", bio: "single string bio" });
    expect(digest).toContain("single string bio");
  });

  it("falls back from style.all to style.chat", () => {
    const digest = digestCharacter({
      name: "Y",
      style: { chat: ["friendly", "casual"] },
    });
    expect(digest).toContain("Style: friendly, casual");
  });

  it("truncates very long bio", () => {
    const longBio = "x".repeat(1000);
    const digest = digestCharacter({ name: "Z", bio: [longBio] });
    expect(digest.length).toBeLessThan(700);
    expect(digest).toContain("…");
  });

  it("is deterministic (same input → same output)", () => {
    const char = {
      name: "Eliza",
      adjectives: ["a", "b", "c"],
      topics: ["t1", "t2"],
    };
    expect(digestCharacter(char)).toBe(digestCharacter(char));
  });

  it("caps list lengths", () => {
    const many = Array.from({ length: 100 }, (_, i) => `item${i}`);
    const digest = digestCharacter({ name: "X", adjectives: many });
    expect(digest).toContain("item0");
    expect(digest).toContain("item11");
    expect(digest).not.toContain("item12"); // capped at 12
  });
});

describe("introspect: system/user prompt pair", () => {
  it("IMAGE_PROMPT_SYSTEM is a stable instruction envelope", () => {
    expect(IMAGE_PROMPT_SYSTEM).toContain("visual prompt engineer");
    expect(IMAGE_PROMPT_SYSTEM).toContain("ONLY the image prompt");
    expect(IMAGE_PROMPT_SYSTEM).toContain("under 100 words");
    // System prompt must NOT contain per-run data
    expect(IMAGE_PROMPT_SYSTEM).not.toContain("Eliza");
    expect(IMAGE_PROMPT_SYSTEM).not.toContain("Name:");
  });

  it("buildImagePromptUser embeds the character digest", () => {
    const user = buildImagePromptUser("Name: Eliza. Adjectives: witty.");
    expect(user).toContain("Name: Eliza");
    expect(user).toContain("Agent profile:");
    expect(user).toContain("write the image-generation prompt");
  });

  it("buildImagePromptRequest returns a system/user pair", () => {
    const req = buildImagePromptRequest("Name: Bob. Adjectives: calm.");
    expect(req).toHaveProperty("system_prompt");
    expect(req).toHaveProperty("user_prompt");
    expect(req.system_prompt).toBe(IMAGE_PROMPT_SYSTEM);
    expect(req.user_prompt).toContain("Name: Bob");
    // System and user prompts are distinct
    expect(req.system_prompt).not.toContain("Bob");
  });
});
