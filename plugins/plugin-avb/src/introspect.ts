/**
 * Character introspection — flatten runtime.character into a prompt-ready digest.
 *
 * The digest is a compact, human-readable string that captures the agent's
 * identity-bearing fields. It is frozen at createRun() and carried through
 * every phase in AvbRunContext, so restarts/retries use the same snapshot.
 *
 * Fields included (from Character proto): name, bio[], topics[], adjectives[],
 * style.all[], system.
 * Fields excluded: id, templates, plugins, settings, secrets, knowledge,
 * messageExamples (too verbose/PII-adjacent).
 *
 * Prompt composition is handled by core's PromptSet — this module only
 * defines the templates and the character-digest builder.
 */

import { PromptSet, toScryptedPayload } from "@elizaos/core";

export interface DigestableCharacter {
  name?: string;
  bio?: string[] | string;
  topics?: string[];
  adjectives?: string[];
  system?: string;
  style?: {
    all?: string[];
    chat?: string[];
    post?: string[];
  };
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

function joinList(items: string[] | undefined, limit = 20): string {
  if (!items || items.length === 0) return "";
  return items.slice(0, limit).join(", ");
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return `${s.slice(0, max - 1)}…`;
}

// ----------------------------------------------------------------------------
// Public
// ----------------------------------------------------------------------------

/**
 * Flatten a Character into a single descriptive paragraph.
 *
 * Output is intentionally deterministic (no randomness, stable ordering)
 * so the same character always yields the same digest — important for
 * idempotency/caching if we add that later.
 */
export function digestCharacter(c: DigestableCharacter): string {
  const parts: string[] = [];

  const name = c.name?.trim() || "The agent";
  parts.push(`Name: ${name}.`);

  // bio: string[] per proto, but accept string defensively
  if (c.bio) {
    const bioText = Array.isArray(c.bio)
      ? c.bio.filter((b) => b?.trim()).join(" ")
      : String(c.bio);
    if (bioText.trim()) {
      parts.push(`Bio: ${truncate(bioText.trim(), 600)}`);
    }
  }

  const adjectives = joinList(c.adjectives, 12);
  if (adjectives) parts.push(`Adjectives: ${adjectives}.`);

  const topics = joinList(c.topics, 15);
  if (topics) parts.push(`Interested in: ${topics}.`);

  // Prefer style.all (general), fall back to chat
  const styleAll = joinList(c.style?.all, 10);
  if (styleAll) {
    parts.push(`Style: ${styleAll}.`);
  } else {
    const styleChat = joinList(c.style?.chat, 10);
    if (styleChat) parts.push(`Style: ${styleChat}.`);
  }

  if (c.system?.trim()) {
    parts.push(`Directive: ${truncate(c.system.trim(), 300)}`);
  }

  return parts.join(" ");
}

/**
 * Model ID for TEXT_PHASE. ScryptedAI's text endpoint is currently Nova Pro
 * only (see plugin-scryptedai/src/constants.ts ENDPOINTS.generations_text_nova_pro).
 * When scryptedai adds more text backends, swap this string (or make it a
 * setting) — limits are resolved from core's model-registry.json automatically.
 */
export const IMAGE_PROMPT_MODEL = "amazon.nova-pro-v1:0";

/**
 * PromptSet for TEXT_PHASE — system envelope + user template.
 * System is stable across runs; user embeds the per-run character digest.
 * Limits are resolved from the model registry via IMAGE_PROMPT_MODEL.
 */
export const imagePromptSet = new PromptSet({
  system:
    "You are an expert visual prompt engineer. Your job: read an AI agent's " +
    "character profile and design ONE vivid, concrete image-generation prompt " +
    "that personifies the agent as a visual avatar (portrait or symbolic figure). " +
    "Output ONLY the image prompt — no preamble, no explanation, no quotes, no " +
    "markdown. Be specific about visual style, mood, lighting, composition, and " +
    "medium. Keep it under 100 words. Do not include the agent's literal name.",
  user:
    "Agent profile:\n{{CHARACTER_DIGEST}}\n\n" +
    "Now write the image-generation prompt for this agent's avatar.",
  model: IMAGE_PROMPT_MODEL,
});

/** @deprecated Read `imagePromptSet.systemTemplate` instead. */
export const IMAGE_PROMPT_SYSTEM = imagePromptSet.systemTemplate;

/** @deprecated Use `imagePromptSet.render({ CHARACTER_DIGEST })` instead. */
export function buildImagePromptUser(characterDigest: string): string {
  return imagePromptSet.render({ CHARACTER_DIGEST: characterDigest }).user;
}

/**
 * Render the full scryptedai payload for TEXT_PHASE.
 *
 * @deprecated Prefer `toScryptedPayload(imagePromptSet.render({ CHARACTER_DIGEST }))`
 * directly; kept for existing callers and tests.
 */
export function buildImagePromptRequest(characterDigest: string): {
  system_prompt: string;
  user_prompt: string;
} {
  const rendered = imagePromptSet.render({ CHARACTER_DIGEST: characterDigest });
  const { system_prompt, user_prompt } = toScryptedPayload(rendered);
  return { system_prompt, user_prompt };
}
