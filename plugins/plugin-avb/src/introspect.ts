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
 */

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
 * System prompt for TEXT_PHASE — defines the model's role and output
 * contract. Stable across runs; does not embed character data.
 */
export const IMAGE_PROMPT_SYSTEM =
  "You are an expert visual prompt engineer. Your job: read an AI agent's " +
  "character profile and design ONE vivid, concrete image-generation prompt " +
  "that personifies the agent as a visual avatar (portrait or symbolic figure). " +
  "Output ONLY the image prompt — no preamble, no explanation, no quotes, no " +
  "markdown. Be specific about visual style, mood, lighting, composition, and " +
  "medium. Keep it under 100 words. Do not include the agent's literal name.";

/**
 * User prompt for TEXT_PHASE — the per-run character data.
 * Pairs with IMAGE_PROMPT_SYSTEM.
 */
export function buildImagePromptUser(characterDigest: string): string {
  return `Agent profile:\n${characterDigest}\n\nNow write the image-generation prompt for this agent's avatar.`;
}

/**
 * Convenience: produce the full { system_prompt, user_prompt } payload
 * for scryptedai's nova-pro text endpoint.
 */
export function buildImagePromptRequest(characterDigest: string): {
  system_prompt: string;
  user_prompt: string;
} {
  return {
    system_prompt: IMAGE_PROMPT_SYSTEM,
    user_prompt: buildImagePromptUser(characterDigest),
  };
}
