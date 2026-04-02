/**
 * PromptSet — a reusable system/user prompt pair with tag replacement and
 * length sanitization.
 *
 * Decouples prompt-template management from callers: plugins define a
 * PromptSet once (with `{{TAG}}` placeholders), then call `.render(values)`
 * to get a concrete system/user payload trimmed to the target model's
 * token limits.
 *
 * Design goals:
 * - Zero runtime dependencies (browser-safe; no Handlebars, no tokenizer model calls)
 * - Deterministic (same input → same output; idempotent render)
 * - Defensive against adversarial/malformed input (fuzz-safe)
 * - Compatible with the `{{placeholder}}` convention used throughout ElizaOS
 */

import {
  DEFAULT_MODEL_LIMITS,
  getModelLimits,
  type ModelLimits,
} from "./model-registry";
import { estimateTokens } from "./utils";

// Re-export the registry types/constants so existing callers importing
// from @elizaos/core keep working without adding a new import path.
export {
  DEFAULT_MODEL_ID,
  DEFAULT_MODEL_LIMITS,
  getModelEntry,
  getModelLimits,
  hasModel,
  listModels,
  type ModelLimits,
  type ModelRegistryEntry,
} from "./model-registry";

/**
 * @deprecated Use `getModelLimits("amazon.nova-pro-v1:0")`.
 * Limits now come from the model registry (src/model-registry.json) so adding
 * a new model is a data change, not a code change.
 */
export const NOVA_PRO_LIMITS: ModelLimits = getModelLimits(
  "amazon.nova-pro-v1:0",
);

// ----------------------------------------------------------------------------
// Tag replacement
// ----------------------------------------------------------------------------

/**
 * Matches `{{TAG}}` placeholders. Tag names may contain letters, digits,
 * underscores, and surrounding whitespace (trimmed before lookup).
 * Non-matching inner content (e.g. `{{#if}}`, `{{!comment}}`) is left as-is.
 */
const TAG_PATTERN = /\{\{([^{}]+)\}\}/g;

/** A safe identifier: letters, digits, underscores, dots, hyphens. */
const SAFE_TAG_NAME = /^[A-Za-z0-9_.-]+$/;

/**
 * Hard ceiling on the rendered output length. Prevents multiplicative
 * expansion attacks (many tags × huge values → runtime string-length
 * overflow). Set well above the largest model context (300k tokens ≈
 * 1.2M chars) so legitimate prompts are never affected; sanitize()
 * trims further to the actual model budget.
 */
const RENDER_OUTPUT_CEILING_CHARS = 4_000_000;

/**
 * Replace `{{TAG}}` placeholders in `template` with values from `values`.
 *
 * - Tag lookup is case-sensitive after trimming inner whitespace.
 * - Missing tags are replaced with an empty string (consistent with
 *   composePrompt's behavior in core utils).
 * - Tags whose name is not a safe identifier are left unreplaced
 *   (avoids accidentally consuming Handlebars-style block helpers or
 *   attacker-supplied brace sequences inside values).
 * - Replacement is single-pass: values containing `{{...}}` are NOT
 *   re-expanded (prevents recursive-expansion attacks).
 * - Output is hard-capped at RENDER_OUTPUT_CEILING_CHARS to defeat
 *   multiplicative expansion (the sanitize step trims further to the
 *   model's actual budget).
 */
export function replaceTags(
  template: string,
  values: Readonly<Record<string, unknown>>,
): string {
  if (!template) return "";

  const parts: string[] = [];
  let outLen = 0;
  let lastIndex = 0;
  TAG_PATTERN.lastIndex = 0;

  const push = (s: string): boolean => {
    const remaining = RENDER_OUTPUT_CEILING_CHARS - outLen;
    if (remaining <= 0) return false;
    if (s.length > remaining) {
      parts.push(s.slice(0, remaining));
      outLen = RENDER_OUTPUT_CEILING_CHARS;
      return false;
    }
    parts.push(s);
    outLen += s.length;
    return true;
  };

  for (
    let match = TAG_PATTERN.exec(template);
    match !== null;
    match = TAG_PATTERN.exec(template)
  ) {
    // Literal text between previous match and this one
    if (match.index > lastIndex) {
      if (!push(template.slice(lastIndex, match.index))) break;
    }
    const key = match[1].trim();
    if (!SAFE_TAG_NAME.test(key)) {
      if (!push(match[0])) break;
    } else {
      const value = values[key];
      if (value !== undefined && value !== null) {
        if (!push(String(value))) break;
      }
      // undefined/null → replace with empty (push nothing)
    }
    lastIndex = TAG_PATTERN.lastIndex;
  }

  // Trailing literal text
  if (outLen < RENDER_OUTPUT_CEILING_CHARS && lastIndex < template.length) {
    push(template.slice(lastIndex));
  }

  return parts.join("");
}

// ----------------------------------------------------------------------------
// Length sanitization
// ----------------------------------------------------------------------------

/** Ellipsis appended to truncated text so callers can detect truncation. */
const TRUNCATION_MARK = "…";

/**
 * Truncate `text` to fit within `maxTokens` using the fast char/4 heuristic.
 * Cuts from the END (keeps the prefix — system prompts and instruction
 * headers are typically front-loaded). Guarantees the result never exceeds
 * `maxTokens` by estimate.
 */
function truncateToTokenBudget(text: string, maxTokens: number): string {
  if (maxTokens <= 0) return "";
  if (estimateTokens(text) <= maxTokens) return text;

  // chars ≈ tokens * 4; leave one char for the ellipsis.
  const maxChars = Math.max(1, maxTokens * 4 - TRUNCATION_MARK.length);
  const sliced = text.slice(0, maxChars);

  // Try to cut at a word boundary within the last 10% of the window so we
  // don't bisect a word. If no boundary is found, hard-cut.
  const searchStart = Math.floor(maxChars * 0.9);
  const boundary = sliced.lastIndexOf(" ", maxChars);
  const cut = boundary > searchStart ? boundary : maxChars;

  return sliced.slice(0, cut) + TRUNCATION_MARK;
}

// ----------------------------------------------------------------------------
// RenderedPromptSet — the concrete output
// ----------------------------------------------------------------------------

/** A fully-rendered system/user prompt pair, ready to send to a model. */
export interface RenderedPromptSet {
  /** The system prompt (role/instruction envelope). */
  readonly system: string;
  /** The user prompt (per-request data). */
  readonly user: string;
  /** Model limits in effect for this render (client-side input budgeting only). */
  readonly limits: ModelLimits;
  /** True if sanitization truncated either prompt to fit the input budget. */
  readonly truncated: boolean;
}

// ----------------------------------------------------------------------------
// PromptSet
// ----------------------------------------------------------------------------

export interface PromptSetOptions {
  /** System prompt template (may contain `{{TAG}}` placeholders). */
  system: string;
  /** User prompt template (may contain `{{TAG}}` placeholders). */
  user: string;
  /**
   * Model ID (exact registry key, e.g. "amazon.nova-pro-v1:0"). Limits
   * are looked up from the model registry. Ignored if `limits` is set.
   */
  model?: string;
  /**
   * Explicit token limits. Overrides `model`. Defaults to the registry's
   * default model when neither `model` nor `limits` is given.
   */
  limits?: ModelLimits;
  /**
   * Fraction of the input budget reserved for the system prompt when
   * combined prompts overflow and both need trimming. Default 0.25
   * (system prompts are usually short instruction envelopes; user prompts
   * carry the bulk of per-request data).
   */
  systemReserveFraction?: number;
}

/**
 * A paired system/user prompt template with automatic tag replacement and
 * length sanitization.
 *
 * @example
 * const prompts = new PromptSet({
 *   system: "You are a {{ROLE}}. Be concise.",
 *   user: "Context: {{CONTEXT}}\n\nTask: {{TASK}}",
 * });
 *
 * const { system, user } = prompts.render({
 *   ROLE: "visual prompt engineer",
 *   CONTEXT: characterDigest,
 *   TASK: "Write an image prompt.",
 * });
 */
export class PromptSet {
  public readonly systemTemplate: string;
  public readonly userTemplate: string;
  public readonly limits: ModelLimits;
  private readonly systemReserveFraction: number;

  constructor(opts: PromptSetOptions) {
    this.systemTemplate = String(opts.system ?? "");
    this.userTemplate = String(opts.user ?? "");

    // Resolution order: explicit limits > model ID lookup > registry default.
    const limits =
      opts.limits ??
      (opts.model ? getModelLimits(opts.model) : DEFAULT_MODEL_LIMITS);
    this.limits = {
      maxInputTokens: clampPositive(limits.maxInputTokens, 1),
      maxOutputTokens: clampPositive(limits.maxOutputTokens, 1),
    };

    this.systemReserveFraction = clampFraction(
      opts.systemReserveFraction ?? 0.25,
    );
  }

  /**
   * Render the templates with the given tag values, then sanitize the
   * combined length to fit within `limits.maxInputTokens`.
   *
   * Missing tags resolve to empty strings. Calling `render` with no
   * arguments is equivalent to `render({})`.
   */
  render(values: Readonly<Record<string, unknown>> = {}): RenderedPromptSet {
    const rawSystem = replaceTags(this.systemTemplate, values);
    const rawUser = replaceTags(this.userTemplate, values);
    return this.sanitize(rawSystem, rawUser);
  }

  /**
   * Length-sanitize a concrete system/user pair against this set's limits.
   * Exposed so callers who build prompts dynamically (no templates) can
   * still benefit from the trimming logic.
   */
  sanitize(system: string, user: string): RenderedPromptSet {
    const sys = system ?? "";
    const usr = user ?? "";
    const budget = this.limits.maxInputTokens;

    const sysTokens = estimateTokens(sys);
    const usrTokens = estimateTokens(usr);

    // Fast path: already within budget.
    if (sysTokens + usrTokens <= budget) {
      return Object.freeze({
        system: sys,
        user: usr,
        limits: this.limits,
        truncated: false,
      });
    }

    // Overflow: split the budget. System prompt gets its reserved fraction
    // (or less if it's already shorter), user prompt gets the rest.
    const sysBudget = Math.min(
      sysTokens,
      Math.max(1, Math.floor(budget * this.systemReserveFraction)),
    );
    const usrBudget = Math.max(1, budget - sysBudget);

    const trimmedSystem = truncateToTokenBudget(sys, sysBudget);
    const trimmedUser = truncateToTokenBudget(usr, usrBudget);

    return Object.freeze({
      system: trimmedSystem,
      user: trimmedUser,
      limits: this.limits,
      truncated: true,
    });
  }

  /** List all `{{TAG}}` placeholders present in either template. */
  getTags(): string[] {
    const tags = new Set<string>();
    const collect = (tpl: string) => {
      for (const m of tpl.matchAll(TAG_PATTERN)) {
        const key = m[1].trim();
        if (SAFE_TAG_NAME.test(key)) tags.add(key);
      }
    };
    collect(this.systemTemplate);
    collect(this.userTemplate);
    return [...tags];
  }

  /** Return a new PromptSet using different model limits. */
  withLimits(limits: ModelLimits): PromptSet {
    return new PromptSet({
      system: this.systemTemplate,
      user: this.userTemplate,
      limits,
      systemReserveFraction: this.systemReserveFraction,
    });
  }

  /** Return a new PromptSet targeting a different model (registry lookup). */
  withModel(model: string): PromptSet {
    return this.withLimits(getModelLimits(model));
  }
}

/**
 * Adapter: convert a rendered set to ScryptedAI's snake_case wire shape.
 * Kept separate from the class so `PromptSet` itself stays provider-agnostic.
 *
 * ## Required wire fields (verified live against /generations/text/nova-pro, 2026-04)
 *
 * `auto_calculate_tokens: false` and an explicit `max_tokens` are BOTH
 * mandatory. The recipe schema advertises `max_tokens` default 2000 and
 * `auto_calculate_tokens` default true, but the observed behaviour is:
 *
 *   request body                                  → metadata.max_tokens_requested  finish_reason
 *   ─────────────────────────────────────────────   ─────────────────────────────  ─────────────
 *   { max_tokens: 10000 }                          → 100                            max_tokens
 *   { }                              (pure default) → 100                            max_tokens
 *   { auto_calculate_tokens: false }               → 100                            max_tokens
 *   { max_tokens: 2000, auto_calculate_tokens:false } → 2000                         end_turn ✓
 *   { max_tokens: 9000, auto_calculate_tokens:false } → 9000                         end_turn ✓
 *
 * i.e. auto-calc silently overrides any caller value with 100, and with
 * auto-calc disabled there is no real server-side default. We therefore:
 *   - always send `auto_calculate_tokens: false`
 *   - always send `max_tokens` = the model's registry `maxOutputTokens`
 *     (9000 for nova-pro — see model-registry.json)
 *
 * PromptSet limits remain primarily for client-side INPUT budgeting (we
 * pre-estimate and truncate so the prompt is guaranteed to fit); the output
 * ceiling is forwarded only because the server default is unusable.
 */
export function toScryptedPayload(r: RenderedPromptSet): {
  system_prompt: string;
  user_prompt: string;
  max_tokens: number;
  auto_calculate_tokens: false;
} {
  return {
    system_prompt: r.system,
    user_prompt: r.user,
    max_tokens: r.limits.maxOutputTokens,
    auto_calculate_tokens: false,
  };
}

// ----------------------------------------------------------------------------
// Internal helpers
// ----------------------------------------------------------------------------

function clampPositive(n: number, floor: number): number {
  if (!Number.isFinite(n) || n < floor) return floor;
  if (n > Number.MAX_SAFE_INTEGER) return Number.MAX_SAFE_INTEGER;
  return Math.floor(n);
}

function clampFraction(f: number): number {
  if (!Number.isFinite(f)) return 0.25;
  if (f < 0) return 0;
  if (f > 1) return 1;
  return f;
}
