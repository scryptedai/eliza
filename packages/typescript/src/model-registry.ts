/**
 * Model registry — central lookup for per-model token limits.
 *
 * The registry is the *single source of truth* for model limits. Adding
 * a new model is a data change (one entry in model-registry.json), not
 * a code change. PromptSet and downstream consumers look up by model ID;
 * they never hardcode numbers.
 *
 * Data lives in model-registry.json so it's trivially editable/diffable
 * and can be loaded/overridden at runtime if needed later. This module
 * wraps it with a typed lookup surface.
 *
 * Model IDs are exact-match only — no aliases. Versioned IDs like
 * "amazon.nova-pro-v1:0" vs "amazon.nova-pro-v2:0" must stay distinct;
 * short names would collide across versions/modalities.
 *
 * Schema:
 *   {
 *     "default": "<model-id>",
 *     "models": {
 *       "<model-id>": {
 *         "maxInputTokens": <number>,
 *         "maxOutputTokens": <number>,
 *         "provider": "<string>",   // optional, informational
 *         "notes": "<string>"       // optional, e.g. doc URL
 *       }
 *     }
 *   }
 */

import registryData from "./model-registry.json";

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------

/** Per-model token limits (prompt budget + completion ceiling). */
export interface ModelLimits {
  /** Max input tokens (system + user combined). */
  maxInputTokens: number;
  /** Max output tokens (completion length ceiling). */
  maxOutputTokens: number;
}

/** Full registry entry. Only the limits are load-bearing; rest is metadata. */
export interface ModelRegistryEntry extends ModelLimits {
  /** Hosting provider (informational only). */
  provider?: string;
  /** Free-form notes — typically a doc URL for limit provenance. */
  notes?: string;
}

interface RegistryShape {
  default: string;
  models: Record<string, ModelRegistryEntry>;
}

// ----------------------------------------------------------------------------
// Registry data
// ----------------------------------------------------------------------------

const REGISTRY: RegistryShape = registryData as RegistryShape;

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------

/** ID of the default model (used when no model is specified). */
export const DEFAULT_MODEL_ID: string = REGISTRY.default;

/**
 * True if `modelId` is a known registry entry. Exact-match — no aliases.
 */
export function hasModel(modelId: string): boolean {
  return Object.hasOwn(REGISTRY.models, modelId);
}

/**
 * Look up token limits for a model by exact ID.
 * Falls back to the registry's default model if the ID is unknown
 * (so callers always get usable limits — unknown models are a soft
 * miss rather than a hard break in the prompt pipeline).
 */
export function getModelLimits(modelId?: string): ModelLimits {
  const entry =
    modelId !== undefined && Object.hasOwn(REGISTRY.models, modelId)
      ? REGISTRY.models[modelId]
      : REGISTRY.models[DEFAULT_MODEL_ID];
  return {
    maxInputTokens: entry.maxInputTokens,
    maxOutputTokens: entry.maxOutputTokens,
  };
}

/**
 * Fetch the full entry (limits + metadata). Returns `undefined` for
 * unknown IDs — use this when you need to distinguish "unknown model"
 * from "default fallback".
 */
export function getModelEntry(modelId: string): ModelRegistryEntry | undefined {
  return Object.hasOwn(REGISTRY.models, modelId)
    ? REGISTRY.models[modelId]
    : undefined;
}

/** List all model IDs in the registry. */
export function listModels(): string[] {
  return Object.keys(REGISTRY.models);
}

/** The default model's limits. Shorthand for getModelLimits(DEFAULT_MODEL_ID). */
export const DEFAULT_MODEL_LIMITS: ModelLimits = getModelLimits();
