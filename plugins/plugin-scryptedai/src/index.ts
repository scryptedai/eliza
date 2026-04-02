/**
 * @elizaos/plugin-scryptedai
 *
 * ElizaOS plugin for ScryptedAI (api.scrypted.ai) multimodal generation.
 *
 * Provides:
 * - ScryptedClient: HTTP client with per-endpoint invoke methods (images/video/text)
 * - ScryptedAIService: job lifecycle tracking, webhook ingestion, polling fallback
 * - Webhook route: mounted at /scryptedai/webhook with HMAC-SHA256 verification
 * - Response adapter: normalizes variant API shapes into a single canonical form
 *
 * Environment variables (read via runtime.getSetting):
 * - SCRYPTEDAI_BEARER_TOKEN (required) — bearer token, must start with `scrypted_`
 * - SCRYPTEDAI_WEBHOOK_SECRET (optional) — default HMAC secret for webhook verification
 * - SCRYPTEDAI_BASE_URL (optional) — override API base URL
 */

import type { Plugin } from "@elizaos/core";
import { scryptedaiModelHandlers } from "./models.ts";
import { ScryptedAIService } from "./service.ts";
import { scryptedaiWebhookRoute } from "./webhook.ts";

// ----------------------------------------------------------------------------
// Plugin definition
// ----------------------------------------------------------------------------

export const scryptedaiPlugin: Plugin = {
  name: "scryptedai",
  description:
    "ScryptedAI multimodal generation provider — images, videos, and text via api.scrypted.ai. " +
    "Handles async job lifecycle with webhook callbacks and polling fallback.",
  services: [ScryptedAIService],
  routes: [scryptedaiWebhookRoute],
  // Model handlers: invoked via runtime.useModel(ModelType.TEXT_LARGE|IMAGE, ...)
  // Cast: core's Plugin.models type depends on generated proto types; handlers
  // are structurally compatible (see models.ts).
  models: scryptedaiModelHandlers as unknown as Plugin["models"],
};

export default scryptedaiPlugin;

// ----------------------------------------------------------------------------
// Public API — re-export everything callers may need
// ----------------------------------------------------------------------------

// Response normalization adapter
export {
  extractAssetUrl,
  extractError,
  extractImageUrl,
  extractText,
  extractVideoUrl,
  isTerminalStatus,
  mergeResult,
  normalizeJobPayload,
  normalizeJobStatusResponse,
  normalizeStatus,
} from "./adapter.ts";

// Client
export {
  generateIdempotencyKey,
  retryWithBackoff,
  ScryptedClient,
  type ScryptedClientOptions,
  validateBearerToken,
} from "./client.ts";
// Constants
export {
  API_BASE_URL,
  DEFAULT_IMAGE_MODEL,
  ENDPOINTS,
  ENV_BASE_URL,
  ENV_BEARER_TOKEN,
  ENV_IMAGE_MODEL,
  ENV_WEBHOOK_SECRET,
  type ImageModelName,
  type JobType,
  POLLING_WINDOWS,
  type PollingWindow,
  SCRYPTEDAI_SERVICE_TYPE,
} from "./constants.ts";
// Exceptions
export {
  ScryptedAPIError,
  ScryptedAuthenticationError,
  ScryptedError,
  ScryptedNetworkError,
  ScryptedPaymentError,
  ScryptedRateLimitError,
  ScryptedTimeoutError,
  ScryptedValidationError,
} from "./exceptions.ts";
// Model handlers (invoked via runtime.useModel)
export {
  handleImage,
  handleTextLarge,
  scryptedaiModelHandlers,
} from "./models.ts";
// Standalone polling utility (reusable independently)
export {
  type PollOptions,
  type PollResult,
  pollJobToCompletion,
} from "./polling.ts";
// Service
export {
  type IngestResult,
  ScryptedAIService,
  type StartJobResult,
} from "./service.ts";
// Types
export type {
  AccountResponse,
  InvokeOptions,
  JobRecord,
  JobStatusResponse,
  JobTerminalListener,
  NewAccountResponse,
  NormalizedJobResult,
  NormalizedStatus,
  RecipeExecutionResponse,
  RecipeInterface,
} from "./types.ts";
// Standalone webhook utilities (reusable independently)
export {
  type ParsedWebhookPayload,
  parseWebhookPayload,
  scryptedaiWebhookRoute,
  verifyWebhookSignature,
} from "./webhook.ts";
