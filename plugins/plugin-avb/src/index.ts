/**
 * @elizaos/plugin-avb
 *
 * Autonomous Virtual Being pipeline for ElizaOS.
 *
 * Provides:
 * - GENERATE_AVATAR action: user-facing trigger (returns immediately)
 * - AvbService: db-persisted state machine coordinating text→image→deliver
 * - Task workers: idempotent phase executors driven by core TaskService
 *
 * Pipeline:
 *   createRun() → TEXT_PHASE (introspect character, draft image prompt)
 *              → IMAGE_PHASE (scryptedai image gen)
 *              → DELIVER (post image to room, emit MESSAGE_SENT)
 *
 * Fault-tolerance:
 * - Every phase is a persisted Task row; survives restart
 * - Workers are idempotent (safe to re-tick)
 * - resumePolling() re-arms scryptedai after restart
 * - Per-phase deadline guards against jobs that never return
 * - Webhook fast-path eager-executes on terminal (no 5s tick wait)
 *
 * Dependencies:
 * - @elizaos/plugin-scryptedai (provider service for actual generation)
 */

import type { Plugin } from "@elizaos/core";
import { generateAvatarAction } from "./action.ts";
import { AvbService } from "./service.ts";
import { Slm16Service } from "./slm16/index.ts";

// ----------------------------------------------------------------------------
// Plugin definition
// ----------------------------------------------------------------------------

export const avbPlugin: Plugin = {
  name: "avb",
  description:
    "Autonomous Virtual Being pipeline — character introspection → avatar " +
    "generation via db-persisted state machine. Calls scryptedai for actual " +
    "generation. Includes SLM16: a background trainer for a 16MB language " +
    "model (OpenAI Parameter Golf 9×512 U-Net GPT) with Nova Pro intelligence scoring.",
  dependencies: ["scryptedai"],
  services: [AvbService, Slm16Service],
  actions: [generateAvatarAction],
};

export default avbPlugin;

// ----------------------------------------------------------------------------
// Public API re-exports
// ----------------------------------------------------------------------------

// Action
export { generateAvatarAction } from "./action.ts";
// Constants
export {
  AVB_SERVICE_TYPE,
  BASE_TAGS,
  DEFAULT_IMAGE_METHOD,
  ENV_AVB_AUTOGEN_ON_BOOT,
  ENV_AVB_IMAGE_METHOD,
  PHASE_TICK_INTERVAL_MS,
  PIPELINE,
  tagForJob,
  tagForRun,
  WORKER_NAMES,
} from "./constants.ts";

// Character introspection (reusable independently)
export {
  buildImagePromptRequest,
  buildImagePromptUser,
  type DigestableCharacter,
  digestCharacter,
  IMAGE_PROMPT_MODEL,
  IMAGE_PROMPT_SYSTEM,
  imagePromptSet,
} from "./introspect.ts";
// Service
export { AvbService } from "./service.ts";

// Types
export type {
  AvbPhaseMetadata,
  AvbRunContext,
  AvbRuntimeSurface,
  PhaseCompleteReport,
  PhaseFailedReport,
  PhaseName,
  PhaseSpec,
} from "./types.ts";

// SLM16 — background trainer + intelligence scoring
export {
  ARCH as SLM16_ARCH,
  QUANT as SLM16_QUANT,
  SCORING as SLM16_SCORING,
  SLM16_ENV,
  SLM16_PATHS,
  SLM16_SERVICE_TYPE,
  SLM16_SETTING,
  Slm16Service,
  TRAIN_DEFAULTS as SLM16_TRAIN_DEFAULTS,
} from "./slm16/index.ts";
export type {
  Slm16Event,
  Slm16Meta,
  Slm16ScoringResult,
  Slm16TrainingConfig,
} from "./slm16/index.ts";
