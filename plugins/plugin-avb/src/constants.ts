/**
 * Constants and configuration for the AVB pipeline.
 *
 * The PIPELINE graph is the single source of truth for phase ordering
 * and timing. AvbService.onPhaseComplete() consults it to decide what
 * Task to spawn next.
 */

import type { PhaseName, PhaseSpec } from "./types.ts";

// ----------------------------------------------------------------------------
// Service / worker identity
// ----------------------------------------------------------------------------

export const AVB_SERVICE_TYPE = "avb" as const;

/** TaskWorker names (must be unique across all plugins). */
export const WORKER_NAMES = {
  TEXT_PHASE: "AVB_TEXT_PHASE",
  IMAGE_PHASE: "AVB_IMAGE_PHASE",
  DELIVER: "AVB_DELIVER",
} as const;

// ----------------------------------------------------------------------------
// Task tags
// ----------------------------------------------------------------------------

/** Base tags on every AVB phase task. */
export const BASE_TAGS = ["queue", "repeat", "avb"] as const;

/** Tag prefix for run-scoped lookup. */
export const tagForRun = (runId: string): string => `avb:run:${runId}`;

/** Tag prefix for jobId → task reverse lookup (webhook fast-path). */
export const tagForJob = (jobId: string): string => `avb:job:${jobId}`;

// ----------------------------------------------------------------------------
// Pipeline graph
//
// Deadlines track scryptedai's polling windows (constants.ts: 60s text,
// 300s image) with small headroom. DELIVER is local-only → short deadline.
// ----------------------------------------------------------------------------

export const PIPELINE: Record<PhaseName, PhaseSpec> = {
  TEXT_PHASE: {
    workerName: WORKER_NAMES.TEXT_PHASE,
    jobType: "text",
    deadlineMs: 90_000, // 60s scryptedai window + 30s headroom
    next: "IMAGE_PHASE",
  },
  IMAGE_PHASE: {
    workerName: WORKER_NAMES.IMAGE_PHASE,
    jobType: "image",
    deadlineMs: 360_000, // 300s scryptedai window + 60s headroom
    next: "DELIVER",
  },
  DELIVER: {
    workerName: WORKER_NAMES.DELIVER,
    jobType: "unknown",
    deadlineMs: 15_000,
    next: null,
  },
};

// ----------------------------------------------------------------------------
// Worker tick interval
//
// TaskService ticks every 1s; setting updateInterval lets us space out
// re-executions. 5s balances latency vs. churn. Webhook fast-path bypasses
// this entirely by eager-executing on terminal events.
// ----------------------------------------------------------------------------

export const PHASE_TICK_INTERVAL_MS = 5_000;

/**
 * FIFO cap on AvbService.handledTasks (in-memory idempotency set guarding
 * the eager-execute / TaskService-tick race). Each entry corresponds to one
 * completed phase task. 256 entries comfortably exceeds any plausible race
 * window (a few ticks × a handful of concurrent runs) while keeping the set
 * bounded for long-lived agents.
 */
export const HANDLED_TASKS_CAP = 256;

// ----------------------------------------------------------------------------
// Environment variables
// ----------------------------------------------------------------------------

/** Optional override for the scryptedai image method used in IMAGE_PHASE. */
export const ENV_AVB_IMAGE_METHOD = "AVB_IMAGE_METHOD";

/**
 * Autonomous avatar generation on boot. Defaults to enabled: if no avatar
 * exists and no run is in flight, AvbService.start() will self-trigger
 * createRun() targeting the agent's own room. Set to "false" to opt out.
 */
export const ENV_AVB_AUTOGEN_ON_BOOT = "AVB_AUTOGEN_ON_BOOT";

/**
 * ScryptedAIService.startImageGeneration method names that AVB may select
 * via AVB_IMAGE_METHOD. This list is type-checked against the scryptedai
 * service signature (see ImageMethodName below) — adding an invalid name
 * here is a compile error. If scryptedai adds a new image method and you
 * want AVB to support it, add it to this array.
 */
export const IMAGE_METHODS = [
  "invokeImageGeneration",
  "invokeNanoBananaGeneration",
  "invokeNanoBananaProGeneration",
  "invokeNanoBananaEditGeneration",
  "invokeNanoBananaProEditGeneration",
  "invokeSeedream4Generation",
  "invokeFlux2ProGeneration",
  "invokeGrokImagineImageGeneration",
] as const;

/** Union of valid AVB_IMAGE_METHOD values. */
export type ImageMethodName = (typeof IMAGE_METHODS)[number];

/** Runtime check: is `value` a valid AVB_IMAGE_METHOD? */
export function isImageMethodName(value: unknown): value is ImageMethodName {
  return (
    typeof value === "string" &&
    (IMAGE_METHODS as readonly string[]).includes(value)
  );
}

/** Default scryptedai client method for image generation (Seedream 4). */
export const DEFAULT_IMAGE_METHOD: ImageMethodName = "invokeSeedream4Generation";
