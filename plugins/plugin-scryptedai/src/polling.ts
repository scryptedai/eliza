/**
 * Standalone polling utility for ScryptedAI jobs.
 *
 * Exported for direct use independently of the ScryptedAIService — useful for
 * scripts, tests, or integrations that want blocking "await completion"
 * semantics without the full service lifecycle.
 *
 * Per integration guide §4.3:
 * - Type-based min/max wait windows (text ~1min, image ~5min, video ~30min)
 * - Backoff between polls (last interval repeats)
 * - 502/503/504 and gateway-style errors are TRANSIENT — retry, don't fail
 */

import { isTerminalStatus, normalizeJobStatusResponse } from "./adapter.ts";
import type { ScryptedClient } from "./client.ts";
import {
  type JobType,
  POLLING_WINDOWS,
  TRANSIENT_GATEWAY_STATUS,
} from "./constants.ts";
import {
  ScryptedAPIError,
  ScryptedNetworkError,
  ScryptedTimeoutError,
} from "./exceptions.ts";
import type { NormalizedJobResult } from "./types.ts";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export interface PollOptions {
  /** Job type — determines polling window (text/image/video/unknown). */
  jobType?: JobType;
  /** Override max wait in seconds. */
  maxWaitSeconds?: number;
  /** Override polling intervals (seconds). */
  intervalsSeconds?: number[];
  /** AbortSignal for early cancellation. */
  signal?: AbortSignal;
  /** Callback invoked after each poll with the current (possibly non-terminal) result. */
  onPoll?: (result: NormalizedJobResult) => void;
}

export interface PollResult {
  /** Final normalized result (terminal or timeout placeholder). */
  result: NormalizedJobResult;
  /** True if polling completed normally (terminal status reached). */
  completed: boolean;
  /** True if maxWait was exceeded before a terminal status. */
  timedOut: boolean;
  /** Number of poll attempts made. */
  attempts: number;
}

/**
 * Poll a ScryptedAI job until it reaches a terminal state or the max-wait
 * window elapses.
 *
 * Transient gateway errors (502/503/504) and network/timeout errors are
 * absorbed and retried — they do NOT cause the poll to fail or the job to
 * be marked failed.
 */
export async function pollJobToCompletion(
  client: ScryptedClient,
  jobId: string,
  opts: PollOptions = {},
): Promise<PollResult> {
  const window = POLLING_WINDOWS[opts.jobType ?? "unknown"];
  const maxWaitMs = (opts.maxWaitSeconds ?? window.maxWaitSeconds) * 1000;
  const intervals = (opts.intervalsSeconds ?? window.intervalsSeconds).map(
    (s) => s * 1000,
  );
  // Cap minAge at maxWait so we always make at least one poll attempt
  const minAgeMs = Math.min(window.minAgeSeconds * 1000, maxWaitMs / 2);

  const startedAt = Date.now();
  let attempts = 0;
  let intervalIdx = 0;
  let lastResult: NormalizedJobResult = {
    jobId,
    status: "pending",
  };

  // Initial delay before the first poll (respect minimum age)
  if (minAgeMs > 0) {
    await sleep(minAgeMs);
    if (opts.signal?.aborted) {
      return {
        result: lastResult,
        completed: false,
        timedOut: false,
        attempts,
      };
    }
  }

  while (Date.now() - startedAt < maxWaitMs) {
    if (opts.signal?.aborted) {
      return {
        result: lastResult,
        completed: false,
        timedOut: false,
        attempts,
      };
    }

    attempts++;
    try {
      const raw = await client.getJobStatus(jobId);
      lastResult = normalizeJobStatusResponse(raw);
      opts.onPoll?.(lastResult);

      if (isTerminalStatus(lastResult.status)) {
        return {
          result: lastResult,
          completed: true,
          timedOut: false,
          attempts,
        };
      }
    } catch (error) {
      // Transient gateway failures: treat as "still processing", retry later
      const isTransient =
        error instanceof ScryptedAPIError &&
        TRANSIENT_GATEWAY_STATUS.has(error.statusCode);

      // Also treat network/timeout errors as transient (guide §6.1)
      const isNetworkLike =
        error instanceof ScryptedNetworkError ||
        error instanceof ScryptedTimeoutError;

      if (!isTransient && !isNetworkLike) {
        // Non-transient error (auth, validation, 4xx other than 429) → rethrow
        throw error;
      }
      // Otherwise: absorb and continue to next poll cycle
    }

    // Sleep using backoff intervals; last interval repeats
    const waitMs = intervals[Math.min(intervalIdx, intervals.length - 1)];
    intervalIdx++;
    await sleep(waitMs);
  }

  // Max wait exceeded
  return {
    result: lastResult,
    completed: false,
    timedOut: true,
    attempts,
  };
}
