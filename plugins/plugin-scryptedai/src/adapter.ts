/**
 * Response normalization adapter for ScryptedAI payloads.
 *
 * Per integration guide §5:
 * - Status may arrive in any casing (`FAILED`, `Completed`, etc.) → always lowercase
 * - Result may live at `result`, `result_data`, or `output` → merge into one
 * - Error may be a string, an object with `.message`, or `error_message` → single string
 * - Image/video URLs have multiple variant shapes → priority-ordered extraction
 *
 * This adapter is pure — no I/O, no logging. Used by both the webhook handler
 * and the polling loop so the rest of the plugin sees exactly one shape.
 */

import type {
  JobStatusResponse,
  NormalizedJobResult,
  NormalizedStatus,
} from "./types.ts";

// ----------------------------------------------------------------------------
// Status normalization
// ----------------------------------------------------------------------------

const TERMINAL_STATUSES = new Set<NormalizedStatus>([
  "completed",
  "failed",
  "cancelled",
]);

/** Normalize an arbitrary status string to a known lowercase enum value. */
export function normalizeStatus(raw: unknown): NormalizedStatus {
  const s = String(raw ?? "")
    .toLowerCase()
    .trim();
  switch (s) {
    case "completed":
    case "complete":
    case "success":
    case "succeeded":
      return "completed";
    case "failed":
    case "failure":
    case "error":
      return "failed";
    case "cancelled":
    case "canceled":
      return "cancelled";
    case "processing":
    case "running":
    case "in_progress":
      return "processing";
    default:
      return "pending";
  }
}

export function isTerminalStatus(status: NormalizedStatus): boolean {
  return TERMINAL_STATUSES.has(status);
}

// ----------------------------------------------------------------------------
// Error extraction
// ----------------------------------------------------------------------------

/**
 * Collapse variant error shapes to a single string.
 * Priority: error_message → error.message → error (string) → undefined.
 */
export function extractError(
  payload: Record<string, unknown>,
): string | undefined {
  if (typeof payload.error_message === "string" && payload.error_message) {
    return payload.error_message;
  }

  const err = payload.error;
  if (err == null) return undefined;

  if (typeof err === "string") return err || undefined;

  if (typeof err === "object") {
    const msg = (err as Record<string, unknown>).message;
    if (typeof msg === "string" && msg) return msg;
    // Fall back to stringify for opaque error objects
    try {
      return JSON.stringify(err);
    } catch {
      return String(err);
    }
  }

  return String(err);
}

// ----------------------------------------------------------------------------
// Result merging (result_data / result / output → one object)
// ----------------------------------------------------------------------------

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * Merge variant result containers into a single object.
 * Precedence: result_data > result > output (later sources do not overwrite earlier).
 */
export function mergeResult(
  payload: Record<string, unknown>,
): Record<string, unknown> | undefined {
  const containers = [payload.result_data, payload.result, payload.output];
  let merged: Record<string, unknown> | undefined;

  for (const container of containers) {
    if (!isRecord(container)) continue;
    if (!merged) {
      merged = { ...container };
    } else {
      for (const [k, v] of Object.entries(container)) {
        if (!(k in merged)) merged[k] = v;
      }
    }
  }

  return merged;
}

// ----------------------------------------------------------------------------
// Media URL extraction
// ----------------------------------------------------------------------------

/** Keys to probe on an image/video item for the asset URL, in priority order. */
const URL_FIELD_PRIORITY = [
  "asset_url",
  "cdn_url",
  "cloudfront_url",
  "url",
] as const;

function probeUrl(item: unknown): string | undefined {
  if (typeof item === "string") return item || undefined;
  if (!isRecord(item)) return undefined;
  for (const key of URL_FIELD_PRIORITY) {
    const v = item[key];
    if (typeof v === "string" && v.length > 0) return v;
  }
  return undefined;
}

/**
 * Extract the first image URL from a normalized-or-raw payload.
 *
 * Checks in priority order (per guide §5.2):
 * 1. result_data.images[0].{asset_url|cdn_url|cloudfront_url|url}
 * 2. result.images[0].{...}
 * 3. Legacy: result.image.url, result.image_url, result.imageUrl
 */
export function extractImageUrl(
  payload: Record<string, unknown>,
): string | undefined {
  // Check result_data and result for images[] array
  for (const containerKey of ["result_data", "result", "output"] as const) {
    const container = payload[containerKey];
    if (!isRecord(container)) continue;

    const images = container.images;
    if (Array.isArray(images) && images.length > 0) {
      const url = probeUrl(images[0]);
      if (url) return url;
    }
  }

  // Legacy shapes on result.*
  const result = payload.result;
  if (isRecord(result)) {
    // result.image.url
    if (isRecord(result.image)) {
      const url = probeUrl(result.image);
      if (url) return url;
    }
    // result.image_url
    if (typeof result.image_url === "string" && result.image_url) {
      return result.image_url;
    }
    // result.imageUrl
    if (typeof result.imageUrl === "string" && result.imageUrl) {
      return result.imageUrl;
    }
  }

  // Also try top-level merged result (e.g. if caller passed pre-merged payload)
  const images = payload.images;
  if (Array.isArray(images) && images.length > 0) {
    const url = probeUrl(images[0]);
    if (url) return url;
  }

  return undefined;
}

/**
 * Extract the first video URL from a normalized-or-raw payload.
 *
 * Checks in priority order (per guide §5.2):
 * 1. result_data.video (singular) or result_data.videos[0]
 * 2. result.video, result.videos[0]
 * 3. Legacy: result.video_url, result.videoUrl
 */
export function extractVideoUrl(
  payload: Record<string, unknown>,
): string | undefined {
  for (const containerKey of ["result_data", "result", "output"] as const) {
    const container = payload[containerKey];
    if (!isRecord(container)) continue;

    // Singular .video
    const video = container.video;
    const singularUrl = probeUrl(video);
    if (singularUrl) return singularUrl;

    // Plural .videos[0]
    const videos = container.videos;
    if (Array.isArray(videos) && videos.length > 0) {
      const url = probeUrl(videos[0]);
      if (url) return url;
    }
  }

  // Legacy shapes on result.*
  const result = payload.result;
  if (isRecord(result)) {
    if (typeof result.video_url === "string" && result.video_url) {
      return result.video_url;
    }
    if (typeof result.videoUrl === "string" && result.videoUrl) {
      return result.videoUrl;
    }
  }

  // Also try top-level (e.g. pre-merged)
  const topVideo = probeUrl(payload.video);
  if (topVideo) return topVideo;
  const topVideos = payload.videos;
  if (Array.isArray(topVideos) && topVideos.length > 0) {
    const url = probeUrl(topVideos[0]);
    if (url) return url;
  }

  return undefined;
}

/**
 * Extract text from a normalized-or-raw payload.
 * Per guide §5.2: often `result.text`, `result_data.text`, or nested under `output`.
 */
export function extractText(
  payload: Record<string, unknown>,
): string | undefined {
  for (const containerKey of ["result_data", "result", "output"] as const) {
    const container = payload[containerKey];
    if (!isRecord(container)) continue;
    if (typeof container.text === "string" && container.text) {
      return container.text;
    }
  }
  if (typeof payload.text === "string" && payload.text) return payload.text;
  return undefined;
}

// ----------------------------------------------------------------------------
// Top-level normalization
// ----------------------------------------------------------------------------

/**
 * Normalize any ScryptedAI job payload (invoke response, polling response,
 * or webhook body) into a single canonical shape.
 *
 * Accepts raw shapes — job_id/result_data/etc. — and produces camelCase.
 */
export function normalizeJobPayload(
  raw: Record<string, unknown>,
): NormalizedJobResult {
  const jobId = String(raw.job_id ?? raw.jobId ?? "");
  const status = normalizeStatus(raw.status);
  const result = mergeResult(raw);
  const error = extractError(raw);

  // Only probe for media if we have a result or are terminal; cheap anyway.
  const imageUrl = extractImageUrl(raw);
  const videoUrl = extractVideoUrl(raw);
  const text = extractText(raw);

  return {
    jobId,
    status,
    result,
    error,
    imageUrl,
    videoUrl,
    text,
  };
}

/** Convenience wrapper for the typed polling response. */
export function normalizeJobStatusResponse(
  resp: JobStatusResponse,
): NormalizedJobResult {
  return normalizeJobPayload(resp as unknown as Record<string, unknown>);
}
