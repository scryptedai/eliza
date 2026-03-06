/**
 * Constants and configuration for the ScryptedAI plugin.
 *
 * Mirrors the reference SDK constants (see references/constants.ts)
 * and adds plugin-specific operational values (polling windows, etc).
 */

// ----------------------------------------------------------------------------
// API configuration
// ----------------------------------------------------------------------------

export const API_BASE_URL = "https://api.scrypted.ai";
export const DEFAULT_TIMEOUT_SECONDS = 30;
export const MAX_RETRIES = 3;
export const BACKOFF_FACTOR_MS = 1000; // 1s base (1s, 2s, 4s, ...)
export const MAX_BACKOFF_MS = 60_000;

// ----------------------------------------------------------------------------
// Bearer token validation
// ----------------------------------------------------------------------------

export const BEARER_PREFIX = "Bearer ";
export const TOKEN_PREFIX = "scrypted_";
export const TOKEN_MIN_LENGTH = 20;

// ----------------------------------------------------------------------------
// Endpoints (mirrors references/constants.ts ENDPOINTS)
// ----------------------------------------------------------------------------

export const ENDPOINTS = {
  // Account management (permissionless bootstrap)
  accounts: "/accounts",
  account_me: "/accounts/me",

  recipes: "/recipes",
  jobs: "/jobs",

  // Direct image generation
  generations_images_aws_canvas: "/generations/images/aws-canvas",
  generations_images_nano_banana: "/generations/images/nano-banana",
  generations_images_nano_banana_pro:
    "/generations/images/nano-banana-pro-image",
  generations_images_nano_banana_edit:
    "/generations/images/nano-banana-edit-image",
  generations_images_nano_banana_pro_edit:
    "/generations/images/nano-banana-pro-edit-image",
  generations_images_seedream_4: "/generations/images/seedream-4",
  generations_images_flux_2_pro: "/generations/images/flux-2-pro",
  generations_images_grok_imagine_image:
    "/generations/images/grok-imagine-image",

  // Direct video generation
  generations_videos: "/generations/videos",
  generations_videos_nova_reel: "/generations/videos/aws-novareel",
  generations_videos_hailuo_2_3: "/generations/videos/hailuo-2-3-i2v",
  generations_videos_veo_3: "/generations/videos/veo-3-i2v",
  generations_videos_veo_3_1: "/generations/videos/veo-3-1-i2v",
  generations_videos_sora2: "/generations/videos/sora2-pro-i2v",
  generations_videos_sora2_i2v: "/generations/videos/sora2-i2v",
  generations_videos_grok_imagine_i2v: "/generations/videos/grok-imagine-i2v",
  generations_videos_tools_upscale_topaz:
    "/generations/videos/tools/upscale/topaz",

  // Text generation
  generations_text_nova_pro: "/generations/text/nova-pro",
} as const;

// ----------------------------------------------------------------------------
// HTTP headers
// ----------------------------------------------------------------------------

export const DEFAULT_HEADERS: Record<string, string> = {
  "Content-Type": "application/json",
  "User-Agent": "ElizaOS-ScryptedAI-Plugin/0.1.0",
};

// ----------------------------------------------------------------------------
// Polling windows by job type (seconds)
//
// Per integration guide §4.3: use type-based min/max wait windows.
// - minAge: don't start treating as "stuck" before this
// - maxWait: give up after this, mark failed
// - intervals: backoff sequence between polls
// ----------------------------------------------------------------------------

export type JobType = "text" | "image" | "video" | "unknown";

export interface PollingWindow {
  /** Seconds before the first poll attempt */
  minAgeSeconds: number;
  /** Total seconds to wait before giving up */
  maxWaitSeconds: number;
  /** Poll interval backoff sequence (seconds). Last value repeats. */
  intervalsSeconds: number[];
}

export const POLLING_WINDOWS: Record<JobType, PollingWindow> = {
  text: {
    minAgeSeconds: 2,
    maxWaitSeconds: 60,
    intervalsSeconds: [2, 4, 8, 15],
  },
  image: {
    minAgeSeconds: 5,
    maxWaitSeconds: 300, // 5 min
    intervalsSeconds: [5, 10, 20, 30],
  },
  video: {
    minAgeSeconds: 15,
    maxWaitSeconds: 1800, // 30 min
    intervalsSeconds: [15, 30, 60, 60],
  },
  unknown: {
    minAgeSeconds: 5,
    maxWaitSeconds: 600, // 10 min
    intervalsSeconds: [5, 15, 30, 60],
  },
};

/** HTTP status codes that indicate a transient gateway failure during polling. */
export const TRANSIENT_GATEWAY_STATUS = new Set([502, 503, 504]);

// ----------------------------------------------------------------------------
// Plugin service / route identity
// ----------------------------------------------------------------------------

export const SCRYPTEDAI_SERVICE_TYPE = "scryptedai" as const;
export const WEBHOOK_ROUTE_PATH = "/webhook"; // mounts at /scryptedai/webhook (runtime namespaces by plugin name)

// ----------------------------------------------------------------------------
// Environment variable names
// ----------------------------------------------------------------------------

export const ENV_BEARER_TOKEN = "SCRYPTEDAI_BEARER_TOKEN";
export const ENV_WEBHOOK_SECRET = "SCRYPTEDAI_WEBHOOK_SECRET";
export const ENV_BASE_URL = "SCRYPTEDAI_BASE_URL";
/** Select which image endpoint `runtime.useModel(IMAGE, ...)` uses. Default: nano-banana. */
export const ENV_IMAGE_MODEL = "SCRYPTEDAI_IMAGE_MODEL";

// ----------------------------------------------------------------------------
// Model handler configuration
// ----------------------------------------------------------------------------

/** Image endpoint names selectable via SCRYPTEDAI_IMAGE_MODEL. */
export type ImageModelName =
  | "nano-banana"
  | "nano-banana-pro"
  | "seedream-4"
  | "flux-2-pro"
  | "aws-canvas"
  | "grok-imagine";

export const DEFAULT_IMAGE_MODEL: ImageModelName = "nano-banana";

/** Max time the IMAGE model handler will block awaiting a terminal result (ms). */
export const MODEL_IMAGE_TIMEOUT_MS = 300_000;
/** Max time the TEXT_LARGE model handler will block awaiting a terminal result (ms). */
export const MODEL_TEXT_TIMEOUT_MS = 60_000;
