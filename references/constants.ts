/**
 * Constants and configuration for ScryptedAI SDK.
 *
 * This module contains all the constants, configuration values, and endpoint
 * definitions used throughout the SDK.
 */

export const API_BASE_URL = "https://api.scrypted.ai";
export const API_VERSION = "v1";
export const DEFAULT_TIMEOUT = 30;
export const MAX_RETRIES = 3;

// Endpoints
export const ENDPOINTS: Record<string, string> = {
  accounts: "/accounts",
  account_me: "/accounts/me",
  health: "/healthz",
  recipe_interface: "/recipes", // Base path for recipe interface and execution
  generations_videos: "/generations/videos", // Video generation endpoint
  generations_videos_nova_reel: "/generations/videos/aws-novareel", // AWS Nova Reel video generation endpoint
  generations_videos_hailuo_2_3: "/generations/videos/hailuo-2-3-i2v", // Hailuo 2.3 image-to-video generation endpoint
  generations_videos_veo_3: "/generations/videos/veo-3-i2v", // Google Veo 3 Fast image-to-video generation endpoint
  generations_videos_veo_3_1: "/generations/videos/veo-3-1-i2v", // Google Veo 3.1 Fast image-to-video generation endpoint
  generations_videos_sora2: "/generations/videos/sora2-pro-i2v", // OpenAI Sora 2 Pro image-to-video generation endpoint
  generations_videos_sora2_i2v: "/generations/videos/sora2-i2v", // OpenAI Sora 2 (non-Pro) image-to-video, 720p only, lower cost
  generations_videos_grok_imagine_i2v: "/generations/videos/grok-imagine-i2v", // xAI Grok Imagine image-to-video generation endpoint
  generations_videos_tools_upscale_topaz: "/generations/videos/tools/upscale/topaz", // Topaz video upscale (FAL) tool
  generations_images_aws_canvas: "/generations/images/aws-canvas", // AWS Canvas image generation endpoint
  generations_images_nano_banana: "/generations/images/nano-banana", // Google Nano-Banana image generation endpoint
  generations_images_nano_banana_pro: "/generations/images/nano-banana-pro-image", // Google Nano-Banana Pro image generation endpoint
  generations_images_nano_banana_edit: "/generations/images/nano-banana-edit-image", // Google Nano-Banana image editing endpoint
  generations_images_nano_banana_pro_edit: "/generations/images/nano-banana-pro-edit-image", // Google Nano-Banana Pro image editing endpoint
  generations_images_seedream_4: "/generations/images/seedream-4", // Seedream 4.0 image generation endpoint
  generations_images_flux_2_pro: "/generations/images/flux-2-pro", // FLUX 2 Pro (Black Forest Labs) image generation endpoint
  generations_images_grok_imagine_image: "/generations/images/grok-imagine-image", // xAI Grok Imagine Image text-to-image (FAL) endpoint
  generations_text_nova_pro: "/generations/text/nova-pro", // Nova Pro text generation endpoint
  jobs: "/jobs", // Base path for job status and result retrieval
};

// Headers
export const DEFAULT_HEADERS: Record<string, string> = {
  "Content-Type": "application/json",
  "User-Agent": "ScryptedAI-SDK-TypeScript/1.0.0",
};

// Bearer Token Configuration
export const BEARER_PREFIX = "Bearer ";
export const TOKEN_PREFIX = "scrypted_";
export const TOKEN_MIN_LENGTH = 20;

// Retry Configuration
export const BACKOFF_FACTOR = 1.0;
export const MAX_BACKOFF_TIME = 60.0;

// Logging Configuration
export const LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s";
export const LOG_LEVEL = "INFO";

