/**
 * ElizaOS model handlers for ScryptedAI.
 *
 * Registered under Plugin.models so callers can dispatch via
 * `runtime.useModel(ModelType.TEXT_LARGE, { prompt })` and
 * `runtime.useModel(ModelType.IMAGE, { prompt, count? })` without knowing
 * anything about ScryptedAI directly.
 *
 * Design:
 * - Handlers are BLOCKING: invoke → await terminal via service.awaitJob().
 *   Webhook completion and polling both funnel through the same path.
 * - Param shapes mirror the proto-derived core types but are declared
 *   locally to avoid a hard dependency on generated protobuf output.
 * - Image endpoint is selectable via SCRYPTEDAI_IMAGE_MODEL (default: nano-banana).
 *
 * Not implemented: VIDEO. The core VideoProcessingParams type is oriented
 * around *processing* an existing video_url, not text-to-video generation.
 * Use the service or client directly for video generation.
 */

import type { IAgentRuntime } from "@elizaos/core";
import {
  DEFAULT_IMAGE_MODEL,
  ENV_IMAGE_MODEL,
  type ImageModelName,
  MODEL_IMAGE_TIMEOUT_MS,
  MODEL_TEXT_TIMEOUT_MS,
  SCRYPTEDAI_SERVICE_TYPE,
} from "./constants.ts";
import type { ScryptedAIService } from "./service.ts";

// ----------------------------------------------------------------------------
// Local param shapes (proto-compatible, declared here to avoid proto dep)
// ----------------------------------------------------------------------------

/** Structural subset of core GenerateTextParams. */
interface TextParams {
  prompt: string;
  maxTokens?: number;
  temperature?: number;
}

/** Structural subset of core ImageGenerationParams. */
interface ImageParams {
  prompt: string;
  size?: string;
  count?: number;
}

/** Structural subset of core ImageGenerationResult. */
interface ImageResult {
  url: string;
}

// ----------------------------------------------------------------------------
// Internals
// ----------------------------------------------------------------------------

function requireService(runtime: IAgentRuntime): ScryptedAIService {
  const svc = runtime.getService<ScryptedAIService>(SCRYPTEDAI_SERVICE_TYPE);
  if (!svc) {
    throw new Error(
      "ScryptedAI service not available. Ensure @elizaos/plugin-scryptedai " +
        "is registered and SCRYPTEDAI_BEARER_TOKEN is set.",
    );
  }
  return svc;
}

/** Map env-selected image model name → service method name. */
const IMAGE_MODEL_METHOD: Record<
  ImageModelName,
  Parameters<ScryptedAIService["startImageGeneration"]>[0]
> = {
  "nano-banana": "invokeNanoBananaGeneration",
  "nano-banana-pro": "invokeNanoBananaProGeneration",
  "seedream-4": "invokeSeedream4Generation",
  "flux-2-pro": "invokeFlux2ProGeneration",
  "aws-canvas": "invokeImageGeneration",
  "grok-imagine": "invokeGrokImagineImageGeneration",
};

function resolveImageMethod(
  runtime: IAgentRuntime,
): Parameters<ScryptedAIService["startImageGeneration"]>[0] {
  const raw = runtime.getSetting(ENV_IMAGE_MODEL);
  if (typeof raw === "string" && raw in IMAGE_MODEL_METHOD) {
    return IMAGE_MODEL_METHOD[raw as ImageModelName];
  }
  return IMAGE_MODEL_METHOD[DEFAULT_IMAGE_MODEL];
}

// ----------------------------------------------------------------------------
// TEXT_LARGE handler
// ----------------------------------------------------------------------------

export async function handleTextLarge(
  runtime: IAgentRuntime,
  params: TextParams,
): Promise<string> {
  const svc = requireService(runtime);

  const inputData: Record<string, unknown> = {
    user_prompt: params.prompt,
  };
  if (typeof params.maxTokens === "number") {
    inputData.max_tokens = params.maxTokens;
  }
  if (typeof params.temperature === "number") {
    inputData.temperature = params.temperature;
  }

  const { jobId } = await svc.startTextGeneration(inputData);
  const result = await svc.awaitJob(jobId, MODEL_TEXT_TIMEOUT_MS);

  if (result.status !== "completed") {
    throw new Error(
      `ScryptedAI text generation ${result.status}: ${result.error ?? "(no error message)"}`,
    );
  }
  if (typeof result.text !== "string" || result.text.length === 0) {
    throw new Error(
      "ScryptedAI text generation completed but no text in result",
    );
  }
  return result.text;
}

// ----------------------------------------------------------------------------
// IMAGE handler
// ----------------------------------------------------------------------------

export async function handleImage(
  runtime: IAgentRuntime,
  params: ImageParams,
): Promise<ImageResult[]> {
  const svc = requireService(runtime);
  const method = resolveImageMethod(runtime);

  const inputData: Record<string, unknown> = {
    prompt: params.prompt,
  };
  if (typeof params.count === "number" && params.count > 0) {
    inputData.num_images = params.count;
  }
  if (typeof params.size === "string" && params.size) {
    // Pass-through; upstream endpoints that don't support `size` ignore it.
    inputData.size = params.size;
  }

  const { jobId } = await svc.startImageGeneration(method, inputData);
  const result = await svc.awaitJob(jobId, MODEL_IMAGE_TIMEOUT_MS);

  if (result.status !== "completed") {
    throw new Error(
      `ScryptedAI image generation ${result.status}: ${result.error ?? "(no error message)"}`,
    );
  }

  // Collect all image URLs from the result payload (not just the first).
  const urls: string[] = [];
  const images = result.result?.images;
  if (Array.isArray(images)) {
    for (const img of images) {
      if (typeof img === "string" && img) urls.push(img);
      else if (typeof img === "object" && img !== null) {
        const rec = img as Record<string, unknown>;
        for (const key of ["asset_url", "cdn_url", "cloudfront_url", "url"]) {
          const v = rec[key];
          if (typeof v === "string" && v) {
            urls.push(v);
            break;
          }
        }
      }
    }
  }

  // Fallback to the adapter's single-url extraction
  if (urls.length === 0 && result.imageUrl) {
    urls.push(result.imageUrl);
  }

  if (urls.length === 0) {
    throw new Error(
      "ScryptedAI image generation completed but no image URL in result",
    );
  }

  return urls.map((url) => ({ url }));
}

// ----------------------------------------------------------------------------
// Export for Plugin.models (string keys match ModelType values)
// ----------------------------------------------------------------------------

export const scryptedaiModelHandlers = {
  TEXT_LARGE: handleTextLarge,
  IMAGE: handleImage,
} as const;
