/**
 * ScryptedAI HTTP client.
 *
 * Ports the reference SDK client (references/client.ts) with ElizaOS-appropriate
 * adaptations:
 * - Native fetch (Node 23.3+) instead of undici — no extra deps
 * - No console.log chatter — callers use runtime.logger if they need telemetry
 * - Per-endpoint invoke methods preserved (user requirement: endpoints are
 *   significant; do not collapse to a single generic call)
 * - Common POST mechanics extracted to a private helper to avoid 18× duplication
 *   while keeping the public surface area identical to the reference SDK
 *
 * Security:
 * - Bearer token is never logged
 * - Token format validated at construction time
 */

import { randomBytes } from "node:crypto";
import {
  API_BASE_URL,
  BACKOFF_FACTOR_MS,
  BEARER_PREFIX,
  DEFAULT_HEADERS,
  DEFAULT_TIMEOUT_SECONDS,
  ENDPOINTS,
  MAX_BACKOFF_MS,
  MAX_RETRIES,
  TOKEN_MIN_LENGTH,
  TOKEN_PREFIX,
} from "./constants.ts";
import {
  ScryptedAPIError,
  ScryptedAuthenticationError,
  ScryptedError,
  ScryptedNetworkError,
  ScryptedPaymentError,
  ScryptedRateLimitError,
  ScryptedTimeoutError,
  ScryptedValidationError,
} from "./exceptions.ts";
import type {
  AccountResponse,
  InvokeOptions,
  JobStatusResponse,
  NewAccountResponse,
  RecipeExecutionResponse,
  RecipeInterface,
} from "./types.ts";

// ----------------------------------------------------------------------------
// Standalone helpers
// ----------------------------------------------------------------------------

/** Validate bearer token format: must start with `scrypted_` and be >= 20 chars. */
export function validateBearerToken(token: unknown): token is string {
  if (typeof token !== "string") return false;
  if (!token.startsWith(TOKEN_PREFIX)) return false;
  if (token.length < TOKEN_MIN_LENGTH) return false;
  return true;
}

/** Generate a 256-bit hex idempotency key. */
export function generateIdempotencyKey(): string {
  return randomBytes(32).toString("hex");
}

/** Sleep helper for retry backoff. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Retry a function with exponential backoff.
 *
 * Per integration guide §6.1:
 * - NEVER retries: ScryptedAuthenticationError, ScryptedValidationError, ScryptedPaymentError
 * - Retries everything else up to maxRetries with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = MAX_RETRIES,
  backoffFactorMs: number = BACKOFF_FACTOR_MS,
): Promise<T> {
  let lastError: Error | undefined;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Non-retriable error types — rethrow immediately
      if (
        error instanceof ScryptedAuthenticationError ||
        error instanceof ScryptedValidationError ||
        error instanceof ScryptedPaymentError
      ) {
        throw error;
      }

      if (attempt >= maxRetries) throw lastError;

      const waitMs = Math.min(backoffFactorMs * 2 ** attempt, MAX_BACKOFF_MS);
      await sleep(waitMs);
    }
  }

  throw lastError ?? new Error("retryWithBackoff: exhausted with no error");
}

// ----------------------------------------------------------------------------
// Internal: HTTP response → typed error mapping
// ----------------------------------------------------------------------------

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let responseData: unknown;
    try {
      const text = await response.text();
      try {
        responseData = JSON.parse(text);
      } catch {
        responseData = text;
      }
    } catch {
      responseData = undefined;
    }

    switch (response.status) {
      case 401:
        throw new ScryptedAPIError("Authentication failed", 401, responseData);
      case 402:
        throw new ScryptedPaymentError(
          "Payment required",
          response.headers.get("Location") ?? undefined,
          response.headers.get("X-Payment-Amount") ?? undefined,
        );
      case 429: {
        const retryAfter = Number.parseInt(
          response.headers.get("Retry-After") ?? "60",
          10,
        );
        throw new ScryptedRateLimitError(
          "Rate limit exceeded",
          Number.isNaN(retryAfter) ? undefined : retryAfter,
        );
      }
      default:
        throw new ScryptedAPIError(
          `API error: ${response.status} ${response.statusText}`,
          response.status,
          responseData,
        );
    }
  }

  try {
    return (await response.json()) as T;
  } catch (error) {
    throw new ScryptedNetworkError(
      `Failed to parse response JSON: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

// ----------------------------------------------------------------------------
// ScryptedClient
// ----------------------------------------------------------------------------

export interface ScryptedClientOptions {
  /**
   * Bearer token. Optional — you can construct tokenless and call
   * `createAccount()` to obtain one (permissionless bootstrap flow).
   * If provided, must start with `scrypted_` and be >= 20 chars.
   */
  bearerToken?: string;
  baseUrl?: string;
  timeoutSeconds?: number;
  maxRetries?: number;
}

export class ScryptedClient {
  private bearerToken: string | undefined;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;

  constructor(opts: ScryptedClientOptions = {}) {
    // Validate token ONLY if provided — tokenless construction is allowed
    // for the permissionless createAccount() bootstrap flow.
    if (opts.bearerToken !== undefined) {
      if (!validateBearerToken(opts.bearerToken)) {
        throw new ScryptedAuthenticationError(
          `Invalid bearer token format (must start with '${TOKEN_PREFIX}' and be at least ${TOKEN_MIN_LENGTH} characters)`,
        );
      }
      this.bearerToken = opts.bearerToken;
    }

    this.baseUrl = (opts.baseUrl ?? API_BASE_URL).replace(/\/$/, "");
    this.timeoutMs = (opts.timeoutSeconds ?? DEFAULT_TIMEOUT_SECONDS) * 1000;
    this.maxRetries = opts.maxRetries ?? MAX_RETRIES;
  }

  /** Current bearer token (undefined if tokenless / pre-createAccount). */
  getBearerToken(): string | undefined {
    return this.bearerToken;
  }

  // --------------------------------------------------------------------------
  // Private HTTP mechanics
  // --------------------------------------------------------------------------

  /** Throw if no bearer token is set. Called by all authenticated endpoints. */
  private requireToken(): string {
    if (!this.bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Call createAccount() first or construct the client with a bearerToken.",
      );
    }
    return this.bearerToken;
  }

  private buildHeaders(extra?: Record<string, string>): Record<string, string> {
    return {
      ...DEFAULT_HEADERS,
      ...(extra ?? {}),
      Authorization: `${BEARER_PREFIX}${this.requireToken()}`,
    };
  }

  /**
   * Core POST helper shared by all invoke* methods.
   * Handles: body assembly (idempotency/webhook fields), timeout, error mapping, retry.
   */
  private async postJob(
    endpoint: string,
    body: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    return retryWithBackoff(async () => {
      const requestBody: Record<string, unknown> = { ...body };
      if (opts?.idempotencyKey)
        requestBody.idempotency_key = opts.idempotencyKey;
      if (opts?.webhookUrl) requestBody.webhook_url = opts.webhookUrl;
      if (opts?.webhookSecret) requestBody.webhook_secret = opts.webhookSecret;

      // Strip undefined fields
      for (const k of Object.keys(requestBody)) {
        if (requestBody[k] === undefined) delete requestBody[k];
      }

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      try {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers: this.buildHeaders(),
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
        return await handleResponse<RecipeExecutionResponse>(response);
      } catch (error) {
        if (error instanceof ScryptedError) throw error;
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError(
            `Request timeout after ${this.timeoutMs}ms`,
          );
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : String(error)}`,
        );
      } finally {
        clearTimeout(timer);
      }
    }, this.maxRetries);
  }

  private async getJson<T>(endpoint: string): Promise<T> {
    return retryWithBackoff(async () => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      try {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
          method: "GET",
          headers: this.buildHeaders(),
          signal: controller.signal,
        });
        return await handleResponse<T>(response);
      } catch (error) {
        if (error instanceof ScryptedError) throw error;
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError(
            `Request timeout after ${this.timeoutMs}ms`,
          );
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : String(error)}`,
        );
      } finally {
        clearTimeout(timer);
      }
    }, this.maxRetries);
  }

  // --------------------------------------------------------------------------
  // Input validation helpers
  // --------------------------------------------------------------------------

  private requireNonEmptyString(
    body: Record<string, unknown>,
    field: string,
  ): void {
    const v = body[field];
    if (typeof v !== "string" || v.trim().length === 0) {
      throw new ScryptedValidationError(
        `${field} is required and must be a non-empty string`,
      );
    }
  }

  private requireNonEmptyArray(
    body: Record<string, unknown>,
    field: string,
  ): void {
    const v = body[field];
    if (!Array.isArray(v) || v.length === 0) {
      throw new ScryptedValidationError(
        `${field} is required and must be a non-empty array`,
      );
    }
  }

  // --------------------------------------------------------------------------
  // Account management (permissionless bootstrap)
  // --------------------------------------------------------------------------

  /**
   * Create a new permissionless account.
   *
   * POST /accounts — no auth required, only an Idempotency-Key header.
   * The returned bearer_token is auto-adopted for all subsequent calls.
   *
   * Use this to bootstrap a client with no pre-existing token:
   *   const client = new ScryptedClient();
   *   const account = await client.createAccount();
   *   // client is now authenticated; account.bearer_token is your token
   */
  async createAccount(): Promise<NewAccountResponse> {
    return retryWithBackoff(async () => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      try {
        const response = await fetch(`${this.baseUrl}${ENDPOINTS.accounts}`, {
          method: "POST",
          headers: {
            ...DEFAULT_HEADERS,
            "Idempotency-Key": generateIdempotencyKey(),
          },
          signal: controller.signal,
        });
        const data = await handleResponse<NewAccountResponse>(response);

        // Auto-adopt the returned token (reference SDK behavior)
        if (validateBearerToken(data.bearer_token)) {
          this.bearerToken = data.bearer_token;
        }

        return data;
      } catch (error) {
        if (error instanceof ScryptedError) throw error;
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError(
            `Request timeout after ${this.timeoutMs}ms`,
          );
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : String(error)}`,
        );
      } finally {
        clearTimeout(timer);
      }
    }, this.maxRetries);
  }

  /**
   * Get current account info (GET /accounts/me). Requires a bearer token.
   */
  async getAccountInfo(): Promise<AccountResponse> {
    return this.getJson<AccountResponse>(ENDPOINTS.account_me);
  }

  // --------------------------------------------------------------------------
  // Recipe execution (generic)
  // --------------------------------------------------------------------------

  /**
   * Execute a recipe by ID. Recipe IDs may contain colons
   * (e.g. `scrypted:converter:eli5-title-cleaner`) and are URL-encoded here.
   */
  async invokeRecipe(
    recipeId: string,
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    if (typeof recipeId !== "string" || recipeId.length === 0) {
      throw new ScryptedValidationError("recipeId is required");
    }
    const encoded = encodeURIComponent(recipeId);
    return this.postJob(
      `${ENDPOINTS.recipes}/${encoded}`,
      { input: inputData },
      opts,
    );
  }

  /** Get a recipe's input/output JSON schemas (no steps). */
  async getRecipeInterface(recipeId: string): Promise<RecipeInterface> {
    if (typeof recipeId !== "string" || recipeId.length === 0) {
      throw new ScryptedValidationError("recipeId is required");
    }
    const encoded = encodeURIComponent(recipeId);
    return this.getJson<RecipeInterface>(`${ENDPOINTS.recipes}/${encoded}`);
  }

  // --------------------------------------------------------------------------
  // Job lifecycle
  // --------------------------------------------------------------------------

  /** Fetch current status for a job by ID. */
  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    if (typeof jobId !== "string" || jobId.length === 0) {
      throw new ScryptedValidationError("jobId is required");
    }
    return this.getJson<JobStatusResponse>(
      `${ENDPOINTS.jobs}/${encodeURIComponent(jobId)}`,
    );
  }

  /** Cancel a job. Idempotent on already-terminal jobs. */
  async cancelJob(jobId: string): Promise<JobStatusResponse> {
    if (typeof jobId !== "string" || jobId.length === 0) {
      throw new ScryptedValidationError("jobId is required");
    }
    const endpoint = `${ENDPOINTS.jobs}/${encodeURIComponent(jobId)}/cancel`;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: "POST",
        headers: this.buildHeaders(),
        signal: controller.signal,
      });
      return await handleResponse<JobStatusResponse>(response);
    } catch (error) {
      if (error instanceof ScryptedError) throw error;
      if (error instanceof Error && error.name === "AbortError") {
        throw new ScryptedTimeoutError(
          `Request timeout after ${this.timeoutMs}ms`,
        );
      }
      throw new ScryptedNetworkError(
        `Network error: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      clearTimeout(timer);
    }
  }

  // --------------------------------------------------------------------------
  // Direct generation — IMAGES
  // --------------------------------------------------------------------------

  /** AWS Canvas image generation (/generations/images/aws-canvas). */
  async invokeImageGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    return this.postJob(
      ENDPOINTS.generations_images_aws_canvas,
      inputData,
      opts,
    );
  }

  /** Google Nano-Banana image generation (/generations/images/nano-banana). */
  async invokeNanoBananaGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    return this.postJob(
      ENDPOINTS.generations_images_nano_banana,
      inputData,
      opts,
    );
  }

  /** Google Nano-Banana Pro image generation (/generations/images/nano-banana-pro-image). */
  async invokeNanoBananaProGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    return this.postJob(
      ENDPOINTS.generations_images_nano_banana_pro,
      inputData,
      opts,
    );
  }

  /** Google Nano-Banana image edit (/generations/images/nano-banana-edit-image). */
  async invokeNanoBananaEditGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    this.requireNonEmptyArray(inputData, "image_urls");
    return this.postJob(
      ENDPOINTS.generations_images_nano_banana_edit,
      inputData,
      opts,
    );
  }

  /** Google Nano-Banana Pro image edit (/generations/images/nano-banana-pro-edit-image). */
  async invokeNanoBananaProEditGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    this.requireNonEmptyArray(inputData, "image_urls");
    return this.postJob(
      ENDPOINTS.generations_images_nano_banana_pro_edit,
      inputData,
      opts,
    );
  }

  /** Seedream 4.0 image generation (/generations/images/seedream-4). */
  async invokeSeedream4Generation(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    return this.postJob(
      ENDPOINTS.generations_images_seedream_4,
      inputData,
      opts,
    );
  }

  /** FLUX 2 Pro image generation (/generations/images/flux-2-pro). */
  async invokeFlux2ProGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    return this.postJob(
      ENDPOINTS.generations_images_flux_2_pro,
      inputData,
      opts,
    );
  }

  /** xAI Grok Imagine text-to-image (/generations/images/grok-imagine-image). */
  async invokeGrokImagineImageGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    const body: Record<string, unknown> = {
      prompt: inputData.prompt,
      aspect_ratio: inputData.aspect_ratio ?? "1:1",
      num_images: inputData.num_images ?? 1,
      output_format: inputData.output_format ?? "jpeg",
      retention: inputData.retention,
      reference_id: inputData.reference_id,
    };
    return this.postJob(
      ENDPOINTS.generations_images_grok_imagine_image,
      body,
      opts,
    );
  }

  // --------------------------------------------------------------------------
  // Direct generation — VIDEOS
  // --------------------------------------------------------------------------

  /** Generic video generation (/generations/videos). */
  async invokeVideoGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    return this.postJob(ENDPOINTS.generations_videos, inputData, opts);
  }

  /** AWS Nova Reel video generation (/generations/videos/aws-novareel). */
  async invokeNovaReelGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    return this.postJob(
      ENDPOINTS.generations_videos_nova_reel,
      inputData,
      opts,
    );
  }

  /** Hailuo 2.3 image-to-video (/generations/videos/hailuo-2-3-i2v). */
  async invokeHailuo23Generation(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    this.requireNonEmptyString(inputData, "image_url");
    return this.postJob(
      ENDPOINTS.generations_videos_hailuo_2_3,
      inputData,
      opts,
    );
  }

  /** Google Veo 3 Fast image-to-video (/generations/videos/veo-3-i2v). */
  async invokeVeo3Generation(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    this.requireNonEmptyString(inputData, "image_url");
    return this.postJob(ENDPOINTS.generations_videos_veo_3, inputData, opts);
  }

  /** Google Veo 3.1 Fast image-to-video (/generations/videos/veo-3-1-i2v). */
  async invokeVeo31Generation(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    this.requireNonEmptyString(inputData, "image_url");
    return this.postJob(ENDPOINTS.generations_videos_veo_3_1, inputData, opts);
  }

  /** OpenAI Sora 2 Pro image-to-video (/generations/videos/sora2-pro-i2v). */
  async invokeSora2Generation(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    this.requireNonEmptyString(inputData, "image_url");
    return this.postJob(ENDPOINTS.generations_videos_sora2, inputData, opts);
  }

  /** OpenAI Sora 2 (non-Pro) image-to-video (/generations/videos/sora2-i2v). 720p only. */
  async invokeSora2I2VGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    this.requireNonEmptyString(inputData, "image_url");
    return this.postJob(
      ENDPOINTS.generations_videos_sora2_i2v,
      inputData,
      opts,
    );
  }

  /** xAI Grok Imagine image-to-video (/generations/videos/grok-imagine-i2v). */
  async invokeGrokImagineI2VGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "prompt");
    this.requireNonEmptyString(inputData, "image_url");
    return this.postJob(
      ENDPOINTS.generations_videos_grok_imagine_i2v,
      inputData,
      opts,
    );
  }

  /** Topaz video upscale (/generations/videos/tools/upscale/topaz). */
  async invokeTopazVideoUpscale(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "video_url");
    const body: Record<string, unknown> = {
      video_url: inputData.video_url,
      upscale_factor: inputData.upscale_factor ?? 2.0,
      target_fps: inputData.target_fps,
      H264_output: inputData.H264_output ?? false,
      retention: inputData.retention ?? 259200,
      reference_id: inputData.reference_id,
      payment_method: inputData.payment_method ?? "balance",
    };
    return this.postJob(
      ENDPOINTS.generations_videos_tools_upscale_topaz,
      body,
      opts,
    );
  }

  // --------------------------------------------------------------------------
  // Direct generation — TEXT
  // --------------------------------------------------------------------------

  /** Nova Pro text generation (/generations/text/nova-pro). */
  async invokeTextGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions,
  ): Promise<RecipeExecutionResponse> {
    this.requireNonEmptyString(inputData, "user_prompt");
    return this.postJob(ENDPOINTS.generations_text_nova_pro, inputData, opts);
  }
}
