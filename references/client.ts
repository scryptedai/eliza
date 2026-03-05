/**
 * Main ScryptedClient class for ScryptedAI SDK.
 *
 * This module contains the main ScryptedClient class that provides
 * enterprise-grade access to the ScryptedAI API with automatic
 * bearer token management and comprehensive error handling.
 */

import {
  API_BASE_URL,
  ENDPOINTS,
  DEFAULT_TIMEOUT,
} from "./constants";
import {
  AccountResponse,
  NewAccountResponse,
  IdempotencyKey,
  FundingResponse,
  X402Invoice,
  RecipeInterface,
  RecipeExecutionRequest,
  RecipeExecutionResponse,
  JobStatusResponse,
} from "./models";
import {
  ScryptedError,
  ScryptedAPIError,
  ScryptedAuthenticationError,
  ScryptedPaymentError,
  ScryptedValidationError,
  ScryptedNetworkError,
  ScryptedTimeoutError,
  ScryptedRateLimitError,
} from "./exceptions";
import { X402PaymentHandler } from "./x402-handler";
import {
  validateBearerToken,
  formatHeaders,
  handleResponse,
  retryWithBackoff,
} from "./utils";
import { Agent, fetch as undiciFetch } from "undici";

export interface ScryptedClientTransportOptions {
  /** Reuse HTTP connections for high-throughput server workloads. */
  keepAlive?: boolean;
  /** Maximum concurrent TCP connections per origin. */
  connections?: number;
  /** Keep-alive idle timeout in milliseconds. */
  keepAliveTimeoutMs?: number;
  /** Maximum keep-alive timeout in milliseconds. */
  keepAliveMaxTimeoutMs?: number;
  /** Enable request pipelining on reused connections. */
  pipelining?: number;
}

/**
 * Enterprise-grade ScryptedAI API client with automatic bearer token management.
 *
 * This client provides comprehensive access to the ScryptedAI API with support
 * for both explicit account management (Mode 1) and permissionless flows (Mode 2).
 *
 * Key Features:
 * - Auto-adoption of bearer tokens from account creation
 * - Comprehensive error handling with custom exceptions
 * - Automatic retry logic with exponential backoff
 * - Type-safe idempotency key generation
 * - Full RESTful API support
 * - Enterprise-grade logging and monitoring
 *
 * Mode 1 (Explicit Account Management):
 *     Used for testing, development, and automation where you need
 *     explicit control over account creation and bearer token management.
 *
 *     Example:
 *         const client = new ScryptedClient();
 *         const account = await client.createAccount(); // Auto-adopts bearer token
 *         const info = await client.getAccountInfo(); // Uses adopted token
 *
 * Mode 2 (Permissionless Flow):
 *     Used for end-user applications where account creation happens
 *     automatically through the x402 payment challenge flow.
 *
 *     Example:
 *         const client = new ScryptedClient();
 *         // client.someApiCall() // Handles 402 Payment Required automatically
 */
export class ScryptedClient {
  private _bearerToken?: string;
  private baseUrl: string;
  private timeout: number;
  private maxRetries: number;
  private readonly transportOptions: Required<ScryptedClientTransportOptions>;
  private readonly agent: Agent;
  public x402Handler: X402PaymentHandler;

  /**
   * Initialize ScryptedAI client.
   *
   * @param bearerToken - Optional bearer token for authentication
   * @param baseUrl - API base URL (default: https://api.scrypted.ai)
   * @param timeout - Request timeout in seconds
   * @param maxRetries - Maximum retry attempts for failed requests
   *
   * @throws ScryptedAuthenticationError - If provided bearer token is invalid
   */
  constructor(
    bearerToken?: string,
    baseUrl: string = API_BASE_URL,
    timeout: number = DEFAULT_TIMEOUT,
    maxRetries: number = 3,
    transportOptions: ScryptedClientTransportOptions = {}
  ) {
    this._bearerToken = bearerToken;
    this.baseUrl = baseUrl.replace(/\/$/, ""); // Remove trailing slash
    this.timeout = timeout;
    this.maxRetries = maxRetries;
    this.transportOptions = {
      keepAlive: transportOptions.keepAlive ?? true,
      connections: transportOptions.connections ?? 256,
      keepAliveTimeoutMs: transportOptions.keepAliveTimeoutMs ?? 15_000,
      keepAliveMaxTimeoutMs: transportOptions.keepAliveMaxTimeoutMs ?? 60_000,
      pipelining: transportOptions.pipelining ?? 1,
    };
    this.agent = new Agent({
      connections: this.transportOptions.connections,
      keepAliveTimeout: this.transportOptions.keepAlive
        ? this.transportOptions.keepAliveTimeoutMs
        : 1,
      keepAliveMaxTimeout: this.transportOptions.keepAlive
        ? this.transportOptions.keepAliveMaxTimeoutMs
        : 1,
      pipelining: this.transportOptions.pipelining,
    });

    // Validate bearer token if provided
    if (bearerToken && !validateBearerToken(bearerToken)) {
      throw new ScryptedAuthenticationError("Invalid bearer token format");
    }

    // Initialize x402 payment handler
    this.x402Handler = new X402PaymentHandler(this);
  }

  /**
   * Close underlying HTTP resources.
   * - graceful (default): waits for in-flight requests/sockets to drain.
   * - force: immediately destroys sockets (best for short-lived scripts/tests).
   */
  async close(options: { force?: boolean } = {}): Promise<void> {
    if (options.force) {
      await this.agent.destroy();
      return;
    }
    await this.agent.close();
  }

  private async fetch(url: string, init: RequestInit): Promise<Response> {
    const headers = new Headers(init.headers ?? {});
    if (!this.transportOptions.keepAlive) {
      headers.set("connection", "close");
    }
    return undiciFetch(url, {
      ...init,
      headers,
      dispatcher: this.agent,
    } as RequestInit & { dispatcher: Agent });
  }

  /**
   * Get the current bearer token.
   */
  get bearerToken(): string | undefined {
    return this._bearerToken;
  }

  /**
   * Create a new permissionless account.
   *
   * This method creates a new account and automatically adopts the returned
   * bearer token for subsequent API calls. This is the primary method for
   * Mode 1 (explicit account management) workflows.
   *
   * @returns Account details including bearer token
   *
   * @throws ScryptedAPIError - If account creation fails
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async createAccount(): Promise<NewAccountResponse> {
    return retryWithBackoff(async () => {
      const idempotencyKey = IdempotencyKey.generate();
      const headers = formatHeaders(undefined, {
        "Idempotency-Key": idempotencyKey.toString(),
      });

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${ENDPOINTS.accounts}`, {
          method: "POST",
          headers,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        const data = await handleResponse(response);

        // Auto-adopt bearer token for subsequent calls
        this._bearerToken = data.bearer_token;

        return data as NewAccountResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Get current account information.
   *
   * This method retrieves account information using the bearer token
   * that was either provided during initialization or adopted from
   * account creation.
   *
   * @returns Account information (no bearer token included)
   *
   * @throws ScryptedAuthenticationError - If no bearer token is available
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async getAccountInfo(): Promise<AccountResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(
          `${this.baseUrl}${ENDPOINTS.account_me}`,
          {
            method: "GET",
            headers,
            signal: controller.signal,
          }
        );

        clearTimeout(timeoutId);

        return (await handleResponse(response)) as AccountResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Generate a new 256-bit idempotency key.
   *
   * This method generates a cryptographically secure random idempotency
   * key that can be used for API requests that require idempotency.
   *
   * @returns Type-safe idempotency key
   */
  generateIdempotencyKey(): IdempotencyKey {
    return IdempotencyKey.generate();
  }

  /**
   * Set bearer token for authentication.
   *
   * This method allows you to set or update the bearer token used
   * for API authentication. The token is validated before being set.
   *
   * @param bearerToken - Bearer token string
   *
   * @throws ScryptedAuthenticationError - If token format is invalid
   */
  setBearerToken(bearerToken: string): void {
    if (!validateBearerToken(bearerToken)) {
      throw new ScryptedAuthenticationError("Invalid bearer token format");
    }

    this._bearerToken = bearerToken;
  }

  /**
   * Check if client has a bearer token.
   *
   * This method checks whether the client has a bearer token available
   * for authentication. This is useful for determining whether certain
   * operations can be performed.
   *
   * @returns True if bearer token is available, False otherwise
   */
  hasBearerToken(): boolean {
    return this._bearerToken !== undefined;
  }

  /**
   * Clear the current bearer token.
   *
   * This method removes the current bearer token from the client.
   * This can be useful for testing or when you want to force
   * re-authentication.
   */
  clearBearerToken(): void {
    this._bearerToken = undefined;
  }

  /**
   * Get the current bearer token.
   *
   * This method returns the current bearer token if available.
   * This is useful for storing the token for later use or for
   * debugging purposes.
   *
   * @returns The current bearer token or undefined if not set
   */
  getBearerToken(): string | undefined {
    return this._bearerToken;
  }

  /**
   * Fund account with USDC using x402 payments.
   *
   * @param amount - Amount to fund in USDC (e.g., 0.50)
   * @param paymentType - "balance" (use account balance) or "x402" (blockchain payment)
   * @param invoice - x402 invoice from previous 402 response (required for paymentType="x402")
   * @param proofOfPayment - Transaction hash from blockchain (required for paymentType="x402")
   *
   * @returns FundingResponse if successful
   *
   * @throws ScryptedPaymentError - If payment is required
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedAPIError - If funding fails
   */
  async fundAccount(
    amount: number,
    paymentType: "balance" | "x402" = "balance",
    invoice?: X402Invoice,
    proofOfPayment?: string
  ): Promise<FundingResponse> {
    const headers: Record<string, string> = {};

    // Add bearer token if available
    if (this._bearerToken) {
      headers["Authorization"] = `Bearer ${this._bearerToken}`;
    }

    // If x402 payment with proof, construct X-PAYMENT header
    if (paymentType === "x402" && invoice && proofOfPayment) {
      const xPayment = await this.x402Handler.constructXPaymentHeader(
        invoice,
        proofOfPayment
      );
      headers["X-PAYMENT"] = xPayment;
    }

    return retryWithBackoff(async () => {
      const fullHeaders = formatHeaders(this._bearerToken, headers);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}/accounts/funding`, {
          method: "POST",
          headers: fullHeaders,
          body: JSON.stringify({ amount }),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        // Handle 402 Payment Required response
        if (response.status === 402) {
          const data = await response.json();
          const [invoiceData] = await this.x402Handler.extractInvoiceFrom402(data);
          const invoiceObj = invoiceData as X402Invoice;

          // Return invoice for user to pay
          throw new ScryptedPaymentError(
            `Payment required: $${amount} USDC`,
            undefined,
            String(amount)
          );
        }

        // Handle successful response
        return (await handleResponse(response)) as FundingResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Get recipe or ingredient interface (input/output schemas only, NO steps).
   *
   * This method retrieves the interface definition for a recipe or ingredient,
   * including its input and output schemas. Recipe steps are NOT included.
   *
   * Works with both recipes and atomic ingredients.
   * Recipe IDs with colons (e.g., scrypted:converter:eli5-title-cleaner)
   * are automatically URL-encoded.
   *
   * @param recipeId - Recipe or ingredient identifier (e.g., "scrypted:converter:eli5-title-cleaner")
   *
   * @returns Recipe interface with input_schema and output_schema
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedValidationError - If recipe not found or invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async getRecipeInterface(recipeId: string): Promise<RecipeInterface> {
    return retryWithBackoff(async () => {
      // URL-encode the recipe ID (colons become %3A)
      const encodedRecipeId = encodeURIComponent(recipeId);
      const endpoint = `${ENDPOINTS.recipe_interface}/${encodedRecipeId}`;

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "GET",
          headers,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        return (await handleResponse(response)) as RecipeInterface;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke a recipe or atomic ingredient with input validation.
   *
   * This method executes a recipe or ingredient with the provided input data.
   * The input is validated against the recipe's input schema before execution.
   * Returns a job ID for tracking execution progress.
   *
   * Works with both recipes and atomic ingredients.
   * Recipe IDs with colons (e.g., scrypted:converter:eli5-title-cleaner)
   * are automatically URL-encoded.
   *
   * @param recipeId - Recipe or ingredient identifier (e.g., "scrypted:converter:eli5-title-cleaner")
   * @param inputData - Input data matching the recipe's input schema
   * @param idempotencyKey - Optional idempotency key for deduplication
   * @param webhookUrl - Optional webhook URL for notifications
   * @param webhookSecret - Optional HMAC secret for webhook signing
   *
   * @returns Execution response with job_id and status
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedValidationError - If input validation fails
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeRecipe(
    recipeId: string,
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    return retryWithBackoff(async () => {
      // URL-encode the recipe ID (colons become %3A)
      const encodedRecipeId = encodeURIComponent(recipeId);
      const endpoint = `${ENDPOINTS.recipe_interface}/${encodedRecipeId}`;

      // Build request body
      const requestBody: RecipeExecutionRequest = {
        input: inputData,
        idempotency_key: idempotencyKey,
        webhook_url: webhookUrl,
        webhook_secret: webhookSecret,
      };

      // Remove undefined fields
      Object.keys(requestBody).forEach((key) => {
        if ((requestBody as any)[key] === undefined) {
          delete (requestBody as any)[key];
        }
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke video generation via ScryptedAI's /generations/videos endpoint
   *
   * This method calls the ScryptedAI API's video generation endpoint
   * with intent and orientation parameters.
   *
   * @param inputData - Input data with intent and orientation
   * @param idempotencyKey - Optional idempotency key for deduplication
   * @param webhookUrl - Optional webhook URL for notifications
   * @param webhookSecret - Optional HMAC secret for webhook signing
   *
   * @returns Execution response with job_id and status
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeVideoGeneration(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_videos;

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke image generation via ScryptedAI's /generations/images/aws-canvas endpoint
   *
   * This method calls the ScryptedAI API's AWS Canvas image generation endpoint
   * with prompt and generation parameters.
   *
   * @param inputData - Input data with prompt, orientation, quality, number_of_images, seed, retention
   * @param idempotencyKey - Optional idempotency key for deduplication
   * @param webhookUrl - Optional webhook URL for notifications
   * @param webhookSecret - Optional HMAC secret for webhook signing
   *
   * @returns Execution response with job_id and status
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeImageGeneration(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_images_aws_canvas;

      // Log inputData before building request body
      console.log('🖼️ [ScryptedClient] Received inputData:', {
        hasPrompt: !!inputData.prompt,
        promptType: typeof inputData.prompt,
        promptValue: inputData.prompt,
        promptLength: inputData.prompt?.length,
        allKeys: Object.keys(inputData),
        fullInputData: JSON.stringify(inputData, null, 2)
      });

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields, but keep empty strings for prompt
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      // Validate prompt exists before sending (image generation only)
      if (endpoint === ENDPOINTS.generations_images_aws_canvas) {
        if (!requestBody.prompt || (typeof requestBody.prompt === 'string' && requestBody.prompt.trim().length === 0)) {
          console.error('🖼️ [ScryptedClient] ERROR: Prompt is missing or empty in request body!', {
            hasPrompt: !!requestBody.prompt,
            promptType: typeof requestBody.prompt,
            promptValue: requestBody.prompt,
            allKeys: Object.keys(requestBody),
            fullRequestBody: JSON.stringify(requestBody, null, 2)
          });
          throw new ScryptedAPIError('Prompt is required', 400, { error: 'Prompt is required' });
        }
        
        console.log('🖼️ [ScryptedClient] Image generation request body:', {
          endpoint: `${this.baseUrl}${endpoint}`,
          requestBody: JSON.stringify(requestBody, null, 2),
          hasPrompt: !!requestBody.prompt,
          promptType: typeof requestBody.prompt,
          promptLength: requestBody.prompt?.length,
          promptValue: requestBody.prompt,
          allKeys: Object.keys(requestBody)
        });
      }

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        // Log error responses for debugging (image generation)
        if (!response.ok && endpoint === ENDPOINTS.generations_images_aws_canvas) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          console.error('🖼️ [ScryptedClient] Image generation API error response:', {
            status: response.status,
            statusText: response.statusText,
            errorData: errorData
          });
        } else if (endpoint === ENDPOINTS.generations_images_aws_canvas) {
          console.log('🖼️ [ScryptedClient] Image generation response status:', {
            status: response.status,
            statusText: response.statusText,
            ok: response.ok
          });
        }

        const result = await handleResponse(response) as RecipeExecutionResponse;
        
        // Log response result for debugging (image generation only)
        if (endpoint === ENDPOINTS.generations_images_aws_canvas) {
          console.log('🖼️ [ScryptedClient] Image generation response result:', {
            job_id: result.job_id,
            status: result.status,
            hasResult: !!result.result
          });
        }
        
        return result;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke Google Nano-Banana image generation via ScryptedAI's /generations/images/nano-banana endpoint
   *
   * This method calls the ScryptedAI API's Nano-Banana image generation endpoint
   * with prompt and generation parameters.
   *
   * @param inputData - Input data with prompt, aspect_ratio, num_images, output_format, retention
   * @param idempotencyKey - Optional idempotency key for deduplication
   * @param webhookUrl - Optional webhook URL for notifications
   * @param webhookSecret - Optional HMAC secret for webhook signing
   *
   * @returns Execution response with job_id and status
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeNanoBananaGeneration(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_images_nano_banana;

      // Log inputData before building request body
      console.log('🍌 [ScryptedClient] Received inputData:', {
        hasPrompt: !!inputData.prompt,
        promptType: typeof inputData.prompt,
        promptValue: inputData.prompt,
        promptLength: inputData.prompt?.length,
        aspectRatio: inputData.aspect_ratio,
        allKeys: Object.keys(inputData),
        fullInputData: JSON.stringify(inputData, null, 2)
      });

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields, but keep empty strings for prompt
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      // Validate prompt exists before sending
      if (!requestBody.prompt || (typeof requestBody.prompt === 'string' && requestBody.prompt.trim().length === 0)) {
        console.error('🍌 [ScryptedClient] ERROR: Prompt is missing or empty in request body!', {
          hasPrompt: !!requestBody.prompt,
          promptType: typeof requestBody.prompt,
          promptValue: requestBody.prompt,
          allKeys: Object.keys(requestBody),
          fullRequestBody: JSON.stringify(requestBody, null, 2)
        });
        throw new ScryptedAPIError('Prompt is required', 400, { error: 'Prompt is required' });
      }
      
      console.log('🍌 [ScryptedClient] Nano-Banana generation request body:', {
        endpoint: `${this.baseUrl}${endpoint}`,
        requestBody: JSON.stringify(requestBody, null, 2),
        hasPrompt: !!requestBody.prompt,
        promptType: typeof requestBody.prompt,
        promptLength: requestBody.prompt?.length,
        promptValue: requestBody.prompt,
        aspectRatio: requestBody.aspect_ratio,
        allKeys: Object.keys(requestBody)
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        // Log error responses for debugging
        if (!response.ok) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          console.error('🍌 [ScryptedClient] Nano-Banana generation API error response:', {
            status: response.status,
            statusText: response.statusText,
            errorData: errorData
          });
        } else {
          console.log('🍌 [ScryptedClient] Nano-Banana generation response status:', {
            status: response.status,
            statusText: response.statusText,
            ok: response.ok
          });
        }

        const result = await handleResponse(response) as RecipeExecutionResponse;
        
        // Log response result for debugging
        console.log('🍌 [ScryptedClient] Nano-Banana generation response result:', {
          job_id: result.job_id,
          status: result.status,
          hasResult: !!result.result
        });
        
        return result;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke Google Nano-Banana Pro image generation via ScryptedAI's /generations/images/nano-banana-pro-image endpoint
   *
   * This method calls the ScryptedAI API's Nano-Banana Pro image generation endpoint
   * with prompt, resolution, and generation parameters.
   *
   * @param inputData - Input data with prompt, aspect_ratio, resolution, num_images, output_format, retention
   * @param idempotencyKey - Optional idempotency key for deduplication
   * @param webhookUrl - Optional webhook URL for notifications
   * @param webhookSecret - Optional HMAC secret for webhook signing
   *
   * @returns Execution response with job_id and status
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeNanoBananaProGeneration(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_images_nano_banana_pro;

      // Log inputData before building request body
      console.log('🍌✨ [ScryptedClient] Nano-Banana Pro generation input:', {
        hasPrompt: !!inputData.prompt,
        promptType: typeof inputData.prompt,
        promptValue: inputData.prompt,
        promptLength: inputData.prompt?.length,
        aspectRatio: inputData.aspect_ratio,
        resolution: inputData.resolution,
        allKeys: Object.keys(inputData),
        fullInputData: JSON.stringify(inputData, null, 2)
      });

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields, but keep empty strings for prompt
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      // Validate prompt exists before sending
      if (!requestBody.prompt || (typeof requestBody.prompt === 'string' && requestBody.prompt.trim().length === 0)) {
        console.error('🍌✨ [ScryptedClient] ERROR: Prompt is missing or empty in request body!', {
          hasPrompt: !!requestBody.prompt,
          promptType: typeof requestBody.prompt,
          promptValue: requestBody.prompt,
          allKeys: Object.keys(requestBody),
          fullRequestBody: JSON.stringify(requestBody, null, 2)
        });
        throw new ScryptedAPIError('Prompt is required', 400, { error: 'Prompt is required' });
      }
      
      console.log('🍌✨ [ScryptedClient] Nano-Banana Pro generation request body:', {
        endpoint: `${this.baseUrl}${endpoint}`,
        requestBody: JSON.stringify(requestBody, null, 2),
        hasPrompt: !!requestBody.prompt,
        promptType: typeof requestBody.prompt,
        promptLength: requestBody.prompt?.length,
        promptValue: requestBody.prompt,
        aspectRatio: requestBody.aspect_ratio,
        resolution: requestBody.resolution,
        allKeys: Object.keys(requestBody)
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        // Log error responses for debugging
        if (!response.ok) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          console.error('🍌✨ [ScryptedClient] Nano-Banana Pro generation API error response:', {
            status: response.status,
            statusText: response.statusText,
            errorData: errorData
          });
        } else {
          console.log('🍌✨ [ScryptedClient] Nano-Banana Pro generation response status:', {
            status: response.status,
            statusText: response.statusText,
            ok: response.ok
          });
        }

        const result = await handleResponse(response) as RecipeExecutionResponse;
        
        // Log response result for debugging
        console.log('🍌✨ [ScryptedClient] Nano-Banana Pro generation response result:', {
          job_id: result.job_id,
          status: result.status,
          hasResult: !!result.result
        });
        
        return result;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke Google Nano-Banana image editing via ScryptedAI's /generations/images/nano-banana-edit-image endpoint
   *
   * This method calls the ScryptedAI API's Nano-Banana image editing endpoint
   * with prompt, image URLs, and editing parameters.
   *
   * @param inputData - Input data with prompt, image_urls, aspect_ratio, num_images, output_format, retention
   * @param idempotencyKey - Optional idempotency key for deduplication
   * @param webhookUrl - Optional webhook URL for notifications
   * @param webhookSecret - Optional HMAC secret for webhook signing
   *
   * @returns Execution response with job_id and status
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeNanoBananaEditGeneration(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_images_nano_banana_edit;

      // Log inputData before building request body
      console.log('🍌✏️ [ScryptedClient] Nano-Banana Edit generation input:', {
        hasPrompt: !!inputData.prompt,
        promptType: typeof inputData.prompt,
        promptLength: inputData.prompt?.length,
        imageUrlsCount: Array.isArray(inputData.image_urls) ? inputData.image_urls.length : 0,
        aspectRatio: inputData.aspect_ratio,
        allKeys: Object.keys(inputData)
      });

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      // Validate required fields
      if (!requestBody.prompt || (typeof requestBody.prompt === 'string' && requestBody.prompt.trim().length === 0)) {
        console.error('🍌✏️ [ScryptedClient] ERROR: Prompt is missing or empty!');
        throw new ScryptedAPIError('Prompt is required', 400, { error: 'Prompt is required' });
      }

      if (!Array.isArray(requestBody.image_urls) || requestBody.image_urls.length === 0) {
        console.error('🍌✏️ [ScryptedClient] ERROR: image_urls is required and must contain at least one URL!');
        throw new ScryptedAPIError('image_urls is required', 400, { error: 'image_urls is required and must contain at least one URL' });
      }

      console.log('🍌✏️ [ScryptedClient] Nano-Banana Edit generation request body:', {
        endpoint: `${this.baseUrl}${endpoint}`,
        promptLength: requestBody.prompt?.length,
        imageUrlsCount: requestBody.image_urls?.length,
        aspectRatio: requestBody.aspect_ratio
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          console.error('🍌✏️ [ScryptedClient] Nano-Banana Edit generation API error response:', {
            status: response.status,
            statusText: response.statusText,
            errorData: errorData
          });
        } else {
          console.log('🍌✏️ [ScryptedClient] Nano-Banana Edit generation response status:', {
            status: response.status,
            statusText: response.statusText,
            ok: response.ok
          });
        }

        const result = await handleResponse(response) as RecipeExecutionResponse;
        
        console.log('🍌✏️ [ScryptedClient] Nano-Banana Edit generation response result:', {
          job_id: result.job_id,
          status: result.status,
          hasResult: !!result.result
        });
        
        return result;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke Google Nano-Banana Pro image editing via ScryptedAI's /generations/images/nano-banana-pro-edit-image endpoint
   *
   * This method calls the ScryptedAI API's Nano-Banana Pro image editing endpoint
   * with prompt, image URLs, resolution, and editing parameters.
   *
   * @param inputData - Input data with prompt, image_urls, aspect_ratio, resolution, num_images, output_format, retention
   * @param idempotencyKey - Optional idempotency key for deduplication
   * @param webhookUrl - Optional webhook URL for notifications
   * @param webhookSecret - Optional HMAC secret for webhook signing
   *
   * @returns Execution response with job_id and status
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeNanoBananaProEditGeneration(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_images_nano_banana_pro_edit;

      // Log inputData before building request body
      console.log('🍌✨✏️ [ScryptedClient] Nano-Banana Pro Edit generation input:', {
        hasPrompt: !!inputData.prompt,
        promptType: typeof inputData.prompt,
        promptLength: inputData.prompt?.length,
        imageUrlsCount: Array.isArray(inputData.image_urls) ? inputData.image_urls.length : 0,
        aspectRatio: inputData.aspect_ratio,
        resolution: inputData.resolution,
        allKeys: Object.keys(inputData)
      });

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      // Validate required fields
      if (!requestBody.prompt || (typeof requestBody.prompt === 'string' && requestBody.prompt.trim().length === 0)) {
        console.error('🍌✨✏️ [ScryptedClient] ERROR: Prompt is missing or empty!');
        throw new ScryptedAPIError('Prompt is required', 400, { error: 'Prompt is required' });
      }

      if (!Array.isArray(requestBody.image_urls) || requestBody.image_urls.length === 0) {
        console.error('🍌✨✏️ [ScryptedClient] ERROR: image_urls is required and must contain at least one URL!');
        throw new ScryptedAPIError('image_urls is required', 400, { error: 'image_urls is required and must contain at least one URL' });
      }

      console.log('🍌✨✏️ [ScryptedClient] Nano-Banana Pro Edit generation request body:', {
        endpoint: `${this.baseUrl}${endpoint}`,
        promptLength: requestBody.prompt?.length,
        imageUrlsCount: requestBody.image_urls?.length,
        aspectRatio: requestBody.aspect_ratio,
        resolution: requestBody.resolution
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          console.error('🍌✨✏️ [ScryptedClient] Nano-Banana Pro Edit generation API error response:', {
            status: response.status,
            statusText: response.statusText,
            errorData: errorData
          });
        } else {
          console.log('🍌✨✏️ [ScryptedClient] Nano-Banana Pro Edit generation response status:', {
            status: response.status,
            statusText: response.statusText,
            ok: response.ok
          });
        }

        const result = await handleResponse(response) as RecipeExecutionResponse;
        
        console.log('🍌✨✏️ [ScryptedClient] Nano-Banana Pro Edit generation response result:', {
          job_id: result.job_id,
          status: result.status,
          hasResult: !!result.result
        });
        
        return result;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke Seedream 4.0 image generation via ScryptedAI's /generations/images/seedream-4 endpoint
   *
   * This method calls the ScryptedAI API's Seedream 4.0 image generation endpoint
   * with prompt and generation parameters.
   *
   * @param inputData - Input data with prompt, orientation, num_images, seed, enable_safety_checker, retention
   * @param idempotencyKey - Optional idempotency key for deduplication
   * @param webhookUrl - Optional webhook URL for notifications
   * @param webhookSecret - Optional HMAC secret for webhook signing
   *
   * @returns Execution response with job_id and status
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeSeedream4Generation(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_images_seedream_4;

      // Log inputData before building request body
      console.log('🌱 [ScryptedClient] Received inputData:', {
        hasPrompt: !!inputData.prompt,
        promptType: typeof inputData.prompt,
        promptValue: inputData.prompt,
        promptLength: inputData.prompt?.length,
        orientation: inputData.orientation,
        seed: inputData.seed,
        allKeys: Object.keys(inputData),
        fullInputData: JSON.stringify(inputData, null, 2)
      });

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields, but keep empty strings for prompt
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      // Validate prompt exists before sending
      if (!requestBody.prompt || (typeof requestBody.prompt === 'string' && requestBody.prompt.trim().length === 0)) {
        console.error('🌱 [ScryptedClient] ERROR: Prompt is missing or empty in request body!', {
          hasPrompt: !!requestBody.prompt,
          promptType: typeof requestBody.prompt,
          promptValue: requestBody.prompt,
          allKeys: Object.keys(requestBody),
          fullRequestBody: JSON.stringify(requestBody, null, 2)
        });
        throw new ScryptedAPIError('Prompt is required', 400, { error: 'Prompt is required' });
      }
      
      console.log('🌱 [ScryptedClient] Seedream 4 generation request body:', {
        endpoint: `${this.baseUrl}${endpoint}`,
        requestBody: JSON.stringify(requestBody, null, 2),
        hasPrompt: !!requestBody.prompt,
        promptType: typeof requestBody.prompt,
        promptLength: requestBody.prompt?.length,
        promptValue: requestBody.prompt,
        orientation: requestBody.orientation,
        seed: requestBody.seed,
        allKeys: Object.keys(requestBody)
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        // Log error responses for debugging
        if (!response.ok) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          console.error('🌱 [ScryptedClient] Seedream 4 generation API error response:', {
            status: response.status,
            statusText: response.statusText,
            errorData: errorData
          });
        } else {
          console.log('🌱 [ScryptedClient] Seedream 4 generation response status:', {
            status: response.status,
            statusText: response.statusText,
            ok: response.ok
          });
        }

        const result = await handleResponse(response) as RecipeExecutionResponse;
        
        // Log response result for debugging
        console.log('🌱 [ScryptedClient] Seedream 4 generation response result:', {
          job_id: result.job_id,
          status: result.status,
          hasResult: !!result.result
        });
        
        return result;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke FLUX 2 Pro image generation via ScryptedAI's /generations/images/flux-2-pro endpoint
   *
   * @param inputData - Input data with prompt, orientation (landscape_hd | portrait_hd | square_hd or square), etc.
   * @param idempotencyKey - Optional idempotency key
   * @param webhookUrl - Optional webhook URL
   * @param webhookSecret - Optional webhook secret
   * @returns Execution response with job_id and status
   */
  async invokeFlux2ProGeneration(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_images_flux_2_pro;
      const requestBody: Record<string, any> = { ...inputData };
      if (idempotencyKey) requestBody.idempotency_key = idempotencyKey;
      if (webhookUrl) requestBody.webhook_url = webhookUrl;
      if (webhookSecret) requestBody.webhook_secret = webhookSecret;
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) delete requestBody[key];
      });
      if (!requestBody.prompt || (typeof requestBody.prompt === "string" && requestBody.prompt.trim().length === 0)) {
        throw new ScryptedAPIError("Prompt is required", 400, { error: "Prompt is required" });
      }
      const headers = formatHeaders(this._bearerToken);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);
      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) throw error;
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke Grok Imagine Image text-to-image via ScryptedAI's /generations/images/grok-imagine-image endpoint.
   * Uses xAI Grok Imagine Image (FAL) with aspect_ratio and output_format support.
   */
  async invokeGrokImagineImageGeneration(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_images_grok_imagine_image;
      const requestBody: Record<string, any> = {
        prompt: inputData.prompt,
        aspect_ratio: inputData.aspect_ratio ?? "1:1",
        num_images: inputData.num_images ?? 1,
        output_format: inputData.output_format ?? "jpeg",
      };
      if (idempotencyKey) requestBody.idempotency_key = idempotencyKey;
      if (webhookUrl) requestBody.webhook_url = webhookUrl;
      if (webhookSecret) requestBody.webhook_secret = webhookSecret;
      if (inputData.retention != null) requestBody.retention = inputData.retention;
      if (inputData.reference_id != null) requestBody.reference_id = inputData.reference_id;
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) delete requestBody[key];
      });
      if (!requestBody.prompt || (typeof requestBody.prompt === "string" && requestBody.prompt.trim().length === 0)) {
        throw new ScryptedAPIError("Prompt is required", 400, { error: "Prompt is required" });
      }
      const headers = formatHeaders(this._bearerToken);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);
      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) throw error;
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke Nova Reel video generation via ScryptedAI's /generations/video/nova-reel endpoint
   *
   * This method calls the ScryptedAI API's Nova Reel video generation endpoint
   * with an image URL and prompt.
   *
   * @param inputData - Input data with image_url and prompt
   * @param idempotencyKey - Optional idempotency key for deduplication
   * @param webhookUrl - Optional webhook URL for notifications
   * @param webhookSecret - Optional HMAC secret for webhook signing
   *
   * @returns Execution response with job_id and status
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeNovaReelGeneration(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_videos_nova_reel;

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      // Log request body for debugging
      console.log('🎬 [ScryptedClient] Nova Reel request body:', {
        endpoint: `${this.baseUrl}${endpoint}`,
        requestBody: JSON.stringify(requestBody, null, 2),
        hasPrompt: !!requestBody.prompt,
        promptType: typeof requestBody.prompt,
        promptLength: requestBody.prompt?.length,
        promptValue: requestBody.prompt,
        hasImageUrl: !!requestBody.image_url,
        allKeys: Object.keys(requestBody)
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        // Log error responses for debugging (nova-reel only)
        if (!response.ok && endpoint === ENDPOINTS.generations_videos_nova_reel) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          console.error('🎬 [ScryptedClient] Nova Reel API error response:', {
            status: response.status,
            statusText: response.statusText,
            statusCode: response.status,
            errorData: errorData,
            fullErrorText: errorText
          });
        }

        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke Hailuo 2.3 image-to-video generation via ScryptedAI's /generations/videos/hailuo-2-3-i2v endpoint
   *
   * This method calls the ScryptedAI API's Hailuo 2.3 i2v endpoint
   * with an image URL and prompt.
   *
   * @param inputData - Input data with image_url, prompt, prompt_optimizer, retention
   * @param idempotencyKey - Optional idempotency key for deduplication
   * @param webhookUrl - Optional webhook URL for notifications
   * @param webhookSecret - Optional HMAC secret for webhook signing
   *
   * @returns Execution response with job_id and status
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeHailuo23Generation(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_videos_hailuo_2_3;

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      // Validate required fields
      if (!requestBody.prompt || (typeof requestBody.prompt === 'string' && requestBody.prompt.trim().length === 0)) {
        throw new ScryptedAPIError('Prompt is required', 400, { error: 'Prompt is required' });
      }
      if (!requestBody.image_url || (typeof requestBody.image_url === 'string' && requestBody.image_url.trim().length === 0)) {
        throw new ScryptedAPIError('Image URL is required', 400, { error: 'Image URL is required' });
      }

      console.log('🌊 [ScryptedClient] Hailuo 2.3 generation request body:', {
        endpoint: `${this.baseUrl}${endpoint}`,
        requestBody: JSON.stringify(requestBody, null, 2),
        hasPrompt: !!requestBody.prompt,
        hasImageUrl: !!requestBody.image_url
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          console.error('🌊 [ScryptedClient] Hailuo 2.3 generation API error response:', {
            status: response.status,
            statusText: response.statusText,
            errorData: errorData
          });
        }

        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke Google Veo 3 Fast image-to-video generation via ScryptedAI's /generations/videos/veo-3-i2v endpoint
   *
   * This method calls the ScryptedAI API's Veo 3 Fast i2v endpoint
   * with an image URL and prompt.
   * 
   * @param inputData - Video generation parameters (prompt, image_url, aspect_ratio, duration, generate_audio, resolution)
   * @param idempotencyKey - Optional idempotency key to prevent duplicate requests
   * @param webhookUrl - Optional webhook URL for async job completion notifications
   * @param webhookSecret - Optional webhook secret for signature verification
   * @returns Recipe execution response with job ID
   * @throws ScryptedAuthenticationError - If bearer token is missing
   * @throws ScryptedAPIError - If API returns an error
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeVeo3Generation(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_videos_veo_3;

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      // Validate required fields
      if (!requestBody.prompt || (typeof requestBody.prompt === 'string' && requestBody.prompt.trim().length === 0)) {
        throw new ScryptedAPIError('Prompt is required', 400, { error: 'Prompt is required' });
      }
      if (!requestBody.image_url || (typeof requestBody.image_url === 'string' && requestBody.image_url.trim().length === 0)) {
        throw new ScryptedAPIError('Image URL is required', 400, { error: 'Image URL is required' });
      }

      console.log('🎬 [ScryptedClient] Veo 3 Fast generation request body:', {
        endpoint: `${this.baseUrl}${endpoint}`,
        requestBody: JSON.stringify(requestBody, null, 2),
        hasPrompt: !!requestBody.prompt,
        hasImageUrl: !!requestBody.image_url,
        aspectRatio: requestBody.aspect_ratio,
        resolution: requestBody.resolution
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          console.error('🎬 [ScryptedClient] Veo 3 Fast generation API error response:', {
            status: response.status,
            statusText: response.statusText,
            errorData: errorData
          });
        }

        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke Google Veo 3.1 Fast image-to-video generation via ScryptedAI's /generations/videos/veo-3-1-i2v endpoint
   *
   * This method calls the ScryptedAI API's Veo 3.1 i2v endpoint
   * with an image URL and prompt.
   *
   * @param inputData - Input data with image_url, prompt, aspect_ratio, duration, generate_audio, resolution, retention
   * @param idempotencyKey - Optional idempotency key for deduplication
   * @param webhookUrl - Optional webhook URL for notifications
   * @param webhookSecret - Optional HMAC secret for webhook signing
   *
   * @returns Execution response with job_id and status
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeVeo31Generation(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_videos_veo_3_1;

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      // Validate required fields
      if (!requestBody.prompt || (typeof requestBody.prompt === 'string' && requestBody.prompt.trim().length === 0)) {
        throw new ScryptedAPIError('Prompt is required', 400, { error: 'Prompt is required' });
      }
      if (!requestBody.image_url || (typeof requestBody.image_url === 'string' && requestBody.image_url.trim().length === 0)) {
        throw new ScryptedAPIError('Image URL is required', 400, { error: 'Image URL is required' });
      }

      console.log('🎬 [ScryptedClient] Veo 3.1 generation request body:', {
        endpoint: `${this.baseUrl}${endpoint}`,
        requestBody: JSON.stringify(requestBody, null, 2),
        hasPrompt: !!requestBody.prompt,
        hasImageUrl: !!requestBody.image_url,
        aspectRatio: requestBody.aspect_ratio,
        resolution: requestBody.resolution
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          console.error('🎬 [ScryptedClient] Veo 3.1 generation API error response:', {
            status: response.status,
            statusText: response.statusText,
            errorData: errorData
          });
        }

        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  async invokeSora2Generation(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_videos_sora2;

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      // Validate required fields
      if (!requestBody.prompt || (typeof requestBody.prompt === 'string' && requestBody.prompt.trim().length === 0)) {
        throw new ScryptedAPIError('Prompt is required', 400, { error: 'Prompt is required' });
      }
      if (!requestBody.image_url || (typeof requestBody.image_url === 'string' && requestBody.image_url.trim().length === 0)) {
        throw new ScryptedAPIError('Image URL is required', 400, { error: 'Image URL is required' });
      }

      console.log('🎬 [ScryptedClient] Sora2 generation request body:', {
        endpoint: `${this.baseUrl}${endpoint}`,
        requestBody: JSON.stringify(requestBody, null, 2),
        hasPrompt: !!requestBody.prompt,
        hasImageUrl: !!requestBody.image_url,
        aspectRatio: requestBody.aspect_ratio,
        resolution: requestBody.resolution,
        duration: requestBody.duration
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          console.error('🎬 [ScryptedClient] Sora2 generation API error response:', {
            status: response.status,
            statusText: response.statusText,
            errorData: errorData
          });
        }

        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke ScryptedAI Sora 2 (non-Pro) image-to-video via /generations/videos/sora2-i2v.
   * 720p or auto resolution only; lower cost than Sora 2 Pro.
   */
  async invokeSora2I2VGeneration(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_videos_sora2_i2v;

      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      if (!requestBody.prompt || (typeof requestBody.prompt === 'string' && requestBody.prompt.trim().length === 0)) {
        throw new ScryptedAPIError('Prompt is required', 400, { error: 'Prompt is required' });
      }
      if (!requestBody.image_url || (typeof requestBody.image_url === 'string' && requestBody.image_url.trim().length === 0)) {
        throw new ScryptedAPIError('Image URL is required', 400, { error: 'Image URL is required' });
      }

      const headers = formatHeaders(this._bearerToken);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        if (!response.ok) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          throw new ScryptedAPIError(
            `Sora 2 i2v request failed: ${response.status}`,
            response.status,
            errorData
          );
        }
        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke Grok Imagine image-to-video generation via ScryptedAI's /generations/videos/grok-imagine-i2v endpoint
   *
   * @param inputData - Request body: prompt, image_url, optional duration (1-15s), aspect_ratio, resolution (480p|720p)
   * @param idempotencyKey - Optional idempotency key
   * @param webhookUrl - Optional webhook URL
   * @param webhookSecret - Optional webhook secret
   * @returns Recipe execution response with job_id
   */
  async invokeGrokImagineI2VGeneration(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_videos_grok_imagine_i2v;

      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      if (!requestBody.prompt || (typeof requestBody.prompt === 'string' && requestBody.prompt.trim().length === 0)) {
        throw new ScryptedAPIError('Prompt is required', 400, { error: 'Prompt is required' });
      }
      if (!requestBody.image_url || (typeof requestBody.image_url === 'string' && requestBody.image_url.trim().length === 0)) {
        throw new ScryptedAPIError('Image URL is required', 400, { error: 'Image URL is required' });
      }

      const headers = formatHeaders(this._bearerToken);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.clone().text();
          let errorData;
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = errorText;
          }
          console.error('[ScryptedClient] Grok Imagine I2V API error:', {
            status: response.status,
            statusText: response.statusText,
            errorData,
          });
        }

        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Invoke ScryptedAI Topaz video upscale via /generations/videos/tools/upscale/topaz.
   * Uses FAL.ai Topaz to upscale video resolution (e.g. 720p → 1080p) with optional frame interpolation.
   *
   * @param inputData - Request body: video_url (required), upscale_factor, target_fps, H264_output, retention, reference_id, payment_method
   * @param idempotencyKey - Optional idempotency key
   * @param webhookUrl - Optional webhook URL
   * @param webhookSecret - Optional webhook secret
   * @returns Recipe execution response with job_id and status
   */
  async invokeTopazVideoUpscale(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_videos_tools_upscale_topaz;

      const requestBody: Record<string, any> = {
        video_url: inputData.video_url,
        upscale_factor: inputData.upscale_factor ?? 2.0,
        target_fps: inputData.target_fps,
        H264_output: inputData.H264_output ?? false,
        retention: inputData.retention ?? 259200,
        reference_id: inputData.reference_id,
        payment_method: inputData.payment_method ?? "balance",
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      if (!requestBody.video_url || (typeof requestBody.video_url === "string" && requestBody.video_url.trim().length === 0)) {
        throw new ScryptedAPIError("video_url is required", 400, { error: "video_url is required" });
      }

      const headers = formatHeaders(this._bearerToken);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Get job status by job ID.
   *
   * This method retrieves the current status of a job execution,
   * including the result if completed or error if failed.
   *
   * @param jobId - Job identifier returned from invokeRecipe()
   *
   * @returns Job status response with current status and optional result/error
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = `${ENDPOINTS.jobs}/${jobId}`;
      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "GET",
          headers,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        return (await handleResponse(response)) as JobStatusResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Cancel a job by ID (POST /jobs/{job_id}/cancel).
   * Idempotent: 200 with current job if already COMPLETED/FAILED/CANCELLED.
   * Only the job owner can cancel (403 otherwise).
   *
   * @param jobId - UUID of the job to cancel (same ID returned at creation)
   * @returns Current job (status CANCELLED or unchanged if already terminal)
   */
  async cancelJob(jobId: string): Promise<JobStatusResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }
    if (!jobId || typeof jobId !== "string") {
      throw new ScryptedValidationError("jobId is required and must be a non-empty string");
    }

    const endpoint = `${ENDPOINTS.jobs}/${encodeURIComponent(jobId)}/cancel`;
    const headers = formatHeaders(this._bearerToken);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

    try {
      const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
        method: "POST",
        headers,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      return (await handleResponse(response)) as JobStatusResponse;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof ScryptedError) {
        throw error;
      }
      if (error instanceof Error && error.name === "AbortError") {
        throw new ScryptedTimeoutError("Request timeout");
      }
      throw new ScryptedNetworkError(
        `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  }

  /**
   * Invoke text generation via ScryptedAI's /generations/text/nova-pro endpoint
   *
   * This method calls the ScryptedAI API's Nova Pro text generation endpoint
   * with user and system prompts.
   *
   * @param inputData - Input data with user_prompt, system_prompt, model, max_tokens, temperature, etc.
   * @param idempotencyKey - Optional idempotency key for deduplication
   * @param webhookUrl - Optional webhook URL for notifications
   * @param webhookSecret - Optional HMAC secret for webhook signing
   *
   * @returns Execution response with job_id and status
   *
   * @throws ScryptedAPIError - If API request fails
   * @throws ScryptedAuthenticationError - If bearer token is invalid
   * @throws ScryptedNetworkError - If network request fails
   * @throws ScryptedTimeoutError - If request times out
   */
  async invokeTextGeneration(
    inputData: Record<string, any>,
    idempotencyKey?: string,
    webhookUrl?: string,
    webhookSecret?: string
  ): Promise<RecipeExecutionResponse> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = ENDPOINTS.generations_text_nova_pro;

      // Build request body
      const requestBody: Record<string, any> = {
        ...inputData,
      };

      if (idempotencyKey) {
        requestBody.idempotency_key = idempotencyKey;
      }
      if (webhookUrl) {
        requestBody.webhook_url = webhookUrl;
      }
      if (webhookSecret) {
        requestBody.webhook_secret = webhookSecret;
      }

      // Remove undefined fields
      Object.keys(requestBody).forEach((key) => {
        if (requestBody[key] === undefined) {
          delete requestBody[key];
        }
      });

      // Validate user_prompt exists
      if (!requestBody.user_prompt || (typeof requestBody.user_prompt === 'string' && requestBody.user_prompt.trim().length === 0)) {
        console.error('📝 [ScryptedClient] ERROR: user_prompt is missing or empty!', {
          hasUserPrompt: !!requestBody.user_prompt,
          userPromptType: typeof requestBody.user_prompt,
          allKeys: Object.keys(requestBody)
        });
        throw new ScryptedAPIError('user_prompt is required', 400, { error: 'user_prompt is required' });
      }

      console.log('📝 [ScryptedClient] Text generation request body:', {
        endpoint: `${this.baseUrl}${endpoint}`,
        hasUserPrompt: !!requestBody.user_prompt,
        hasSystemPrompt: !!requestBody.system_prompt,
        userPromptLength: requestBody.user_prompt?.length,
        systemPromptLength: requestBody.system_prompt?.length,
        model: requestBody.model,
        maxTokens: requestBody.max_tokens,
        temperature: requestBody.temperature,
        allKeys: Object.keys(requestBody)
      });

      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        return (await handleResponse(response)) as RecipeExecutionResponse;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * Create a moderation assessment job.
   *
   * Uses the same managed transport/dispatcher lifecycle as all other API calls.
   */
  async createModerationAssessment(
    inputData: Record<string, any>
  ): Promise<{ job_id: string; status: string }> {
    if (!this._bearerToken) {
      throw new ScryptedAuthenticationError(
        "Bearer token required for this operation. " +
          "Create an account first or provide a bearer token."
      );
    }

    return retryWithBackoff(async () => {
      const endpoint = "/moderations/text/assessments";
      const headers = formatHeaders(this._bearerToken);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 1000);

      try {
        const response = await this.fetch(`${this.baseUrl}${endpoint}`, {
          method: "POST",
          headers,
          body: JSON.stringify(inputData),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);
        return (await handleResponse(response)) as { job_id: string; status: string };
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof ScryptedError) {
          throw error;
        }
        if (error instanceof Error && error.name === "AbortError") {
          throw new ScryptedTimeoutError("Request timeout");
        }
        throw new ScryptedNetworkError(
          `Network error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
      }
    }, this.maxRetries);
  }

  /**
   * String representation of client.
   *
   * @returns String representation showing base URL and token status
   */
  toString(): string {
    const tokenStatus = this._bearerToken ? "with token" : "without token";
    return `ScryptedClient(base_url=${this.baseUrl}, ${tokenStatus})`;
  }
}

