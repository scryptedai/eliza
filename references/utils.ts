/**
 * Utility functions for ScryptedAI SDK.
 *
 * This module contains utility functions for bearer token validation,
 * header formatting, response handling, and other common operations
 * used throughout the SDK.
 */

import {
  BEARER_PREFIX,
  TOKEN_PREFIX,
  TOKEN_MIN_LENGTH,
  DEFAULT_HEADERS,
} from "./constants";
import {
  ScryptedPaymentError,
  ScryptedAuthenticationError,
  ScryptedRateLimitError,
  ScryptedAPIError,
  ScryptedTimeoutError,
  ScryptedNetworkError,
  ScryptedValidationError,
} from "./exceptions";

/**
 * Validate bearer token format.
 *
 * This function validates that a bearer token has the correct format
 * and meets minimum length requirements.
 *
 * @param token - The bearer token to validate
 * @returns True if the token is valid, False otherwise
 */
export function validateBearerToken(token: string): boolean {
  if (typeof token !== "string") {
    return false;
  }

  if (!token.startsWith(TOKEN_PREFIX)) {
    return false;
  }

  if (token.length < TOKEN_MIN_LENGTH) {
    return false;
  }

  return true;
}

/**
 * Format request headers with optional bearer token.
 *
 * This function creates a properly formatted headers dictionary
 * for API requests, including authentication headers if a bearer
 * token is provided.
 *
 * @param bearerToken - Optional bearer token for authentication
 * @param additionalHeaders - Additional headers to include
 * @returns Formatted headers dictionary
 */
export function formatHeaders(
  bearerToken?: string,
  additionalHeaders?: Record<string, string>
): Record<string, string> {
  const headers: Record<string, string> = { ...DEFAULT_HEADERS };
  
  if (additionalHeaders) {
    Object.assign(headers, additionalHeaders);
  }

  if (bearerToken) {
    headers["Authorization"] = `${BEARER_PREFIX}${bearerToken}`;
  }

  return headers;
}

/**
 * Handle HTTP response with proper error handling.
 *
 * This function processes HTTP responses and raises appropriate
 * exceptions for different error conditions, including payment
 * required, authentication failures, and rate limiting.
 *
 * @param response - The HTTP response object
 * @returns Parsed JSON response data
 * @throws ScryptedPaymentError - For 402 Payment Required responses
 * @throws ScryptedAuthenticationError - For 401 Unauthorized responses
 * @throws ScryptedRateLimitError - For 429 Too Many Requests responses
 * @throws ScryptedAPIError - For other HTTP error responses
 * @throws ScryptedTimeoutError - For timeout errors
 * @throws ScryptedNetworkError - For network connection errors
 */
export async function handleResponse(response: Response): Promise<any> {
  // Check if response is ok (status 200-299)
  if (!response.ok) {
    let errorData: any;
    try {
      const errorText = await response.clone().text();
      try {
        errorData = JSON.parse(errorText);
      } catch {
        errorData = errorText;
      }
      // Log detailed error for 422 validation errors
      if (response.status === 422) {
        console.error('🔴 [ScryptedClient] Validation error (422):', JSON.stringify(errorData, null, 2));
      }
    } catch {
      errorData = null;
    }

    if (response.status === 402) {
      throw new ScryptedPaymentError(
        "Payment required",
        response.headers.get("Location") || undefined,
        response.headers.get("X-Payment-Amount") || undefined
      );
    } else if (response.status === 401) {
      throw new ScryptedAPIError(
        "Authentication failed",
        response.status,
        errorData
      );
    } else if (response.status === 429) {
      const retryAfter = parseInt(
        response.headers.get("Retry-After") || "60",
        10
      );
      throw new ScryptedRateLimitError(
        "Rate limit exceeded",
        retryAfter
      );
    } else {
      throw new ScryptedAPIError(
        `API error: ${response.statusText}`,
        response.status,
        errorData
      );
    }
  }

  // Parse and return JSON response
  try {
    return await response.json();
  } catch (error) {
    throw new ScryptedNetworkError(
      `Failed to parse response: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}

/**
 * Sleep for a specified number of milliseconds.
 *
 * @param ms - Milliseconds to sleep
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Retry a function with exponential backoff.
 *
 * @param fn - Function to retry
 * @param maxRetries - Maximum number of retries
 * @param backoffFactor - Factor for exponential backoff
 * @returns Result of the function
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  backoffFactor: number = 1.0
): Promise<T> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry on certain errors
      if (
        error instanceof ScryptedAuthenticationError ||
        error instanceof ScryptedValidationError ||
        error instanceof ScryptedPaymentError
      ) {
        throw error;
      }

      // If this was the last attempt, throw the error
      if (attempt >= maxRetries) {
        throw error;
      }

      // Calculate wait time with exponential backoff
      const waitTime = Math.min(
        backoffFactor * Math.pow(2, attempt),
        60.0 * 1000 // Cap at 60 seconds in milliseconds
      );

      await sleep(waitTime);
    }
  }

  throw lastError || new Error("Retry failed");
}

