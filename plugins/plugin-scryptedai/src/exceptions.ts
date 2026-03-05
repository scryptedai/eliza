/**
 * Custom exception hierarchy for the ScryptedAI plugin.
 *
 * Mirrors the reference SDK exception hierarchy (references/exceptions.ts).
 *
 * Retry semantics (enforced in client.ts retryWithBackoff):
 * - NEVER retry: ScryptedAuthenticationError, ScryptedValidationError, ScryptedPaymentError
 * - RETRY: ScryptedAPIError (5xx), ScryptedNetworkError, ScryptedTimeoutError, ScryptedRateLimitError
 */

export interface ErrorDetails {
  [key: string]: unknown;
}

/** Base exception for all ScryptedAI plugin errors. */
export class ScryptedError extends Error {
  public errorCode?: string;
  public details: ErrorDetails;

  constructor(message: string, errorCode?: string, details?: ErrorDetails) {
    super(message);
    this.name = "ScryptedError";
    this.errorCode = errorCode;
    this.details = details || {};
  }
}

/** HTTP-level API errors. Includes status code and parsed response body. */
export class ScryptedAPIError extends ScryptedError {
  public statusCode: number;
  public responseData?: unknown;

  constructor(message: string, statusCode: number, responseData?: unknown) {
    super(message, `HTTP_${statusCode}`, {
      status_code: statusCode,
      response: responseData,
    });
    this.name = "ScryptedAPIError";
    this.statusCode = statusCode;
    this.responseData = responseData;
  }
}

/**
 * Authentication failures (invalid/missing bearer token).
 * DO NOT RETRY — fix config.
 */
export class ScryptedAuthenticationError extends ScryptedError {
  constructor(message: string) {
    super(message);
    this.name = "ScryptedAuthenticationError";
  }
}

/**
 * 402 Payment Required (x402 permissionless flow).
 * DO NOT RETRY blindly — handle via payment flow or fail.
 */
export class ScryptedPaymentError extends ScryptedAPIError {
  public paymentUrl?: string;
  public amount?: string;

  constructor(message: string, paymentUrl?: string, amount?: string) {
    super(message, 402, { payment_url: paymentUrl, amount });
    this.name = "ScryptedPaymentError";
    this.paymentUrl = paymentUrl;
    this.amount = amount;
  }
}

/**
 * Input validation failures (bad token format, malformed request).
 * DO NOT RETRY — fix input.
 */
export class ScryptedValidationError extends ScryptedError {
  constructor(message: string) {
    super(message);
    this.name = "ScryptedValidationError";
  }
}

/** Network-level errors (connection refused, DNS failure, etc). RETRY. */
export class ScryptedNetworkError extends ScryptedError {
  constructor(message: string) {
    super(message);
    this.name = "ScryptedNetworkError";
  }
}

/** Request timeout. RETRY. */
export class ScryptedTimeoutError extends ScryptedNetworkError {
  constructor(message: string) {
    super(message);
    this.name = "ScryptedTimeoutError";
  }
}

/** 429 Too Many Requests. Includes retry-after hint. RETRY with backoff. */
export class ScryptedRateLimitError extends ScryptedAPIError {
  public retryAfter?: number;

  constructor(message: string, retryAfter?: number) {
    super(message, 429, { retry_after: retryAfter });
    this.name = "ScryptedRateLimitError";
    this.retryAfter = retryAfter;
  }
}
