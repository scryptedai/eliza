/**
 * Custom exception hierarchy for ScryptedAI SDK.
 *
 * This module defines a comprehensive exception hierarchy that provides
 * detailed error information and proper error handling for different
 * types of failures that can occur when interacting with the ScryptedAI API.
 */

export interface ErrorDetails {
  [key: string]: any;
}

/**
 * Base exception for all ScryptedAI SDK errors.
 *
 * This is the root exception class that all other SDK exceptions inherit from.
 * It provides a consistent interface for error handling across the SDK.
 */
export class ScryptedError extends Error {
  message: string;
  errorCode?: string;
  details: ErrorDetails;

  constructor(
    message: string,
    errorCode?: string,
    details?: ErrorDetails
  ) {
    super(message);
    this.name = "ScryptedError";
    this.message = message;
    this.errorCode = errorCode;
    this.details = details || {};
  }
}

/**
 * API-related errors from the ScryptedAI service.
 *
 * This exception is raised when the API returns an error response.
 * It includes the HTTP status code and response data for debugging.
 */
export class ScryptedAPIError extends ScryptedError {
  statusCode: number;
  responseData?: any;

  constructor(
    message: string,
    statusCode: number,
    responseData?: any
  ) {
    super(
      message,
      `HTTP_${statusCode}`,
      { status_code: statusCode, response: responseData }
    );
    this.name = "ScryptedAPIError";
    this.statusCode = statusCode;
    this.responseData = responseData;
  }
}

/**
 * Authentication-related errors.
 *
 * This exception is raised when authentication fails, such as when
 * an invalid bearer token is provided or when no bearer token is
 * available for a protected endpoint.
 */
export class ScryptedAuthenticationError extends ScryptedError {
  constructor(message: string) {
    super(message);
    this.name = "ScryptedAuthenticationError";
  }
}

/**
 * Payment-related errors (402 Payment Required).
 *
 * This exception is raised when the API returns a 402 Payment Required
 * response, typically in permissionless flows where payment is needed
 * to create an account or continue processing.
 */
export class ScryptedPaymentError extends ScryptedAPIError {
  paymentUrl?: string;
  amount?: string;

  constructor(
    message: string,
    paymentUrl?: string,
    amount?: string
  ) {
    super(
      message,
      402,
      { payment_url: paymentUrl, amount }
    );
    this.name = "ScryptedPaymentError";
    this.paymentUrl = paymentUrl;
    this.amount = amount;
  }
}

/**
 * Input validation errors.
 *
 * This exception is raised when input validation fails, such as
 * invalid bearer token format or malformed request data.
 */
export class ScryptedValidationError extends ScryptedError {
  constructor(message: string) {
    super(message);
    this.name = "ScryptedValidationError";
  }
}

/**
 * Network-related errors.
 *
 * This exception is raised when network-level errors occur, such as
 * connection failures or DNS resolution issues.
 */
export class ScryptedNetworkError extends ScryptedError {
  constructor(message: string) {
    super(message);
    this.name = "ScryptedNetworkError";
  }
}

/**
 * Request timeout errors.
 *
 * This exception is raised when a request times out, either due to
 * network latency or server processing time exceeding the timeout limit.
 */
export class ScryptedTimeoutError extends ScryptedNetworkError {
  constructor(message: string) {
    super(message);
    this.name = "ScryptedTimeoutError";
  }
}

/**
 * Rate limiting errors.
 *
 * This exception is raised when the API rate limit is exceeded.
 * It includes the retry-after time for when the client can retry.
 */
export class ScryptedRateLimitError extends ScryptedAPIError {
  retryAfter?: number;

  constructor(
    message: string,
    retryAfter?: number
  ) {
    super(
      message,
      429,
      { retry_after: retryAfter }
    );
    this.name = "ScryptedRateLimitError";
    this.retryAfter = retryAfter;
  }
}

