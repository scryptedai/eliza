/**
 * ScryptedAI SDK - Enterprise-grade TypeScript client for ScryptedAI API.
 *
 * This SDK provides a comprehensive, type-safe interface to the ScryptedAI API
 * with automatic bearer token management, comprehensive error handling, and
 * support for both explicit account management (Mode 1) and permissionless
 * flows (Mode 2).
 *
 * Key Features:
 * - Auto-adoption of bearer tokens from account creation
 * - Comprehensive error handling with custom exceptions
 * - Automatic retry logic with exponential backoff
 * - Type-safe idempotency key generation
 * - Full RESTful API support
 * - Enterprise-grade logging and monitoring
 *
 * Usage:
 *     import { ScryptedClient } from './vendor/scryptedai';
 *
 *     // Mode 1: Explicit account management
 *     const client = new ScryptedClient();
 *     const account = await client.createAccount();
 *     const info = await client.getAccountInfo();
 *
 *     // Mode 2: Permissionless flow (future)
 *     const client = new ScryptedClient();
 *     // client.someApiCall() // Handles 402 Payment Required automatically
 */

export { ScryptedClient } from "./client";
export {
  IdempotencyKey,
  CurrencyCode,
  AccountStatus,
} from "./models";
export type {
  AccountResponse,
  NewAccountResponse,
  Balance,
  FundingResponse,
  X402Invoice,
  RecipeInterface,
  RecipeExecutionRequest,
  RecipeExecutionResponse,
  JobStatusResponse,
} from "./models";
export {
  JobStatus,
} from "./models";
export {
  ScryptedError,
  ScryptedAPIError,
  ScryptedAuthenticationError,
  ScryptedPaymentError,
  ScryptedValidationError,
  ScryptedNetworkError,
  ScryptedTimeoutError,
  ScryptedRateLimitError,
} from "./exceptions";
export { X402PaymentHandler } from "./x402-handler";

export const VERSION = "1.0.0";
export const AUTHOR = "ScryptedAI";
export const EMAIL = "support@scrypted.ai";

