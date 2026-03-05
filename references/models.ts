/**
 * Type definitions and models for ScryptedAI SDK.
 *
 * This module contains all the data models, enums, and type definitions
 * used throughout the SDK for type safety and validation.
 */

import { randomBytes } from "crypto";

/**
 * Supported currency codes.
 *
 * This enum defines all the currency codes supported by the ScryptedAI API.
 * Currently supports USDC as the primary currency, with extensibility for
 * additional currencies in the future.
 */
export enum CurrencyCode {
  USDC = "USDC",
  USD = "USD",
  BTC = "BTC",
  ETH = "ETH",
}

/**
 * Account status enumeration.
 *
 * This enum defines the possible states of an account in the ScryptedAI system.
 */
export enum AccountStatus {
  ACTIVE = "active",
  SUSPENDED = "suspended",
  PENDING = "pending",
}

/**
 * Individual currency balance.
 *
 * This model represents a single currency balance with proper validation
 * to ensure balances cannot be negative.
 */
export interface Balance {
  currency: CurrencyCode;
  amount: string; // Decimal as string for precision
}

/**
 * Account information response model.
 *
 * This model represents the account information returned by the API,
 * including user ID, balances, creation timestamp, and status.
 */
export interface AccountResponse {
  user_id: string;
  balances: Record<CurrencyCode, string>; // Currency balances as formatted strings
  created_at: string; // ISO 8601 timestamp
  status: AccountStatus;
}

/**
 * Response for newly created account.
 *
 * This model extends AccountResponse to include the bearer token that
 * is only returned when creating a new account.
 */
export interface NewAccountResponse extends AccountResponse {
  bearer_token: string;
}

/**
 * Response from account funding request.
 *
 * This model represents the response when funding an account,
 * including success status, amount funded, and updated balances.
 */
export interface FundingResponse {
  success: boolean;
  amount: number;
  currency: string; // Default: USDC
  balances: Record<string, string>;
  legacy_balance: string;
  funded_at: string;
  user_id: string;
  bearer_token?: string; // Optional bearer token (if new account created)
  message?: string;
}

/**
 * x402 payment invoice extracted from 402 response.
 *
 * This model represents the x402 payment challenge containing
 * payment requirements and authorization details.
 */
export interface X402Invoice {
  scheme: string;
  network: string;
  maxAmountRequired: string;
  resource: string;
  description: string;
  mimeType?: string;
  payTo: string;
  maxTimeoutSeconds: number;
  asset: string;
  extra?: Record<string, any>;
}

/**
 * Type-safe idempotency key.
 *
 * This class provides a type-safe wrapper around idempotency keys
 * with automatic generation of 256-bit random hex strings.
 */
export class IdempotencyKey {
  private value: string;

  private constructor(value: string) {
    this.value = value;
  }

  /**
   * Generate a new 256-bit idempotency key.
   *
   * @returns A new randomly generated idempotency key
   */
  static generate(): IdempotencyKey {
    // Generate 32 random bytes (256 bits) and convert to hex
    const bytes = randomBytes(32);
    const hex = bytes.toString('hex');
    return new IdempotencyKey(hex);
  }

  toString(): string {
    return this.value;
  }

  valueOf(): string {
    return this.value;
  }
}

/**
 * Recipe or ingredient interface schema (input/output only, no steps).
 *
 * This model represents the interface of a recipe or ingredient,
 * including its input and output schemas. Recipe steps are NOT included.
 */
export interface RecipeInterface {
  id: string;
  name: string;
  description?: string;
  input_schema: Record<string, any>; // JSON Schema format
  output_schema: Record<string, any>; // JSON Schema format
}

/**
 * Request body for recipe or ingredient execution.
 *
 * This model represents the request body for executing a recipe or ingredient,
 * including the input data that must match the recipe's input schema.
 */
export interface RecipeExecutionRequest {
  input: Record<string, any>; // Input data matching the recipe's input schema
  idempotency_key?: string; // Idempotency key for deduplication
  webhook_url?: string; // Webhook URL for notifications
  webhook_secret?: string; // HMAC secret for webhook signing
}

/**
 * Response for recipe or ingredient execution.
 *
 * This model represents the response from executing a recipe or ingredient,
 * including the job ID for tracking and optionally the result if completed synchronously.
 */
export interface RecipeExecutionResponse {
  job_id: string;
  status: string; // pending, processing, completed, failed
  result?: Record<string, any>; // Execution result (if completed synchronously)
  estimated_seconds?: number; // Estimated completion time in seconds
}

/**
 * Job status enumeration.
 *
 * This enum defines the possible states of a job in the ScryptedAI system.
 */
export enum JobStatus {
  PENDING = "pending",
  PROCESSING = "processing",
  COMPLETED = "completed",
  FAILED = "failed",
}

/**
 * Response for job status polling.
 *
 * This model represents the response from checking a job's status,
 * including current status and optionally the result if completed.
 */
export interface JobStatusResponse {
  job_id: string;
  status: JobStatus | string; // pending, processing, completed, failed
  result?: Record<string, any>; // Execution result (if completed)
  error?: string; // Error message (if failed)
  estimated_seconds?: number; // Estimated completion time in seconds
  created_at?: string; // ISO 8601 timestamp
  updated_at?: string; // ISO 8601 timestamp
}

