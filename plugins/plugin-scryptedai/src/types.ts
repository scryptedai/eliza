/**
 * Type definitions for the ScryptedAI plugin.
 *
 * Covers both raw API shapes (snake_case, as returned by api.scrypted.ai)
 * and normalized plugin-internal shapes (camelCase).
 */

import type { JobType } from "./constants.ts";

// ----------------------------------------------------------------------------
// Raw API shapes (snake_case — as sent/received over the wire)
// ----------------------------------------------------------------------------

export type RawJobStatus =
  | "pending"
  | "processing"
  | "completed"
  | "failed"
  | "cancelled"
  | string; // API may return mixed casing; always normalize before comparing

export interface RecipeExecutionResponse {
  job_id: string;
  status: RawJobStatus;
  result?: Record<string, unknown>;
  estimated_seconds?: number;
}

/** GET /accounts/me — account info for an authenticated client. */
export interface AccountResponse {
  user_id: string;
  balances: Record<string, string>; // currency code → formatted decimal string
  created_at: string; // ISO 8601
  status?: string; // "active" | "suspended" | "pending" — not always present on new accounts
}

/** POST /accounts — newly created account (includes bearer token exactly once). */
export interface NewAccountResponse extends AccountResponse {
  bearer_token: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: RawJobStatus;
  result?: Record<string, unknown>;
  result_data?: Record<string, unknown>;
  output?: Record<string, unknown>;
  error?: string | { message?: string; [k: string]: unknown };
  error_message?: string;
  estimated_seconds?: number;
  created_at?: string;
  updated_at?: string;
}

export interface RecipeInterface {
  id: string;
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
  output_schema: Record<string, unknown>;
}

// ----------------------------------------------------------------------------
// Normalized shapes (camelCase — produced by adapter.ts for internal use)
// ----------------------------------------------------------------------------

export type NormalizedStatus =
  | "pending"
  | "processing"
  | "completed"
  | "failed"
  | "cancelled";

export interface NormalizedJobResult {
  jobId: string;
  status: NormalizedStatus;
  /** Merged result payload (result_data / result / output unified) */
  result?: Record<string, unknown>;
  /** Single error string (error_message / error.message / error collapsed) */
  error?: string;
  /** Convenience extractions (may be undefined if not present) */
  imageUrl?: string;
  videoUrl?: string;
  text?: string;
}

// ----------------------------------------------------------------------------
// Job store (service-internal)
// ----------------------------------------------------------------------------

export interface JobRecord {
  jobId: string;
  jobType: JobType;
  status: NormalizedStatus;
  /** epoch ms when the job was first tracked */
  createdAt: number;
  /** epoch ms of last state update */
  updatedAt: number;
  /** Normalized result once terminal; undefined while pending/processing */
  result?: NormalizedJobResult;
  /** Full server-side error message (not for end-user display) */
  rawError?: string;
  /** Optional metadata supplied at registration time */
  metadata?: Record<string, unknown>;
  /** Whether a poller is actively watching this job */
  polling: boolean;
}

// ----------------------------------------------------------------------------
// Invocation options (shared across all invoke* methods)
// ----------------------------------------------------------------------------

export interface InvokeOptions {
  idempotencyKey?: string;
  webhookUrl?: string;
  webhookSecret?: string;
}

/** Terminal-event listener signature. */
export type JobTerminalListener = (result: NormalizedJobResult) => void;
