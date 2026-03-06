/**
 * Test-time mock for @elizaos/plugin-scryptedai.
 *
 * Exposes only the types and constants plugin-avb imports.
 * ScryptedAIService instances in tests are plain fake objects —
 * no need to mock the class itself.
 */

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------

export const SCRYPTEDAI_SERVICE_TYPE = "scryptedai" as const;

export type JobType = "text" | "image" | "video" | "unknown";

// ----------------------------------------------------------------------------
// Result types
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
  result?: Record<string, unknown>;
  error?: string;
  imageUrl?: string;
  videoUrl?: string;
  text?: string;
}

export interface JobRecord {
  jobId: string;
  jobType: JobType;
  status: NormalizedStatus;
  createdAt: number;
  updatedAt: number;
  result?: NormalizedJobResult;
  rawError?: string;
  metadata?: Record<string, unknown>;
  polling: boolean;
}

export type JobTerminalListener = (result: NormalizedJobResult) => void;

export interface StartJobResult {
  jobId: string;
  response: unknown;
  record: JobRecord;
}

// ----------------------------------------------------------------------------
// Service class shape (structural placeholder — tests pass fake objects)
// ----------------------------------------------------------------------------

export declare class ScryptedAIService {
  static serviceType: "scryptedai";
  onTerminal(listener: JobTerminalListener): () => void;
  getJob(jobId: string): JobRecord | undefined;
  fetchJobStatus(jobId: string): Promise<NormalizedJobResult>;
  resumePolling(jobId: string, jobType?: JobType): void;
  startTextGeneration(
    inputData: Record<string, unknown>,
    opts?: Record<string, unknown>,
  ): Promise<StartJobResult>;
  startImageGeneration(
    methodName: string,
    inputData: Record<string, unknown>,
    opts?: Record<string, unknown>,
  ): Promise<StartJobResult>;
}
