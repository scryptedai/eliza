/**
 * Test-time mock for @elizaos/plugin-scryptedai.
 *
 * Only the surface plugin-ffm imports: SCRYPTEDAI_SERVICE_TYPE constant
 * and the NormalizedJobResult type. The actual ScryptedAIService is
 * structurally faked per-test (see service.test.ts).
 */

export const SCRYPTEDAI_SERVICE_TYPE = "scryptedai" as const;

export type NormalizedStatus =
  | "pending"
  | "processing"
  | "completed"
  | "failed";

export interface NormalizedJobResult {
  jobId: string;
  status: NormalizedStatus;
  result?: Record<string, unknown>;
  error?: string;
  imageUrl?: string;
  videoUrl?: string;
  text?: string;
}
