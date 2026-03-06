/**
 * Core types for the AVB (Autonomous Virtual Being) pipeline.
 *
 * The pipeline is a db-persisted state machine:
 *   TEXT_PHASE → IMAGE_PHASE → DELIVER → (done)
 *
 * Each phase is a separate Task row. Workers are single-purpose idempotent
 * executors; when a phase completes it reports back to AvbService, which
 * consults the PIPELINE graph and spawns the next Task (carrying context forward).
 */

import type { Task, TaskMetadata, UUID } from "@elizaos/core";
import type { JobType, NormalizedJobResult } from "@elizaos/plugin-scryptedai";

// ----------------------------------------------------------------------------
// Phase names (state machine states)
// ----------------------------------------------------------------------------

export type PhaseName = "TEXT_PHASE" | "IMAGE_PHASE" | "DELIVER";

// ----------------------------------------------------------------------------
// Run context — carried through every phase of a single run
// ----------------------------------------------------------------------------

/**
 * Context accumulated across phases. Serialized into Task.metadata so it
 * survives restarts. Each phase-complete callback appends its output before
 * the next phase is spawned.
 */
export interface AvbRunContext {
  /** Stable identifier for the whole pipeline run (all phases share it). */
  runId: UUID;
  /** Room to deliver the final result into. */
  roomId: UUID;
  /** Message that triggered the run (if any). */
  triggeringMessageId?: UUID;
  /** Flattened character identity string, frozen at createRun(). */
  characterDigest: string;
  /** Output of TEXT_PHASE: the LLM-drafted image prompt. */
  imagePrompt?: string;
  /** Output of IMAGE_PHASE: final asset URL. */
  imageUrl?: string;
  /** Error message if a phase failed (for DELIVER-as-failure-reporter). */
  error?: string;
}

// ----------------------------------------------------------------------------
// Task metadata shape for AVB phase tasks
// ----------------------------------------------------------------------------

/**
 * Metadata stored on each phase Task. Extends core TaskMetadata with our
 * phase-specific fields. Fully serializable (all JSON-safe primitives).
 */
export interface AvbPhaseMetadata extends TaskMetadata {
  /** Which phase this Task represents. */
  phase: PhaseName;
  /** Accumulated run context (persisted with the Task row). */
  runContext: AvbRunContext;
  /** ScryptedAI job id — set on first worker tick after submission. */
  scryptedJobId?: string;
  /** Absolute deadline (epoch ms). If now > deadlineAt, phase is failed. */
  deadlineAt: number;
  /** How often TaskService should re-tick this task (ms). */
  updateInterval: number;
}

// ----------------------------------------------------------------------------
// Pipeline graph spec
// ----------------------------------------------------------------------------

export interface PhaseSpec {
  /** Name of the TaskWorker registered for this phase. */
  workerName: string;
  /**
   * ScryptedAI job type (used for resumePolling windowing). "unknown" for
   * phases that don't submit scryptedai jobs (DELIVER).
   */
  jobType: JobType;
  /** How long (ms) to wait before giving up on this phase. */
  deadlineMs: number;
  /** Next phase to spawn on success, or null for terminal. */
  next: PhaseName | null;
}

// ----------------------------------------------------------------------------
// Terminal phase report (worker → AvbService)
// ----------------------------------------------------------------------------

export interface PhaseCompleteReport {
  phase: PhaseName;
  taskId: UUID;
  /** Updated context with this phase's output merged in. */
  ctx: AvbRunContext;
  /** Raw scryptedai result (for logging/debugging). */
  rawResult?: NormalizedJobResult;
}

export interface PhaseFailedReport {
  phase: PhaseName;
  taskId: UUID;
  ctx: AvbRunContext;
  error: string;
}

// ----------------------------------------------------------------------------
// Minimal runtime surface this plugin actually uses (for structural casts)
// ----------------------------------------------------------------------------

/**
 * Structural subset of IAgentRuntime that AvbService and workers call.
 * Declared locally so tests can pass plain objects matching this shape.
 * Real IAgentRuntime satisfies this interface structurally.
 */
export interface AvbRuntimeSurface {
  agentId: UUID;
  character: {
    name?: string;
    bio?: string[] | string;
    topics?: string[];
    adjectives?: string[];
    system?: string;
    style?: { all?: string[]; chat?: string[]; post?: string[] };
  };
  logger: {
    info: (msg: string) => void;
    warn: (msg: string) => void;
    error: (msg: string) => void;
    debug: (msg: string) => void;
  };
  getService<T = unknown>(type: string): T | undefined;
  getServiceLoadPromise(type: string): Promise<unknown>;
  getSetting(key: string): unknown;
  // Task CRUD (from IDatabaseAdapter)
  createTask(task: Task): Promise<UUID>;
  getTask(id: UUID): Promise<Task | null>;
  getTasks(params: { tags?: string[]; roomId?: UUID }): Promise<Task[]>;
  updateTask(id: UUID, task: Partial<Task>): Promise<void>;
  deleteTask(id: UUID): Promise<void>;
  // Memory delivery
  createMemory(
    memory: {
      entityId?: UUID;
      agentId?: UUID;
      roomId: UUID;
      content: Record<string, unknown>;
      createdAt?: number;
    },
    tableName: string,
  ): Promise<UUID>;
  getMemories(params: {
    roomId: UUID;
    tableName: string;
    count?: number;
  }): Promise<Array<{ content?: Record<string, unknown> }>>;
  emitEvent(name: string, payload: unknown): Promise<void>;
  // Task worker registration
  registerTaskWorker(worker: {
    name: string;
    execute: (
      runtime: unknown,
      options: Record<string, unknown>,
      task: Task,
    ) => Promise<void>;
  }): void;
}
