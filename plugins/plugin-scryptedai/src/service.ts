/**
 * ScryptedAIService — ElizaOS service wrapping the ScryptedAI client.
 *
 * Responsibilities:
 * - Owns the ScryptedClient (bearer token from runtime settings)
 * - Tracks job lifecycle in an in-memory store
 * - Provides a UNIFIED terminal processor used by both webhook ingestion
 *   and polling — guarantees idempotency on (jobId, status)
 * - Exposes start-and-track helpers that invoke + register + optionally poll
 * - Notifies listeners on terminal events
 *
 * Idempotency guarantee (per guide §3.3, §4.4):
 *   processTerminal() is the ONLY path that mutates a job into terminal state.
 *   It checks the (jobId, status) pair against a processed-Set before acting,
 *   so duplicate webhook deliveries and webhook/poll races are both safe.
 *
 * Limitations (V1):
 * - Job store is in-memory; lost on restart. Callers can recover by
 *   invoking getJobStatus(jobId) manually for any job IDs they persisted.
 */

import { type IAgentRuntime, Service } from "@elizaos/core";
import { isTerminalStatus, normalizeJobPayload } from "./adapter.ts";
import { generateIdempotencyKey, ScryptedClient } from "./client.ts";
import {
  API_BASE_URL,
  ENV_BASE_URL,
  ENV_BEARER_TOKEN,
  ENV_WEBHOOK_SECRET,
  type JobType,
  POLLING_WINDOWS,
  SCRYPTEDAI_SERVICE_TYPE,
} from "./constants.ts";
import { pollJobToCompletion } from "./polling.ts";
import type {
  InvokeOptions,
  JobRecord,
  JobTerminalListener,
  NormalizedJobResult,
  RecipeExecutionResponse,
} from "./types.ts";

export interface IngestResult {
  /** True if this (jobId, status) was already processed — no state change applied. */
  idempotent: boolean;
  /** True if this ingest caused a terminal transition. */
  terminal: boolean;
}

export interface StartJobResult {
  jobId: string;
  /** The full invoke response (may include synchronous result). */
  response: RecipeExecutionResponse;
  /** The job record now tracked in the store. */
  record: JobRecord;
}

export class ScryptedAIService extends Service {
  static serviceType = SCRYPTEDAI_SERVICE_TYPE;
  static serviceName = "ScryptedAI";

  public capabilityDescription =
    "ScryptedAI multimodal generation (images, videos, text) via api.scrypted.ai with webhook + polling job lifecycle.";

  private client!: ScryptedClient;
  private defaultWebhookSecret?: string;

  /**
   * How long terminal jobs linger before eviction. Covers the realistic
   * webhook+poll race window (seconds) with generous slack — the only
   * reason a terminal record is kept is so a late duplicate hits the
   * idempotency check instead of recreating a stub.
   */
  private static readonly TERMINAL_TTL_MS = 5 * 60_000;

  /** In-memory job store keyed by jobId. Terminal entries evict after TERMINAL_TTL_MS. */
  private readonly jobs = new Map<string, JobRecord>();

  /** (jobId:status) tuples already processed by the terminal handler. Evicted with their job. */
  private readonly processed = new Set<string>();

  /** Pending eviction timers, keyed by jobId, so stop() can cancel them. */
  private readonly evictions = new Map<string, ReturnType<typeof setTimeout>>();

  /** Terminal-event listeners (simple fan-out). */
  private readonly listeners = new Set<JobTerminalListener>();

  /** Active polling AbortControllers by jobId (for cancellation on stop). */
  private readonly activePolls = new Map<string, AbortController>();

  // --------------------------------------------------------------------------
  // Service lifecycle
  // --------------------------------------------------------------------------

  static async start(runtime: IAgentRuntime): Promise<ScryptedAIService> {
    const svc = new ScryptedAIService(runtime);

    const token = runtime.getSetting(ENV_BEARER_TOKEN);
    if (typeof token !== "string" || !token) {
      throw new Error(
        `${ENV_BEARER_TOKEN} is required to start the ScryptedAI service`,
      );
    }

    const baseUrl = runtime.getSetting(ENV_BASE_URL);
    svc.client = new ScryptedClient({
      bearerToken: token,
      baseUrl: typeof baseUrl === "string" && baseUrl ? baseUrl : API_BASE_URL,
    });

    const secret = runtime.getSetting(ENV_WEBHOOK_SECRET);
    if (typeof secret === "string" && secret) {
      svc.defaultWebhookSecret = secret;
    }

    runtime.logger.info(
      "[scryptedai] Service started (webhooks=" +
        (svc.defaultWebhookSecret ? "configured" : "not-configured") +
        ")",
    );
    return svc;
  }

  async stop(): Promise<void> {
    // Abort all active polls
    for (const [jobId, controller] of this.activePolls.entries()) {
      controller.abort();
      this.runtime.logger.debug(`[scryptedai] Aborted poll for job ${jobId}`);
    }
    this.activePolls.clear();
    this.listeners.clear();
    for (const t of this.evictions.values()) clearTimeout(t);
    this.evictions.clear();
  }

  // --------------------------------------------------------------------------
  // Public: client access + listener registration
  // --------------------------------------------------------------------------

  /** Direct access to the underlying client for advanced use. */
  getClient(): ScryptedClient {
    return this.client;
  }

  /** Register a listener for terminal job events (completed/failed/cancelled). */
  onTerminal(listener: JobTerminalListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Await a specific job's terminal result. Resolves immediately if the job
   * is already terminal; otherwise blocks until a terminal event fires
   * (via webhook OR polling) or the timeout elapses.
   *
   * If `timeoutMs` is omitted, defaults to the polling window's max-wait for
   * the job's tracked type (or `unknown` if untracked) — never unbounded.
   */
  async awaitJob(
    jobId: string,
    timeoutMs?: number,
  ): Promise<NormalizedJobResult> {
    const existing = this.jobs.get(jobId);
    if (existing?.result && isTerminalStatus(existing.status)) {
      return existing.result;
    }

    const effectiveTimeout =
      timeoutMs ??
      POLLING_WINDOWS[existing?.jobType ?? "unknown"].maxWaitSeconds * 1000;

    return new Promise<NormalizedJobResult>((resolve, reject) => {
      const unsubscribe = this.onTerminal((result) => {
        if (result.jobId !== jobId) return;
        unsubscribe();
        clearTimeout(timer);
        resolve(result);
      });

      const timer = setTimeout(() => {
        unsubscribe();
        reject(
          new Error(
            `awaitJob timed out after ${effectiveTimeout}ms (jobId=${jobId})`,
          ),
        );
      }, effectiveTimeout);
    });
  }

  // --------------------------------------------------------------------------
  // Public: job store queries
  // --------------------------------------------------------------------------

  getJob(jobId: string): JobRecord | undefined {
    return this.jobs.get(jobId);
  }

  listJobs(): JobRecord[] {
    return [...this.jobs.values()];
  }

  /** Poll the API for a job's current status and normalize it. Does NOT mutate store. */
  async fetchJobStatus(jobId: string): Promise<NormalizedJobResult> {
    const raw = await this.client.getJobStatus(jobId);
    return normalizeJobPayload(raw as unknown as Record<string, unknown>);
  }

  // --------------------------------------------------------------------------
  // Unified terminal processor (webhook + polling both route here)
  // --------------------------------------------------------------------------

  /**
   * Process a (possibly terminal) job result. IDEMPOTENT on (jobId, status).
   *
   * This is the single "writer" for terminal transitions. Both the webhook
   * route handler and the polling loop call this method. If the same
   * (jobId, status) pair arrives twice (replayed webhook, webhook+poll race),
   * the second call is a no-op and returns { idempotent: true }.
   *
   * Non-terminal updates (pending/processing) just refresh updatedAt.
   */
  processTerminal(normalized: NormalizedJobResult): IngestResult {
    const { jobId, status } = normalized;
    const key = `${jobId}:${status}`;

    if (this.processed.has(key)) {
      return { idempotent: true, terminal: isTerminalStatus(status) };
    }

    let record = this.jobs.get(jobId);
    const now = Date.now();

    if (!record) {
      // Job we never tracked (e.g. invoked outside the service). Create a
      // minimal record so listeners still fire.
      record = {
        jobId,
        jobType: "unknown",
        status,
        createdAt: now,
        updatedAt: now,
        polling: false,
      };
      this.jobs.set(jobId, record);
    }

    // If already terminal, ignore conflicting late-arriving statuses
    // (e.g. stale "processing" webhook after a "completed" poll).
    if (isTerminalStatus(record.status) && !isTerminalStatus(status)) {
      this.processed.add(key);
      return { idempotent: true, terminal: false };
    }

    record.status = status;
    record.updatedAt = now;

    if (isTerminalStatus(status)) {
      record.result = normalized;
      record.rawError = normalized.error;
      record.polling = false;
      this.processed.add(key);

      // Stop any active poller for this job
      const ctrl = this.activePolls.get(jobId);
      if (ctrl) {
        ctrl.abort();
        this.activePolls.delete(jobId);
      }

      // Fan out to listeners
      for (const listener of this.listeners) {
        try {
          listener(normalized);
        } catch (error) {
          this.runtime.logger.warn(
            `[scryptedai] Terminal listener threw: ${
              error instanceof Error ? error.message : String(error)
            }`,
          );
        }
      }

      this.runtime.logger.info(
        `[scryptedai] Job ${jobId} → ${status}` +
          (normalized.error ? " (with error)" : ""),
      );

      this.scheduleEviction(jobId);
      return { idempotent: false, terminal: true };
    }

    // Non-terminal: just mark processed for this particular status tuple
    this.processed.add(key);
    return { idempotent: false, terminal: false };
  }

  /** Webhook entrypoint — called by the route handler. Thin wrapper over processTerminal. */
  ingestWebhook(normalized: NormalizedJobResult): IngestResult {
    return this.processTerminal(normalized);
  }

  /**
   * Schedule a terminal job for eviction from `jobs` and `processed`.
   * Idempotent — re-scheduling resets the timer.
   *
   * Trade-off: a duplicate webhook arriving AFTER the TTL window will be
   * treated as fresh and re-fire listeners. ScryptedAI retry policy caps
   * well under 5 minutes; downstream consumers (AVB's eagerExecuteForJob)
   * are independently idempotent via task-row lookup. Bounded memory wins.
   */
  private scheduleEviction(jobId: string): void {
    const existing = this.evictions.get(jobId);
    if (existing) clearTimeout(existing);

    const timer = setTimeout(() => {
      this.evictions.delete(jobId);
      this.jobs.delete(jobId);
      // Sweep all status tuples for this jobId. The set is small (≤5 statuses
      // per job: pending/processing/completed/failed/cancelled).
      const prefix = `${jobId}:`;
      for (const key of this.processed) {
        if (key.startsWith(prefix)) this.processed.delete(key);
      }
    }, ScryptedAIService.TERMINAL_TTL_MS);

    // Don't keep the process alive just for eviction housekeeping.
    if (typeof timer === "object" && "unref" in timer) {
      (timer as { unref: () => void }).unref();
    }
    this.evictions.set(jobId, timer);
  }

  // --------------------------------------------------------------------------
  // Public: invoke-and-track helpers
  // --------------------------------------------------------------------------

  private resolveInvokeOptions(opts?: InvokeOptions): InvokeOptions {
    return {
      idempotencyKey: opts?.idempotencyKey ?? generateIdempotencyKey(),
      webhookUrl: opts?.webhookUrl,
      webhookSecret: opts?.webhookSecret ?? this.defaultWebhookSecret,
    };
  }

  /**
   * Register a job in the store after invocation and optionally start a
   * polling fallback. Returns immediately — polling (if enabled) runs in
   * the background and routes results through processTerminal().
   */
  private trackJob(
    response: RecipeExecutionResponse,
    jobType: JobType,
    opts: { pollFallback: boolean; metadata?: Record<string, unknown> },
  ): JobRecord {
    const now = Date.now();
    const normalized = normalizeJobPayload(
      response as unknown as Record<string, unknown>,
    );

    const record: JobRecord = {
      jobId: response.job_id,
      jobType,
      status: normalized.status,
      createdAt: now,
      updatedAt: now,
      metadata: opts.metadata,
      polling: false,
    };
    this.jobs.set(response.job_id, record);

    // If the invoke response was already terminal (fast synchronous completion),
    // process it immediately.
    if (isTerminalStatus(normalized.status)) {
      this.processTerminal(normalized);
      return record;
    }

    // Start polling fallback if requested (or if no webhook configured)
    if (opts.pollFallback) {
      this.startPolling(response.job_id, jobType);
    }

    return record;
  }

  private startPolling(jobId: string, jobType: JobType): void {
    if (this.activePolls.has(jobId)) return;

    const controller = new AbortController();
    this.activePolls.set(jobId, controller);

    const record = this.jobs.get(jobId);
    if (record) record.polling = true;

    // Fire-and-forget background poll
    pollJobToCompletion(this.client, jobId, {
      jobType,
      signal: controller.signal,
      onPoll: (result) => {
        // Non-terminal updates refresh the store but do NOT trigger listeners
        if (!isTerminalStatus(result.status)) {
          this.processTerminal(result);
        }
      },
    })
      .then((pollResult) => {
        this.activePolls.delete(jobId);
        if (pollResult.completed) {
          this.processTerminal(pollResult.result);
        } else if (pollResult.timedOut) {
          // Max-wait exceeded — synthesize a failure
          this.runtime.logger.warn(
            `[scryptedai] Job ${jobId} polling timed out after ${pollResult.attempts} attempts`,
          );
          this.processTerminal({
            jobId,
            status: "failed",
            error: `Polling timed out after max-wait window (jobType=${jobType})`,
          });
        }
      })
      .catch((error) => {
        this.activePolls.delete(jobId);
        this.runtime.logger.error(
          `[scryptedai] Poll for job ${jobId} failed: ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
        this.processTerminal({
          jobId,
          status: "failed",
          error: error instanceof Error ? error.message : String(error),
        });
      });
  }

  /**
   * Start (or restart) polling for a known job. Useful for recovery scenarios
   * where the app restarted and the caller has persisted job IDs.
   */
  resumePolling(jobId: string, jobType: JobType = "unknown"): void {
    if (!this.jobs.has(jobId)) {
      const now = Date.now();
      this.jobs.set(jobId, {
        jobId,
        jobType,
        status: "pending",
        createdAt: now,
        updatedAt: now,
        polling: false,
      });
    }
    this.startPolling(jobId, jobType);
  }

  // --------------------------------------------------------------------------
  // High-level invoke helpers — execute + track + (optionally) poll
  //
  // Each wraps the corresponding client method, registers the job in the
  // store, and starts a polling fallback unless the caller provides a
  // webhook URL (in which case polling is optional — controlled by the
  // `pollFallback` flag, default true as a safety net).
  // --------------------------------------------------------------------------

  async startRecipe(
    recipeId: string,
    inputData: Record<string, unknown>,
    opts?: InvokeOptions & {
      jobType?: JobType;
      pollFallback?: boolean;
      metadata?: Record<string, unknown>;
    },
  ): Promise<StartJobResult> {
    const inv = this.resolveInvokeOptions(opts);
    const response = await this.client.invokeRecipe(recipeId, inputData, inv);
    const record = this.trackJob(response, opts?.jobType ?? "unknown", {
      pollFallback: opts?.pollFallback ?? true,
      metadata: opts?.metadata,
    });
    return { jobId: response.job_id, response, record };
  }

  async startImageGeneration(
    methodName:
      | "invokeImageGeneration"
      | "invokeNanoBananaGeneration"
      | "invokeNanoBananaProGeneration"
      | "invokeNanoBananaEditGeneration"
      | "invokeNanoBananaProEditGeneration"
      | "invokeSeedream4Generation"
      | "invokeFlux2ProGeneration"
      | "invokeGrokImagineImageGeneration",
    inputData: Record<string, unknown>,
    opts?: InvokeOptions & {
      pollFallback?: boolean;
      metadata?: Record<string, unknown>;
    },
  ): Promise<StartJobResult> {
    const inv = this.resolveInvokeOptions(opts);
    const response = await this.client[methodName](inputData, inv);
    const record = this.trackJob(response, "image", {
      pollFallback: opts?.pollFallback ?? true,
      metadata: opts?.metadata,
    });
    return { jobId: response.job_id, response, record };
  }

  async startVideoGeneration(
    methodName:
      | "invokeVideoGeneration"
      | "invokeNovaReelGeneration"
      | "invokeHailuo23Generation"
      | "invokeVeo3Generation"
      | "invokeVeo31Generation"
      | "invokeSora2Generation"
      | "invokeSora2I2VGeneration"
      | "invokeGrokImagineI2VGeneration"
      | "invokeTopazVideoUpscale",
    inputData: Record<string, unknown>,
    opts?: InvokeOptions & {
      pollFallback?: boolean;
      metadata?: Record<string, unknown>;
    },
  ): Promise<StartJobResult> {
    const inv = this.resolveInvokeOptions(opts);
    const response = await this.client[methodName](inputData, inv);
    const record = this.trackJob(response, "video", {
      pollFallback: opts?.pollFallback ?? true,
      metadata: opts?.metadata,
    });
    return { jobId: response.job_id, response, record };
  }

  async startTextGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions & {
      pollFallback?: boolean;
      metadata?: Record<string, unknown>;
    },
  ): Promise<StartJobResult> {
    const inv = this.resolveInvokeOptions(opts);
    const response = await this.client.invokeTextGeneration(inputData, inv);
    const record = this.trackJob(response, "text", {
      pollFallback: opts?.pollFallback ?? true,
      metadata: opts?.metadata,
    });
    return { jobId: response.job_id, response, record };
  }
}
