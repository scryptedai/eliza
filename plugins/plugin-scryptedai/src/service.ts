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
 * Memory bounds:
 * - Concurrent non-terminal jobs are soft-capped at MAX_INFLIGHT_JOBS.
 *   start*() calls await a free slot when at the cap (backpressure).
 *   We do NOT spill deferred requests to a disk queue here — see the
 *   rationale in constants.ts: callers that need durability persist their
 *   intent in the ElizaOS runtime task DB and retry idempotently.
 * - Terminal job records are evicted after TERMINAL_RETENTION_MS via a
 *   periodic sweep, so the store cannot grow without bound.
 *
 * Limitations:
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
  EVICTION_SWEEP_INTERVAL_MS,
  type JobType,
  MAX_INFLIGHT_JOBS,
  SCRYPTEDAI_SERVICE_TYPE,
  TERMINAL_RETENTION_MS,
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

  /** In-memory job store keyed by jobId. */
  private readonly jobs = new Map<string, JobRecord>();

  /** (jobId:status) tuples already processed by the terminal handler. */
  private readonly processed = new Set<string>();

  /** Terminal-event listeners (simple fan-out). */
  private readonly listeners = new Set<JobTerminalListener>();

  /** Active polling AbortControllers by jobId (for cancellation on stop). */
  private readonly activePolls = new Map<string, AbortController>();

  /** Count of currently-held in-flight slots (≤ MAX_INFLIGHT_JOBS, soft). */
  private inflight = 0;

  /** Resolvers for callers awaiting an in-flight slot (FIFO). */
  private readonly slotWaiters: Array<() => void> = [];

  /** jobIds that currently hold an in-flight slot (released on terminal). */
  private readonly heldSlots = new Set<string>();

  /** Periodic eviction sweep handle. */
  private evictionTimer?: ReturnType<typeof setInterval>;

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

    // Periodic eviction of terminal job records keeps the store bounded.
    // .unref() so this timer alone doesn't keep the Node process alive.
    svc.evictionTimer = setInterval(
      () => svc.pruneTerminalJobs(),
      EVICTION_SWEEP_INTERVAL_MS,
    );
    svc.evictionTimer.unref?.();

    runtime.logger.info(
      "[scryptedai] Service started (webhooks=" +
        (svc.defaultWebhookSecret ? "configured" : "not-configured") +
        `, inflight-cap=${MAX_INFLIGHT_JOBS})`,
    );
    return svc;
  }

  async stop(): Promise<void> {
    if (this.evictionTimer) {
      clearInterval(this.evictionTimer);
      this.evictionTimer = undefined;
    }
    // Release any callers awaiting a slot so they don't hang forever.
    while (this.slotWaiters.length > 0) {
      this.slotWaiters.shift()?.();
    }
    // Abort all active polls
    for (const [jobId, controller] of this.activePolls.entries()) {
      controller.abort();
      this.runtime.logger.debug(`[scryptedai] Aborted poll for job ${jobId}`);
    }
    this.activePolls.clear();
    this.listeners.clear();
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
   */
  async awaitJob(
    jobId: string,
    timeoutMs?: number,
  ): Promise<NormalizedJobResult> {
    const existing = this.jobs.get(jobId);
    if (existing?.result && isTerminalStatus(existing.status)) {
      return existing.result;
    }

    return new Promise<NormalizedJobResult>((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | undefined;

      const unsubscribe = this.onTerminal((result) => {
        if (result.jobId !== jobId) return;
        unsubscribe();
        if (timer) clearTimeout(timer);
        resolve(result);
      });

      if (timeoutMs !== undefined) {
        timer = setTimeout(() => {
          unsubscribe();
          reject(
            new Error(
              `awaitJob timed out after ${timeoutMs}ms (jobId=${jobId})`,
            ),
          );
        }, timeoutMs);
      }
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

  /** Number of currently-held in-flight slots (for observability/tests). */
  inflightCount(): number {
    return this.inflight;
  }

  /**
   * Evict terminal job records older than `maxAgeMs` (default: configured
   * retention window) and prune their `processed` keys. Returns count evicted.
   * Called periodically by the eviction sweep; also callable directly.
   */
  pruneTerminalJobs(maxAgeMs: number = TERMINAL_RETENTION_MS): number {
    const now = Date.now();
    let evicted = 0;
    for (const [jobId, record] of this.jobs) {
      if (
        record.terminalAt !== undefined &&
        now - record.terminalAt > maxAgeMs
      ) {
        this.jobs.delete(jobId);
        const prefix = `${jobId}:`;
        for (const key of this.processed) {
          if (key.startsWith(prefix)) this.processed.delete(key);
        }
        evicted++;
      }
    }
    if (evicted > 0) {
      this.runtime.logger.debug(
        `[scryptedai] Evicted ${evicted} terminal job record(s) from store`,
      );
    }
    return evicted;
  }

  // --------------------------------------------------------------------------
  // In-flight slot management (backpressure)
  // --------------------------------------------------------------------------

  /**
   * Reserve an in-flight slot. Resolves immediately if below the cap;
   * otherwise enqueues the caller FIFO and resolves when a slot frees
   * (i.e., when a tracked job goes terminal). Soft limit: see constants.ts
   * for why deferred requests are NOT spilled to disk at this layer.
   */
  private async acquireSlot(): Promise<void> {
    if (this.inflight < MAX_INFLIGHT_JOBS) {
      this.inflight++;
      return;
    }
    this.runtime.logger.debug(
      `[scryptedai] In-flight cap (${MAX_INFLIGHT_JOBS}) reached; awaiting slot`,
    );
    await new Promise<void>((resolve) => this.slotWaiters.push(resolve));
    this.inflight++;
  }

  /** Release an in-flight slot and wake the next waiter (if any). */
  private releaseSlot(): void {
    if (this.inflight > 0) this.inflight--;
    const next = this.slotWaiters.shift();
    if (next) next();
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
      record.terminalAt = now;
      this.processed.add(key);

      // Free the in-flight slot this job was holding (if it was started
      // via a start*() helper; webhook-only / resumePolling jobs hold none).
      if (this.heldSlots.delete(jobId)) {
        this.releaseSlot();
      }

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
  //
  // All start*() helpers are gated by the in-flight semaphore: when
  // MAX_INFLIGHT_JOBS non-terminal jobs are already tracked, the call
  // awaits a free slot before invoking the upstream API. The slot is
  // released when the job reaches a terminal state (processTerminal),
  // or immediately if the invoke itself throws.
  // --------------------------------------------------------------------------

  /**
   * Acquire a slot, run `invoke()` to obtain a RecipeExecutionResponse,
   * mark the resulting jobId as holding the slot, then track it.
   *
   * heldSlots.add() must happen BEFORE trackJob() because trackJob may
   * synchronously call processTerminal() (fast-completing job), which
   * needs to find and release the held slot.
   */
  private async startTracked(
    invoke: () => Promise<RecipeExecutionResponse>,
    jobType: JobType,
    track: { pollFallback: boolean; metadata?: Record<string, unknown> },
  ): Promise<StartJobResult> {
    await this.acquireSlot();
    let jobId: string | undefined;
    try {
      const response = await invoke();
      jobId = response.job_id;
      this.heldSlots.add(jobId);
      const record = this.trackJob(response, jobType, track);
      return { jobId, response, record };
    } catch (err) {
      // Invoke failed (or trackJob threw) before the job could reach
      // processTerminal — release the slot we acquired.
      if (jobId === undefined || this.heldSlots.delete(jobId)) {
        this.releaseSlot();
      }
      throw err;
    }
  }

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
    return this.startTracked(
      () => this.client.invokeRecipe(recipeId, inputData, inv),
      opts?.jobType ?? "unknown",
      { pollFallback: opts?.pollFallback ?? true, metadata: opts?.metadata },
    );
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
    return this.startTracked(
      () => this.client[methodName](inputData, inv),
      "image",
      { pollFallback: opts?.pollFallback ?? true, metadata: opts?.metadata },
    );
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
    return this.startTracked(
      () => this.client[methodName](inputData, inv),
      "video",
      { pollFallback: opts?.pollFallback ?? true, metadata: opts?.metadata },
    );
  }

  async startTextGeneration(
    inputData: Record<string, unknown>,
    opts?: InvokeOptions & {
      pollFallback?: boolean;
      metadata?: Record<string, unknown>;
    },
  ): Promise<StartJobResult> {
    const inv = this.resolveInvokeOptions(opts);
    return this.startTracked(
      () => this.client.invokeTextGeneration(inputData, inv),
      "text",
      { pollFallback: opts?.pollFallback ?? true, metadata: opts?.metadata },
    );
  }
}
