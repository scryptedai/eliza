/**
 * AvbService — owns the avatar-generation pipeline.
 *
 * Architecture (db-persisted state machine):
 *
 *   createRun()
 *     └─ createTask(TEXT_PHASE)   ← Task row in db, survives restart
 *
 *   TaskService ticks TEXT_PHASE worker every ~5s:
 *     tick 1: no scryptedJobId → submit text gen, store jobId, return
 *     tick N: check job; if terminal → onPhaseComplete(TEXT_PHASE, ctx)
 *
 *   onPhaseComplete(TEXT_PHASE, ctx):
 *     └─ createTask(IMAGE_PHASE, ctx)   ← carries imagePrompt forward
 *     └─ deleteTask(textTaskId)
 *
 *   ...same pattern for IMAGE_PHASE → DELIVER → done.
 *
 * Fault-tolerance properties:
 * - Every phase is a persisted Task row. Restart re-discovers via tags.
 * - Worker.execute() is idempotent: safe to re-tick any number of times.
 * - resumePolling(jobId) re-arms scryptedai polling after restart.
 * - deadlineAt guards against jobs that never return (3-layer timeout).
 * - Webhook fast-path eager-executes workers on terminal, no 5s wait.
 */

import { randomUUID } from "node:crypto";
import {
  type IAgentRuntime,
  Service,
  type Task,
  toScryptedPayload,
} from "@elizaos/core";
import {
  extractAssetUrl,
  type JobType,
  type NormalizedJobResult,
  SCRYPTEDAI_SERVICE_TYPE,
  type ScryptedAIService,
} from "@elizaos/plugin-scryptedai";
import {
  AVB_SERVICE_TYPE,
  BASE_TAGS,
  DEFAULT_IMAGE_METHOD,
  ENV_AVB_AUTOGEN_ON_BOOT,
  ENV_AVB_IMAGE_METHOD,
  PHASE_TICK_INTERVAL_MS,
  PIPELINE,
  tagForJob,
  tagForRun,
  WORKER_NAMES,
} from "./constants.ts";
import { digestCharacter, imagePromptSet } from "./introspect.ts";
import type {
  AvbPhaseMetadata,
  AvbRunContext,
  AvbRuntimeSurface,
  PhaseCompleteReport,
  PhaseFailedReport,
  PhaseName,
} from "./types.ts";

// ----------------------------------------------------------------------------
// Internal: terminal-status check
// ----------------------------------------------------------------------------

const TERMINAL_STATUSES: ReadonlySet<string> = new Set([
  "completed",
  "failed",
  "cancelled",
]);

function isTerminal(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

// ----------------------------------------------------------------------------
// AvbService
// ----------------------------------------------------------------------------

export class AvbService extends Service {
  static serviceType = AVB_SERVICE_TYPE;
  static serviceName = "AVB";

  public capabilityDescription =
    "Autonomous Virtual Being pipeline: character introspection → avatar image generation via db-persisted state machine.";

  /** Unsubscribe fn from scryptedai onTerminal hook. */
  private unsubscribeTerminal?: () => void;

  /** Cached scryptedai image method name (resolved at start()). */
  private imageMethod: string = DEFAULT_IMAGE_METHOD;

  /**
   * Task IDs that have already reached a terminal-report handler.
   * Protects against double-execution when the webhook fast-path
   * (eagerExecuteForJob) races with a TaskService tick.
   * In-memory only — the race itself is in-memory.
   * Entries are cleared when their Task row is deleted (see onPhaseComplete /
   * onPhaseFailed) so the set stays bounded by in-flight task count.
   */
  private readonly handledTasks = new Set<string>();

  /**
   * Workers built once in start() and reused. Indexed by phase so the
   * webhook fast-path (workerForPhase) reuses the same closures the
   * runtime registered, instead of allocating fresh ones per event.
   */
  private readonly workers = new Map<
    PhaseName,
    {
      name: string;
      execute: (
        rt: unknown,
        opts: Record<string, unknown>,
        t: Task,
      ) => Promise<void>;
    }
  >();

  /**
   * Settles when autoStartIfNeeded() resolves. Tests can `await
   * svc.whenReady()` instead of guessing microtask depth. Resolved
   * immediately if autostart is skipped.
   */
  private autoStartDone: Promise<void> = Promise.resolve();

  /**
   * Runtime surface (structural cast of this.runtime).
   * Saved once in start() so workers and dispatch methods don't re-cast.
   */
  private rt!: AvbRuntimeSurface;

  // --------------------------------------------------------------------------
  // Service lifecycle
  // --------------------------------------------------------------------------

  static async start(runtime: IAgentRuntime): Promise<AvbService> {
    const svc = new AvbService(runtime);
    svc.rt = runtime as unknown as AvbRuntimeSurface;

    // Resolve image method once
    const envMethod = svc.rt.getSetting(ENV_AVB_IMAGE_METHOD);
    if (typeof envMethod === "string" && envMethod) {
      svc.imageMethod = envMethod;
    }

    // Build workers once, register them, AND cache for webhook fast-path reuse.
    svc.workers.set("TEXT_PHASE", svc.buildTextPhaseWorker());
    svc.workers.set("IMAGE_PHASE", svc.buildImagePhaseWorker());
    svc.workers.set("DELIVER", svc.buildDeliverWorker());
    for (const w of svc.workers.values()) {
      svc.rt.registerTaskWorker(w);
    }

    // Hook scryptedai terminal events for webhook fast-path.
    // Service init order isn't guaranteed (avb may start before scryptedai
    // even though it's a plugin dependency) — await the load promise so
    // we reliably arm the fast-path on every boot.
    try {
      const scrypted = (await svc.rt.getServiceLoadPromise(
        SCRYPTEDAI_SERVICE_TYPE,
      )) as ScryptedAIService;
      svc.unsubscribeTerminal = scrypted.onTerminal((result) => {
        // Eager-execute: don't wait for the next 5s tick
        void svc.eagerExecuteForJob(result).catch((err) => {
          svc.rt.logger.warn(
            `[avb] eagerExecuteForJob threw: ${err instanceof Error ? err.message : String(err)}`,
          );
        });
      });
    } catch (err) {
      svc.rt.logger.error(
        `[avb] ScryptedAI service failed to load — webhook fast-path disabled: ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    svc.rt.logger.info(
      `[avb] Service started (image method=${svc.imageMethod})`,
    );

    // Autonomous trigger: generate an avatar on boot if one doesn't exist.
    // Fire-and-forget so we don't block service startup. The promise is
    // stored so tests can await deterministic settlement (see whenReady).
    svc.autoStartDone = svc.autoStartIfNeeded().catch((err) => {
      svc.rt.logger.warn(
        `[avb] Auto-start check failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    });

    return svc;
  }

  async stop(): Promise<void> {
    this.unsubscribeTerminal?.();
    this.unsubscribeTerminal = undefined;
  }

  /**
   * Resolves once the boot-time autoStartIfNeeded() chain has settled.
   * Test seam — production callers don't need this (start() returns
   * before autostart settles by design).
   */
  whenReady(): Promise<void> {
    return this.autoStartDone;
  }

  // --------------------------------------------------------------------------
  // Public: pipeline entry point
  // --------------------------------------------------------------------------

  /**
   * Start a new avatar-generation run. Returns the runId immediately;
   * the actual generation happens in the background via TaskService ticks.
   */
  async createRun(
    roomId: string,
    triggeringMessageId?: string,
  ): Promise<string> {
    const runId = randomUUID();
    const characterDigest = digestCharacter(this.rt.character);

    const ctx: AvbRunContext = {
      runId,
      roomId,
      triggeringMessageId,
      characterDigest,
    };

    await this.spawnPhaseTask("TEXT_PHASE", ctx);
    this.rt.logger.info(
      `[avb] Run ${runId} started (phase=TEXT_PHASE, room=${roomId})`,
    );
    return runId;
  }

  /**
   * Query all phase tasks for a given run. Empty result = run is complete
   * (all phase tasks deleted) or never existed.
   */
  async getRunTasks(runId: string): Promise<Task[]> {
    return this.rt.getTasks({ tags: [tagForRun(runId)] });
  }

  // --------------------------------------------------------------------------
  // Autonomous boot trigger
  // --------------------------------------------------------------------------

  /**
   * Self-trigger avatar generation on boot if the agent has no avatar yet.
   *
   * Skips if:
   * - AVB_AUTOGEN_ON_BOOT setting is explicitly "false" / "0"
   * - An AVB run is already in flight (task rows exist — will resume on tick)
   * - An avatar has already been delivered to the agent's room
   *
   * Target room is the agent's own agentId (identity room) — same convention
   * as the proving script. Fire-and-forget: logged but never throws to caller.
   */
  private async autoStartIfNeeded(): Promise<void> {
    const setting = this.rt.getSetting(ENV_AVB_AUTOGEN_ON_BOOT);
    if (setting === "false" || setting === "0" || setting === false) {
      this.rt.logger.debug("[avb] Auto-start disabled via setting");
      return;
    }

    // In-flight run? (db-persisted tasks survive restart — let them resume)
    const inFlight = await this.rt.getTasks({ tags: ["avb"] });
    if (inFlight.length > 0) {
      this.rt.logger.info(
        `[avb] Auto-start: ${inFlight.length} run task(s) already in flight — resuming, not re-triggering`,
      );
      return;
    }

    // Already delivered? Check the agent's identity room for a prior avatar.
    // The identity room is low-traffic (only the agent writes here), and the
    // most recent AVB delivery — if any — was the last thing written before
    // its Task row was deleted, so it sits near the head of the result set.
    // 200 gives generous headroom against interleaved system messages.
    const roomId = this.rt.agentId;
    const memories = await this.rt.getMemories({
      roomId,
      tableName: "messages",
      count: 200,
    });
    const existing = memories.find((m) => {
      const c = m.content as
        | { source?: string; attachments?: unknown[] }
        | undefined;
      return (
        c?.source === "avb" &&
        Array.isArray(c.attachments) &&
        c.attachments.length > 0
      );
    });
    if (existing) {
      this.rt.logger.info(
        "[avb] Auto-start: avatar already delivered — skipping",
      );
      return;
    }

    this.rt.logger.info(
      `[avb] Auto-start: no avatar found — triggering createRun(room=${roomId})`,
    );
    await this.createRun(roomId);
  }

  // --------------------------------------------------------------------------
  // Pipeline dispatch: workers report here, we decide what's next
  // --------------------------------------------------------------------------

  /**
   * Called by a worker when its phase completes successfully.
   * Spawns the next phase (if any) with the updated context, then
   * deletes the finished phase's Task row.
   *
   * Spawn-before-delete ordering ensures we never lose a run between
   * the two writes (worst case: both tasks exist briefly).
   */
  async onPhaseComplete(report: PhaseCompleteReport): Promise<void> {
    const { phase, taskId, ctx } = report;

    // Idempotency guard: eager-execute + TaskService tick can both reach
    // terminal on the same task. First one wins; subsequent calls no-op.
    if (this.handledTasks.has(taskId)) return;
    this.handledTasks.add(taskId);

    const spec = PIPELINE[phase];

    this.rt.logger.info(
      `[avb] Phase ${phase} complete (run=${ctx.runId}, next=${spec.next ?? "terminal"})`,
    );

    if (spec.next) {
      await this.spawnPhaseTask(spec.next, ctx);
    }
    // Only delete the old task once the next one is durably written.
    await this.rt.deleteTask(taskId);
    // Task row gone → race window for this taskId is closed. Reclaim the
    // idempotency-set entry so handledTasks stays bounded by in-flight count.
    this.handledTasks.delete(taskId);
  }

  /**
   * Called by a worker when its phase fails (job error, deadline, etc).
   * Delivers an error message to the room and deletes the task.
   */
  async onPhaseFailed(report: PhaseFailedReport): Promise<void> {
    const { phase, taskId, ctx, error } = report;

    if (this.handledTasks.has(taskId)) return;
    this.handledTasks.add(taskId);

    this.rt.logger.warn(
      `[avb] Phase ${phase} failed (run=${ctx.runId}): ${error}`,
    );

    await this.deliverFailure(ctx, phase, error);
    await this.rt.deleteTask(taskId);
    this.handledTasks.delete(taskId);
  }

  // --------------------------------------------------------------------------
  // Task spawning
  // --------------------------------------------------------------------------

  private async spawnPhaseTask(
    phase: PhaseName,
    ctx: AvbRunContext,
  ): Promise<string> {
    const spec = PIPELINE[phase];
    const now = Date.now();

    const metadata: AvbPhaseMetadata = {
      phase,
      runContext: ctx,
      deadlineAt: now + spec.deadlineMs,
      updateInterval: PHASE_TICK_INTERVAL_MS,
      updatedAt: now,
    };

    const taskId = await this.rt.createTask({
      name: spec.workerName,
      description: `AVB ${phase} for run ${ctx.runId}`,
      roomId: ctx.roomId,
      tags: [...BASE_TAGS, tagForRun(ctx.runId)],
      metadata: metadata as unknown as Task["metadata"],
    });

    this.rt.logger.debug(
      `[avb] Spawned ${phase} task=${taskId} (deadline=${spec.deadlineMs}ms)`,
    );
    return taskId;
  }

  // --------------------------------------------------------------------------
  // Webhook fast-path: eager-execute on terminal
  // --------------------------------------------------------------------------

  /**
   * When scryptedai reports a terminal result, find the phase task that
   * submitted that job (via tag reverse lookup) and eager-execute its
   * worker — no waiting for the next 5s TaskService tick.
   */
  private async eagerExecuteForJob(result: NormalizedJobResult): Promise<void> {
    const tasks = await this.rt.getTasks({
      tags: [tagForJob(result.jobId)],
    });
    if (tasks.length === 0) return; // Not one of ours, or already completed

    for (const task of tasks) {
      const md = task.metadata as unknown as AvbPhaseMetadata | undefined;
      if (!md?.phase) continue;

      const worker = this.workerForPhase(md.phase);
      if (!worker) continue;

      this.rt.logger.debug(
        `[avb] Eager-executing ${md.phase} for job ${result.jobId}`,
      );
      // Pass the terminal result through options so the worker can skip
      // a redundant fetchJobStatus() round-trip.
      await worker.execute(this.runtime, { terminalHint: result }, task);
    }
  }

  private workerForPhase(phase: PhaseName):
    | {
        execute: (
          rt: unknown,
          opts: Record<string, unknown>,
          t: Task,
        ) => Promise<void>;
      }
    | undefined {
    return this.workers.get(phase);
  }

  // --------------------------------------------------------------------------
  // TEXT_PHASE worker
  //
  // Idempotent contract:
  //   - If no scryptedJobId: submit text gen, persist jobId to task, return.
  //   - If deadline passed: report fail.
  //   - If job not terminal: ensure polling armed, return.
  //   - If terminal completed: extract text → ctx.imagePrompt, report complete.
  //   - If terminal failed/cancelled: report fail.
  // --------------------------------------------------------------------------

  private buildTextPhaseWorker() {
    const svc = this;
    return {
      name: WORKER_NAMES.TEXT_PHASE,
      async execute(
        _runtime: unknown,
        options: Record<string, unknown>,
        task: Task,
      ): Promise<void> {
        await svc.executePhase(task, options, {
          phase: "TEXT_PHASE",
          jobType: "text",
          submit: async (scrypted, ctx) => {
            const rendered = imagePromptSet.render({
              CHARACTER_DIGEST: ctx.characterDigest,
            });
            const { jobId } = await scrypted.startTextGeneration(
              toScryptedPayload(rendered),
            );
            return jobId;
          },
          extract: (result, ctx) => {
            const text = result.text?.trim();
            if (!text) {
              return {
                ok: false,
                error: "text generation completed but returned empty text",
              };
            }
            return { ok: true, ctx: { ...ctx, imagePrompt: text } };
          },
        });
      },
    };
  }

  // --------------------------------------------------------------------------
  // IMAGE_PHASE worker
  // --------------------------------------------------------------------------

  private buildImagePhaseWorker() {
    const svc = this;
    return {
      name: WORKER_NAMES.IMAGE_PHASE,
      async execute(
        _runtime: unknown,
        options: Record<string, unknown>,
        task: Task,
      ): Promise<void> {
        await svc.executePhase(task, options, {
          phase: "IMAGE_PHASE",
          jobType: "image",
          submit: async (scrypted, ctx) => {
            if (!ctx.imagePrompt) {
              throw new Error(
                "IMAGE_PHASE reached without imagePrompt in context",
              );
            }
            const { jobId } = await scrypted.startImageGeneration(
              svc.imageMethod as Parameters<
                ScryptedAIService["startImageGeneration"]
              >[0],
              { prompt: ctx.imagePrompt, num_images: 1 },
            );
            return jobId;
          },
          extract: (result, ctx) => {
            // Prefer adapter's pre-extracted url; fall back to merged result.images[0]
            // (extractImageUrl ran on the RAW payload pre-merge; result.result is
            // the merged container — they can legitimately differ).
            let url = result.imageUrl;
            if (!url && Array.isArray(result.result?.images)) {
              url = extractAssetUrl(result.result.images[0]);
            }
            if (!url) {
              return {
                ok: false,
                error: "image generation completed but no URL in result",
              };
            }
            return { ok: true, ctx: { ...ctx, imageUrl: url } };
          },
        });
      },
    };
  }

  // --------------------------------------------------------------------------
  // DELIVER worker
  //
  // No scryptedai job — just writes the result memory and emits MESSAGE_SENT.
  // --------------------------------------------------------------------------

  private buildDeliverWorker() {
    const svc = this;
    return {
      name: WORKER_NAMES.DELIVER,
      async execute(
        _runtime: unknown,
        _options: Record<string, unknown>,
        task: Task,
      ): Promise<void> {
        const { md, taskId } = svc.readPhaseTask(task);
        if (!md) return;

        const ctx = md.runContext;

        if (!ctx.imageUrl) {
          await svc.onPhaseFailed({
            phase: "DELIVER",
            taskId,
            ctx,
            error: "DELIVER reached without imageUrl in context",
          });
          return;
        }

        await svc.deliverSuccess(ctx);
        await svc.onPhaseComplete({
          phase: "DELIVER",
          taskId,
          ctx,
        });
      },
    };
  }

  // --------------------------------------------------------------------------
  // Shared phase executor (submit → poll-check → extract → report)
  //
  // Centralizes the idempotent worker lifecycle for phases that submit
  // a scryptedai job. TEXT_PHASE and IMAGE_PHASE both call through here
  // with phase-specific submit/extract callbacks.
  // --------------------------------------------------------------------------

  private async executePhase(
    task: Task,
    options: Record<string, unknown>,
    phaseConfig: {
      phase: PhaseName;
      jobType: JobType;
      submit: (
        scrypted: ScryptedAIService,
        ctx: AvbRunContext,
      ) => Promise<string>;
      extract: (
        result: NormalizedJobResult,
        ctx: AvbRunContext,
      ) => { ok: true; ctx: AvbRunContext } | { ok: false; error: string };
    },
  ): Promise<void> {
    const { md: staleMd, taskId } = this.readPhaseTask(task);
    if (!staleMd) return;

    // --- Double-execution guard ---
    // The webhook fast-path (eagerExecuteForJob) and TaskService ticks can
    // race. Read-fresh: if the task row is already gone, another executor
    // handled it. Also picks up jobId persisted by a concurrent first-tick.
    const fresh = await this.rt.getTask(taskId);
    if (!fresh) {
      this.rt.logger.debug(
        `[avb] ${staleMd.phase}: task ${taskId} already gone (handled by concurrent executor)`,
      );
      return;
    }
    const md =
      (fresh.metadata as unknown as AvbPhaseMetadata | undefined) ?? staleMd;

    const { phase, jobType, submit, extract } = phaseConfig;
    const ctx = md.runContext;

    // --- Step 0: deadline check ---
    if (Date.now() > md.deadlineAt) {
      await this.onPhaseFailed({
        phase,
        taskId,
        ctx,
        error: `deadline exceeded (${PIPELINE[phase].deadlineMs}ms)`,
      });
      return;
    }

    // --- Step 1: ensure scryptedai service is available ---
    const scrypted = this.rt.getService<ScryptedAIService>(
      SCRYPTEDAI_SERVICE_TYPE,
    );
    if (!scrypted) {
      // Transient — maybe service hasn't started yet. Let the next tick retry.
      this.rt.logger.debug(
        `[avb] ${phase}: scryptedai service not yet available`,
      );
      return;
    }

    // --- Step 2: if no job yet, submit it ---
    if (!md.scryptedJobId) {
      try {
        const jobId = await submit(scrypted, ctx);
        await this.persistJobId(fresh, md, jobId);
        this.rt.logger.info(
          `[avb] ${phase} submitted (run=${ctx.runId}, job=${jobId})`,
        );
      } catch (err) {
        await this.onPhaseFailed({
          phase,
          taskId,
          ctx,
          error: `submit failed: ${err instanceof Error ? err.message : String(err)}`,
        });
      }
      return; // Next tick will poll.
    }

    // --- Step 3: check job status ---
    // Webhook fast-path: if eager-executor passed the terminal result, use it.
    const hint = options.terminalHint as NormalizedJobResult | undefined;
    let result: NormalizedJobResult | undefined =
      hint && hint.jobId === md.scryptedJobId ? hint : undefined;

    // Check the service's in-memory record (populated by webhook/polling).
    if (!result) {
      const rec = scrypted.getJob(md.scryptedJobId);
      if (rec?.result && isTerminal(rec.status)) {
        result = rec.result;
      }
    }

    // Restart case: in-memory record gone. Re-arm polling, then do ONE
    // direct fetch so we don't have to wait for the next poll interval.
    if (!result && !scrypted.getJob(md.scryptedJobId)) {
      scrypted.resumePolling(md.scryptedJobId, jobType);
      try {
        const fetched = await scrypted.fetchJobStatus(md.scryptedJobId);
        if (isTerminal(fetched.status)) {
          result = fetched;
        }
      } catch (err) {
        this.rt.logger.debug(
          `[avb] ${phase}: fetchJobStatus transient error: ${err instanceof Error ? err.message : String(err)}`,
        );
        // Not fatal — polling is armed, next tick will check again.
      }
    }

    // --- Step 4: if not terminal yet, just return (next tick will re-check) ---
    if (!result) return;

    // --- Step 5: terminal — extract output and report ---
    if (result.status !== "completed") {
      await this.onPhaseFailed({
        phase,
        taskId,
        ctx,
        error: `scryptedai job ${result.status}: ${result.error ?? "(no error message)"}`,
      });
      return;
    }

    const extracted = extract(result, ctx);
    if (!extracted.ok) {
      await this.onPhaseFailed({
        phase,
        taskId,
        ctx,
        error: extracted.error,
      });
      return;
    }

    await this.onPhaseComplete({
      phase,
      taskId,
      ctx: extracted.ctx,
      rawResult: result,
    });
  }

  // --------------------------------------------------------------------------
  // Task metadata helpers
  // --------------------------------------------------------------------------

  /** Read + validate phase metadata from a Task. Returns undefined on malformed. */
  private readPhaseTask(task: Task): {
    md: AvbPhaseMetadata | undefined;
    taskId: string;
  } {
    const taskId = String(task.id ?? "");
    if (!taskId) {
      this.rt.logger.warn("[avb] Worker received task with no id — skipping");
      return { md: undefined, taskId };
    }
    const md = task.metadata as unknown as AvbPhaseMetadata | undefined;
    if (!md || !md.phase || !md.runContext) {
      this.rt.logger.warn(
        `[avb] Task ${taskId} missing phase metadata — skipping`,
      );
      return { md: undefined, taskId };
    }
    return { md, taskId };
  }

  /**
   * Persist the submitted scryptedai jobId onto the task (metadata + tag).
   * The tag enables webhook fast-path reverse lookup.
   *
   * Also writes updatedAt so the TaskService's pre-execute updatedAt bump
   * (which uses stale metadata) doesn't accidentally clobber our jobId.
   */
  private async persistJobId(
    task: Task,
    md: AvbPhaseMetadata,
    jobId: string,
  ): Promise<void> {
    const newMd: AvbPhaseMetadata = {
      ...md,
      scryptedJobId: jobId,
      updatedAt: Date.now(),
    };
    const newTags = [
      ...(task.tags ?? []).filter((t) => !t.startsWith("avb:job:")),
      tagForJob(jobId),
    ];
    await this.rt.updateTask(String(task.id), {
      metadata: newMd as unknown as Task["metadata"],
      tags: newTags,
    });
  }

  // --------------------------------------------------------------------------
  // Delivery (final output → room)
  // --------------------------------------------------------------------------

  private async deliverSuccess(ctx: AvbRunContext): Promise<void> {
    const attachmentId = randomUUID();
    const responseMemory = {
      entityId: this.rt.agentId,
      agentId: this.rt.agentId,
      roomId: ctx.roomId,
      createdAt: Date.now(),
      content: {
        text: ctx.imagePrompt ?? "",
        thought: `Generated avatar for "${this.rt.character.name ?? "agent"}" via scryptedai.`,
        actions: ["GENERATE_AVATAR"],
        attachments: [
          {
            id: attachmentId,
            url: ctx.imageUrl as string,
            title: `Avatar_${ctx.runId}.png`,
            contentType: "image",
          },
        ],
        inReplyTo: ctx.triggeringMessageId,
        source: "avb",
      },
    };

    await this.rt.createMemory(responseMemory, "messages");
    await this.rt.emitEvent("MESSAGE_SENT", {
      runtime: this.runtime,
      message: responseMemory,
      source: "avb",
    });

    this.rt.logger.info(
      `[avb] Delivered avatar for run ${ctx.runId} → ${ctx.imageUrl}`,
    );
  }

  private async deliverFailure(
    ctx: AvbRunContext,
    phase: PhaseName,
    error: string,
  ): Promise<void> {
    const responseMemory = {
      entityId: this.rt.agentId,
      agentId: this.rt.agentId,
      roomId: ctx.roomId,
      createdAt: Date.now(),
      content: {
        text: `Avatar generation failed during ${phase}: ${error}`,
        actions: ["GENERATE_AVATAR"],
        inReplyTo: ctx.triggeringMessageId,
        source: "avb",
      },
    };

    await this.rt.createMemory(responseMemory, "messages");
    await this.rt.emitEvent("MESSAGE_SENT", {
      runtime: this.runtime,
      message: responseMemory,
      source: "avb",
    });
  }
}
