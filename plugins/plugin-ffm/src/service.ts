/**
 * FfmService — non-blocking personality engine.
 *
 * Two operation classes:
 *
 *   Class A — pure derivation (instant, ≤10μs):
 *     deriveProfile(seed?) → FfmProfile
 *     Pure math; no I/O. Wrapped in a resolved Promise for API uniformity
 *     but never blocks the event loop.
 *
 *   Class B — LLM expansion (slow, 10–60s):
 *     startNarrativeExpansion(profile, name) → {jobId}   ← returns immediately
 *     startVoiceExpansion(profile, narrative, name) → {jobId}
 *     awaitExpansion(jobId, timeoutMs?) → Expansion       ← optional blocking variant
 *     onExpansionTerminal(listener) → unsubscribe
 *
 *   The Class-B job lifecycle rides ScryptedAIService's existing
 *   webhook+polling machinery. FfmService is a thin tracking layer on top:
 *   it remembers which scryptedai jobId corresponds to which expansion kind,
 *   parses the LLM text into structured types, and fans out to its own
 *   listeners. It owns NO polling loops, NO timers, NO HTTP — all of that
 *   is delegated to scryptedai.
 *
 *   start() returns immediately. stop() unsubscribes from scryptedai and
 *   clears listeners. No background work to cancel.
 */

import { type IAgentRuntime, Service, toScryptedPayload } from "@elizaos/core";
import {
  type NormalizedJobResult,
  SCRYPTEDAI_SERVICE_TYPE,
} from "@elizaos/plugin-scryptedai";
import { classifyTraits } from "./archetype.ts";
import { FFM_SERVICE_TYPE } from "./constants.ts";
import { deriveTraits, generateSeed, normalizeSeed } from "./prng.ts";
import {
  parseNarrative,
  parseVoice,
  renderNarrativePrompt,
  renderVoicePrompt,
} from "./promptset.ts";
import type {
  Expansion,
  ExpansionKind,
  ExpansionRecord,
  ExpansionTerminalListener,
  FfmProfile,
  FfmRuntimeSurface,
  NarrativeExpansion,
  ScryptedAILike,
} from "./types.ts";

// ----------------------------------------------------------------------------
// Service
// ----------------------------------------------------------------------------

export class FfmService extends Service {
  static serviceType = FFM_SERVICE_TYPE;
  static serviceName = "FFM";

  public capabilityDescription =
    "Five Factor Model (OCEAN) personality engine — seeded trait derivation with empirical distributions, plus LLM-driven character expansion.";

  private rt!: FfmRuntimeSurface;
  private scrypted?: ScryptedAILike;
  private unsubscribeScrypted?: () => void;

  /** jobId → expansion record. In-memory only; cleared on restart. */
  private readonly expansions = new Map<string, ExpansionRecord>();
  /** Subscribers to expansion-terminal events. */
  private readonly listeners = new Set<ExpansionTerminalListener>();

  // --------------------------------------------------------------------------
  // Lifecycle
  // --------------------------------------------------------------------------

  static async start(runtime: IAgentRuntime): Promise<FfmService> {
    const svc = new FfmService(runtime);
    svc.rt = runtime as unknown as FfmRuntimeSurface;

    // Wait for scryptedai to be ready, then subscribe to its terminal stream.
    // Same pattern as AvbService.start() — getServiceLoadPromise blocks until
    // the dependency's start() resolves, but we're inside our own start()
    // here so the runtime is already expecting us to take a moment.
    //
    // If scryptedai isn't loaded (e.g. token missing), Class-A derivation
    // still works; only Class-B expansion fails. Don't crash on its account.
    try {
      const scrypted = (await svc.rt.getServiceLoadPromise(
        SCRYPTEDAI_SERVICE_TYPE,
      )) as ScryptedAILike;
      svc.scrypted = scrypted;
      svc.unsubscribeScrypted = scrypted.onTerminal((result) => {
        // Fan terminal results into our own tracker. Wrapped in void+catch
        // so a parse error in one job doesn't crash the listener fan-out.
        try {
          svc.handleScryptedTerminal(result);
        } catch (err) {
          svc.rt.logger.warn(
            `[ffm] terminal handler threw for job ${result.jobId}: ${
              err instanceof Error ? err.message : String(err)
            }`,
          );
        }
      });
      svc.rt.logger.info("[ffm] Service started (scryptedai linked)");
    } catch (err) {
      svc.rt.logger.warn(
        `[ffm] scryptedai unavailable; expansion disabled. Trait derivation still works. (${
          err instanceof Error ? err.message : String(err)
        })`,
      );
    }

    return svc;
  }

  async stop(): Promise<void> {
    this.unsubscribeScrypted?.();
    this.unsubscribeScrypted = undefined;
    this.listeners.clear();
    // We don't clear `expansions` — if anything's still pending, the records
    // are dead anyway (we've unsubscribed). Leaving them costs nothing and
    // lets late inspectors at least see what was in flight.
  }

  // --------------------------------------------------------------------------
  // Class A — pure derivation (instant, deterministic)
  // --------------------------------------------------------------------------

  /**
   * Derive a full FFM profile from a seed.
   *
   * If `seed` is omitted, a fresh random 256-bit seed is generated.
   * If `seed` is provided, it's normalized (trim, lowercase, validate)
   * and the SAME profile is produced every call.
   *
   * Pure math — no I/O, no awaits inside. Returns a resolved Promise for
   * API uniformity with the Class-B methods.
   */
  // biome-ignore lint/suspicious/useAwait: API uniformity with Class-B; sync body is the point.
  async deriveProfile(seed?: string): Promise<FfmProfile> {
    const seedHex = seed === undefined ? generateSeed() : normalizeSeed(seed);
    const traits = deriveTraits(seedHex);
    const archetype = classifyTraits(traits);
    return Object.freeze({ seed: seedHex, traits, archetype });
  }

  // --------------------------------------------------------------------------
  // Class B — LLM expansion (non-blocking start; optional blocking await)
  // --------------------------------------------------------------------------

  /**
   * Start narrative expansion (call #1). Renders the prompt, fires it at
   * ScryptedAI, registers the jobId, and returns IMMEDIATELY. The result
   * arrives later via onExpansionTerminal or awaitExpansion.
   */
  async startNarrativeExpansion(
    profile: FfmProfile,
    agentName: string,
  ): Promise<{ jobId: string }> {
    return this.startExpansion("narrative", profile, agentName, undefined);
  }

  /**
   * Start voice expansion (call #2). Requires the narrative output from
   * call #1. Same non-blocking contract as startNarrativeExpansion.
   */
  async startVoiceExpansion(
    profile: FfmProfile,
    narrative: NarrativeExpansion,
    agentName: string,
  ): Promise<{ jobId: string }> {
    return this.startExpansion("voice", profile, agentName, narrative);
  }

  private async startExpansion(
    kind: ExpansionKind,
    profile: FfmProfile,
    agentName: string,
    narrative: NarrativeExpansion | undefined,
  ): Promise<{ jobId: string }> {
    if (!this.scrypted) {
      throw new Error(
        "[ffm] scryptedai service unavailable — cannot start LLM expansion",
      );
    }

    const rendered =
      kind === "narrative"
        ? renderNarrativePrompt(profile, agentName)
        : renderVoicePrompt(
            profile,
            narrative as NarrativeExpansion,
            agentName,
          );

    // toScryptedPayload gives {system_prompt, user_prompt, max_tokens}, but the
    // /generations/text/nova-pro endpoint defaults auto_calculate_tokens=true
    // which floors output at ~100 tokens and ignores max_tokens entirely. We
    // must send auto_calculate_tokens=false for max_tokens to be honored.
    // The registry's maxOutputTokens is Nova Pro's hard ceiling (10k); request
    // below it. Narrative lands ~600 tokens, voice ~1500 — 4000 is generous.
    const base = toScryptedPayload(rendered);
    const payload = {
      ...base,
      max_tokens: Math.min(base.max_tokens, 4000),
      auto_calculate_tokens: false,
    };

    // Fire the text-gen call. ScryptedAI's startTextGeneration tracks the
    // job, starts a polling fallback, and returns immediately with a jobId.
    const { jobId } = await this.scrypted.startTextGeneration(payload, {
      pollFallback: true,
      metadata: { source: "ffm", kind, archetype: profile.archetype.sloan },
    });

    const now = Date.now();
    const record: ExpansionRecord = {
      jobId,
      kind,
      profile,
      status: "pending",
      createdAt: now,
      updatedAt: now,
    };
    this.expansions.set(jobId, record);

    this.rt.logger.debug(
      `[ffm] ${kind} expansion started: job=${jobId} archetype=${profile.archetype.sloan}`,
    );

    return { jobId };
  }

  /**
   * Subscribe to expansion-terminal events. Called once per terminal job
   * (success OR failure). Returns an unsubscribe function.
   */
  onExpansionTerminal(listener: ExpansionTerminalListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Block until an expansion job reaches terminal state, or until
   * `timeoutMs` elapses. If the job already terminated, resolves
   * immediately. Throws on failure (parse error or API error) and on
   * timeout.
   *
   * This is the OPTIONAL blocking variant — for fire-and-forget use,
   * subscribe via onExpansionTerminal instead. Pattern mirrors
   * ScryptedAIService.awaitJob exactly.
   */
  async awaitExpansion(jobId: string, timeoutMs?: number): Promise<Expansion> {
    // Fast path: already terminal.
    const existing = this.expansions.get(jobId);
    if (existing && existing.status !== "pending") {
      return this.unwrapRecord(existing);
    }

    return new Promise<Expansion>((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | undefined;

      const unsubscribe = this.onExpansionTerminal((record) => {
        if (record.jobId !== jobId) return;
        unsubscribe();
        if (timer) clearTimeout(timer);
        try {
          resolve(this.unwrapRecord(record));
        } catch (err) {
          reject(err);
        }
      });

      if (timeoutMs !== undefined) {
        timer = setTimeout(() => {
          unsubscribe();
          reject(
            new Error(
              `[ffm] awaitExpansion timed out after ${timeoutMs}ms for job ${jobId}`,
            ),
          );
        }, timeoutMs);
      }
    });
  }

  private unwrapRecord(record: ExpansionRecord): Expansion {
    if (record.status === "failed" || !record.result) {
      throw new Error(
        `[ffm] expansion job ${record.jobId} failed: ${record.error ?? "unknown error"}`,
      );
    }
    return record.result;
  }

  /** Snapshot of a tracked expansion (for debugging / introspection). */
  getExpansion(jobId: string): ExpansionRecord | undefined {
    return this.expansions.get(jobId);
  }

  // --------------------------------------------------------------------------
  // ScryptedAI terminal sink
  // --------------------------------------------------------------------------

  /**
   * Handle a scryptedai terminal result. We only care about jobs we
   * started — anything else is silently ignored (this listener fires for
   * ALL scryptedai jobs, including ones started by other plugins).
   */
  private handleScryptedTerminal(result: NormalizedJobResult): void {
    const record = this.expansions.get(result.jobId);
    if (!record) return; // not ours

    record.updatedAt = Date.now();
    record.rawText = result.text;

    // Failure path: scryptedai reported an error, or there's no text.
    if (result.status === "failed" || result.error || !result.text) {
      record.status = "failed";
      record.error =
        result.error ?? `scryptedai job ${result.status} with no text output`;
      this.rt.logger.warn(
        `[ffm] ${record.kind} expansion failed: job=${result.jobId} ${record.error}`,
      );
      this.fanOut(record);
      return;
    }

    // Success path: parse the LLM output into the right structured type.
    try {
      record.result =
        record.kind === "narrative"
          ? { kind: "narrative", ...parseNarrative(result.text) }
          : { kind: "voice", ...parseVoice(result.text) };
      record.status = "completed";
      this.rt.logger.info(
        `[ffm] ${record.kind} expansion completed: job=${result.jobId}`,
      );
    } catch (err) {
      record.status = "failed";
      record.error = err instanceof Error ? err.message : String(err);
      this.rt.logger.warn(
        `[ffm] ${record.kind} expansion parse failed: job=${result.jobId} — ${record.error}`,
      );
    }

    this.fanOut(record);
  }

  private fanOut(record: ExpansionRecord): void {
    for (const listener of this.listeners) {
      try {
        listener(record);
      } catch (err) {
        this.rt.logger.warn(
          `[ffm] listener threw for job ${record.jobId}: ${
            err instanceof Error ? err.message : String(err)
          }`,
        );
      }
    }
  }
}
