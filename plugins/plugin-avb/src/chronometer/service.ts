/**
 * ChronometerService — the agent's subjective clock.
 *
 * Owns the mining loop:
 *   start() → load chain → validate → (mine genesis if empty) → spawn worker
 *   loop:    collect events → submit job → onSolved → seal block → append → adjust difficulty → resubmit
 *
 * The service exposes recordEvent() for the rest of AVB to log actions.
 * Events accumulate in a mempool; when a block is solved, the mempool is
 * sealed into that block and cleared. Events arriving after submission
 * but before solve go into the *next* block (we don't restart the miner
 * for late-arriving events — that would let a flood of events stall
 * mining indefinitely).
 *
 * Tamper warnings:
 *   - On load, every chain issue is logged at WARN with a [chronometer:tamper]
 *     prefix. Hard issues (PoW/hash failures) also set a sticky flag
 *     that anyone can query via getStatus().
 *   - During operation, hostTimestamp regression vs. the previous block
 *     is also flagged — catches an admin rewinding the clock mid-run.
 */

import { join } from "node:path";
import { EventType, type IAgentRuntime, Service } from "@elizaos/core";
import { appendBlock, loadChain, tipOf, validateBlock } from "./chain.ts";
import {
  type DifficultyState,
  GENESIS_BITS,
  initialDifficulty,
  retarget,
  TARGET_BLOCK_MS,
} from "./difficulty.ts";
import {
  buildGenesisEvent,
  buildGenesisHeader,
  computeEventsRoot,
  ZERO_HASH,
} from "./encoding.ts";
import { Miner, mineSync } from "./miner.ts";
import type {
  Block,
  ChainTip,
  ChronoEvent,
  MiningResult,
  ValidationReport,
} from "./types.ts";
import { BLOCK_VERSION, EventKind } from "./types.ts";

// ----------------------------------------------------------------------------
// Service identity
// ----------------------------------------------------------------------------

export const CHRONOMETER_SERVICE_TYPE = "avb-chronometer" as const;

/** Override for the chain file path. Defaults to <ELIZA_DATA_DIR>/avb/chronometer-<agentId>.chain */
export const ENV_AVB_CHRONOMETER_PATH = "AVB_CHRONOMETER_PATH";

/** Set to "false" to disable the chronometer entirely. */
export const ENV_AVB_CHRONOMETER_ENABLED = "AVB_CHRONOMETER_ENABLED";

// ----------------------------------------------------------------------------
// Status surface
// ----------------------------------------------------------------------------

export interface ChronometerStatus {
  enabled: boolean;
  tip?: ChainTip;
  /** Block height (== tip.height + 1, or 0 if no tip). Number of blocks. */
  height: number;
  /** Subjective uptime: height × target block time. Trustworthy. */
  subjectiveUptimeMs: number;
  /** Last validation report (from load). */
  lastValidation?: ValidationReport;
  /** Sticky: any hard tamper signal seen this session? */
  tamperDetected: boolean;
  /** All tamper warnings accumulated this session. */
  warnings: string[];
  /** Events waiting in the mempool for the next block. */
  pendingEvents: number;
  /** Current mining difficulty. */
  bits: number;
}

// ----------------------------------------------------------------------------
// Minimal logger surface (matches AvbRuntimeSurface convention)
// ----------------------------------------------------------------------------

interface Log {
  info: (msg: string) => void;
  warn: (msg: string) => void;
  error: (msg: string) => void;
  debug: (msg: string) => void;
}

// ----------------------------------------------------------------------------
// Runtime event bridge
// ----------------------------------------------------------------------------

/**
 * Runtime event names we DON'T witness. These are infrastructure noise:
 *   - HOOK_*       fire on every tool call / command — would dwarf signal
 *   - EMBEDDING_*  per-message vectorization, not agent behavior
 *   - CONTROL/FORM UI plumbing
 * Everything else in EventType is "the agent did something notable" and
 * gets recorded automatically.
 */
function isExcludedRuntimeEvent(name: string): boolean {
  return (
    name.startsWith("HOOK_") ||
    name.startsWith("EMBEDDING_") ||
    name.startsWith("FORM_") ||
    name === "CONTROL_MESSAGE"
  );
}

/**
 * Reduce a runtime event payload to a small identifying digest.
 *
 * The chronometer is a timestamp witness, not a content archive. We prove
 * "MESSAGE_SENT happened around chain height N" with enough metadata to
 * correlate against the real Memory store — NOT the full message body.
 * This keeps the event log from ballooning and avoids stuffing PII into
 * a file that's specifically designed to be hard to delete.
 *
 * Payloads always include `runtime` (circular, huge — must skip) and
 * usually `source`. Beyond that, shapes vary by event type; we
 * opportunistically grab common ID fields.
 */
export function digestRuntimeEvent(
  eventName: string,
  payload: unknown,
): string {
  const out: Record<string, unknown> = { event: eventName };

  if (payload && typeof payload === "object") {
    const p = payload as Record<string, unknown>;

    // String-valued identifying fields common across payload shapes.
    // (RunEventPayload, ActionEventPayload, InvokePayload, etc. all have
    // overlapping subsets of these.)
    for (const key of [
      "source",
      "runId",
      "messageId",
      "roomId",
      "worldId",
      "entityId",
      "status",
    ]) {
      if (typeof p[key] === "string") out[key] = p[key];
    }

    // MessagePayload nests a Memory object. Pull just the ID + size hint.
    const msg = p.message as
      | { id?: unknown; roomId?: unknown; content?: { text?: unknown } }
      | undefined;
    if (msg && typeof msg === "object") {
      if (typeof msg.id === "string") out.messageId = msg.id;
      if (typeof msg.roomId === "string" && !out.roomId)
        out.roomId = msg.roomId;
      const text = msg.content?.text;
      if (typeof text === "string") out.textLen = text.length;
    }

    // ActionEventPayload puts the action name inside content.actions[].
    const content = p.content as { actions?: unknown } | undefined;
    if (content && Array.isArray(content.actions)) {
      out.actions = content.actions.filter((a) => typeof a === "string");
    }
  }

  // Defensive cap: if some payload shape we didn't anticipate produces
  // a giant digest, truncate. 512 bytes is plenty for identifying metadata.
  const json = JSON.stringify(out);
  return json.length > 512 ? `${json.slice(0, 509)}…"}` : json;
}

// ----------------------------------------------------------------------------
// ChronometerService
// ----------------------------------------------------------------------------

export class ChronometerService extends Service {
  static serviceType = CHRONOMETER_SERVICE_TYPE;
  static serviceName = "AVB Chronometer";

  public capabilityDescription =
    "Proof-of-work timestamp chain. Provides tamper-evident subjective time for AVB agents.";

  private log!: Log;
  private chainPath = "";
  private eventLogPath = "";
  private enabled = false;

  private miner?: Miner;
  private blocks: Block[] = [];
  private difficulty: DifficultyState = initialDifficulty();

  /** Events recorded since the last block was sealed. */
  private mempool: ChronoEvent[] = [];

  /** Snapshot of mempool when the current job was submitted. Sealed on solve. */
  private inFlightEvents: ChronoEvent[] = [];
  private inFlightJobId = -1;
  private inFlightSubmittedAt = 0;

  /** Sticky tamper state. */
  private tamperDetected = false;
  private warnings: string[] = [];
  private lastValidation?: ValidationReport;

  // --------------------------------------------------------------------------
  // Lifecycle
  // --------------------------------------------------------------------------

  static async start(runtime: IAgentRuntime): Promise<ChronometerService> {
    const svc = new ChronometerService(runtime);
    const rt = runtime as unknown as {
      agentId: string;
      logger: Log;
      getSetting: (k: string) => unknown;
      registerEvent: (
        event: string,
        handler: (params: unknown) => Promise<void>,
      ) => void;
    };
    svc.log = rt.logger;

    // Resolve enable flag.
    const enabledSetting = rt.getSetting(ENV_AVB_CHRONOMETER_ENABLED);
    if (
      enabledSetting === "false" ||
      enabledSetting === "0" ||
      enabledSetting === false
    ) {
      svc.log.info("[chronometer] disabled via setting");
      svc.enabled = false;
      return svc;
    }
    svc.enabled = true;

    // Resolve chain path. Event log lives alongside it with .events ext.
    const explicitPath = rt.getSetting(ENV_AVB_CHRONOMETER_PATH);
    if (typeof explicitPath === "string" && explicitPath) {
      svc.chainPath = explicitPath;
    } else {
      const dataDir =
        process.env.ELIZA_DATA_DIR || join(process.cwd(), ".eliza");
      svc.chainPath = join(dataDir, "avb", `chronometer-${rt.agentId}.chain`);
    }
    // foo.chain → foo.events; foo → foo.events. The two files always sit
    // together so they're easy to back up / inspect / delete as a pair.
    svc.eventLogPath = `${svc.chainPath.replace(/\.chain$/, "")}.events`;

    // Load + validate existing chain.
    await svc.loadAndValidate();

    // If no chain → mine genesis synchronously (it's trivially easy at
    // GENESIS_BITS, finishes in microseconds, and we want a tip before
    // we spawn the worker).
    if (svc.blocks.length === 0) {
      await svc.mineGenesis();
    }

    // Spawn the worker and kick off the first real mining job.
    svc.miner = new Miner({
      onSolved: (r) => void svc.onSolved(r),
      onError: (e) => svc.log.error(`[chronometer] worker error: ${e.message}`),
    });
    svc.miner.start();
    svc.submitNext();

    // Bridge the runtime event bus into the chain. We register for every
    // EventType the core publishes, minus infrastructure noise — this is
    // automatic discovery, so when @elizaos/core ships a new event type
    // it's witnessed here without a code change. The runtime offers no
    // unsubscribe handle; stop() flips `enabled` so these become no-ops.
    let bridged = 0;
    for (const name of Object.values(EventType)) {
      if (typeof name !== "string" || isExcludedRuntimeEvent(name)) continue;
      rt.registerEvent(name, async (payload) => {
        svc.recordEvent(EventKind.RUNTIME, digestRuntimeEvent(name, payload));
      });
      bridged++;
    }
    svc.log.debug(
      `[chronometer] subscribed to ${bridged} runtime event type(s)`,
    );

    svc.log.info(
      `[chronometer] started (height=${svc.blocks.length}, bits=0x${svc.difficulty.bits.toString(16)}, path=${svc.chainPath})`,
    );
    return svc;
  }

  async stop(): Promise<void> {
    // registerEvent has no unsubscribe — handlers stay registered on the
    // runtime forever. Flip enabled so recordEvent() short-circuits.
    this.enabled = false;
    await this.miner?.stop();
    this.miner = undefined;
  }

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  /**
   * Record an event into the next block. Returns immediately; the event
   * sits in the mempool until a block is mined.
   */
  recordEvent(kind: EventKind, data: string): void {
    if (!this.enabled) return;
    this.mempool.push({ kind, at: Date.now(), data });
  }

  getStatus(): ChronometerStatus {
    const tip = tipOf(this.blocks);
    const height = this.blocks.length;
    return {
      enabled: this.enabled,
      tip,
      height,
      subjectiveUptimeMs: height * TARGET_BLOCK_MS,
      lastValidation: this.lastValidation,
      tamperDetected: this.tamperDetected,
      warnings: [...this.warnings],
      pendingEvents: this.mempool.length + this.inFlightEvents.length,
      bits: this.difficulty.bits,
    };
  }

  // --------------------------------------------------------------------------
  // Load + validation
  // --------------------------------------------------------------------------

  private async loadAndValidate(): Promise<void> {
    const { blocks, report } = loadChain(this.chainPath, this.eventLogPath);
    this.blocks = blocks;
    this.lastValidation = report;

    if (report.issues.length > 0) {
      for (const issue of report.issues) {
        const msg = `[chronometer:tamper] ${issue.code} at height ${issue.height}: ${issue.detail}`;
        this.warnings.push(msg);
        this.log.warn(msg);
      }
    }
    if (!report.ok) {
      this.tamperDetected = true;
      this.log.error(
        `[chronometer:tamper] CHAIN INTEGRITY FAILURE — ${report.issues.length} issue(s), only ${report.validBlocks} block(s) validated. Chain truncated to last good block.`,
      );
    } else if (blocks.length > 0) {
      this.log.info(
        `[chronometer] chain validated: ${blocks.length} block(s) clean`,
      );
    }

    // Seed difficulty from the tip if we have one — otherwise the
    // adjuster starts from GENESIS_BITS and reconverges, wasting a few
    // blocks. The tip's bits is what was used to mine it, which is also
    // what should be used to mine the next one (modulo one retarget step
    // we're skipping because we don't have the last solve time on disk —
    // acceptable, the EMA absorbs one missed step).
    const tip = tipOf(blocks);
    if (tip) {
      this.difficulty = { bits: tip.bits, emaBlockMs: TARGET_BLOCK_MS };
    }
  }

  // --------------------------------------------------------------------------
  // Genesis
  // --------------------------------------------------------------------------

  private async mineGenesis(): Promise<void> {
    this.log.info("[chronometer] no chain found — mining genesis");
    const ts = Math.floor(Date.now() / 1000);
    const template = buildGenesisHeader(GENESIS_BITS, ts);
    const { header, hash } = mineSync(template);
    const genesis: Block = { header, hash, height: 0 };
    // Genesis event goes to the event log; the chain header only carries
    // the commitment. buildGenesisHeader already baked computeEventsRoot
    // of this exact event into eventsRoot.
    const genesisEvents = [buildGenesisEvent(ts)];

    // Sanity-check our own work before persisting.
    const issue = validateBlock(genesis, ZERO_HASH, genesisEvents);
    if (issue) {
      throw new Error(`[chronometer] genesis self-check failed: ${issue.code}`);
    }

    await appendBlock(
      this.chainPath,
      this.eventLogPath,
      genesis,
      genesisEvents,
    );
    this.blocks = [genesis];
    this.log.info(
      `[chronometer] genesis mined: ${hash.toString("hex").slice(0, 16)}…`,
    );
  }

  // --------------------------------------------------------------------------
  // Mining loop
  // --------------------------------------------------------------------------

  private submitNext(): void {
    if (!this.miner) return;
    const tip = this.blocks[this.blocks.length - 1];

    // Snapshot the mempool. Events arriving after this point go into
    // the NEXT block — we don't restart mining for late events.
    this.inFlightEvents = this.mempool;
    this.mempool = [];
    this.inFlightSubmittedAt = Date.now();

    const eventsRoot = computeEventsRoot(this.inFlightEvents);
    const hostTimestamp = Math.floor(Date.now() / 1000);

    // Catch mid-run clock manipulation: if host time went backwards
    // relative to the last sealed block, that's exactly what an admin
    // rewinding the clock looks like.
    if (hostTimestamp < tip.header.hostTimestamp) {
      const msg = `[chronometer:tamper] host clock regression: ${tip.header.hostTimestamp} → ${hostTimestamp} (Δ=${hostTimestamp - tip.header.hostTimestamp}s)`;
      this.warnings.push(msg);
      this.log.warn(msg);
    }

    this.inFlightJobId = this.miner.submit({
      prevHash: tip.hash,
      eventsRoot,
      hostTimestamp,
      bits: this.difficulty.bits,
    });
  }

  private async onSolved(result: MiningResult): Promise<void> {
    if (result.jobId !== this.inFlightJobId) {
      // Stale result from a superseded job. Shouldn't happen in normal
      // operation (we never supersede mid-flight) but defend anyway.
      this.log.debug(
        `[chronometer] discarding stale result for job ${result.jobId} (current=${this.inFlightJobId})`,
      );
      return;
    }

    const tip = this.blocks[this.blocks.length - 1];
    const sealedEvents = this.inFlightEvents;
    const block: Block = {
      header: {
        version: BLOCK_VERSION,
        prevHash: tip.hash,
        eventsRoot: computeEventsRoot(sealedEvents),
        hostTimestamp: Math.floor(this.inFlightSubmittedAt / 1000),
        bits: this.difficulty.bits,
        nonce: result.nonce,
      },
      hash: Buffer.from(result.hash),
      height: tip.height + 1,
    };

    // Verify the worker didn't lie to us. Cheap, and catches both bugs
    // and hypothetical worker compromise.
    const issue = validateBlock(block, tip.hash, sealedEvents);
    if (issue) {
      this.log.error(
        `[chronometer] worker returned invalid block: ${issue.code} — ${issue.detail}. Discarding and resubmitting.`,
      );
      // Put the events back and try again.
      this.mempool = [...sealedEvents, ...this.mempool];
      this.inFlightEvents = [];
      this.submitNext();
      return;
    }

    // Persist. Event log first, then chain header — see appendBlock for
    // why the order matters under crashes.
    try {
      await appendBlock(this.chainPath, this.eventLogPath, block, sealedEvents);
    } catch (e) {
      this.log.error(
        `[chronometer] append failed: ${e instanceof Error ? e.message : String(e)} — block dropped`,
      );
      // Don't push events back: append failure usually means disk full
      // or permissions, retrying immediately won't help. Just move on.
      this.inFlightEvents = [];
      this.submitNext();
      return;
    }

    this.blocks.push(block);
    this.inFlightEvents = [];

    // Retarget. We use the actual elapsed wall-clock time (including
    // throttle sleeps), not the worker's reported time, because the
    // worker's clock could be tampered with too. Date.now() on the main
    // thread is the same untrusted host clock, but at least we're
    // consistently untrusting it from one place.
    const elapsedMs = Date.now() - this.inFlightSubmittedAt;
    this.difficulty = retarget(this.difficulty, elapsedMs);

    this.log.debug(
      `[chronometer] block ${block.height} sealed: ${block.hash.toString("hex").slice(0, 16)}… (${sealedEvents.length} events, ${elapsedMs}ms, next bits=0x${this.difficulty.bits.toString(16)})`,
    );

    // Next block.
    this.submitNext();
  }
}

// Re-export for the difficulty type used in the public status surface.
export type { DifficultyState } from "./difficulty.ts";
