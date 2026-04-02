/**
 * ChronometerService — on-device PoW timestamp server for the AVB.
 *
 * Lifecycle:
 *   start():
 *     - resolve chain path / cpu allocation / block-time target
 *     - load chain file, verify integrity, raise warning flags
 *     - if empty: mint genesis (height 0, trivial PoW) synchronously
 *     - record a BOOT event
 *     - spawn the miner worker and dispatch the first MineJob
 *
 *   on worker "sealed":
 *     - append header‖payload to chain file (and in-memory chain)
 *     - retarget difficulty toward TARGET_BLOCK_MS using wallElapsedMs
 *     - drain pendingEvents into the next job's payload, dispatch
 *
 *   recordEvent(): push into pendingEvents; sealed into the next block.
 *
 * The service exposes getTip()/getHeight()/getIntrinsicTimeMs() so the
 * rest of the AVB can reason about "how much proof-of-work time has
 * accumulated since genesis" independently of the host clock.
 */

import os from "node:os";
import path from "node:path";
import { Worker } from "node:worker_threads";
import { type IAgentRuntime, Service } from "@elizaos/core";

import {
  bitsToTarget,
  bytesEqual,
  deserializeHeader,
  encodeEvent,
  hashToBigInt,
  INITIAL_DIFFICULTY_BITS,
  PROTOCOL_VERSION,
  retarget,
  serializeHeader,
  sha256d,
  splitEvents,
  toHex,
  ZERO_HASH,
} from "./block.ts";
import {
  appendBlockBytes,
  appendEventSegment,
  loadChain,
  loadEventSegments,
  verifyBlocks,
  verifySegments,
} from "./chain.ts";
import {
  type MerkleStep,
  merkleProof,
  merkleRoot,
  rootFromProof,
} from "./merkle.ts";
import {
  type Block,
  type BlockHeader,
  type ChronoEvent,
  ChronoEventType,
  type IntegrityIssue,
  type MineJob,
  type MineResult,
  type MinerOutbound,
} from "./types.ts";

// ----------------------------------------------------------------------------
// Constants / settings
// ----------------------------------------------------------------------------

export const CHRONO_SERVICE_TYPE = "avb-chrono" as const;

/** Runtime event emitted whenever an integrity issue is detected. */
export const CHRONO_INTEGRITY_EVENT = "AVB_CHRONO_INTEGRITY_WARNING" as const;
/** Runtime event emitted each time a block is sealed. */
export const CHRONO_BLOCK_SEALED_EVENT = "AVB_CHRONO_BLOCK_SEALED" as const;

/**
 * Per-agent chronometer data directory. Inside it the service writes:
 *   ./logs/events.chain — header-only PoW chain (108 B/block)
 *   ./logs/events.bin   — off-chain event segments (Merkle-committed)
 * Default: ${ELIZA_DATA_DIR}/avb-chrono/<agentId>
 */
export const ENV_CHRONO_DATA_DIR = "AVB_CHRONO_DATA_DIR";
export const ENV_CHRONO_CPU_PERCENT = "AVB_CHRONO_CPU_PERCENT";
export const ENV_CHRONO_BLOCK_MS = "AVB_CHRONO_BLOCK_MS";
/** Set to "false"/"0" to disable the miner (tests, CI). */
export const ENV_CHRONO_ENABLED = "AVB_CHRONO_ENABLED";

const DEFAULT_CPU_PERCENT = 5;
const DEFAULT_BLOCK_MS = 60_000;

const EMPTY = new Uint8Array(0);

function concat(parts: readonly Uint8Array[]): Uint8Array {
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

// Structural runtime surface this service uses (mirrors AvbRuntimeSurface
// pattern so tests can pass plain objects).
interface ChronoRuntimeSurface {
  agentId: string;
  logger: {
    info: (msg: string) => void;
    warn: (msg: string) => void;
    error: (msg: string) => void;
    debug: (msg: string) => void;
  };
  getSetting(key: string): unknown;
  emitEvent(name: string, payload: unknown): Promise<void>;
}

// ----------------------------------------------------------------------------
// Service
// ----------------------------------------------------------------------------

export class ChronometerService extends Service {
  static serviceType = CHRONO_SERVICE_TYPE;
  static serviceName = "AVB Chronometer";

  public capabilityDescription =
    "On-device proof-of-work timestamp server: seals AVB event logs into a " +
    "hash-linked chain at ~1 block/min using a fixed CPU allocation, giving " +
    "the agent a tamper-evident intrinsic clock independent of host time.";

  // ---- config (frozen at start) ----
  private dataDir = "";
  private chainPath = "";
  private eventsPath = "";
  private cpuFraction = DEFAULT_CPU_PERCENT / 100;
  private targetBlockMs = DEFAULT_BLOCK_MS;
  private enabled = true;

  // ---- state ----
  private rt!: ChronoRuntimeSurface;
  private chain: Block[] = [];
  private pendingEvents: ChronoEvent[] = [];
  private currentBits = INITIAL_DIFFICULTY_BITS;
  private worker?: Worker;
  /** Integrity issues observed at any point (load-time + runtime). */
  private issues: IntegrityIssue[] = [];
  /**
   * Event segment for the block currently being mined, held here so
   * it can be written to events.bin once the worker returns the
   * sealed header (the worker only sees the Merkle root).
   */
  private inFlight?: { height: number; segment: Uint8Array };

  // --------------------------------------------------------------------------
  // Lifecycle
  // --------------------------------------------------------------------------

  static async start(runtime: IAgentRuntime): Promise<ChronometerService> {
    const svc = new ChronometerService(runtime);
    svc.rt = runtime as unknown as ChronoRuntimeSurface;

    svc.resolveConfig();
    await svc.loadAndVerify();

    if (svc.chain.length === 0) {
      await svc.mintGenesis();
    } else {
      // Resume difficulty from the tip so retargeting continues smoothly.
      svc.currentBits = svc.tip().header.difficultyBits;
    }

    svc.recordEvent(
      ChronoEventType.BOOT,
      `agent=${svc.rt.agentId} height=${svc.getHeight()} hostTs=${Date.now()}`,
    );

    if (svc.enabled) {
      svc.spawnMiner();
      svc.dispatchNextJob();
    } else {
      svc.rt.logger.info("[avb-chrono] miner disabled via setting");
    }

    const cores = Math.max(1, os.cpus()?.length ?? 1);
    svc.rt.logger.info(
      `[avb-chrono] started (dir=${svc.dataDir} height=${svc.getHeight()} ` +
        `cpu=${((svc.cpuFraction * 100) / cores).toFixed(1)}% total ` +
        `[${(svc.cpuFraction * 100).toFixed(0)}% of 1/${cores} cores] ` +
        `target=${svc.targetBlockMs}ms)`,
    );
    return svc;
  }

  async stop(): Promise<void> {
    this.recordEvent(ChronoEventType.SHUTDOWN, "service stop");
    const w = this.worker;
    this.worker = undefined; // stop dispatchNextJob() from re-arming
    if (w) {
      // The worker was unref()'d so it wouldn't keep the process alive
      // while idle; re-ref it now so the event loop can't drain out from
      // under `await terminate()` (observed under Bun).
      w.ref();
      w.removeAllListeners("message");
      w.postMessage({ kind: "stop" });
      await Promise.race([
        w.terminate(),
        new Promise((r) => setTimeout(r, 2000)),
      ]);
    }
  }

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  /** Buffer an event to be sealed into the next block. */
  recordEvent(type: ChronoEventType, detail: string): void {
    this.pendingEvents.push({
      type,
      hostTimestampMs: Date.now(),
      detail,
    });
  }

  /** Tip block (genesis or later). Throws if called before start(). */
  tip(): Block {
    const t = this.chain[this.chain.length - 1];
    if (!t) throw new Error("[avb-chrono] tip() called on empty chain");
    return t;
  }

  getHeight(): number {
    return this.chain.length === 0 ? -1 : this.tip().header.height;
  }

  /**
   * Intrinsic time: cumulative miner-measured wall time across all blocks.
   * This is the AVB's "I have personally witnessed this many milliseconds
   * of proof-of-work" clock — immune to host-clock edits.
   */
  getIntrinsicTimeMs(): number {
    let sum = 0;
    for (const b of this.chain) sum += b.header.wallElapsedMs;
    return sum;
  }

  /** All integrity issues seen so far (load-time + any later re-checks). */
  getIntegrityIssues(): readonly IntegrityIssue[] {
    return this.issues;
  }

  /** Re-verify the in-memory chain; new issues are flagged and returned. */
  verify(): IntegrityIssue[] {
    const found = verifyBlocks(this.chain);
    for (const i of found) this.flagIssue(i);
    return found;
  }

  getDataDir(): string {
    return this.dataDir;
  }

  getChainPath(): string {
    return this.chainPath;
  }

  getEventsPath(): string {
    return this.eventsPath;
  }

  /**
   * Cross-check the off-chain event log against sealed Merkle roots.
   * Loads events.bin, recomputes each segment's root, flags
   * MERKLE_ROOT_MISMATCH for any divergence. Heavier than verify()
   * (touches the full event log) so it is on-demand only.
   */
  async verifyEventLog(): Promise<IntegrityIssue[]> {
    const segments = await loadEventSegments(this.eventsPath);
    const found = verifySegments(this.chain, segments, (seg) =>
      merkleRoot(splitEvents(seg)),
    );
    for (const i of found) this.flagIssue(i);
    return found;
  }

  /**
   * Build an inclusion proof for event `eventIndex` of block `height`,
   * sufficient to convince a third party holding only the chain (or
   * even just the one block header) that the event was sealed.
   */
  async proveInclusion(
    height: number,
    eventIndex: number,
  ): Promise<InclusionProof> {
    const block = this.chain.find((b) => b.header.height === height);
    if (!block) throw new Error(`proveInclusion: no block at height ${height}`);
    const segments = await loadEventSegments(this.eventsPath);
    const seg = segments.get(height);
    if (!seg) {
      throw new Error(
        `proveInclusion: off-chain segment for #${height} missing`,
      );
    }
    const leaves = splitEvents(seg);
    if (eventIndex < 0 || eventIndex >= leaves.length) {
      throw new RangeError(
        `proveInclusion: event index ${eventIndex} out of range`,
      );
    }
    return {
      header: block.header,
      encodedEvent: leaves[eventIndex],
      path: merkleProof(leaves, eventIndex),
    };
  }

  // --------------------------------------------------------------------------
  // Config
  // --------------------------------------------------------------------------

  private resolveConfig(): void {
    const s = (k: string): string | undefined => {
      const v = this.rt.getSetting(k);
      return typeof v === "string" && v.length > 0 ? v : undefined;
    };

    // Per-agent directory so multiple agents on one host don't collide.
    const elizaData =
      (typeof process !== "undefined" && process.env?.ELIZA_DATA_DIR) ||
      path.join(process.cwd(), ".eliza");
    this.dataDir =
      s(ENV_CHRONO_DATA_DIR) ??
      path.join(elizaData, "avb-chrono", String(this.rt.agentId));
    this.chainPath = path.join(this.dataDir, "logs", "events.chain");
    this.eventsPath = path.join(this.dataDir, "logs", "events.bin");

    // CPU allocation is a percentage of TOTAL machine capacity. The
    // miner is a single thread, so translate to a duty-cycle fraction
    // of one core: 5% of an N-core box = N·5% of one core, capped at
    // 100% (we don't spawn additional miner threads).
    const cores = Math.max(1, os.cpus()?.length ?? 1);
    const totalPct = Number(s(ENV_CHRONO_CPU_PERCENT) ?? DEFAULT_CPU_PERCENT);
    const perCore = (totalPct / 100) * cores;
    this.cpuFraction = Math.min(1, Math.max(0.001, perCore));
    if (perCore > 1) {
      this.rt.logger.warn(
        `[avb-chrono] ${totalPct}% of ${cores} cores = ${(perCore * 100).toFixed(0)}% ` +
          `of one core; miner is single-threaded so capping at 100%`,
      );
    }

    const blk = Number(s(ENV_CHRONO_BLOCK_MS) ?? DEFAULT_BLOCK_MS);
    this.targetBlockMs =
      Number.isFinite(blk) && blk > 0 ? blk : DEFAULT_BLOCK_MS;

    const en = this.rt.getSetting(ENV_CHRONO_ENABLED);
    this.enabled = !(en === "false" || en === "0" || en === false);
  }

  // --------------------------------------------------------------------------
  // Load + verify
  // --------------------------------------------------------------------------

  private async loadAndVerify(): Promise<void> {
    const { blocks, issues } = await loadChain(this.chainPath);
    this.chain = blocks;
    for (const i of issues) this.flagIssue(i);
    if (blocks.length > 0) {
      this.rt.logger.info(
        `[avb-chrono] loaded ${blocks.length} block(s), tip=${toHex(this.tip().hash).slice(0, 16)}…`,
      );
    }
  }

  private flagIssue(issue: IntegrityIssue): void {
    // Dedupe on (code, height) so verify() re-runs don't spam.
    const key = `${issue.code}@${issue.height}`;
    if (this.issues.some((i) => `${i.code}@${i.height}` === key)) return;
    this.issues.push(issue);
    this.rt.logger.warn(
      `[avb-chrono] INTEGRITY ${issue.code} @${issue.height}: ${issue.message}`,
    );
    void this.rt
      .emitEvent(CHRONO_INTEGRITY_EVENT, { ...issue })
      .catch(() => undefined);
    // Also feed the warning into the event stream so it is itself sealed
    // into the next block — the agent's suspicion becomes part of its
    // tamper-evident history.
    this.recordEvent(
      ChronoEventType.INTEGRITY_WARNING,
      `${issue.code}: ${issue.message}`,
    );
  }

  // --------------------------------------------------------------------------
  // Genesis
  // --------------------------------------------------------------------------

  private async mintGenesis(): Promise<void> {
    const events: ChronoEvent[] = [
      {
        type: ChronoEventType.GENESIS,
        hostTimestampMs: Date.now(),
        detail: `genesis agent=${this.rt.agentId}`,
      },
    ];
    const leaves = events.map(encodeEvent);
    const segment = concat(leaves);
    const header: BlockHeader = {
      version: PROTOCOL_VERSION,
      height: 0,
      prevHash: ZERO_HASH,
      eventsHash: merkleRoot(leaves),
      hostTimestampMs: Date.now(),
      wallElapsedMs: 0,
      difficultyBits: INITIAL_DIFFICULTY_BITS,
      nonce: 0n,
      eventCount: events.length,
      eventBytes: segment.length,
    };
    // Genesis difficulty is trivially satisfiable; grind a few nonces in
    // case nonce=0 happens not to land under the (huge) target.
    let headerBytes = serializeHeader(header);
    let hash = sha256d(headerBytes);
    const target = bitsToTarget(header.difficultyBits);
    while (hashToBigInt(hash) > target) {
      header.nonce++;
      headerBytes = serializeHeader(header);
      hash = sha256d(headerBytes);
    }

    await appendBlockBytes(this.chainPath, headerBytes, EMPTY);
    await appendEventSegment(this.eventsPath, 0, segment);
    this.chain.push({ header, payload: EMPTY, hash });
    this.currentBits = INITIAL_DIFFICULTY_BITS;
    this.rt.logger.info(
      `[avb-chrono] genesis minted hash=${toHex(hash).slice(0, 16)}…`,
    );
  }

  // --------------------------------------------------------------------------
  // Miner thread
  // --------------------------------------------------------------------------

  private spawnMiner(): void {
    // Resolve the worker entry relative to this file so it works whether
    // the plugin is consumed from src/ or a future dist/ build.
    const url = new URL("./miner-worker.ts", import.meta.url);
    this.worker = new Worker(url);
    this.worker.unref();
    this.worker.on("message", (msg: MinerOutbound) => {
      if (msg.kind === "error") {
        this.rt.logger.error(`[avb-chrono] miner error: ${msg.message}`);
        return;
      }
      void this.onSealed(msg).catch((err) => {
        this.rt.logger.error(
          `[avb-chrono] onSealed failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      });
    });
    this.worker.on("error", (err: Error) => {
      this.rt.logger.error(`[avb-chrono] worker crashed: ${err.message}`);
    });
  }

  private dispatchNextJob(): void {
    if (!this.worker) return;
    const tip = this.tip();
    // Drain pending events into this job's segment.
    const events = this.pendingEvents;
    this.pendingEvents = [];
    const leaves = events.map(encodeEvent);
    const segment = concat(leaves);
    const height = tip.header.height + 1;
    this.inFlight = { height, segment };

    const job: MineJob = {
      kind: "mine",
      version: PROTOCOL_VERSION,
      height,
      prevHash: tip.hash,
      eventsHash: merkleRoot(leaves),
      eventCount: events.length,
      eventBytes: segment.length,
      difficultyBits: this.currentBits,
      hostTimestampMs: Date.now(),
      cpuFraction: this.cpuFraction,
    };
    this.worker.postMessage(job);
  }

  private async onSealed(msg: MineResult): Promise<void> {
    const { headerBytes, hash } = msg;
    // Pair the sealed header with the segment we held back at dispatch.
    const seg =
      this.inFlight && this.inFlight.height === msg.height
        ? this.inFlight.segment
        : EMPTY;
    this.inFlight = undefined;

    await appendBlockBytes(this.chainPath, headerBytes, EMPTY);
    await appendEventSegment(this.eventsPath, msg.height, seg);
    const block: Block = {
      header: deserializeHeader(headerBytes),
      payload: EMPTY,
      hash,
    };
    this.chain.push(block);

    // Retarget toward targetBlockMs using the miner's monotonic measurement.
    this.currentBits = retarget(
      block.header.difficultyBits,
      msg.wallElapsedMs,
      this.targetBlockMs,
    );

    const rate = msg.attempts / Math.max(1, msg.wallElapsedMs / 1000);
    this.rt.logger.info(
      `[avb-chrono] sealed #${block.header.height} ` +
        `hash=${toHex(hash).slice(0, 12)}… ` +
        `events=${block.header.eventCount} ` +
        `took=${(msg.wallElapsedMs / 1000).toFixed(1)}s ` +
        `(~${rate.toFixed(0)} H/s, next bits=0x${this.currentBits.toString(16)})`,
    );
    void this.rt
      .emitEvent(CHRONO_BLOCK_SEALED_EVENT, {
        height: block.header.height,
        hash: toHex(hash),
        hostTimestampMs: block.header.hostTimestampMs,
        wallElapsedMs: msg.wallElapsedMs,
        eventCount: block.header.eventCount,
      })
      .catch(() => undefined);

    this.dispatchNextJob();
  }
}

// ----------------------------------------------------------------------------
// Inclusion proofs
// ----------------------------------------------------------------------------

export interface InclusionProof {
  header: BlockHeader;
  encodedEvent: Uint8Array;
  path: MerkleStep[];
}

/**
 * Verify an inclusion proof against nothing but the proof itself:
 * checks the header's PoW and that walking the Merkle path from
 * `encodedEvent` reproduces `header.eventsHash`. Caller may
 * additionally check `header.prevHash` against a trusted tip.
 */
export function verifyInclusion(p: InclusionProof): boolean {
  const headerBytes = serializeHeader(p.header);
  const h = sha256d(headerBytes);
  if (hashToBigInt(h) > bitsToTarget(p.header.difficultyBits)) return false;
  const root = rootFromProof(p.encodedEvent, p.path);
  return bytesEqual(root, p.header.eventsHash);
}
