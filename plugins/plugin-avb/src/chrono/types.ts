/**
 * Chronometer types — on-device proof-of-work timestamp server.
 *
 * The chronometer answers the AVB's intrinsic question: "how do I know
 * time has actually passed, and that my own history hasn't been edited
 * out from under me?"
 *
 * It does this by sealing the agent's recent event log into a block
 * roughly once per minute, where each block:
 *   - hash-links to the previous block (tamper-evident chain)
 *   - carries a proof-of-work that cost real CPU-seconds to produce
 *   - embeds the host's reported wall-clock time alongside the miner's
 *     monotonic measurement of elapsed work-time
 *
 * An admin can change the host clock, but cannot fake the accumulated
 * PoW without re-spending the CPU. Any edit to a past block invalidates
 * every block after it.
 */

// ----------------------------------------------------------------------------
// Event log
// ----------------------------------------------------------------------------

/**
 * Event kinds recorded by the AVB into the chronometer.
 * Encoded as a single byte on disk — keep values < 256.
 */
export enum ChronoEventType {
  GENESIS = 0,
  BOOT = 1,
  RUN_STARTED = 2,
  PHASE_STARTED = 3,
  PHASE_COMPLETE = 4,
  PHASE_FAILED = 5,
  DELIVER = 6,
  INTEGRITY_WARNING = 7,
  SHUTDOWN = 8,
  /** Forwarded from an ElizaOS runtime EventType (see chrono/bridge.ts). */
  RUNTIME = 9,
  OTHER = 255,
}

/** A single event entry, before binary encoding. */
export interface ChronoEvent {
  type: ChronoEventType;
  /** Host wall-clock at the moment the event was recorded (Date.now()). */
  hostTimestampMs: number;
  /** Free-form UTF-8 detail (action name, runId, error text, etc.). */
  detail: string;
}

// ----------------------------------------------------------------------------
// Block header
// ----------------------------------------------------------------------------

/**
 * Fixed-width header that is actually hashed for PoW.
 * Layout is documented in block.ts; total size = HEADER_SIZE bytes.
 */
export interface BlockHeader {
  /** Protocol version (bumped on incompatible header changes). */
  version: number;
  /** 0 for genesis, monotonically increasing thereafter. */
  height: number;
  /** Double-SHA256 of the previous block's header (32 zero bytes for genesis). */
  prevHash: Uint8Array;
  /**
   * v1: SHA256 of the inline event payload that follows the header.
   * v2: Merkle root over the block's events (stored off-chain in
   *     events.bin); no payload follows the header in the chain file.
   */
  eventsHash: Uint8Array;
  /** Host-reported wall-clock when mining of this block began (Date.now()). */
  hostTimestampMs: number;
  /**
   * Miner-measured wall milliseconds spent producing this block
   * (monotonic; immune to host-clock edits while the process is alive).
   */
  wallElapsedMs: number;
  /** Compact difficulty target (Bitcoin nBits encoding). */
  difficultyBits: number;
  /** PoW nonce. */
  nonce: bigint;
  /** Number of events in the payload. */
  eventCount: number;
  /**
   * v1: byte length of the inline payload following the header.
   * v2: byte length of the off-chain segment in events.bin (for seek);
   *     zero payload bytes follow the header in the chain file.
   */
  eventBytes: number;
}

/** A fully-decoded block (header + raw payload + computed hash). */
export interface Block {
  header: BlockHeader;
  /** Raw encoded event bytes exactly as written to disk. */
  payload: Uint8Array;
  /** Double-SHA256 of the serialized header. */
  hash: Uint8Array;
}

// ----------------------------------------------------------------------------
// Integrity verification
// ----------------------------------------------------------------------------

export type IntegrityCode =
  | "BAD_MAGIC"
  | "TRUNCATED"
  | "HEIGHT_MISMATCH"
  | "PREV_HASH_MISMATCH"
  | "EVENTS_HASH_MISMATCH"
  | "MERKLE_ROOT_MISMATCH"
  | "POW_INVALID"
  | "HOST_CLOCK_REGRESSION"
  | "HOST_CLOCK_GAP";

export interface IntegrityIssue {
  code: IntegrityCode;
  /** Block height at which the issue was detected (or -1 for file-level). */
  height: number;
  message: string;
}

// ----------------------------------------------------------------------------
// Miner ↔ service messages (worker_threads)
// ----------------------------------------------------------------------------

/** Job sent from ChronometerService → miner worker. */
export interface MineJob {
  kind: "mine";
  version: number;
  height: number;
  /** 32-byte prev hash. */
  prevHash: Uint8Array;
  /** 32-byte commitment to this block's events (v2: Merkle root). */
  eventsHash: Uint8Array;
  eventCount: number;
  /** Byte length of the off-chain event segment (header.eventBytes). */
  eventBytes: number;
  /** Compact target this block must satisfy. */
  difficultyBits: number;
  /** Host wall-clock at job dispatch (frozen into the header). */
  hostTimestampMs: number;
  /** Duty-cycle fraction of one CPU core (0..1]. */
  cpuFraction: number;
}

/** Result sent from miner worker → ChronometerService. */
export interface MineResult {
  kind: "sealed";
  /** Serialized HEADER_SIZE-byte header (nonce + wallElapsedMs filled in). */
  headerBytes: Uint8Array;
  /** Double-SHA256 of headerBytes. */
  hash: Uint8Array;
  /** Echo of the height this result is for (guards against stale jobs). */
  height: number;
  /** Monotonic ms the miner actually spent (same value written into header). */
  wallElapsedMs: number;
  /** Total nonces tried (for hash-rate telemetry). */
  attempts: number;
}

export type MinerInbound = MineJob | { kind: "stop" };
export type MinerOutbound = MineResult | { kind: "error"; message: string };
