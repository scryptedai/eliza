/**
 * Chronometer types — on-device proof-of-work timestamp chain.
 *
 * Concept:
 * An AVB agent cannot trust the host's wallclock; an admin can rewrite it
 * at will. What an agent CAN trust is *accumulated work*: if a chain of N
 * blocks exists where each block (a) hash-links to its predecessor and
 * (b) satisfies a proof-of-work target that we know costs ~60 seconds of
 * CPU at the configured allocation, then the chain itself encodes "at
 * least N minutes of unforgeable wall-clock effort have elapsed since
 * genesis."
 *
 * This is Bitcoin's timestamp-server primitive [Nakamoto 2008] applied to
 * a single-process subjective clock instead of a distributed ledger. We
 * are not seeking consensus — we are seeking *subjective continuity*: the
 * agent's own sense of "I have been running uninterrupted for this long,
 * and here is the receipt."
 *
 * Storage is split across two files:
 *
 *   CHAIN FILE — header-only, fixed 80-byte stride. Stays small: a year
 *   at one block/min is ~42 MB regardless of event volume.
 *     [ 4]  uint32   version
 *     [32]  bytes    prevHash      (sha256d of previous header)
 *     [32]  bytes    eventsRoot    (sha256d of serialized event payload — commitment only)
 *     [ 4]  uint32   hostTimestamp (host's reported epoch seconds — recorded, not trusted)
 *     [ 4]  uint32   bits          (compact difficulty target, Bitcoin nBits encoding)
 *     [ 4]  uint32   nonce
 *
 *   EVENT LOG — sparse, variable-length batches. The actual event payloads.
 *   Anchored to the chain by eventsRoot: tamper the log → root mismatch on
 *   validation. Lose the log → chain still proves *something* was committed
 *   at each height, you just can't verify what.
 *     [ 4]  uint32   height        (which chain block this batch belongs to)
 *     [ 4]  uint32   bodyLen
 *     [ N]  body     (serialized events; see encoding.serializeEvents)
 *
 * Tamper detection:
 *   - Modifying any block's events  → eventsRoot mismatch
 *   - Modifying any block's header  → next block's prevHash mismatch
 *   - Removing blocks               → prevHash chain break
 *   - Inserting fake blocks         → must redo PoW for entire suffix
 *   - Wallclock manipulation        → hostTimestamp jumps but block height
 *                                     doesn't (work cannot be faked)
 */

// ----------------------------------------------------------------------------
// Wire constants
// ----------------------------------------------------------------------------

/** Header is exactly 80 bytes. Hashing always covers the full header. */
export const HEADER_SIZE = 80;

/** Current block format version. Bump on incompatible layout changes. */
export const BLOCK_VERSION = 1;

/** Magic bytes prefixing the chain file (uppercase "AVBC" → AVB Chronometer). */
export const CHAIN_MAGIC = 0x41564243;

/** Magic bytes prefixing the event log file ("AVBE" → AVB Events). */
export const EVENTS_MAGIC = 0x41564245;

// ----------------------------------------------------------------------------
// Event log entry
// ----------------------------------------------------------------------------

/** Discriminant for binary event encoding. uint8 on the wire. */
export enum EventKind {
  /** Catch-all for events without a dedicated kind. */
  GENERIC = 0,
  /** AVB pipeline phase started (TEXT_PHASE / IMAGE_PHASE / DELIVER). */
  PHASE_START = 1,
  /** AVB pipeline phase completed successfully. */
  PHASE_COMPLETE = 2,
  /** AVB pipeline phase failed. */
  PHASE_FAILED = 3,
  /** Service-level lifecycle event (start/stop/etc). */
  SERVICE = 4,
  /**
   * ElizaOS runtime event bridged from runtime.emitEvent — MESSAGE_SENT,
   * ACTION_STARTED, RUN_ENDED, etc. The chronometer subscribes to the
   * runtime event bus and witnesses these automatically.
   */
  RUNTIME = 5,
}

/**
 * One event recorded into the current block's mempool.
 * `at` is host time — like `hostTimestamp`, recorded but not trusted.
 * `data` is freeform UTF-8; callers decide encoding (typically JSON).
 */
export interface ChronoEvent {
  kind: EventKind;
  /** Host epoch milliseconds when the event was recorded. */
  at: number;
  /** Freeform UTF-8 payload. Caller-defined; chronometer treats it opaquely. */
  data: string;
}

// ----------------------------------------------------------------------------
// Block header
// ----------------------------------------------------------------------------

/**
 * The 80-byte mined header. The PoW hash is sha256(sha256(serialize(this))).
 * `prevHash` and `eventsRoot` are stored as Buffers (32 bytes each).
 */
export interface BlockHeader {
  version: number;
  prevHash: Buffer;
  eventsRoot: Buffer;
  /** Host-reported epoch seconds. Embedded, not trusted. */
  hostTimestamp: number;
  /** Compact difficulty target (Bitcoin nBits format). */
  bits: number;
  nonce: number;
}

// ----------------------------------------------------------------------------
// Full block
// ----------------------------------------------------------------------------

/**
 * A mined, sealed block. Header-only — events live in the separate event
 * log and are anchored by `header.eventsRoot`. `hash` and `height` are
 * derived (height = position in chain, hash = sha256d(header)) but cached
 * on the in-memory struct for convenience.
 */
export interface Block {
  header: BlockHeader;
  /** sha256d(header). Cached after mine/load. */
  hash: Buffer;
  /** 0 = genesis. Cached after load. */
  height: number;
}

// ----------------------------------------------------------------------------
// Mining job (main thread → worker)
// ----------------------------------------------------------------------------

/**
 * Everything the worker needs to grind a header. The worker does NOT see
 * raw events — main thread serializes events, computes eventsRoot, and
 * passes only the root. This keeps the worker dumb and the protocol clean
 * (no "what counts as the same event payload" ambiguity across the IPC
 * boundary).
 */
export interface MiningJob {
  /** Monotonic id so we can correlate results when jobs are superseded. */
  jobId: number;
  prevHash: Buffer;
  eventsRoot: Buffer;
  hostTimestamp: number;
  bits: number;
}

// ----------------------------------------------------------------------------
// Mining result (worker → main thread)
// ----------------------------------------------------------------------------

export interface MiningResult {
  jobId: number;
  /** The nonce that satisfied the target. */
  nonce: number;
  /** sha256d of the winning header (worker computes it anyway; saves a re-hash). */
  hash: Buffer;
  /** How many hash attempts the worker burned. Feeds difficulty adjustment. */
  attempts: number;
  /** Wall-clock milliseconds the worker spent (including throttle sleeps). */
  elapsedMs: number;
}

// ----------------------------------------------------------------------------
// Worker control messages
// ----------------------------------------------------------------------------

export type WorkerInbound = { type: "mine"; job: MiningJob } | { type: "stop" };

export type WorkerOutbound =
  | { type: "solved"; result: MiningResult }
  | { type: "progress"; jobId: number; attempts: number };

// ----------------------------------------------------------------------------
// Validation result
// ----------------------------------------------------------------------------

export type ChainIssueCode =
  | "BAD_MAGIC"
  | "TRUNCATED"
  | "PREV_HASH_MISMATCH"
  | "EVENTS_ROOT_MISMATCH"
  | "POW_INSUFFICIENT"
  | "TIMESTAMP_REGRESSION"
  /**
   * Event log file missing or unreadable. Chain integrity (linkage, PoW,
   * timestamps) still holds — the commitments are intact in the headers —
   * but the committed payloads cannot be verified. Warning, not error.
   */
  | "EVENTS_LOG_MISSING";

export interface ChainIssue {
  code: ChainIssueCode;
  /** Block height where the issue was detected (or -1 for file-level). */
  height: number;
  detail: string;
}

export interface ValidationReport {
  ok: boolean;
  /** Number of blocks that validated cleanly before the first issue (or all). */
  validBlocks: number;
  issues: ChainIssue[];
}

// ----------------------------------------------------------------------------
// Tip summary (for status queries)
// ----------------------------------------------------------------------------

export interface ChainTip {
  height: number;
  hash: Buffer;
  bits: number;
  hostTimestamp: number;
  /** Sum of (1 / target) across all blocks — total accumulated work. */
  cumulativeWork: number;
}
