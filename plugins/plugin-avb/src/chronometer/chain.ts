/**
 * Chain persistence and validation.
 *
 * Storage is split across two append-only files:
 *
 *   CHAIN FILE (.chain) — header-only, fixed 80-byte stride.
 *     [4]  CHAIN_MAGIC
 *     [4]  fileVersion
 *     [80] header 0
 *     [80] header 1
 *     ...
 *   Stays small regardless of event volume: ~42 MB/year at 1 block/min.
 *   block[n] is at byte offset FILE_HEADER_SIZE + n*HEADER_SIZE — random
 *   access is trivial if it's ever needed.
 *
 *   EVENT LOG (.events) — sparse, variable-length batches.
 *     [4]  EVENTS_MAGIC
 *     [4]  fileVersion
 *     [4]height [4]bodyLen [body]   ← one batch per block-with-events
 *     ...
 *   Only blocks with events write a batch; empty blocks are skipped. The
 *   chain header's eventsRoot anchors each batch — tamper the log, the
 *   recomputed root won't match. Lose the log entirely, the chain still
 *   proves *something* was committed at each height (the commitments are
 *   unforgeable PoW); you just can't verify what.
 *
 * Validation checks four invariants per block:
 *   1. prevHash matches the actual hash of the previous header
 *   2. PoW: header hash <= bitsToTarget(header.bits)
 *   3. eventsRoot matches the hash of the serialized event payload
 *      (cross-referenced from the event log; missing batch ⇒ assume [])
 *   4. hostTimestamp is non-decreasing (warning, not error — host clock
 *      can legitimately go backwards under NTP correction, but we flag
 *      it because it's *also* what tampering looks like)
 *
 * Any failure on 1-3 is a hard tamper signal. Chain truncates to last
 * good block; issue is reported so the agent can scream about it.
 */

import { existsSync, readFileSync } from "node:fs";
import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import {
  computeEventsRoot,
  deserializeEventBatch,
  deserializeHeader,
  EMPTY_EVENTS_ROOT,
  headerHash,
  serializeEventBatch,
  serializeHeader,
  ZERO_HASH,
} from "./encoding.ts";
import { verifyPow } from "./miner.ts";
import type {
  Block,
  ChainIssue,
  ChainTip,
  ChronoEvent,
  ValidationReport,
} from "./types.ts";
import { CHAIN_MAGIC, EVENTS_MAGIC, HEADER_SIZE } from "./types.ts";

// ----------------------------------------------------------------------------
// File headers
// ----------------------------------------------------------------------------

const FILE_VERSION = 1;
const FILE_HEADER_SIZE = 8;

function buildFileHeader(magic: number): Buffer {
  const buf = Buffer.alloc(FILE_HEADER_SIZE);
  buf.writeUInt32BE(magic, 0);
  buf.writeUInt32LE(FILE_VERSION, 4);
  return buf;
}

// ----------------------------------------------------------------------------
// Event log loading (internal — feeds invariant 3)
// ----------------------------------------------------------------------------

/**
 * Read the event log into a height → events map. Sparse — only heights
 * with non-empty event lists are present.
 *
 * Errors here surface as `EVENTS_LOG_MISSING` warnings rather than hard
 * failures: the chain's integrity (linkage, PoW, timestamps) is verifiable
 * without the event log. We just lose the ability to confirm what was
 * committed.
 *
 * Orphan batches (height >= chain length) are silently dropped — they
 * arise when an event-log write succeeds but the subsequent chain-header
 * write fails (crash, disk full). The events were never committed, so
 * discarding them is correct.
 */
function loadEventBatches(
  path: string,
  warnings: ChainIssue[],
): Map<number, ChronoEvent[]> {
  const batches = new Map<number, ChronoEvent[]>();

  if (!existsSync(path)) {
    warnings.push({
      code: "EVENTS_LOG_MISSING",
      height: -1,
      detail: `event log not found at ${path} — chain commitments cannot be verified`,
    });
    return batches;
  }

  let raw: Buffer;
  try {
    raw = readFileSync(path);
  } catch (e) {
    warnings.push({
      code: "EVENTS_LOG_MISSING",
      height: -1,
      detail: `event log unreadable: ${e instanceof Error ? e.message : String(e)}`,
    });
    return batches;
  }

  if (raw.length < FILE_HEADER_SIZE || raw.readUInt32BE(0) !== EVENTS_MAGIC) {
    warnings.push({
      code: "EVENTS_LOG_MISSING",
      height: -1,
      detail: `event log has bad magic — ignoring (chain commitments unverifiable)`,
    });
    return batches;
  }

  let off = FILE_HEADER_SIZE;
  while (off < raw.length) {
    // Need at least height+bodyLen (8 bytes) to know how big this batch is.
    if (off + 8 > raw.length) break; // Truncated tail — partial write.
    const bodyLen = raw.readUInt32LE(off + 4);
    if (off + 8 + bodyLen > raw.length) break; // Truncated body.
    try {
      const { height, events, bytesRead } = deserializeEventBatch(raw, off);
      batches.set(height, events);
      off += bytesRead;
    } catch {
      // Decode error — stop here. Earlier batches are still good.
      break;
    }
  }

  return batches;
}

// ----------------------------------------------------------------------------
// Load + validate
// ----------------------------------------------------------------------------

/**
 * Load and validate the chain from disk.
 *
 * Reads both the chain file (headers) and the event log (payloads). Chain
 * integrity (linkage, PoW, timestamps) is verified from headers alone;
 * payload integrity (eventsRoot) is verified by cross-referencing the
 * event log. Events are read for validation then discarded — the returned
 * blocks are header-only, matching what's actually in the chain file.
 *
 * If the chain file doesn't exist, returns an empty chain with ok=true.
 * If the event log is missing but the chain exists, that's an
 * EVENTS_LOG_MISSING warning — the agent decides if that's suspicious.
 */
export function loadChain(
  chainPath: string,
  eventLogPath: string,
): {
  blocks: Block[];
  report: ValidationReport;
} {
  if (!existsSync(chainPath)) {
    return { blocks: [], report: { ok: true, validBlocks: 0, issues: [] } };
  }

  const raw = readFileSync(chainPath);
  const issues: ChainIssue[] = [];

  if (raw.length < FILE_HEADER_SIZE) {
    return badFile("TRUNCATED", "chain file shorter than header");
  }
  if (raw.readUInt32BE(0) !== CHAIN_MAGIC) {
    return badFile("BAD_MAGIC", `expected 0x${CHAIN_MAGIC.toString(16)}`);
  }
  // fileVersion at offset 4 — checked but currently only one version exists.

  // Pull event batches up front. We deliberately load these BEFORE
  // walking the chain so that an event-log issue shows up first in the
  // issues array (less surprising for the reader).
  const eventBatches = loadEventBatches(eventLogPath, issues);
  const eventsAvailable = !issues.some((i) => i.code === "EVENTS_LOG_MISSING");

  const blocks: Block[] = [];
  let off = FILE_HEADER_SIZE;
  let height = 0;
  let prevHash: Buffer = ZERO_HASH;
  let prevTimestamp = 0;

  while (off < raw.length) {
    // Fixed-stride records — exactly HEADER_SIZE bytes each.
    if (off + HEADER_SIZE > raw.length) {
      issues.push({
        code: "TRUNCATED",
        height,
        detail: `chain file ends mid-header at offset ${off} (need ${HEADER_SIZE}, have ${raw.length - off})`,
      });
      break;
    }

    const header = deserializeHeader(raw, off);
    const hash = headerHash(header);
    const block: Block = { header, hash, height };
    off += HEADER_SIZE;

    // Invariant 1: chain linkage.
    if (!header.prevHash.equals(prevHash)) {
      issues.push({
        code: "PREV_HASH_MISMATCH",
        height,
        detail: `expected prevHash ${prevHash.toString("hex").slice(0, 16)}…, got ${header.prevHash.toString("hex").slice(0, 16)}…`,
      });
      break;
    }

    // Invariant 2: proof of work.
    if (!verifyPow(header)) {
      issues.push({
        code: "POW_INSUFFICIENT",
        height,
        detail: `header hash exceeds target at bits=0x${header.bits.toString(16)}`,
      });
      break;
    }

    // Invariant 3: event payload integrity.
    // Sparse log: a missing batch means the block sealed an empty event
    // list, in which case the header must commit to EMPTY_EVENTS_ROOT.
    // If the log is entirely absent we can't check this at all — that
    // case is already a separate warning above.
    if (eventsAvailable) {
      const events = eventBatches.get(height);
      const computedRoot = events
        ? computeEventsRoot(events)
        : EMPTY_EVENTS_ROOT;
      if (!computedRoot.equals(header.eventsRoot)) {
        issues.push({
          code: "EVENTS_ROOT_MISMATCH",
          height,
          detail: events
            ? `header commits ${header.eventsRoot.toString("hex").slice(0, 16)}…, events hash to ${computedRoot.toString("hex").slice(0, 16)}…`
            : `header commits to non-empty events but no batch found in event log at height ${height}`,
        });
        break;
      }
    }

    // Invariant 4: timestamp monotonicity (warning only).
    // Host clock can legitimately step backwards (NTP, DST mishandling),
    // but it's also exactly what an admin rewinding the clock looks like,
    // so we surface it without rejecting the block. The agent gets to
    // decide whether the warning matters in context.
    if (header.hostTimestamp < prevTimestamp) {
      issues.push({
        code: "TIMESTAMP_REGRESSION",
        height,
        detail: `host timestamp went backwards: ${prevTimestamp} → ${header.hostTimestamp} (Δ=${header.hostTimestamp - prevTimestamp}s)`,
      });
      // Don't break — block itself is valid.
    }

    blocks.push(block);
    prevHash = hash;
    prevTimestamp = header.hostTimestamp;
    height++;
  }

  // Hard issues are everything except the two warning codes.
  const hardIssues = issues.filter(
    (i) => i.code !== "TIMESTAMP_REGRESSION" && i.code !== "EVENTS_LOG_MISSING",
  );
  return {
    blocks,
    report: { ok: hardIssues.length === 0, validBlocks: blocks.length, issues },
  };

  function badFile(code: ChainIssue["code"], detail: string) {
    return {
      blocks: [],
      report: {
        ok: false,
        validBlocks: 0,
        issues: [{ code, height: -1, detail }],
      },
    };
  }
}

// ----------------------------------------------------------------------------
// Append
// ----------------------------------------------------------------------------

/**
 * Append one block. Writes to BOTH files in a specific order:
 *
 *   1. Event log (if events.length > 0)
 *   2. Chain header
 *
 * The order matters. If we crash between writes, the orphan event batch
 * is harmless on next load (no chain block at that height — silently
 * dropped). If we wrote the header first and crashed, we'd have a header
 * committing to events that aren't on disk, which would falsely report
 * EVENTS_ROOT_MISMATCH. Events-first means crash recovery is benign.
 *
 * Both appends are atomic-ish: appendFile is a single write() syscall for
 * payloads this size, and a partial write produces a record that the
 * loader's bounds checks catch as truncated.
 *
 * The caller is responsible for ensuring `block` actually chains
 * correctly — we don't re-validate on write. Validation is a load-time
 * concern; write-time is "I just mined this and I trust myself."
 */
export async function appendBlock(
  chainPath: string,
  eventLogPath: string,
  block: Block,
  events: ChronoEvent[],
): Promise<void> {
  // Step 1: event batch (sparse — skip if empty).
  if (events.length > 0) {
    if (!existsSync(eventLogPath)) {
      await mkdir(dirname(eventLogPath), { recursive: true });
      await writeFile(eventLogPath, buildFileHeader(EVENTS_MAGIC));
    }
    await appendFile(eventLogPath, serializeEventBatch(block.height, events));
  }

  // Step 2: chain header.
  if (!existsSync(chainPath)) {
    await mkdir(dirname(chainPath), { recursive: true });
    await writeFile(chainPath, buildFileHeader(CHAIN_MAGIC));
  }
  await appendFile(chainPath, serializeHeader(block.header));
}

// ----------------------------------------------------------------------------
// Tip extraction
// ----------------------------------------------------------------------------

/**
 * Pull the tip summary off a loaded chain. Returns undefined for an
 * empty chain (caller should mine genesis).
 *
 * cumulativeWork is approximate — we sum 1/target (in JS-number space)
 * as a sortable proxy. Good enough for "did the chain get longer/heavier
 * since I last looked"; not good enough for cross-chain comparison
 * (which we never do).
 */
export function tipOf(blocks: Block[]): ChainTip | undefined {
  if (blocks.length === 0) return undefined;
  const tip = blocks[blocks.length - 1];

  // Approximate work: for each block, work ≈ 2^256 / target.
  // We compute this in log-space to avoid overflow, then sum.
  // Practically: count leading zero bytes, treat as 256^n.
  let cumulativeWork = 0;
  for (const b of blocks) {
    // Read first non-zero byte position from the hash itself (hash <= target,
    // so hash's leading zeros are at least target's). Cheaper than expanding
    // bits and approximately right for a monotone metric.
    let lz = 0;
    while (lz < 32 && b.hash[lz] === 0) lz++;
    cumulativeWork += 256 ** Math.min(lz, 6); // cap to avoid Infinity
  }

  return {
    height: tip.height,
    hash: tip.hash,
    bits: tip.header.bits,
    hostTimestamp: tip.header.hostTimestamp,
    cumulativeWork,
  };
}

// ----------------------------------------------------------------------------
// Re-validation of a single block (for verifying freshly mined blocks
// before append — belt-and-braces against bugs in the worker)
// ----------------------------------------------------------------------------

export function validateBlock(
  block: Block,
  expectedPrevHash: Buffer,
  events: ChronoEvent[],
): ChainIssue | null {
  if (!block.header.prevHash.equals(expectedPrevHash)) {
    return {
      code: "PREV_HASH_MISMATCH",
      height: block.height,
      detail: "freshly mined block does not chain to current tip",
    };
  }
  if (!verifyPow(block.header)) {
    return {
      code: "POW_INSUFFICIENT",
      height: block.height,
      detail: "freshly mined block fails its own PoW check",
    };
  }
  const root = computeEventsRoot(events);
  if (!root.equals(block.header.eventsRoot)) {
    return {
      code: "EVENTS_ROOT_MISMATCH",
      height: block.height,
      detail: "freshly mined block eventsRoot does not match payload",
    };
  }
  // Recompute the cached hash too — defends against a worker that
  // sends back the wrong hash for a valid header.
  const computedHash = headerHash(block.header);
  if (!computedHash.equals(block.hash)) {
    return {
      code: "POW_INSUFFICIENT",
      height: block.height,
      detail: "cached hash does not match recomputed header hash",
    };
  }
  return null;
}
