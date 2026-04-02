/**
 * Chain file I/O and integrity verification.
 *
 * The chain file is a flat append-only concatenation of blocks
 * (header‖payload, header‖payload, …). There is no separate index;
 * a reader walks forward using each header's `eventBytes` field.
 *
 * Verification checks, per block:
 *   - magic bytes present
 *   - height == prev.height + 1 (or 0 for first)
 *   - prevHash == hash(prev header)
 *   - eventsHash == SHA256(payload)
 *   - sha256d(header) <= target(difficultyBits)
 *
 * And cross-block heuristics:
 *   - hostTimestampMs strictly non-decreasing (else HOST_CLOCK_REGRESSION)
 *   - hostTimestamp gap not wildly larger than wallElapsedMs
 *     (else HOST_CLOCK_GAP — possible interruption / restore-from-snapshot)
 *
 * None of these checks abort the load; they are returned as IntegrityIssue
 * records so the ChronometerService can raise warning flags while still
 * resuming from whatever tip exists.
 */

import { promises as fs } from "node:fs";
import path from "node:path";
import {
  bitsToTarget,
  bytesEqual,
  deserializeHeader,
  HEADER_SIZE,
  hashToBigInt,
  MAGIC,
  sha256,
  sha256d,
} from "./block.ts";
import type { Block, IntegrityIssue } from "./types.ts";

// ----------------------------------------------------------------------------
// Tunables
// ----------------------------------------------------------------------------

/**
 * If host-clock delta between consecutive blocks exceeds the miner's
 * monotonic measurement by more than this factor (and by more than the
 * absolute floor), flag HOST_CLOCK_GAP. Generous because legitimate
 * process restarts also produce gaps.
 */
const GAP_RATIO = 8;
const GAP_ABS_FLOOR_MS = 5 * 60_000;

// ----------------------------------------------------------------------------
// Load + parse
// ----------------------------------------------------------------------------

export interface LoadResult {
  blocks: Block[];
  issues: IntegrityIssue[];
  /** Bytes successfully consumed; if < file size, tail was truncated/corrupt. */
  bytesConsumed: number;
}

/**
 * Read and parse the entire chain file. Missing file → empty chain, no issues.
 */
export async function loadChain(filePath: string): Promise<LoadResult> {
  let raw: Uint8Array;
  try {
    raw = new Uint8Array(await fs.readFile(filePath));
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") {
      return { blocks: [], issues: [], bytesConsumed: 0 };
    }
    throw err;
  }
  return parseChain(raw);
}

/** Parse a raw chain buffer (exposed for tests). */
export function parseChain(raw: Uint8Array): LoadResult {
  const blocks: Block[] = [];
  const issues: IntegrityIssue[] = [];
  let off = 0;

  while (off + HEADER_SIZE <= raw.length) {
    const headerBytes = raw.subarray(off, off + HEADER_SIZE);

    // Magic check — if wrong, we've lost framing; stop here.
    if (!bytesEqual(headerBytes.subarray(0, 4), MAGIC)) {
      issues.push({
        code: "BAD_MAGIC",
        height: blocks.length,
        message: `bad magic at byte ${off}`,
      });
      break;
    }

    const header = deserializeHeader(headerBytes);
    // v1 inlines `eventBytes` of payload after the header; v2+ stores
    // events off-chain so the header is followed by nothing.
    const inlineLen = header.version < 2 ? header.eventBytes : 0;
    const payloadEnd = off + HEADER_SIZE + inlineLen;
    if (payloadEnd > raw.length) {
      issues.push({
        code: "TRUNCATED",
        height: header.height,
        message: `block ${header.height} payload truncated`,
      });
      break;
    }
    const payload = raw.slice(off + HEADER_SIZE, payloadEnd);
    const hash = sha256d(headerBytes);
    blocks.push({ header, payload, hash });
    off = payloadEnd;
  }

  if (off < raw.length && issues.length === 0) {
    issues.push({
      code: "TRUNCATED",
      height: blocks.length,
      message: `${raw.length - off} trailing bytes after last full block`,
    });
  }

  issues.push(...verifyBlocks(blocks));
  return { blocks, issues, bytesConsumed: off };
}

// ----------------------------------------------------------------------------
// Verification
// ----------------------------------------------------------------------------

export function verifyBlocks(blocks: readonly Block[]): IntegrityIssue[] {
  const issues: IntegrityIssue[] = [];

  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    const h = b.header;
    const prev = i > 0 ? blocks[i - 1] : undefined;

    // Height continuity
    const expectedHeight = prev ? prev.header.height + 1 : 0;
    if (h.height !== expectedHeight) {
      issues.push({
        code: "HEIGHT_MISMATCH",
        height: h.height,
        message: `expected height ${expectedHeight}, got ${h.height}`,
      });
    }

    // prevHash linkage
    if (prev && !bytesEqual(h.prevHash, prev.hash)) {
      issues.push({
        code: "PREV_HASH_MISMATCH",
        height: h.height,
        message: `block ${h.height} prevHash does not match block ${prev.header.height} hash`,
      });
    }

    // Events hash — only checkable here for v1 (payload inline). For
    // v2 the eventsHash is a Merkle root over off-chain data; that is
    // verified on demand via verifySegments().
    if (h.version < 2) {
      const eh = sha256(b.payload);
      if (!bytesEqual(h.eventsHash, eh)) {
        issues.push({
          code: "EVENTS_HASH_MISMATCH",
          height: h.height,
          message: `block ${h.height} eventsHash does not match payload`,
        });
      }
    }

    // Proof of work
    if (hashToBigInt(b.hash) > bitsToTarget(h.difficultyBits)) {
      issues.push({
        code: "POW_INVALID",
        height: h.height,
        message: `block ${h.height} hash exceeds target (invalid PoW)`,
      });
    }

    // Host-clock heuristics (advisory)
    if (prev) {
      const hostDelta = h.hostTimestampMs - prev.header.hostTimestampMs;
      if (hostDelta < 0) {
        issues.push({
          code: "HOST_CLOCK_REGRESSION",
          height: h.height,
          message: `host clock went backward by ${-hostDelta}ms at block ${h.height}`,
        });
      } else if (
        hostDelta > GAP_ABS_FLOOR_MS &&
        hostDelta > h.wallElapsedMs * GAP_RATIO
      ) {
        issues.push({
          code: "HOST_CLOCK_GAP",
          height: h.height,
          message: `host clock jumped ${hostDelta}ms but miner measured ${h.wallElapsedMs}ms — possible interruption/restore`,
        });
      }
    }
  }

  return issues;
}

// ----------------------------------------------------------------------------
// Append
// ----------------------------------------------------------------------------

/**
 * Append a single serialized block (header‖payload) to the chain file.
 * For v2 blocks `payload` is empty. Creates parent directories on
 * first write.
 */
export async function appendBlockBytes(
  filePath: string,
  headerBytes: Uint8Array,
  payload: Uint8Array,
): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const buf = new Uint8Array(headerBytes.length + payload.length);
  buf.set(headerBytes, 0);
  buf.set(payload, headerBytes.length);
  await fs.appendFile(filePath, buf);
}

// ----------------------------------------------------------------------------
// Off-chain event log (v2)
//
// events.bin is a flat append-only sequence of segments, one per block:
//   u32 height (LE) | u32 segLen (LE) | segLen bytes of encoded events
// ----------------------------------------------------------------------------

export const SEGMENT_FRAME_SIZE = 8;

/** Append one block's encoded-event segment to the off-chain log. */
export async function appendEventSegment(
  eventsPath: string,
  height: number,
  encodedEvents: Uint8Array,
): Promise<void> {
  await fs.mkdir(path.dirname(eventsPath), { recursive: true });
  const buf = new Uint8Array(SEGMENT_FRAME_SIZE + encodedEvents.length);
  const dv = new DataView(buf.buffer);
  dv.setUint32(0, height >>> 0, true);
  dv.setUint32(4, encodedEvents.length >>> 0, true);
  buf.set(encodedEvents, SEGMENT_FRAME_SIZE);
  await fs.appendFile(eventsPath, buf);
}

/**
 * Load every segment from events.bin into a height→payload map.
 * Missing file → empty map. A truncated trailing segment is skipped.
 */
export async function loadEventSegments(
  eventsPath: string,
): Promise<Map<number, Uint8Array>> {
  let raw: Uint8Array;
  try {
    raw = new Uint8Array(await fs.readFile(eventsPath));
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return new Map();
    throw err;
  }
  const out = new Map<number, Uint8Array>();
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  let off = 0;
  while (off + SEGMENT_FRAME_SIZE <= raw.length) {
    const height = dv.getUint32(off, true);
    const segLen = dv.getUint32(off + 4, true);
    const end = off + SEGMENT_FRAME_SIZE + segLen;
    if (end > raw.length) break;
    out.set(height, raw.slice(off + SEGMENT_FRAME_SIZE, end));
    off = end;
  }
  return out;
}

/**
 * Cross-check off-chain segments against sealed Merkle roots.
 * `merkleRootOf` recomputes the root from a segment's encoded-event
 * bytes (injected so this module stays cycle-free with merkle.ts).
 *
 * Flags MERKLE_ROOT_MISMATCH for any v2 block whose segment is missing,
 * has the wrong byte length, or whose recomputed root differs from
 * `header.eventsHash`.
 */
export function verifySegments(
  blocks: readonly Block[],
  segments: ReadonlyMap<number, Uint8Array>,
  merkleRootOf: (segment: Uint8Array) => Uint8Array,
): IntegrityIssue[] {
  const issues: IntegrityIssue[] = [];
  for (const b of blocks) {
    const h = b.header;
    if (h.version < 2) continue;
    const seg = segments.get(h.height);
    if (!seg) {
      if (h.eventCount > 0) {
        issues.push({
          code: "MERKLE_ROOT_MISMATCH",
          height: h.height,
          message: `block ${h.height}: off-chain segment missing (expected ${h.eventCount} events)`,
        });
      }
      continue;
    }
    if (seg.length !== h.eventBytes) {
      issues.push({
        code: "MERKLE_ROOT_MISMATCH",
        height: h.height,
        message: `block ${h.height}: segment length ${seg.length} ≠ header.eventBytes ${h.eventBytes}`,
      });
      continue;
    }
    const root = merkleRootOf(seg);
    if (!bytesEqual(root, h.eventsHash)) {
      issues.push({
        code: "MERKLE_ROOT_MISMATCH",
        height: h.height,
        message: `block ${h.height}: recomputed Merkle root does not match sealed header`,
      });
    }
  }
  return issues;
}
