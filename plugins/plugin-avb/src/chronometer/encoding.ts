/**
 * Binary encoding/decoding for blocks, headers, and events.
 *
 * All multi-byte integers are little-endian (matches Bitcoin wire format
 * and Node's default for Buffer.writeUInt32LE etc — least-surprise on
 * both axes).
 *
 * Hashing is sha256d (double-SHA256), again matching Bitcoin. Single
 * SHA256 would be fine for our threat model (no length-extension attack
 * surface here), but doubling costs ~nothing per block and means anyone
 * who knows Bitcoin can read this code without translation.
 *
 * Compact difficulty (`bits`/nBits): we use Bitcoin's encoding because
 * (a) it packs a 256-bit target into 4 bytes, (b) the format is well-
 * documented, and (c) it makes our headers exactly 80 bytes. The encoding
 * is `target = mantissa * 2^(8*(exponent - 3))` where exponent is the
 * high byte and mantissa is the low 3 bytes.
 */

import { createHash } from "node:crypto";
import {
  BLOCK_VERSION,
  type BlockHeader,
  type ChronoEvent,
  HEADER_SIZE,
} from "./types.ts";

// ----------------------------------------------------------------------------
// Hashing
// ----------------------------------------------------------------------------

/** sha256(sha256(buf)). The Bitcoin primitive. */
export function sha256d(buf: Buffer): Buffer {
  const inner = createHash("sha256").update(buf).digest();
  return createHash("sha256").update(inner).digest();
}

// ----------------------------------------------------------------------------
// Compact difficulty target (nBits)
// ----------------------------------------------------------------------------

/**
 * Decode a 4-byte compact target into a 32-byte big-endian threshold.
 * A header hash (interpreted as a 256-bit big-endian integer) must be
 * <= this threshold to be valid PoW.
 */
export function bitsToTarget(bits: number): Buffer {
  const exponent = bits >>> 24;
  const mantissa = bits & 0x007fffff;
  const target = Buffer.alloc(32);

  if (exponent <= 3) {
    // Mantissa is right-shifted; rare at our difficulty levels but
    // handle it correctly so the encoding is total.
    const shifted = mantissa >>> (8 * (3 - exponent));
    target.writeUIntBE(shifted, 29, 3);
  } else {
    // Mantissa occupies 3 bytes; exponent positions its low byte.
    // Position is from the left: byte index = 32 - exponent.
    const pos = 32 - exponent;
    if (pos >= 0 && pos + 3 <= 32) {
      target.writeUIntBE(mantissa, pos, 3);
    }
    // Out-of-range exponents leave target = 0 (impossible to satisfy),
    // which is the safe failure mode.
  }
  return target;
}

/**
 * Encode a 32-byte big-endian target into compact form. Inverse of
 * bitsToTarget for the values we actually produce (lossy in general
 * because mantissa is only 3 bytes — but our adjuster only ever scales
 * by integer-ish factors so precision loss is sub-percent).
 */
export function targetToBits(target: Buffer): number {
  // Find the most significant non-zero byte.
  let msb = 0;
  while (msb < 32 && target[msb] === 0) msb++;
  let size = 32 - msb;

  if (size === 0) return 0; // All-zero target — degenerate.

  // Read up to 3 bytes of mantissa starting at msb.
  let mantissa = 0;
  for (let i = 0; i < 3 && msb + i < 32; i++) {
    mantissa = (mantissa << 8) | target[msb + i];
  }

  // If the high bit of the mantissa is set, Bitcoin shifts right by one
  // byte and bumps the exponent (sign-bit avoidance). We do the same so
  // round-tripping through Bitcoin tooling works.
  if (mantissa & 0x00800000) {
    mantissa >>>= 8;
    size++;
  }

  return (size << 24) | mantissa;
}

/** Compare two 32-byte big-endian integers. Returns true if a <= b. */
export function lte256(a: Buffer, b: Buffer): boolean {
  for (let i = 0; i < 32; i++) {
    if (a[i] < b[i]) return true;
    if (a[i] > b[i]) return false;
  }
  return true; // Equal.
}

/**
 * Approximate difficulty as a JS number: maxTarget / target.
 * Used only for human-readable logging and cumulative-work bookkeeping;
 * actual PoW comparison uses lte256() on the full 256-bit values.
 */
export function bitsToDifficulty(bits: number): number {
  const target = bitsToTarget(bits);
  // Find first non-zero byte and read up to 6 bytes as a JS-safe integer.
  let i = 0;
  while (i < 32 && target[i] === 0) i++;
  if (i === 32) return Infinity;
  const len = Math.min(6, 32 - i);
  let mantissa = 0;
  for (let j = 0; j < len; j++) mantissa = mantissa * 256 + target[i + j];
  // Difficulty ~ 2^(leading_zero_bits) / mantissa_normalized.
  // Simpler: 2^(8*i + (48-len*8)) / mantissa, but we just want a sortable
  // monotone metric so use 2^(8 * leadingZeroBytes) scaled.
  const scale = 2 ** (8 * (32 - i - len));
  return 2 ** (8 * 32) / (mantissa * scale + 1);
}

// ----------------------------------------------------------------------------
// Header serialization (the 80 bytes that get hashed)
// ----------------------------------------------------------------------------

export function serializeHeader(h: BlockHeader): Buffer {
  const buf = Buffer.alloc(HEADER_SIZE);
  let off = 0;
  off = buf.writeUInt32LE(h.version, off);
  off += h.prevHash.copy(buf, off, 0, 32);
  off += h.eventsRoot.copy(buf, off, 0, 32);
  off = buf.writeUInt32LE(h.hostTimestamp, off);
  off = buf.writeUInt32LE(h.bits, off);
  buf.writeUInt32LE(h.nonce, off);
  return buf;
}

export function deserializeHeader(buf: Buffer, offset = 0): BlockHeader {
  let off = offset;
  const version = buf.readUInt32LE(off);
  off += 4;
  const prevHash = Buffer.from(buf.subarray(off, off + 32));
  off += 32;
  const eventsRoot = Buffer.from(buf.subarray(off, off + 32));
  off += 32;
  const hostTimestamp = buf.readUInt32LE(off);
  off += 4;
  const bits = buf.readUInt32LE(off);
  off += 4;
  const nonce = buf.readUInt32LE(off);
  return { version, prevHash, eventsRoot, hostTimestamp, bits, nonce };
}

/** sha256d of the serialized header. The block's identity. */
export function headerHash(h: BlockHeader): Buffer {
  return sha256d(serializeHeader(h));
}

// ----------------------------------------------------------------------------
// Event serialization
// ----------------------------------------------------------------------------

/**
 * Wire format for one event:
 *   [1]  uint8    kind
 *   [8]  float64  at (epoch ms; double because JS)
 *   [4]  uint32   dataLen
 *   [N]  bytes    data (UTF-8)
 */
export function serializeEvent(e: ChronoEvent): Buffer {
  const data = Buffer.from(e.data, "utf8");
  const buf = Buffer.alloc(1 + 8 + 4 + data.length);
  let off = 0;
  off = buf.writeUInt8(e.kind, off);
  off = buf.writeDoubleLE(e.at, off);
  off = buf.writeUInt32LE(data.length, off);
  data.copy(buf, off);
  return buf;
}

export function deserializeEvent(
  buf: Buffer,
  offset: number,
): { event: ChronoEvent; bytesRead: number } {
  let off = offset;
  const kind = buf.readUInt8(off);
  off += 1;
  const at = buf.readDoubleLE(off);
  off += 8;
  const dataLen = buf.readUInt32LE(off);
  off += 4;
  const data = buf.subarray(off, off + dataLen).toString("utf8");
  off += dataLen;
  return { event: { kind, at, data }, bytesRead: off - offset };
}

/** Serialize the full event list. Length-prefixed array. */
export function serializeEvents(events: ChronoEvent[]): Buffer {
  const parts = events.map(serializeEvent);
  const total = parts.reduce((n, p) => n + p.length, 0);
  const buf = Buffer.alloc(4 + total);
  buf.writeUInt32LE(events.length, 0);
  let off = 4;
  for (const p of parts) {
    p.copy(buf, off);
    off += p.length;
  }
  return buf;
}

export function deserializeEvents(
  buf: Buffer,
  offset: number,
): { events: ChronoEvent[]; bytesRead: number } {
  let off = offset;
  const count = buf.readUInt32LE(off);
  off += 4;
  const events: ChronoEvent[] = [];
  for (let i = 0; i < count; i++) {
    const { event, bytesRead } = deserializeEvent(buf, off);
    events.push(event);
    off += bytesRead;
  }
  return { events, bytesRead: off - offset };
}

/**
 * The eventsRoot committed in the header. We hash the serialized event
 * payload directly rather than building a Merkle tree — with one block
 * per minute and a handful of events each, tree partial-proofs offer no
 * value (no light clients) and the simpler hash-the-blob is harder to
 * get wrong.
 *
 * The chain stores ONLY this 32-byte commitment per block; the events
 * themselves live in the separate event log. This is the rollup hash
 * that anchors the log to the chain.
 */
export function computeEventsRoot(events: ChronoEvent[]): Buffer {
  return sha256d(serializeEvents(events));
}

/**
 * Root for an empty event list. Precomputed because the event log is
 * sparse — blocks with no events write nothing, and on validation a
 * missing batch is checked against this constant.
 */
export const EMPTY_EVENTS_ROOT = computeEventsRoot([]);

// ----------------------------------------------------------------------------
// Event log batch serialization (separate file, sparse)
// ----------------------------------------------------------------------------

/**
 * One batch in the event log:
 *   [4]   uint32  height    (which chain block this batch is for)
 *   [4]   uint32  bodyLen   (so the reader can skip without parsing events)
 *   [N]   body    (serializeEvents output)
 *
 * The chain file is fixed-stride headers only; event batches go here.
 * Sparse: only written when events.length > 0.
 */
export function serializeEventBatch(
  height: number,
  events: ChronoEvent[],
): Buffer {
  const body = serializeEvents(events);
  const out = Buffer.alloc(4 + 4 + body.length);
  out.writeUInt32LE(height, 0);
  out.writeUInt32LE(body.length, 4);
  body.copy(out, 8);
  return out;
}

export function deserializeEventBatch(
  buf: Buffer,
  offset: number,
): { height: number; events: ChronoEvent[]; bytesRead: number } {
  let off = offset;
  const height = buf.readUInt32LE(off);
  off += 4;
  const bodyLen = buf.readUInt32LE(off);
  off += 4;
  const { events } = deserializeEvents(buf, off);
  // Advance by declared bodyLen (resilient to event codec evolution).
  return { height, events, bytesRead: 8 + bodyLen };
}

// ----------------------------------------------------------------------------
// Genesis
// ----------------------------------------------------------------------------

/** All-zero hash. prevHash of the genesis block. */
export const ZERO_HASH = Buffer.alloc(32);

/**
 * The synthetic event recorded at genesis — the agent's first moment of
 * self-awareness. Returned alongside the header so the caller can write
 * it to the event log (the chain itself only carries the commitment).
 */
export function buildGenesisEvent(hostTimestamp: number): ChronoEvent {
  return {
    kind: 4, // EventKind.SERVICE
    at: hostTimestamp * 1000,
    data: "genesis",
  };
}

/**
 * Build an unmined genesis header. Caller still has to grind the nonce
 * (genesis is not exempt from PoW — that would be a free tampering vector
 * for anyone who deletes the chain file and restarts).
 *
 * The genesis event is committed via eventsRoot but stored in the event
 * log, not the chain. Caller writes it there separately.
 */
export function buildGenesisHeader(
  bits: number,
  hostTimestamp: number,
): BlockHeader {
  return {
    version: BLOCK_VERSION,
    prevHash: ZERO_HASH,
    eventsRoot: computeEventsRoot([buildGenesisEvent(hostTimestamp)]),
    hostTimestamp,
    bits,
    nonce: 0,
  };
}
