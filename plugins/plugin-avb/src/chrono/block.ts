/**
 * Block serialization, hashing, and difficulty math.
 *
 * On-disk block layout (little-endian):
 *
 *   offset  size  field
 *   ------  ----  ------------------------------------------------
 *        0     4  magic           ASCII "AVBC" (0x41 56 42 43)
 *        4     4  version         u32
 *        8     4  height          u32
 *       12    32  prevHash        raw bytes
 *       44    32  eventsHash      SHA256(payload)
 *       76     8  hostTimestampMs i64 (BigInt64LE)
 *       84     4  wallElapsedMs   u32
 *       88     4  difficultyBits  u32 (compact nBits)
 *       92     8  nonce           u64 (BigUint64LE)
 *      100     4  eventCount      u32
 *      104     4  eventBytes      u32
 *   ------  ----
 *      108         HEADER_SIZE
 *
 *   Followed immediately by `eventBytes` bytes of encoded event payload.
 *
 * Block hash = SHA256(SHA256(header[0..108])).
 * PoW: block hash interpreted as a big-endian 256-bit unsigned integer
 * must be <= the target derived from difficultyBits.
 *
 * Event payload encoding (per event, concatenated):
 *   u8  type
 *   i64 hostTimestampMs (BigInt64LE)
 *   u16 detailLen
 *   ..  detailLen bytes UTF-8
 */

import { createHash } from "node:crypto";
import type { Block, BlockHeader, ChronoEvent } from "./types.ts";

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------

export const MAGIC = Uint8Array.of(0x41, 0x56, 0x42, 0x43); // "AVBC"
export const HEADER_SIZE = 108;
export const HASH_SIZE = 32;
/**
 * v1: events inlined after the header in the chain file;
 *     eventsHash = SHA256(payload).
 * v2: events stored off-chain (events.bin); eventsHash = Merkle root
 *     over per-event leaves; zero payload bytes follow the header.
 * Header layout is identical across versions.
 */
export const PROTOCOL_VERSION = 2;

/** All-zero hash used as genesis prevHash. */
export const ZERO_HASH = new Uint8Array(HASH_SIZE);

/**
 * Easiest possible compact target. Exponent 0x20 (=32 bytes), mantissa
 * 0x00ffff → target ≈ 0x00ffff << (8*(32−3)) which is just under 2^256.
 * Practically every hash satisfies it; the retarget loop tightens from here.
 */
export const INITIAL_DIFFICULTY_BITS = 0x2000ffff;

/** Absolute ceiling for retargets — never loosen past this. */
export const MAX_TARGET = bitsToTarget(INITIAL_DIFFICULTY_BITS);

// ----------------------------------------------------------------------------
// Hashing
// ----------------------------------------------------------------------------

export function sha256(data: Uint8Array): Uint8Array {
  return new Uint8Array(createHash("sha256").update(data).digest());
}

export function sha256d(data: Uint8Array): Uint8Array {
  return sha256(sha256(data));
}

/** Interpret a 32-byte hash as a big-endian unsigned bigint. */
export function hashToBigInt(hash: Uint8Array): bigint {
  let n = 0n;
  for (let i = 0; i < hash.length; i++) {
    n = (n << 8n) | BigInt(hash[i]);
  }
  return n;
}

export function toHex(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) {
    s += bytes[i].toString(16).padStart(2, "0");
  }
  return s;
}

// ----------------------------------------------------------------------------
// Compact difficulty (Bitcoin nBits)
//
// nBits = (exponent << 24) | mantissa, where
//   target = mantissa * 2^(8*(exponent − 3))
// Mantissa is at most 0x7fffff (sign bit reserved). We never set the sign.
// ----------------------------------------------------------------------------

export function bitsToTarget(bits: number): bigint {
  const u = bits >>> 0;
  const exponent = u >>> 24;
  const mantissa = BigInt(u & 0x007fffff);
  if (exponent <= 3) {
    return mantissa >> (8n * BigInt(3 - exponent));
  }
  return mantissa << (8n * BigInt(exponent - 3));
}

export function targetToBits(target: bigint): number {
  if (target <= 0n) return 0x03000001; // degenerate floor: target=1
  // Count bytes
  let size = 0;
  let t = target;
  while (t > 0n) {
    size++;
    t >>= 8n;
  }
  let mantissa: bigint;
  if (size <= 3) {
    mantissa = target << (8n * BigInt(3 - size));
  } else {
    mantissa = target >> (8n * BigInt(size - 3));
  }
  // If the high bit of mantissa is set, shift right and bump exponent so
  // the sign bit stays clear.
  if (mantissa & 0x00800000n) {
    mantissa >>= 8n;
    size++;
  }
  return ((size << 24) | Number(mantissa & 0x007fffffn)) >>> 0;
}

export function meetsTarget(hash: Uint8Array, bits: number): boolean {
  return hashToBigInt(hash) <= bitsToTarget(bits);
}

/**
 * Retarget toward `targetMs` block time given the last block actually
 * took `actualMs`. Adjustment is clamped to ¼×–4× per step (Bitcoin's
 * bound) so a single outlier can't whipsaw difficulty.
 */
export function retarget(
  prevBits: number,
  actualMs: number,
  targetMs: number,
): number {
  const prevTarget = bitsToTarget(prevBits);
  const ratioNum = BigInt(Math.max(1, Math.round(actualMs)));
  const ratioDen = BigInt(Math.max(1, Math.round(targetMs)));
  // Clamp ratio to [1/4, 4]
  let num = ratioNum;
  let den = ratioDen;
  if (num * 1n > den * 4n) {
    num = 4n;
    den = 1n;
  } else if (num * 4n < den * 1n) {
    num = 1n;
    den = 4n;
  }
  let next = (prevTarget * num) / den;
  if (next < 1n) next = 1n;
  if (next > MAX_TARGET) next = MAX_TARGET;
  return targetToBits(next);
}

// ----------------------------------------------------------------------------
// Header serialization
// ----------------------------------------------------------------------------

export function serializeHeader(h: BlockHeader): Uint8Array {
  const buf = new Uint8Array(HEADER_SIZE);
  const dv = new DataView(buf.buffer);
  buf.set(MAGIC, 0);
  dv.setUint32(4, h.version, true);
  dv.setUint32(8, h.height, true);
  buf.set(h.prevHash, 12);
  buf.set(h.eventsHash, 44);
  dv.setBigInt64(76, BigInt(h.hostTimestampMs), true);
  dv.setUint32(84, h.wallElapsedMs >>> 0, true);
  dv.setUint32(88, h.difficultyBits >>> 0, true);
  dv.setBigUint64(92, h.nonce, true);
  dv.setUint32(100, h.eventCount >>> 0, true);
  dv.setUint32(104, h.eventBytes >>> 0, true);
  return buf;
}

export function deserializeHeader(buf: Uint8Array): BlockHeader {
  if (buf.length < HEADER_SIZE) {
    throw new Error(`header too short: ${buf.length} < ${HEADER_SIZE}`);
  }
  const dv = new DataView(buf.buffer, buf.byteOffset, HEADER_SIZE);
  return {
    version: dv.getUint32(4, true),
    height: dv.getUint32(8, true),
    prevHash: buf.slice(12, 44),
    eventsHash: buf.slice(44, 76),
    hostTimestampMs: Number(dv.getBigInt64(76, true)),
    wallElapsedMs: dv.getUint32(84, true),
    difficultyBits: dv.getUint32(88, true),
    nonce: dv.getBigUint64(92, true),
    eventCount: dv.getUint32(100, true),
    eventBytes: dv.getUint32(104, true),
  };
}

export function hashHeader(h: BlockHeader): Uint8Array {
  return sha256d(serializeHeader(h));
}

// ----------------------------------------------------------------------------
// Event payload encoding
// ----------------------------------------------------------------------------

const enc = /* @__PURE__ */ new TextEncoder();
const dec = /* @__PURE__ */ new TextDecoder();

/** Max detail length per event (u16 length field). */
const MAX_DETAIL = 0xffff;

/** Encode a single event: u8 type | i64 hostTs | u16 len | detail. */
export function encodeEvent(e: ChronoEvent): Uint8Array {
  let d = enc.encode(e.detail);
  if (d.length > MAX_DETAIL) d = d.subarray(0, MAX_DETAIL);
  const out = new Uint8Array(1 + 8 + 2 + d.length);
  const dv = new DataView(out.buffer);
  out[0] = e.type & 0xff;
  dv.setBigInt64(1, BigInt(e.hostTimestampMs), true);
  dv.setUint16(9, d.length, true);
  out.set(d, 11);
  return out;
}

export function encodeEvents(events: readonly ChronoEvent[]): Uint8Array {
  const parts = events.map(encodeEvent);
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

/**
 * Split a concatenated event payload back into the raw per-event
 * encodings (without decoding). Used to recompute Merkle roots and
 * build inclusion proofs from an off-chain segment.
 */
export function splitEvents(payload: Uint8Array): Uint8Array[] {
  const dv = new DataView(
    payload.buffer,
    payload.byteOffset,
    payload.byteLength,
  );
  const out: Uint8Array[] = [];
  let off = 0;
  while (off + 11 <= payload.length) {
    const len = dv.getUint16(off + 9, true);
    const end = off + 11 + len;
    if (end > payload.length) break;
    out.push(payload.subarray(off, end));
    off = end;
  }
  return out;
}

export function decodeEvents(payload: Uint8Array): ChronoEvent[] {
  const dv = new DataView(
    payload.buffer,
    payload.byteOffset,
    payload.byteLength,
  );
  const out: ChronoEvent[] = [];
  let off = 0;
  while (off + 11 <= payload.length) {
    const type = payload[off];
    off += 1;
    const ts = Number(dv.getBigInt64(off, true));
    off += 8;
    const len = dv.getUint16(off, true);
    off += 2;
    if (off + len > payload.length) break; // truncated tail — stop
    const detail = dec.decode(payload.subarray(off, off + len));
    off += len;
    out.push({ type, hostTimestampMs: ts, detail });
  }
  return out;
}

// ----------------------------------------------------------------------------
// Whole-block helpers
// ----------------------------------------------------------------------------

export function buildBlock(
  headerBytes: Uint8Array,
  payload: Uint8Array,
): Block {
  const header = deserializeHeader(headerBytes);
  return { header, payload, hash: sha256d(headerBytes) };
}

export function serializeBlock(block: Block): Uint8Array {
  const headerBytes = serializeHeader(block.header);
  const out = new Uint8Array(HEADER_SIZE + block.payload.length);
  out.set(headerBytes, 0);
  out.set(block.payload, HEADER_SIZE);
  return out;
}

export function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
