/**
 * SLM16 — minimal SentencePiece BPE tokenizer.
 *
 * The Parameter Golf training data uses `fineweb_1024_bpe.model` — a
 * SentencePiece protobuf model with vocab=1024. We don't pull in the full
 * sentencepiece native binding; instead this file hand-decodes the protobuf
 * wire format just enough to extract the vocab table (`pieces` array), then
 * implements:
 *   - decode: ID[] → string (table lookup + byte-fallback unescaping)
 *   - encode: string → ID[] (greedy longest-match — slow O(n·V) but correct,
 *             and we only encode 8 short scoring prompts at boot)
 *
 * Protobuf wire format (just what we need):
 *   ModelProto         = message { repeated SentencePiece pieces = 1; ... }
 *   SentencePiece      = message { string piece = 1; float score = 2; int32 type = 3; }
 *   Wire type 2 = length-delimited (varint length prefix + bytes)
 *   Field tag = (field_number << 3) | wire_type
 *
 * Reference: github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
 *
 * If the .model file is missing, this module falls back to a byte-identity
 * vocab (ID i ↔ byte i for i<256, IDs ≥256 decode to U+FFFD). That keeps
 * inference functional even before the user has downloaded the tokenizer.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { PATHS } from "./config.ts";

// SentencePiece type enum (proto field SentencePiece.type)
const SP_TYPE_NORMAL = 1;
const SP_TYPE_UNKNOWN = 2;
const SP_TYPE_CONTROL = 3;
const SP_TYPE_BYTE = 6;

// SP uses U+2581 (LOWER ONE EIGHTH BLOCK) as the word-boundary marker.
const SP_SPACE = "\u2581";

// ----------------------------------------------------------------------------
// Protobuf wire decoder (just varints + length-delimited)
// ----------------------------------------------------------------------------

class WireReader {
  private pos = 0;
  constructor(private readonly buf: Uint8Array) {}

  eof(): boolean {
    return this.pos >= this.buf.length;
  }

  /** Read a base-128 varint. Returns the decoded unsigned integer. */
  varint(): number {
    let result = 0;
    let shift = 0;
    while (this.pos < this.buf.length) {
      const b = this.buf[this.pos++];
      result |= (b & 0x7f) << shift;
      if ((b & 0x80) === 0) return result >>> 0;
      shift += 7;
      if (shift > 35) throw new Error("varint too long");
    }
    throw new Error("varint truncated");
  }

  /** Read a length-delimited field's payload (returns a subarray view). */
  bytes(): Uint8Array {
    const len = this.varint();
    if (this.pos + len > this.buf.length) {
      throw new Error("length-delimited field truncated");
    }
    const out = this.buf.subarray(this.pos, this.pos + len);
    this.pos += len;
    return out;
  }

  /** Skip a field given its wire type. */
  skip(wireType: number): void {
    switch (wireType) {
      case 0: // varint
        this.varint();
        break;
      case 1: // fixed64
        this.pos += 8;
        break;
      case 2: // length-delimited
        this.bytes();
        break;
      case 5: // fixed32
        this.pos += 4;
        break;
      default:
        throw new Error(`unsupported wire type ${wireType}`);
    }
  }
}

// ----------------------------------------------------------------------------
// SentencePiece model parser
// ----------------------------------------------------------------------------

interface SpPiece {
  piece: string;
  type: number;
}

/**
 * Parse a SentencePiece submessage. Field 1 = piece (string),
 * field 3 = type (varint enum). We ignore field 2 (score) — only
 * needed for sampling, and we use greedy longest-match instead.
 */
function parsePiece(buf: Uint8Array): SpPiece {
  const r = new WireReader(buf);
  let piece = "";
  let type = SP_TYPE_NORMAL;
  const dec = new TextDecoder();
  while (!r.eof()) {
    const tag = r.varint();
    const fieldNum = tag >>> 3;
    const wireType = tag & 7;
    if (fieldNum === 1 && wireType === 2) {
      piece = dec.decode(r.bytes());
    } else if (fieldNum === 3 && wireType === 0) {
      type = r.varint();
    } else {
      r.skip(wireType);
    }
  }
  return { piece, type };
}

/**
 * Parse the top-level ModelProto. Field 1 = repeated SentencePiece (pieces).
 * Other fields (trainer_spec, normalizer_spec, etc.) are skipped.
 */
function parseModel(buf: Uint8Array): SpPiece[] {
  const r = new WireReader(buf);
  const pieces: SpPiece[] = [];
  while (!r.eof()) {
    const tag = r.varint();
    const fieldNum = tag >>> 3;
    const wireType = tag & 7;
    if (fieldNum === 1 && wireType === 2) {
      pieces.push(parsePiece(r.bytes()));
    } else {
      r.skip(wireType);
    }
  }
  return pieces;
}

// ----------------------------------------------------------------------------
// Tokenizer
// ----------------------------------------------------------------------------

export interface Tokenizer {
  vocabSize: number;
  /** True if loaded from a real .model file; false = byte-identity fallback. */
  loaded: boolean;
  encode(text: string): number[];
  decode(ids: number[]): string;
}

/** Decode a SentencePiece byte-fallback piece like "<0x41>" → 0x41. */
function decodeBytePiece(piece: string): number | null {
  const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece);
  return m ? Number.parseInt(m[1], 16) : null;
}

function buildTokenizer(pieces: SpPiece[]): Tokenizer {
  const vocabSize = pieces.length;

  // ID → surface form. For byte pieces we keep the raw byte value separately
  // so decode can reconstruct UTF-8 from a stream of byte-fallback tokens.
  const idToPiece: string[] = new Array(vocabSize);
  const idToByte: (number | null)[] = new Array(vocabSize).fill(null);
  // piece → ID for encoding lookup. Only NORMAL pieces participate in
  // greedy longest-match; UNK/CONTROL/BYTE are excluded.
  const pieceToId = new Map<string, number>();
  // Byte value → ID for the byte-fallback path during encoding.
  const byteToId = new Map<number, number>();

  let unkId = 0;

  for (let id = 0; id < vocabSize; id++) {
    const p = pieces[id];
    idToPiece[id] = p.piece;
    if (p.type === SP_TYPE_BYTE) {
      const bv = decodeBytePiece(p.piece);
      if (bv !== null) {
        idToByte[id] = bv;
        byteToId.set(bv, id);
      }
    } else if (p.type === SP_TYPE_NORMAL) {
      // Strip the SP space marker for encoding-side matching: we'll
      // pre-process input text to insert SP_SPACE at word boundaries
      // before lookup, so the table key keeps the marker intact.
      pieceToId.set(p.piece, id);
    } else if (p.type === SP_TYPE_UNKNOWN) {
      unkId = id;
    }
    // CONTROL pieces (BOS/EOS/PAD) are decode-only; we don't emit them.
  }

  // For greedy longest-match we need to know the max piece length so we
  // can bound the inner loop. Compute once.
  let maxPieceLen = 1;
  for (const k of pieceToId.keys()) {
    if (k.length > maxPieceLen) maxPieceLen = k.length;
  }

  /**
   * Encode: SP normalization → greedy longest-match BPE → byte fallback.
   *
   * SP's normalizer is more involved (NFKC, custom rules), but for the
   * short ASCII scoring prompts in config.ts the only transformation that
   * matters is: replace ' ' with U+2581 and prepend one at the start.
   * Greedy longest-match isn't optimal BPE (real SP uses Viterbi over
   * piece scores) but it's deterministic and good enough for ~50-char
   * prompts feeding a 17M-param model.
   */
  function encode(text: string): number[] {
    // Normalize: prepend space marker, replace internal spaces.
    const normalized = SP_SPACE + text.replace(/ /g, SP_SPACE);
    const ids: number[] = [];
    let i = 0;
    while (i < normalized.length) {
      // Try longest match first.
      let matched = false;
      const cap = Math.min(maxPieceLen, normalized.length - i);
      for (let len = cap; len >= 1; len--) {
        const candidate = normalized.slice(i, i + len);
        const id = pieceToId.get(candidate);
        if (id !== undefined) {
          ids.push(id);
          i += len;
          matched = true;
          break;
        }
      }
      if (!matched) {
        // Byte fallback: encode this codepoint's UTF-8 bytes individually.
        const cp = normalized.codePointAt(i);
        if (cp === undefined) {
          ids.push(unkId);
          i += 1;
          continue;
        }
        // UTF-8 encode the codepoint.
        const utf8 = new TextEncoder().encode(String.fromCodePoint(cp));
        for (const b of utf8) {
          const bid = byteToId.get(b);
          ids.push(bid !== undefined ? bid : unkId);
        }
        i += cp > 0xffff ? 2 : 1; // surrogate pair advance
      }
    }
    return ids;
  }

  /**
   * Decode: ID → piece string, with byte-piece reassembly.
   *
   * Byte pieces are accumulated into a buffer and flushed through a UTF-8
   * decoder when a non-byte piece is encountered (or at the end). This
   * correctly reconstructs multi-byte codepoints that were split across
   * byte-fallback tokens.
   */
  function decode(ids: number[]): string {
    const parts: string[] = [];
    const byteAcc: number[] = [];
    const flushBytes = () => {
      if (byteAcc.length === 0) return;
      parts.push(new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(byteAcc)));
      byteAcc.length = 0;
    };
    for (const id of ids) {
      if (id < 0 || id >= vocabSize) continue;
      const bv = idToByte[id];
      if (bv !== null) {
        byteAcc.push(bv);
        continue;
      }
      flushBytes();
      const piece = idToPiece[id];
      // Skip control tokens (anything wrapped in <...> that wasn't a byte piece).
      if (piece.startsWith("<") && piece.endsWith(">")) continue;
      parts.push(piece);
    }
    flushBytes();
    // Un-normalize: U+2581 → ' '
    return parts.join("").replace(new RegExp(SP_SPACE, "g"), " ").trimStart();
  }

  return { vocabSize, loaded: true, encode, decode };
}

/**
 * Byte-identity fallback when no .model file is available.
 * IDs 0-255 map to raw bytes; everything else decodes to U+FFFD.
 * Encoding is just UTF-8 bytes → IDs.
 */
function byteFallbackTokenizer(): Tokenizer {
  return {
    vocabSize: 1024,
    loaded: false,
    encode(text: string): number[] {
      return Array.from(new TextEncoder().encode(text));
    },
    decode(ids: number[]): string {
      const bytes: number[] = [];
      for (const id of ids) {
        if (id >= 0 && id < 256) bytes.push(id);
        // IDs ≥256 are unknown merges — drop them (better than emitting garbage).
      }
      return new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(bytes));
    },
  };
}

// ----------------------------------------------------------------------------
// Public loader
// ----------------------------------------------------------------------------

/**
 * Load the SentencePiece tokenizer from `${checkpointDir}/fineweb_1024_bpe.model`.
 * Falls back to byte-identity if the file is missing or unparseable.
 *
 * Synchronous because it's called once at service init and the file is ~50KB.
 */
export function loadTokenizer(checkpointDir: string): Tokenizer {
  const path = join(checkpointDir, PATHS.tokenizer);
  if (!existsSync(path)) {
    return byteFallbackTokenizer();
  }
  try {
    const raw = readFileSync(path);
    // Buffer → Uint8Array view (same backing).
    const u8 = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
    const pieces = parseModel(u8);
    if (pieces.length === 0) return byteFallbackTokenizer();
    return buildTokenizer(pieces);
  } catch {
    return byteFallbackTokenizer();
  }
}
