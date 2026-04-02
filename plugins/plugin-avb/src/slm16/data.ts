/**
 * SLM16 — FineWeb .bin shard loader.
 *
 * On-disk format (per OpenAI Parameter Golf data pipeline):
 *   [256 × int32_le header][uint16_le tokens × header[2]]
 *     header[0] = 20240520  (magic)
 *     header[1] = 1         (version)
 *     header[2] = num_tokens
 *     header[3..255] = reserved (zero)
 *
 * Reference: github.com/openai/parameter-golf/blob/main/data/cached_challenge_fineweb.py
 *            train_gpt.py L429–L443 (load_data_shard)
 *
 * Each shard is ~100M tokens × 2 bytes ≈ 200MB. We hold one train shard
 * resident at a time and cycle through the directory. The val slice is
 * carved deterministically from the front of the lexically-first shard
 * (or from val_*.bin if present), so the same model always sees the same
 * holdout regardless of which session it's running in.
 */

import { open, readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { ARCH, SHARD } from "./config.ts";

const HEADER_BYTES = SHARD.headerInts * 4; // 1024

// ----------------------------------------------------------------------------
// Low-level shard I/O
// ----------------------------------------------------------------------------

interface ShardHeader {
  numTokens: number;
}

async function readHeader(path: string): Promise<ShardHeader> {
  const fh = await open(path, "r");
  try {
    const buf = Buffer.alloc(HEADER_BYTES);
    const { bytesRead } = await fh.read(buf, 0, HEADER_BYTES, 0);
    if (bytesRead !== HEADER_BYTES) {
      throw new Error(`shard ${path}: short header read (${bytesRead}/${HEADER_BYTES} bytes)`);
    }
    // 256 × int32_le. We only need the first three.
    const magic = buf.readInt32LE(0);
    const version = buf.readInt32LE(4);
    const numTokens = buf.readInt32LE(8);
    if (magic !== SHARD.magic) {
      throw new Error(`shard ${path}: bad magic ${magic} (expected ${SHARD.magic})`);
    }
    if (version !== SHARD.version) {
      throw new Error(`shard ${path}: unsupported version ${version}`);
    }
    if (numTokens <= 0) {
      throw new Error(`shard ${path}: header reports ${numTokens} tokens`);
    }
    return { numTokens };
  } finally {
    await fh.close();
  }
}

/**
 * Read a contiguous range of tokens from a shard. Returns a fresh Uint16Array
 * (no shared backing — safe to keep across shard transitions).
 */
async function readTokens(
  path: string,
  startToken: number,
  count: number,
): Promise<Uint16Array> {
  const fh = await open(path, "r");
  try {
    const byteOffset = HEADER_BYTES + startToken * SHARD.tokenBytes;
    const byteLength = count * SHARD.tokenBytes;
    const buf = Buffer.alloc(byteLength);
    const { bytesRead } = await fh.read(buf, 0, byteLength, byteOffset);
    if (bytesRead !== byteLength) {
      throw new Error(
        `shard ${path}: short token read at offset ${startToken} ` +
          `(got ${bytesRead}/${byteLength} bytes)`,
      );
    }
    // Buffer.alloc returns a fresh Buffer with byteOffset=0 within its own
    // ArrayBuffer, so re-viewing as Uint16Array is alignment-safe.
    return new Uint16Array(buf.buffer, buf.byteOffset, count);
  } finally {
    await fh.close();
  }
}

// ----------------------------------------------------------------------------
// Shard discovery
// ----------------------------------------------------------------------------

interface ShardEntry {
  path: string;
  numTokens: number;
}

/**
 * Discover all .bin shards under `dir`. Returns sorted by filename so the
 * train/val split is deterministic across runs. Skips zero-length files
 * and anything failing the magic check.
 */
export async function discoverShards(dir: string): Promise<{
  train: ShardEntry[];
  val: ShardEntry | null;
}> {
  let names: string[];
  try {
    names = await readdir(dir);
  } catch {
    // Directory missing — caller will idle with "no_data".
    return { train: [], val: null };
  }

  const binFiles = names.filter((n) => n.endsWith(".bin")).sort();

  const train: ShardEntry[] = [];
  let val: ShardEntry | null = null;

  for (const name of binFiles) {
    const path = join(dir, name);
    let header: ShardHeader;
    try {
      const st = await stat(path);
      if (st.size < HEADER_BYTES + SHARD.tokenBytes) continue;
      header = await readHeader(path);
    } catch {
      continue; // Bad shard — skip silently, log responsibility is on caller.
    }
    const entry: ShardEntry = { path, numTokens: header.numTokens };
    // Convention: val_*.bin or *_val.bin → validation. First match wins.
    if (val === null && /(?:^|_)val(?:_|\.bin$)/i.test(name)) {
      val = entry;
    } else {
      train.push(entry);
    }
  }

  return { train, val };
}

// ----------------------------------------------------------------------------
// Validation slice
// ----------------------------------------------------------------------------

/**
 * Build a fixed validation set. If no val_*.bin shard exists, carve the
 * first `valTokens` tokens off the front of the first train shard (and
 * record that boundary so the train iterator skips it).
 *
 * The val slice is reshaped into [N, seqLen+1] non-overlapping windows;
 * the trailing remainder is discarded. Each row yields one (input, target)
 * pair via [:-1] / [1:].
 */
export interface ValSlice {
  /** Token windows, packed [numWindows × (seqLen+1)] in an Int32Array. */
  windows: Int32Array;
  /** Number of validation windows. */
  numWindows: number;
  /** seqLen + 1 (window stride). */
  windowLen: number;
  /** If carved from train[0], how many tokens at the start to skip there. */
  skipFromFirstTrainShard: number;
}

/** Default val slice size: ~256K tokens → ~256 windows at seqLen=1024. */
const DEFAULT_VAL_TOKENS = 262_144;

export async function buildValSlice(
  shards: { train: ShardEntry[]; val: ShardEntry | null },
  valTokens = DEFAULT_VAL_TOKENS,
): Promise<ValSlice | null> {
  const windowLen = ARCH.seqLen + 1;
  let raw: Uint16Array;
  let skip = 0;

  if (shards.val) {
    const n = Math.min(valTokens, shards.val.numTokens);
    raw = await readTokens(shards.val.path, 0, n);
  } else if (shards.train.length > 0) {
    const first = shards.train[0];
    // Don't consume more than 1/8 of the first shard for val.
    const cap = Math.floor(first.numTokens / 8);
    const n = Math.min(valTokens, cap);
    if (n < windowLen) return null; // Shard too small to carve anything useful.
    raw = await readTokens(first.path, 0, n);
    skip = n;
  } else {
    return null;
  }

  const numWindows = Math.floor(raw.length / windowLen);
  if (numWindows === 0) return null;

  // Pack into Int32Array — tfjs expects int32 token IDs, and the upcast
  // is cheap relative to the forward pass.
  const windows = new Int32Array(numWindows * windowLen);
  for (let i = 0; i < numWindows * windowLen; i++) {
    windows[i] = raw[i];
  }

  return { windows, numWindows, windowLen, skipFromFirstTrainShard: skip };
}

// ----------------------------------------------------------------------------
// Training batch iterator
// ----------------------------------------------------------------------------

export interface BatchSpec {
  /** Sequences per microbatch (B). */
  microBatchSize: number;
  /** Sequence length (T). Always ARCH.seqLen. */
  seqLen: number;
}

export interface Batch {
  /** [B, T] int32 — input token IDs. */
  inputs: Int32Array;
  /** [B, T] int32 — target token IDs (inputs shifted by 1). */
  targets: Int32Array;
  /** Number of training tokens this batch contributes (B × T). */
  numTokens: number;
}

/**
 * Mulberry32 — same fast 32-bit PRNG used in model.ts for init.
 * We seed it from `cfg.seed ^ step` so resumption picks up roughly where
 * the previous session left off in the data stream (good enough for
 * background training; this isn't an exactly-reproducible scientific run).
 */
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Streaming batch iterator. Cycles through train shards in order, sampling
 * non-overlapping windows of seqLen+1 tokens at random offsets within the
 * currently-resident shard. When the current shard's coverage exceeds
 * ~80% (by sampled position count), advance to the next shard.
 *
 * This is NOT the DDP-aware contiguous packer the reference uses on 8×H100;
 * for a single-process background trainer on commodity hardware, random
 * windowing within a shard is simpler and avoids the document-boundary
 * bookkeeping. The gradient signal is equivalent.
 */
export class TrainIterator {
  private readonly shards: ShardEntry[];
  private readonly skipFirst: number;
  private readonly spec: BatchSpec;
  private readonly windowLen: number;
  private readonly rng: () => number;

  private shardIdx = 0;
  /** Resident token buffer for the current shard. */
  private resident: Uint16Array | null = null;
  /** Token offset within resident shard where sampling starts. */
  private residentStart = 0;
  /** Available token count in resident shard (after start offset). */
  private residentLen = 0;
  /** Windows sampled from the current resident shard. */
  private samplesThisShard = 0;
  /** Threshold to advance shard (≈ 80% coverage, computed on load). */
  private advanceThreshold = 0;

  constructor(
    trainShards: ShardEntry[],
    skipFromFirstShard: number,
    spec: BatchSpec,
    seed: number,
  ) {
    if (trainShards.length === 0) {
      throw new Error("TrainIterator: no train shards provided");
    }
    this.shards = trainShards;
    this.skipFirst = skipFromFirstShard;
    this.spec = spec;
    this.windowLen = spec.seqLen + 1;
    this.rng = mulberry32(seed);
  }

  /** Total tokens across all shards (for max-tokens budget tracking). */
  totalTokens(): number {
    let n = 0;
    for (let i = 0; i < this.shards.length; i++) {
      n += this.shards[i].numTokens - (i === 0 ? this.skipFirst : 0);
    }
    return n;
  }

  /** Load the current shard into memory. */
  private async loadShard(): Promise<void> {
    const idx = this.shardIdx % this.shards.length;
    const shard = this.shards[idx];
    const start = idx === 0 ? this.skipFirst : 0;
    const len = shard.numTokens - start;

    // Free previous shard before loading the next (don't hold two at once).
    this.resident = null;

    this.resident = await readTokens(shard.path, start, len);
    this.residentStart = start;
    this.residentLen = len;
    this.samplesThisShard = 0;
    // ~80% of windows that fit, then move on.
    const fits = Math.max(1, Math.floor(len / this.windowLen));
    this.advanceThreshold = Math.max(1, Math.floor(fits * 0.8));
  }

  /** Pull one microbatch. Async because shard transitions hit disk. */
  async next(): Promise<Batch> {
    if (
      this.resident === null ||
      this.samplesThisShard >= this.advanceThreshold
    ) {
      if (this.resident !== null) this.shardIdx += 1;
      await this.loadShard();
    }

    const B = this.spec.microBatchSize;
    const T = this.spec.seqLen;
    const W = this.windowLen;
    const tokens = this.resident as Uint16Array;
    const maxStart = this.residentLen - W;

    const inputs = new Int32Array(B * T);
    const targets = new Int32Array(B * T);

    for (let b = 0; b < B; b++) {
      // Random aligned-ish offset. We don't enforce non-overlap across
      // sequences in the same batch — overlap is fine for SGD.
      const off = Math.floor(this.rng() * (maxStart + 1));
      const inBase = b * T;
      // tokens[off : off+T]   → inputs row b
      // tokens[off+1 : off+W] → targets row b
      for (let t = 0; t < T; t++) {
        inputs[inBase + t] = tokens[off + t];
        targets[inBase + t] = tokens[off + t + 1];
      }
    }

    this.samplesThisShard += B;
    return { inputs, targets, numTokens: B * T };
  }

  /** Release the resident shard buffer. */
  dispose(): void {
    this.resident = null;
  }
}

// ----------------------------------------------------------------------------
// Validation batch helper
// ----------------------------------------------------------------------------

/**
 * Slice a contiguous chunk of validation windows into one batch.
 * Pure CPU — caller wraps in tf.tensor when feeding the model.
 */
export function valBatch(
  slice: ValSlice,
  startWindow: number,
  count: number,
): Batch {
  const B = Math.min(count, slice.numWindows - startWindow);
  const T = slice.windowLen - 1;
  const W = slice.windowLen;

  const inputs = new Int32Array(B * T);
  const targets = new Int32Array(B * T);

  for (let b = 0; b < B; b++) {
    const winBase = (startWindow + b) * W;
    const outBase = b * T;
    for (let t = 0; t < T; t++) {
      inputs[outBase + t] = slice.windows[winBase + t];
      targets[outBase + t] = slice.windows[winBase + t + 1];
    }
  }

  return { inputs, targets, numTokens: B * T };
}
