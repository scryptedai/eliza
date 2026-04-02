/**
 * SLM16 data pipeline — port of `load_data_shard` / `TokenStream` /
 * `DistributedTokenLoader` from openai/parameter-golf train_gpt.py.
 *
 * Shard format (same as modded-nanogpt):
 *   - 256 × int32 little-endian header
 *       header[0] = 20240520 (magic)
 *       header[1] = 1        (version)
 *       header[2] = numTokens
 *   - numTokens × uint16 little-endian token ids
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import {
  DEFAULT_DATA_DIR,
  DEFAULT_TOKENIZER_PATH,
  SHARD_HEADER_INTS,
  SHARD_MAGIC,
} from "./constants.ts";

// ----------------------------------------------------------------------------
// Bootstrap: download FineWeb shards + tokenizer from HuggingFace
//
// Ports the relevant slice of openai/parameter-golf
// `data/cached_challenge_fineweb.py` — instead of going through
// `huggingface_hub`, we hit the public resolve URL directly so there's
// no Python or extra-dep requirement.
// ----------------------------------------------------------------------------

/** HuggingFace dataset repo holding the parameter-golf FineWeb export. */
export const FINEWEB_HF_REPO = "willdepueoai/parameter-golf";
/** Resolve-URL base for files in the repo. */
export const FINEWEB_HF_BASE = `https://huggingface.co/datasets/${FINEWEB_HF_REPO}/resolve/main`;
/** Repo-relative dir containing the sp1024 shards. */
export const FINEWEB_HF_SHARD_DIR = "datasets/datasets/fineweb10B_sp1024";
/** Repo-relative path to the SentencePiece model. */
export const FINEWEB_HF_TOKENIZER =
  "datasets/tokenizers/fineweb_1024_bpe.model";
/** Default number of train shards to fetch (≈191 MB each). */
export const DEFAULT_TRAIN_SHARDS = 2;

const trainShardName = (i: number): string =>
  `fineweb_train_${String(i).padStart(6, "0")}.bin`;
const VAL_SHARD_NAME = "fineweb_val_000000.bin";

export interface BootstrapOptions {
  /** Where shards land (default: ~/.eliza/slm16/data/fineweb10B_sp1024). */
  dataDir?: string;
  /** Where the SentencePiece .model lands. */
  tokenizerPath?: string;
  /** How many train shards to fetch (default 2). */
  trainShards?: number;
  /** Override base URL (e.g. internal mirror). */
  baseUrl?: string;
  /** Called after each file lands. */
  onProgress?: (msg: string) => void;
}

export interface BootstrapResult {
  dataDir: string;
  tokenizerPath: string;
  /** Files that were actually downloaded (not already present). */
  fetched: string[];
  /** Files that already existed and validated. */
  skipped: string[];
}

/** Stream `url` → `dest` via a `.partial` temp + atomic rename. */
async function downloadTo(url: string, dest: string): Promise<void> {
  const tmp = `${dest}.partial`;
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`);
  }
  // bun/node: convert WHATWG ReadableStream → node Readable, pipe to fs.
  const out = fs.createWriteStream(tmp);
  await pipeline(Readable.fromWeb(res.body as never), out);
  fs.renameSync(tmp, dest);
}

/** Cheap header check without loading the whole shard. */
function shardHeaderValid(file: string): boolean {
  try {
    const fd = fs.openSync(file, "r");
    try {
      const buf = Buffer.alloc(8);
      fs.readSync(fd, buf, 0, 8, 0);
      return buf.readInt32LE(0) === SHARD_MAGIC && buf.readInt32LE(4) === 1;
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    return false;
  }
}

/**
 * Ensure FineWeb shards + tokenizer exist locally. Idempotent: files
 * that already exist (and, for shards, have a valid header) are left
 * alone. Returns immediately if everything is in place.
 */
export async function ensureFinewebData(
  opts: BootstrapOptions = {},
): Promise<BootstrapResult> {
  const dataDir = opts.dataDir ?? DEFAULT_DATA_DIR;
  const tokenizerPath = opts.tokenizerPath ?? DEFAULT_TOKENIZER_PATH;
  const trainShards = Math.max(1, opts.trainShards ?? DEFAULT_TRAIN_SHARDS);
  const baseUrl = opts.baseUrl ?? FINEWEB_HF_BASE;
  const log = opts.onProgress ?? (() => {});

  const fetched: string[] = [];
  const skipped: string[] = [];

  const ensureShard = async (name: string): Promise<void> => {
    const dest = path.join(dataDir, name);
    if (fs.existsSync(dest) && shardHeaderValid(dest)) {
      skipped.push(dest);
      return;
    }
    const url = `${baseUrl}/${FINEWEB_HF_SHARD_DIR}/${name}`;
    log(`fetching shard ${name} …`);
    await downloadTo(url, dest);
    if (!shardHeaderValid(dest)) {
      fs.rmSync(dest, { force: true });
      throw new Error(`Downloaded shard failed header check: ${name}`);
    }
    fetched.push(dest);
    log(
      `  ✓ ${name} (${(fs.statSync(dest).size / 1024 / 1024).toFixed(1)} MiB)`,
    );
  };

  // Train shards 000000..N-1
  for (let i = 0; i < trainShards; i++) {
    await ensureShard(trainShardName(i));
  }
  // Single val shard
  await ensureShard(VAL_SHARD_NAME);

  // Tokenizer
  if (fs.existsSync(tokenizerPath) && fs.statSync(tokenizerPath).size > 0) {
    skipped.push(tokenizerPath);
  } else {
    const url = `${baseUrl}/${FINEWEB_HF_TOKENIZER}`;
    log(`fetching tokenizer ${path.basename(tokenizerPath)} …`);
    await downloadTo(url, tokenizerPath);
    fetched.push(tokenizerPath);
    log(`  ✓ ${path.basename(tokenizerPath)}`);
  }

  return { dataDir, tokenizerPath, fetched, skipped };
}

// ----------------------------------------------------------------------------
// Shard IO
// ----------------------------------------------------------------------------

/** Load one .bin shard into a Uint16Array (zero-copy view over the file buffer). */
export function loadDataShard(file: string): Uint16Array {
  const buf = fs.readFileSync(file);
  const headerBytes = SHARD_HEADER_INTS * 4;
  if (buf.length < headerBytes) {
    throw new Error(`Shard too small for header: ${file}`);
  }
  const header = new Int32Array(buf.buffer, buf.byteOffset, SHARD_HEADER_INTS);
  if (header[0] !== SHARD_MAGIC || header[1] !== 1) {
    throw new Error(`Unexpected shard header for ${file}`);
  }
  const numTokens = header[2];
  const expected = headerBytes + numTokens * 2;
  if (buf.length !== expected) {
    throw new Error(
      `Shard size mismatch for ${file}: expected ${expected}, got ${buf.length}`,
    );
  }
  return new Uint16Array(buf.buffer, buf.byteOffset + headerBytes, numTokens);
}

/** Glob `<dir>/<prefix>*.bin` and return sorted absolute paths. */
export function listShards(dir: string, glob: string): string[] {
  // glob is e.g. "fineweb_train_*.bin" — convert to prefix/suffix match.
  const star = glob.indexOf("*");
  const prefix = star >= 0 ? glob.slice(0, star) : glob;
  const suffix = star >= 0 ? glob.slice(star + 1) : "";
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir)
    .filter((f) => f.startsWith(prefix) && f.endsWith(suffix))
    .sort()
    .map((f) => path.join(dir, f));
}

// ----------------------------------------------------------------------------
// TokenStream — sequential, infinite, deterministic
// ----------------------------------------------------------------------------

export class TokenStream {
  private readonly files: string[];
  private fileIdx = 0;
  private tokens: Uint16Array;
  private pos = 0;

  constructor(files: string[]) {
    if (files.length === 0) {
      throw new Error("TokenStream: no shard files provided");
    }
    this.files = files;
    this.tokens = loadDataShard(this.files[0]);
  }

  private advanceFile(): void {
    this.fileIdx = (this.fileIdx + 1) % this.files.length;
    this.tokens = loadDataShard(this.files[this.fileIdx]);
    this.pos = 0;
  }

  /** Take the next `n` tokens, wrapping across shards as needed. */
  take(n: number): Int32Array {
    const out = new Int32Array(n);
    let written = 0;
    while (written < n) {
      const avail = this.tokens.length - this.pos;
      if (avail <= 0) {
        this.advanceFile();
        continue;
      }
      const k = Math.min(n - written, avail);
      for (let i = 0; i < k; i++) out[written + i] = this.tokens[this.pos + i];
      this.pos += k;
      written += k;
    }
    return out;
  }

  /** Snapshot for resumable training. */
  state(): { fileIdx: number; pos: number } {
    return { fileIdx: this.fileIdx, pos: this.pos };
  }

  restore(state: { fileIdx: number; pos: number }): void {
    this.fileIdx = state.fileIdx % this.files.length;
    this.tokens = loadDataShard(this.files[this.fileIdx]);
    this.pos = Math.min(state.pos, this.tokens.length);
  }
}

// ----------------------------------------------------------------------------
// Batch loader (single-process; world_size==1)
// ----------------------------------------------------------------------------

export interface Batch {
  /** [B, T] input ids. */
  x: Int32Array;
  /** [B, T] target ids (x shifted by 1). */
  y: Int32Array;
  /** Number of sequences in the batch. */
  bsz: number;
  /** Sequence length. */
  seqLen: number;
}

export class TokenLoader {
  private readonly stream: TokenStream;

  constructor(files: string[]) {
    this.stream = new TokenStream(files);
  }

  /**
   * Mirrors DistributedTokenLoader.next_batch with world_size=1:
   * take `globalTokens + 1` and slice into x/y by 1-token shift.
   */
  nextBatch(globalTokens: number, seqLen: number): Batch {
    const local = this.stream.take(globalTokens + 1);
    const bsz = Math.floor(globalTokens / seqLen);
    const used = bsz * seqLen;
    return {
      x: local.subarray(0, used),
      y: local.subarray(1, used + 1),
      bsz,
      seqLen,
    };
  }

  state(): { fileIdx: number; pos: number } {
    return this.stream.state();
  }

  restore(state: { fileIdx: number; pos: number }): void {
    this.stream.restore(state);
  }
}

/**
 * Load the full validation split into a single contiguous buffer,
 * truncated to a multiple of seqLen + 1 (mirrors `load_validation_tokens`).
 */
export function loadValidationTokens(
  files: string[],
  seqLen: number,
): Int32Array {
  if (files.length === 0) {
    throw new Error("loadValidationTokens: no validation shards");
  }
  let total = 0;
  const shards = files.map((f) => {
    const t = loadDataShard(f);
    total += t.length;
    return t;
  });
  const out = new Int32Array(total);
  let off = 0;
  for (const t of shards) {
    for (let i = 0; i < t.length; i++) out[off + i] = t[i];
    off += t.length;
  }
  const usable = Math.floor((out.length - 1) / seqLen) * seqLen;
  if (usable <= 0) {
    throw new Error(`Validation split too short for seqLen=${seqLen}`);
  }
  return out.subarray(0, usable + 1);
}
