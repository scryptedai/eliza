/**
 * SLM16 unit tests.
 *
 * These cover the pure-TS layers (quantizer, shard loader, evaluator math)
 * so they run without the native tfjs binding. The model/trainer paths are
 * exercised end-to-end in the project runner; here we only assert the
 * load-bearing invariants:
 *   - int8 quantize → dequantize round-trips within scale error
 *   - 16 MiB ceiling on a baseline-sized state dict
 *   - FineWeb shard header parsing matches the reference format
 *   - cosine similarity behaves
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as zlib from "node:zlib";
import { afterAll, describe, expect, it } from "vitest";

import {
  ARTIFACT_BYTE_LIMIT,
  DEFAULT_HPARAMS,
  SHARD_HEADER_INTS,
  SHARD_MAGIC,
} from "../slm16/constants.ts";
import {
  listShards,
  loadDataShard,
  TokenLoader,
  TokenStream,
} from "../slm16/data.ts";
import { cosineSimilarity, parseJudgeVerdict } from "../slm16/evaluator.ts";
import {
  deserializeInt8,
  type StateDict,
  serializeInt8,
} from "../slm16/quantize.ts";

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "slm16-test-"));
afterAll(() => fs.rmSync(tmpDir, { recursive: true, force: true }));

/** xorshift32 — deterministic PRNG so the test is reproducible. */
function makeRng(seed = 0xdeadbeef): () => number {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) / 0x100000000) * 2 - 1; // [-1, 1)
  };
}

function randF32(n: number, rng: () => number, scale = 0.1): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = rng() * scale;
  return a;
}

function maxAbsErr(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

/** Write a modded-nanogpt-format shard to disk. */
function writeShard(file: string, tokens: Uint16Array): void {
  const header = new Int32Array(SHARD_HEADER_INTS);
  header[0] = SHARD_MAGIC;
  header[1] = 1;
  header[2] = tokens.length;
  const buf = Buffer.concat([
    Buffer.from(header.buffer),
    Buffer.from(tokens.buffer),
  ]);
  fs.writeFileSync(file, buf);
}

// ----------------------------------------------------------------------------
// Quantizer
// ----------------------------------------------------------------------------

describe("slm16/quantize", () => {
  it("round-trips a 2-D matrix within per-row int8 error", () => {
    const rng = makeRng(1);
    // Must exceed INT8_KEEP_FLOAT_MAX_NUMEL (65 536) to hit the quant path.
    const rows = 512;
    const cols = 256;
    const data = randF32(rows * cols, rng, 0.5);
    const sd: StateDict = new Map([
      ["blocks.0.attn.c_q.weight", { shape: [rows, cols], data }],
    ]);
    const { blob, stats } = serializeInt8(sd);
    expect(stats.numQuantized).toBe(1);
    expect(stats.numPassthrough).toBe(0);
    const back = deserializeInt8(blob);
    const out = back.get("blocks.0.attn.c_q.weight");
    expect(out).toBeDefined();
    expect(out!.shape).toEqual([rows, cols]);
    // Symmetric int8 with |x|≤0.5 → step ≤ 0.5/127 ≈ 0.0039.
    expect(maxAbsErr(data, out!.data)).toBeLessThan(0.01);
  });

  it("passes through small/control tensors as fp32", () => {
    const sd: StateDict = new Map([
      ["blocks.0.attn_scale", { shape: [1], data: Float32Array.of(1.5) }],
      ["skip_weights", { shape: [4], data: Float32Array.of(1, 1, 1, 1) }],
    ]);
    const { blob, stats } = serializeInt8(sd);
    expect(stats.numPassthrough).toBe(2);
    expect(stats.numQuantized).toBe(0);
    const back = deserializeInt8(blob);
    expect(back.get("blocks.0.attn_scale")!.data[0]).toBeCloseTo(1.5, 6);
  });

  it("compresses the 512×9 baseline param budget under 16 MiB", { timeout: 60_000 }, () => {
    // Mirror the dominant tensors of the parameter-golf naive baseline:
    // tied embedding + per-layer attn (q/k/v/proj) + mlp (fc/proj). The
    // exact param count differs from the real model (no q_gain etc.) but
    // the byte budget is the same order of magnitude.
    const {
      modelDim: d,
      numLayers: L,
      numHeads,
      numKvHeads,
      vocabSize,
    } = DEFAULT_HPARAMS;
    const headDim = d / numHeads;
    const kvDim = numKvHeads * headDim;
    const rng = makeRng(2);
    const sd: StateDict = new Map();
    sd.set("tok_emb.weight", {
      shape: [vocabSize, d],
      data: randF32(vocabSize * d, rng, 0.005),
    });
    for (let i = 0; i < L; i++) {
      const add = (name: string, rows: number, cols: number) =>
        sd.set(name, {
          shape: [rows, cols],
          data: randF32(rows * cols, rng, 0.05),
        });
      add(`blocks.${i}.attn.c_q.weight`, d, d);
      add(`blocks.${i}.attn.c_k.weight`, kvDim, d);
      add(`blocks.${i}.attn.c_v.weight`, kvDim, d);
      add(`blocks.${i}.attn.proj.weight`, d, d);
      add(`blocks.${i}.mlp.fc.weight`, 2 * d, d);
      add(`blocks.${i}.mlp.proj.weight`, d, 2 * d);
    }
    const { blob, stats } = serializeInt8(sd);
    expect(stats.compressedBytes).toBe(blob.length);
    expect(stats.compressedBytes).toBeLessThan(ARTIFACT_BYTE_LIMIT);
    // Sanity: int8 + zlib should beat raw fp32 by >3×.
    expect(stats.compressedBytes).toBeLessThan(stats.paramCount * 4 * 0.35);
  });

  it("rejects a blob with the wrong format tag", () => {
    const raw = Buffer.from(
      JSON.stringify({
        __quant_format__: "bogus",
        quantized: {},
        passthrough: {},
      }),
    );
    // Manually deflate so deserializeInt8 inflates a valid zlib stream.
    expect(() => deserializeInt8(zlib.deflateSync(raw))).toThrow(
      /Unsupported quant format/,
    );
  });
});

// ----------------------------------------------------------------------------
// Shard loader
// ----------------------------------------------------------------------------

describe("slm16/data", () => {
  const trainA = path.join(tmpDir, "fineweb_train_000.bin");
  const trainB = path.join(tmpDir, "fineweb_train_001.bin");
  const val = path.join(tmpDir, "fineweb_val_000.bin");

  // 0..99 in shard A, 100..149 in shard B, 200..263 in val.
  writeShard(
    trainA,
    Uint16Array.from({ length: 100 }, (_, i) => i),
  );
  writeShard(
    trainB,
    Uint16Array.from({ length: 50 }, (_, i) => 100 + i),
  );
  writeShard(
    val,
    Uint16Array.from({ length: 64 }, (_, i) => 200 + i),
  );

  it("parses the shard header and returns the right token count", () => {
    const t = loadDataShard(trainA);
    expect(t.length).toBe(100);
    expect(t[0]).toBe(0);
    expect(t[99]).toBe(99);
  });

  it("rejects a shard with a bad magic", () => {
    const bad = path.join(tmpDir, "bad.bin");
    const header = new Int32Array(SHARD_HEADER_INTS);
    header[0] = 1234;
    header[1] = 1;
    header[2] = 0;
    fs.writeFileSync(bad, Buffer.from(header.buffer));
    expect(() => loadDataShard(bad)).toThrow(/header/);
  });

  it("listShards globs and sorts", () => {
    const files = listShards(tmpDir, "fineweb_train_*.bin");
    expect(files.map((f) => path.basename(f))).toEqual([
      "fineweb_train_000.bin",
      "fineweb_train_001.bin",
    ]);
  });

  it("TokenStream wraps across shards and is restorable", () => {
    const ts = new TokenStream([trainA, trainB]);
    const a = ts.take(120);
    expect(a[0]).toBe(0);
    expect(a[99]).toBe(99);
    expect(a[100]).toBe(100); // crossed into shard B
    expect(a[119]).toBe(119);
    const snap = ts.state();
    const b = ts.take(10);
    expect(b[0]).toBe(120);
    ts.restore(snap);
    const c = ts.take(10);
    expect(c[0]).toBe(120); // resumed exactly where the snapshot was taken
  });

  it("TokenLoader.nextBatch shifts targets by one", () => {
    const tl = new TokenLoader([trainA, trainB]);
    const batch = tl.nextBatch(32, 16);
    expect(batch.bsz).toBe(2);
    expect(batch.seqLen).toBe(16);
    expect(batch.x.length).toBe(32);
    expect(batch.y.length).toBe(32);
    for (let i = 0; i < 32; i++) {
      expect(batch.y[i]).toBe(batch.x[i] + 1);
    }
  });
});

// ----------------------------------------------------------------------------
// Evaluator math
// ----------------------------------------------------------------------------

describe("slm16/evaluator", () => {
  it("cosineSimilarity is 1 for identical, 0 for orthogonal, -1 for opposite", () => {
    expect(cosineSimilarity([1, 0, 0], [1, 0, 0])).toBeCloseTo(1, 6);
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 6);
    expect(cosineSimilarity([1, 2, 3], [-1, -2, -3])).toBeCloseTo(-1, 6);
  });

  it("cosineSimilarity is scale-invariant", () => {
    expect(cosineSimilarity([1, 2, 3], [2, 4, 6])).toBeCloseTo(1, 6);
  });

  it("parseJudgeVerdict extracts and clamps similarity", () => {
    const v = parseJudgeVerdict('{"similarity": 0.65, "reasoning": "ok"}');
    expect(v.similarity).toBeCloseTo(0.65, 6);
    expect(v.reasoning).toBe("ok");
    // Clamp out-of-range, accept "score" alias, default reasoning.
    expect(parseJudgeVerdict('{"score": 1.4}').similarity).toBe(1);
    expect(parseJudgeVerdict('{"similarity": -0.2}').similarity).toBe(0);
    expect(parseJudgeVerdict('{"similarity": 0.5}').reasoning).toBe(
      "(no reasoning)",
    );
  });

  it("parseJudgeVerdict tolerates code fences and surrounding prose", () => {
    const raw =
      'Sure, here is the verdict:\n```json\n{"similarity":0.3,"reasoning":"weak"}\n```\nDone.';
    const v = parseJudgeVerdict(raw);
    expect(v.similarity).toBeCloseTo(0.3, 6);
    expect(v.reasoning).toBe("weak");
  });

  it("parseJudgeVerdict throws when no JSON / no numeric similarity", () => {
    expect(() => parseJudgeVerdict("no json here")).toThrow(/No JSON object/);
    expect(() => parseJudgeVerdict('{"reasoning":"only"}')).toThrow(
      /missing numeric/,
    );
  });
});
