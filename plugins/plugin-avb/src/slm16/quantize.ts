/**
 * SLM16 int8 post-training quantization + zlib compression.
 *
 * Port of `quantize_state_dict_int8` / `dequantize_state_dict_int8` from
 * openai/parameter-golf train_gpt.py:
 *   - 2-D float tensors → per-row symmetric int8 (clip at 99.99984th pct)
 *   - other float tensors with numel > 65 536 → per-tensor int8
 *   - small / control tensors → fp32 passthrough (cheap, preserves stability)
 *   - everything wrapped in a JSON-friendly envelope and zlib-deflated
 *
 * The serialized blob is the LKG checkpoint: it must stay under
 * ARTIFACT_BYTE_LIMIT (16 MiB) and be round-trippable for both training
 * resumption and inference.
 */

import * as zlib from "node:zlib";
import {
  CONTROL_TENSOR_NAME_PATTERNS,
  INT8_CLIP_PERCENTILE,
  INT8_KEEP_FLOAT_MAX_NUMEL,
  INT8_QUANT_FORMAT,
} from "./constants.ts";

export type StateDict = Map<string, { shape: number[]; data: Float32Array }>;

interface QuantTensor {
  shape: number[];
  /** base64 of Int8Array bytes. */
  q: string;
  /** Per-row scale (length=shape[0]) or single scalar. base64 of Float32Array. */
  s: string;
  scheme: "per_row" | "per_tensor";
}

interface PassTensor {
  shape: number[];
  /** base64 of Float32Array bytes. */
  d: string;
}

interface QuantEnvelope {
  __quant_format__: typeof INT8_QUANT_FORMAT;
  quantized: Record<string, QuantTensor>;
  passthrough: Record<string, PassTensor>;
}

export interface QuantStats {
  paramCount: number;
  numTensors: number;
  numQuantized: number;
  numPassthrough: number;
  rawBytes: number;
  compressedBytes: number;
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

const CLIP_Q = INT8_CLIP_PERCENTILE / 100;

function isControl(name: string): boolean {
  return CONTROL_TENSOR_NAME_PATTERNS.some((p) => name.includes(p));
}

function b64encode(arr: Int8Array | Float32Array): string {
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString(
    "base64",
  );
}

function b64decodeF32(s: string): Float32Array {
  const buf = Buffer.from(s, "base64");
  return new Float32Array(
    buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  );
}

function b64decodeI8(s: string): Int8Array {
  const buf = Buffer.from(s, "base64");
  return new Int8Array(
    buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  );
}

/**
 * Quantile of |x| using nearest-rank on a sorted copy.
 * Mirrors torch.quantile(abs, q).
 */
function absQuantile(data: Float32Array, q: number): number {
  if (data.length === 0) return 0;
  const abs = Float32Array.from(data, Math.abs).sort();
  const idx = Math.min(abs.length - 1, Math.round(q * (abs.length - 1)));
  return abs[idx];
}

// ----------------------------------------------------------------------------
// Quantize
// ----------------------------------------------------------------------------

function quantizePerRow(
  data: Float32Array,
  rows: number,
  cols: number,
): { q: Int8Array; s: Float32Array } {
  const q = new Int8Array(rows * cols);
  const s = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    const row = data.subarray(r * cols, (r + 1) * cols);
    const clip = absQuantile(row, CLIP_Q);
    const scale = Math.max(clip / 127, 1 / 127);
    s[r] = scale;
    for (let c = 0; c < cols; c++) {
      const v = Math.max(-clip, Math.min(clip, row[c]));
      q[r * cols + c] = Math.max(-127, Math.min(127, Math.round(v / scale)));
    }
  }
  return { q, s };
}

function quantizePerTensor(data: Float32Array): {
  q: Int8Array;
  s: Float32Array;
} {
  const clip = absQuantile(data, CLIP_Q);
  const scale = clip > 0 ? clip / 127 : 1;
  const q = new Int8Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const v = Math.max(-clip, Math.min(clip, data[i]));
    q[i] = Math.max(-127, Math.min(127, Math.round(v / scale)));
  }
  return { q, s: Float32Array.of(scale) };
}

/**
 * Quantize a state dict and zlib-compress it. Returns the blob plus stats
 * for size accounting (used to enforce the 16 MB ceiling).
 */
export function serializeInt8(sd: StateDict): {
  blob: Buffer;
  stats: QuantStats;
} {
  const env: QuantEnvelope = {
    __quant_format__: INT8_QUANT_FORMAT,
    quantized: {},
    passthrough: {},
  };
  const stats: QuantStats = {
    paramCount: 0,
    numTensors: 0,
    numQuantized: 0,
    numPassthrough: 0,
    rawBytes: 0,
    compressedBytes: 0,
  };

  for (const [name, { shape, data }] of sd) {
    stats.numTensors++;
    stats.paramCount += data.length;

    const small = data.length <= INT8_KEEP_FLOAT_MAX_NUMEL;
    if (small || isControl(name)) {
      env.passthrough[name] = { shape, d: b64encode(data) };
      stats.numPassthrough++;
      continue;
    }

    if (shape.length === 2) {
      const { q, s } = quantizePerRow(data, shape[0], shape[1]);
      env.quantized[name] = {
        shape,
        q: b64encode(q),
        s: b64encode(s),
        scheme: "per_row",
      };
    } else {
      const { q, s } = quantizePerTensor(data);
      env.quantized[name] = {
        shape,
        q: b64encode(q),
        s: b64encode(s),
        scheme: "per_tensor",
      };
    }
    stats.numQuantized++;
  }

  const raw = Buffer.from(JSON.stringify(env), "utf8");
  const blob = zlib.deflateSync(raw, { level: 9 });
  stats.rawBytes = raw.length;
  stats.compressedBytes = blob.length;
  return { blob, stats };
}

// ----------------------------------------------------------------------------
// Dequantize
// ----------------------------------------------------------------------------

export function deserializeInt8(blob: Buffer): StateDict {
  const raw = zlib.inflateSync(blob);
  const env = JSON.parse(raw.toString("utf8")) as QuantEnvelope;
  if (env.__quant_format__ !== INT8_QUANT_FORMAT) {
    throw new Error(
      `Unsupported quant format: ${env.__quant_format__} (expected ${INT8_QUANT_FORMAT})`,
    );
  }
  const out: StateDict = new Map();

  for (const [name, t] of Object.entries(env.quantized)) {
    const q = b64decodeI8(t.q);
    const s = b64decodeF32(t.s);
    const data = new Float32Array(q.length);
    if (t.scheme === "per_row") {
      const [rows, cols] = t.shape;
      for (let r = 0; r < rows; r++) {
        const sc = s[r];
        for (let c = 0; c < cols; c++) {
          data[r * cols + c] = q[r * cols + c] * sc;
        }
      }
    } else {
      const sc = s[0];
      for (let i = 0; i < q.length; i++) data[i] = q[i] * sc;
    }
    out.set(name, { shape: t.shape, data });
  }

  for (const [name, t] of Object.entries(env.passthrough)) {
    out.set(name, { shape: t.shape, data: b64decodeF32(t.d) });
  }

  return out;
}
