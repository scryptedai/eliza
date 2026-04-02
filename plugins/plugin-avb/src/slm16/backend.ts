/**
 * SLM16 backend selection.
 *
 * TensorFlow.js is the only widely-deployed tensor library that supports
 * gradient-based training from TypeScript with native GPU bindings.
 * Backend resolution order:
 *
 *   1. CUDA  → `@tensorflow/tfjs-node-gpu` (Linux/Windows + NVIDIA driver)
 *   2. Metal → `@tensorflow/tfjs-node` on darwin (TensorFlow C library is
 *              built against Apple Accelerate / MPS; matmuls dispatch to the
 *              GPU via Metal Performance Shaders on Apple-silicon)
 *   3. CPU   → `@tensorflow/tfjs-node` elsewhere
 *
 * VRAM limiting:
 *   tfjs-node-gpu honours the same env knobs as upstream TensorFlow
 *   (`TF_FORCE_GPU_ALLOW_GROWTH`, `TF_GPU_ALLOCATOR`,
 *   `TF_PER_PROCESS_GPU_MEMORY_FRACTION`). We set the fraction *before*
 *   the native binding is loaded so the device allocator picks it up.
 */

import * as os from "node:os";
import {
  DEFAULT_VRAM_FRACTION,
  ENV_SLM16_BACKEND,
  ENV_SLM16_VRAM_FRACTION,
  type Slm16BackendName,
} from "./constants.ts";

/** The subset of the tfjs surface SLM16 actually uses. Keeping this narrow
 *  lets tests inject a fake without pulling in the native binding. */
export type TF = typeof import("@tensorflow/tfjs");

export interface BackendInfo {
  tf: TF;
  name: Slm16BackendName;
  /** Resolved VRAM fraction (echoed back for logging; CPU = 0). */
  vramFraction: number;
}

function clamp01(n: number): number {
  if (!Number.isFinite(n)) return DEFAULT_VRAM_FRACTION;
  return Math.min(1, Math.max(0.05, n));
}

function resolveVramFraction(): number {
  const raw = process.env[ENV_SLM16_VRAM_FRACTION];
  return clamp01(raw ? Number.parseFloat(raw) : DEFAULT_VRAM_FRACTION);
}

function applyGpuEnv(fraction: number): void {
  // Must be set BEFORE the native addon is dlopen'd.
  process.env.TF_FORCE_GPU_ALLOW_GROWTH ??= "true";
  process.env.TF_GPU_ALLOCATOR ??= "cuda_malloc_async";
  process.env.TF_PER_PROCESS_GPU_MEMORY_FRACTION = String(fraction);
  // Quiet the native banner; the service does its own logging.
  process.env.TF_CPP_MIN_LOG_LEVEL ??= "2";
}

/**
 * Resolve and load the best available tfjs backend for the current host.
 * Optional dependencies are imported via dynamic specifier strings so
 * bundlers don't try to inline native .node addons.
 */
export async function loadBackend(
  forced?: Slm16BackendName,
): Promise<BackendInfo> {
  const want = (forced ??
    (process.env[ENV_SLM16_BACKEND] as Slm16BackendName | undefined)) as
    | Slm16BackendName
    | undefined;
  const vramFraction = resolveVramFraction();
  const isDarwin = os.platform() === "darwin";

  // ---- CUDA ----------------------------------------------------------------
  if (want === "cuda" || (!want && !isDarwin)) {
    applyGpuEnv(vramFraction);
    try {
      const spec = "@tensorflow/tfjs-node-gpu";
      const mod = (await import(/* @vite-ignore */ spec)) as TF;
      await mod.ready();
      return { tf: mod, name: "cuda", vramFraction };
    } catch {
      if (want === "cuda") {
        throw new Error(
          "SLM16: CUDA backend requested but @tensorflow/tfjs-node-gpu failed to load",
        );
      }
      // fall through
    }
  }

  // ---- Metal (darwin tfjs-node) -------------------------------------------
  if (want === "metal" || (!want && isDarwin)) {
    applyGpuEnv(vramFraction);
    try {
      const spec = "@tensorflow/tfjs-node";
      const mod = (await import(/* @vite-ignore */ spec)) as TF;
      await mod.ready();
      return {
        tf: mod,
        name: isDarwin ? "metal" : "cpu",
        vramFraction: isDarwin ? vramFraction : 0,
      };
    } catch {
      if (want === "metal") {
        throw new Error(
          "SLM16: Metal backend requested but @tensorflow/tfjs-node failed to load",
        );
      }
    }
  }

  // ---- CPU fallback (pure-JS tfjs) ----------------------------------------
  const spec = "@tensorflow/tfjs";
  const mod = (await import(/* @vite-ignore */ spec)) as TF;
  await mod.setBackend("cpu");
  await mod.ready();
  return { tf: mod, name: "cpu", vramFraction: 0 };
}
