/**
 * SLM16 — worker_threads entry point.
 *
 * This file is the `new Worker(__filename)` target. It receives the resolved
 * training config via `workerData`, loads the right tfjs backend, and runs
 * the training loop until told to stop.
 *
 * Why a worker thread:
 *   - tfjs-node forward/backward is mostly C++ via N-API → it releases the
 *     V8 isolate's microtask queue but still hogs the calling thread's
 *     event loop while waiting on the binding. Putting it in a worker keeps
 *     the agent's main thread fully responsive.
 *   - Crashes (CUDA OOM, NaN explosions) are contained: the parent gets an
 *     'exit' event and can choose to restart, the agent process survives.
 *
 * Communication:
 *   parentPort.postMessage(Slm16Event)  — progress (ready, step, val, lkg, error, idle)
 *   parentPort.on('message', cmd)        — control ({ type: 'stop' } | { type: 'checkpoint' })
 *
 * The worker exits cleanly after emitting 'idle'. Exit code 0 = clean,
 * non-zero = unhandled crash (the parent's 'exit' handler treats this as
 * an error event).
 */

import { parentPort, workerData } from "node:worker_threads";
import { detectBackend } from "./config.ts";
import { loadTfBackend, trainLoop, type TrainerHooks } from "./trainer.ts";
import type { Slm16Command, Slm16Event, Slm16WorkerData } from "./types.ts";

// Top-level await is fine in an ESM worker.

if (parentPort === null) {
  // Loaded directly (not as a worker). This shouldn't happen in normal
  // operation — the service always spawns via `new Worker(...)`. But it's
  // useful for ad-hoc debugging: `bun src/slm16/worker.ts` will print the
  // backend it would load and exit.
  const { label } = detectBackend();
  process.stderr.write(`[slm16-worker] standalone mode (backend would be: ${label})\n`);
  process.exit(0);
}

const port = parentPort;
const wd = workerData as Slm16WorkerData;

// Control flags polled by the trainer loop. We use plain booleans (not
// atomics) because the worker is single-threaded: the message handler
// runs on the same event loop as the trainer's `await setImmediate()` yield.
let stopRequested = false;
let checkpointRequested = false;

port.on("message", (raw: unknown) => {
  const cmd = raw as Slm16Command;
  if (cmd.type === "stop") {
    stopRequested = true;
  } else if (cmd.type === "checkpoint") {
    checkpointRequested = true;
  }
});

const hooks: TrainerHooks = {
  emit: (e: Slm16Event) => port.postMessage(e),
  shouldStop: () => stopRequested,
  shouldCheckpoint: () => {
    if (checkpointRequested) {
      checkpointRequested = false; // one-shot
      return true;
    }
    return false;
  },
};

try {
  const backend = detectBackend();
  const { tf, loaded } = await loadTfBackend(backend.pkg, wd.config.vramFraction);

  // Sanity log on stderr (the parent doesn't intercept stderr; this lands
  // in the agent's terminal). The structured "ready" event carries the
  // same info to the parent for setSetting().
  process.stderr.write(
    `[slm16-worker] tf backend: ${tf.getBackend()} (pkg: ${loaded}, ` +
      `vram_fraction: ${wd.config.vramFraction})\n`,
  );

  await trainLoop(tf, wd.config, hooks);

  // trainLoop returned cleanly → emitted 'idle' already. Exit 0.
  process.exit(0);
} catch (err) {
  const e = err as Error;
  port.postMessage({
    type: "error",
    message: e.message ?? String(err),
    stack: e.stack,
    fatal: true,
  } satisfies Slm16Event);
  process.exit(1);
}
