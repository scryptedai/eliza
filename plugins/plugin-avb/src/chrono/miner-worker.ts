/**
 * PoW miner — runs inside a node:worker_threads Worker.
 *
 * Receives a MineJob from the parent, grinds nonces over a fixed
 * 108-byte header until sha256d(header) <= target, then posts the
 * sealed header back. CPU usage is throttled to `cpuFraction` of one
 * core by alternating short hash bursts with proportional sleeps, so
 * the worker holds (by default) ~5 % of a core continuously.
 *
 * The header buffer is mutated in place: only the nonce (offset 92)
 * and wallElapsedMs (offset 84) fields change between attempts /
 * before the final post.
 *
 * This file is the worker entry point. It must stay dependency-light
 * (only node:crypto + node:worker_threads + ./block.ts) so it can be
 * loaded via `new Worker(new URL(import.meta.url))` under both Bun
 * and Node without a bundling step.
 */

import { parentPort } from "node:worker_threads";
import {
  bitsToTarget,
  HEADER_SIZE,
  hashToBigInt,
  serializeHeader,
  sha256d,
} from "./block.ts";
import type { MineJob, MinerInbound, MinerOutbound } from "./types.ts";

if (!parentPort) {
  throw new Error("miner-worker must be run as a worker thread");
}
const port = parentPort;

// ----------------------------------------------------------------------------
// Duty-cycle scheduler
//
// We hash in bursts of WORK_SLICE_MS, then sleep long enough that the
// long-run average equals cpuFraction. Using a fixed slice (rather than
// a fixed nonce count) keeps the on/off cadence stable across machines
// with very different hash rates.
// ----------------------------------------------------------------------------

const WORK_SLICE_MS = 50;

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ----------------------------------------------------------------------------
// Mining loop
// ----------------------------------------------------------------------------

let stopping = false;

async function mine(job: MineJob): Promise<void> {
  const target = bitsToTarget(job.difficultyBits);
  const cpu = Math.min(1, Math.max(0.001, job.cpuFraction));
  const sleepPerSlice = WORK_SLICE_MS * (1 / cpu - 1);

  // Build the header once; nonce and wallElapsedMs are patched in place.
  const headerBytes = serializeHeader({
    version: job.version,
    height: job.height,
    prevHash: job.prevHash,
    eventsHash: job.eventsHash,
    hostTimestampMs: job.hostTimestampMs,
    wallElapsedMs: 0,
    difficultyBits: job.difficultyBits,
    nonce: 0n,
    eventCount: job.eventCount,
    eventBytes: job.eventBytes,
  });
  const dv = new DataView(headerBytes.buffer);

  const t0 = performance.now();
  let nonce = 0n;
  let attempts = 0;

  while (!stopping) {
    // wallElapsedMs is part of the hashed header. Refresh it once per
    // work slice so the value sealed into a winning block is accurate
    // to within WORK_SLICE_MS without needing a post-hoc re-hash.
    const sliceStart = performance.now();
    const elapsed = Math.max(0, Math.round(sliceStart - t0));
    dv.setUint32(84, elapsed >>> 0, true);

    const sliceEnd = sliceStart + WORK_SLICE_MS;
    while (performance.now() < sliceEnd) {
      dv.setBigUint64(92, nonce, true);
      const hash = sha256d(headerBytes);
      attempts++;
      if (hashToBigInt(hash) <= target) {
        post({
          kind: "sealed",
          headerBytes: headerBytes.slice(0, HEADER_SIZE),
          hash,
          height: job.height,
          wallElapsedMs: elapsed,
          attempts,
        });
        return;
      }
      nonce++;
    }
    if (sleepPerSlice > 0) await sleep(sleepPerSlice);
  }
}

function post(msg: MinerOutbound): void {
  port.postMessage(msg);
}

port.on("message", (msg: MinerInbound) => {
  if (msg.kind === "stop") {
    stopping = true;
    return;
  }
  if (msg.kind === "mine") {
    stopping = false;
    void mine(msg).catch((err) => {
      post({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    });
  }
});
