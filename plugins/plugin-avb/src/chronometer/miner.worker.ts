/**
 * PoW miner — runs in a Worker thread.
 *
 * The throttle is the interesting part. We want ~5% of one core, which we
 * achieve by duty-cycling: hash for a short burst, then sleep for ~19×
 * that burst. Operating systems don't give us a "use 5% of a core"
 * primitive, but they DO honour sleep() — so 5ms work + 95ms sleep gets
 * us very close to 5% on average without needing cgroup/nice tricks that
 * are non-portable.
 *
 * We hash in fixed-count batches rather than fixed-time bursts because
 * fixed-time would require checking the clock inside the inner loop
 * (expensive). Instead we calibrate batch size on the first batch
 * (measure how long N hashes take) and recalibrate every few seconds to
 * track thermal/load drift.
 *
 * Atomics-based cancellation: when the main thread posts a new job, we
 * want the worker to abandon the current grind ASAP. Worker.postMessage
 * is async (lands on the event loop), so the inner loop checks a
 * SharedArrayBuffer flag every batch. SharedArrayBuffer requires
 * crossOriginIsolated in browsers but we're in Node so it Just Works.
 */

import { createHash } from "node:crypto";
import { parentPort } from "node:worker_threads";
import type {
  MiningJob,
  MiningResult,
  WorkerInbound,
  WorkerOutbound,
} from "./types.ts";
import { HEADER_SIZE } from "./types.ts";

// ----------------------------------------------------------------------------
// Throttle config
// ----------------------------------------------------------------------------

/** Fraction of one core to consume. 0.05 = 5%. */
const CPU_ALLOCATION = 0.05;

/** Target work-burst duration. Short enough to be responsive, long
 *  enough that syscall overhead (sleep, clock_gettime) is negligible. */
const TARGET_BURST_MS = 5;

/** Initial guess for hashes/ms. Recalibrated after first burst. */
const INITIAL_HASHES_PER_MS = 500;

/** Recalibrate batch size every N bursts. */
const RECALIBRATE_EVERY = 50;

/** Post a progress message every N bursts (so main thread sees we're alive). */
const PROGRESS_EVERY = 200;

// ----------------------------------------------------------------------------
// Cancellation flag (set by main thread between jobs)
// ----------------------------------------------------------------------------

const cancelBuf = new SharedArrayBuffer(4);
const cancelFlag = new Int32Array(cancelBuf);

// Expose the buffer back to main on first ready so it can poke us.
let sentCancelHandle = false;

// ----------------------------------------------------------------------------
// Hashing primitives (duplicated from encoding.ts so the worker has zero
// project-internal imports beyond types — keeps the bundle the bundler
// has to ship to the worker tiny and avoids any circular surprises)
// ----------------------------------------------------------------------------

function sha256d(buf: Buffer): Buffer {
  const inner = createHash("sha256").update(buf).digest();
  return createHash("sha256").update(inner).digest();
}

function bitsToTarget(bits: number): Buffer {
  const exponent = bits >>> 24;
  const mantissa = bits & 0x007fffff;
  const target = Buffer.alloc(32);
  if (exponent <= 3) {
    const shifted = mantissa >>> (8 * (3 - exponent));
    target.writeUIntBE(shifted, 29, 3);
  } else {
    const pos = 32 - exponent;
    if (pos >= 0 && pos + 3 <= 32) target.writeUIntBE(mantissa, pos, 3);
  }
  return target;
}

function lte256(a: Buffer, b: Buffer): boolean {
  for (let i = 0; i < 32; i++) {
    if (a[i] < b[i]) return true;
    if (a[i] > b[i]) return false;
  }
  return true;
}

// ----------------------------------------------------------------------------
// Header buffer assembly (write once, mutate nonce in place)
// ----------------------------------------------------------------------------

function buildHeaderBuf(job: MiningJob): Buffer {
  const buf = Buffer.alloc(HEADER_SIZE);
  let off = 0;
  off = buf.writeUInt32LE(1, off); // BLOCK_VERSION
  off += Buffer.from(job.prevHash).copy(buf, off, 0, 32);
  off += Buffer.from(job.eventsRoot).copy(buf, off, 0, 32);
  off = buf.writeUInt32LE(job.hostTimestamp, off);
  off = buf.writeUInt32LE(job.bits, off);
  buf.writeUInt32LE(0, off); // nonce (mutated in loop)
  return buf;
}

// ----------------------------------------------------------------------------
// The grind
// ----------------------------------------------------------------------------

const NONCE_OFFSET = HEADER_SIZE - 4;

async function mine(job: MiningJob): Promise<MiningResult | null> {
  Atomics.store(cancelFlag, 0, 0);

  const header = buildHeaderBuf(job);
  const target = bitsToTarget(job.bits);
  const start = Date.now();

  let nonce = 0;
  let attempts = 0;
  let batchSize = Math.max(
    1,
    Math.floor(INITIAL_HASHES_PER_MS * TARGET_BURST_MS),
  );
  let burst = 0;

  // The sleep duration that gives us our duty cycle.
  // work_ms / (work_ms + sleep_ms) = CPU_ALLOCATION
  // → sleep_ms = work_ms * (1 - alloc) / alloc
  const sleepRatio = (1 - CPU_ALLOCATION) / CPU_ALLOCATION;

  while (true) {
    // Cancellation check (cheap — one atomic load per batch).
    if (Atomics.load(cancelFlag, 0) !== 0) return null;

    const burstStart = performance.now();

    for (let i = 0; i < batchSize; i++) {
      header.writeUInt32LE(nonce, NONCE_OFFSET);
      const hash = sha256d(header);
      attempts++;

      if (lte256(hash, target)) {
        return {
          jobId: job.jobId,
          nonce,
          hash,
          attempts,
          elapsedMs: Date.now() - start,
        };
      }

      // Wrap at uint32. If we exhaust the nonce space without solving,
      // bump hostTimestamp by one second and keep grinding — same trick
      // Bitcoin uses (extraNonce in coinbase). At our difficulty levels
      // we'll never get close to 2^32 attempts, but correctness > faith.
      nonce = (nonce + 1) >>> 0;
      if (nonce === 0) {
        const tsOff = NONCE_OFFSET - 8; // 4 (bits) + 4 (ts) before nonce
        header.writeUInt32LE(header.readUInt32LE(tsOff) + 1, tsOff);
      }
    }

    const burstMs = performance.now() - burstStart;
    burst++;

    // Recalibrate batch size toward TARGET_BURST_MS.
    if (burst % RECALIBRATE_EVERY === 0 && burstMs > 0) {
      const measuredHashesPerMs = batchSize / burstMs;
      batchSize = Math.max(
        1,
        Math.floor(measuredHashesPerMs * TARGET_BURST_MS),
      );
    }

    // Heartbeat.
    if (burst % PROGRESS_EVERY === 0) {
      post({ type: "progress", jobId: job.jobId, attempts });
    }

    // Throttle. We sleep proportionally to how long we just worked.
    // Using actual burstMs (not TARGET_BURST_MS) self-corrects: if a
    // burst ran long, we sleep longer to compensate.
    const sleepMs = burstMs * sleepRatio;
    await sleep(sleepMs);
  }
}

// ----------------------------------------------------------------------------
// Plumbing
// ----------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function post(msg: WorkerOutbound): void {
  parentPort?.postMessage(msg);
}

// Job queue (depth 1 — newer job supersedes older).
let activeJob: Promise<void> | null = null;
let pendingJob: MiningJob | null = null;

async function runJob(job: MiningJob): Promise<void> {
  const result = await mine(job);
  if (result) post({ type: "solved", result });
  // null = cancelled; main thread already knows it cancelled us.
}

function scheduleJob(job: MiningJob): void {
  if (activeJob) {
    // Cancel the running grind and stash the new job.
    Atomics.store(cancelFlag, 0, 1);
    pendingJob = job;
    return;
  }
  activeJob = runJob(job).finally(() => {
    activeJob = null;
    if (pendingJob) {
      const next = pendingJob;
      pendingJob = null;
      scheduleJob(next);
    }
  });
}

parentPort?.on("message", (raw: WorkerInbound) => {
  // Main thread can grab the cancel handle from the first message
  // exchange if it wants direct poke access — but we also handle
  // supersession internally via scheduleJob, so this is belt+braces.
  if (!sentCancelHandle) {
    sentCancelHandle = true;
    parentPort?.postMessage({ type: "cancelHandle", buf: cancelBuf });
  }

  if (raw.type === "stop") {
    Atomics.store(cancelFlag, 0, 1);
    return;
  }
  if (raw.type === "mine") {
    scheduleJob(raw.job);
  }
});
