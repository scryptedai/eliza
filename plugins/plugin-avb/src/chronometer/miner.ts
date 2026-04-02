/**
 * Miner facade — main-thread interface to the worker.
 *
 * Two concerns:
 * 1. Worker lifecycle (spawn, message routing, teardown).
 * 2. Synchronous in-thread mining for genesis + tests, where spinning up
 *    a worker is overkill or actively harmful (vitest worker-in-worker
 *    gets weird, and genesis at GENESIS_BITS solves in ~milliseconds).
 *
 * Both share the same mineSync() core so there's exactly one PoW
 * implementation to audit. The worker file duplicates the inner loop for
 * import-hygiene reasons (see miner.worker.ts header) but the algorithm
 * is identical and tests cover both.
 */

import { Worker } from "node:worker_threads";
import {
  bitsToTarget,
  headerHash,
  lte256,
  serializeHeader,
  sha256d,
} from "./encoding.ts";
import type {
  BlockHeader,
  MiningJob,
  MiningResult,
  WorkerOutbound,
} from "./types.ts";
import { HEADER_SIZE } from "./types.ts";

// ----------------------------------------------------------------------------
// Synchronous in-thread miner (genesis, tests, fallback)
// ----------------------------------------------------------------------------

/**
 * Grind the nonce on the calling thread. No throttling — caller is
 * responsible for not invoking this with a difficulty that'll lock up
 * the event loop. At GENESIS_BITS this finishes in microseconds.
 *
 * @param header header with nonce=0 (or any starting nonce)
 * @param maxAttempts safety valve; throws if exceeded
 */
export function mineSync(
  header: BlockHeader,
  maxAttempts = 10_000_000,
): { header: BlockHeader; hash: Buffer; attempts: number } {
  const target = bitsToTarget(header.bits);
  const buf = serializeHeader(header);
  const nonceOff = HEADER_SIZE - 4;

  let nonce = header.nonce >>> 0;
  for (let attempts = 1; attempts <= maxAttempts; attempts++) {
    buf.writeUInt32LE(nonce, nonceOff);
    const hash = sha256d(buf);
    if (lte256(hash, target)) {
      return {
        header: { ...header, nonce },
        hash,
        attempts,
      };
    }
    nonce = (nonce + 1) >>> 0;
  }
  throw new Error(
    `mineSync exhausted ${maxAttempts} attempts at bits=0x${header.bits.toString(16)}`,
  );
}

// ----------------------------------------------------------------------------
// Worker-backed miner
// ----------------------------------------------------------------------------

export interface MinerEvents {
  onSolved: (result: MiningResult) => void;
  onProgress?: (jobId: number, attempts: number) => void;
  onError?: (err: Error) => void;
}

export class Miner {
  private worker?: Worker;
  private nextJobId = 1;
  private readonly events: MinerEvents;

  constructor(events: MinerEvents) {
    this.events = events;
  }

  /**
   * Spawn the worker. We resolve the worker script via import.meta.url
   * so it works regardless of where the package is installed. Bun and
   * Node both honour `new URL(rel, import.meta.url)` for worker entry.
   */
  start(): void {
    if (this.worker) return;

    const workerUrl = new URL("./miner.worker.ts", import.meta.url);
    this.worker = new Worker(workerUrl);

    this.worker.on("message", (msg: WorkerOutbound | { type: string }) => {
      if (msg.type === "solved") {
        const r = (msg as { result: MiningResult }).result;
        // Buffers cross the worker boundary as Uint8Array; rewrap.
        r.hash = Buffer.from(r.hash);
        this.events.onSolved(r);
      } else if (msg.type === "progress") {
        const p = msg as { jobId: number; attempts: number };
        this.events.onProgress?.(p.jobId, p.attempts);
      }
      // "cancelHandle" is informational; we don't currently use direct
      // SAB poking from this side (supersession via postMessage suffices).
    });

    this.worker.on("error", (err: unknown) => {
      this.events.onError?.(
        err instanceof Error ? err : new Error(String(err)),
      );
    });
  }

  /**
   * Submit a header for mining. Returns the jobId so callers can
   * correlate the eventual onSolved callback. If a job is already
   * running, the worker cancels it and starts this one — callers
   * shouldn't expect a solved callback for the superseded job.
   */
  submit(template: Omit<MiningJob, "jobId">): number {
    if (!this.worker) throw new Error("Miner not started");
    const jobId = this.nextJobId++;
    const job: MiningJob = {
      jobId,
      // Worker boundary clones Buffers but the receiving end may see
      // Uint8Array; the worker rewraps with Buffer.from() so we don't
      // need to do anything special on send.
      prevHash: template.prevHash,
      eventsRoot: template.eventsRoot,
      hostTimestamp: template.hostTimestamp,
      bits: template.bits,
    };
    this.worker.postMessage({ type: "mine", job });
    return jobId;
  }

  async stop(): Promise<void> {
    if (!this.worker) return;
    this.worker.postMessage({ type: "stop" });
    await this.worker.terminate();
    this.worker = undefined;
  }
}

// ----------------------------------------------------------------------------
// PoW verification (used by both miner self-check and chain validation)
// ----------------------------------------------------------------------------

/** True if the header's hash meets its own declared difficulty. */
export function verifyPow(header: BlockHeader): boolean {
  const hash = headerHash(header);
  const target = bitsToTarget(header.bits);
  return lte256(hash, target);
}
