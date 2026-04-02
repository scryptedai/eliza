/**
 * SLM16 training worker entrypoint.
 *
 * Spawned via `new Worker(new URL("./worker.ts", import.meta.url))` from
 * `Slm16Service`. Runs the long-lived training loop on its own thread so
 * the agent's main event loop never blocks on a backward pass. All status
 * is forwarded to the parent over `parentPort.postMessage`.
 *
 * The native tfjs binding is loaded *inside* this worker, so the VRAM
 * fraction (set via env before the worker is spawned) bounds only this
 * thread's GPU allocator — the main agent process never touches the GPU.
 */

import { parentPort, workerData } from "node:worker_threads";

import { loadBackend } from "./backend.ts";
import {
  DEFAULT_CKPT_DIR,
  DEFAULT_DATA_DIR,
  DEFAULT_HPARAMS,
  type Slm16BackendName,
} from "./constants.ts";
import { Slm16Trainer, type TrainerReport } from "./trainer.ts";

export interface Slm16WorkerData {
  dataDir?: string;
  ckptDir?: string;
  backend?: Slm16BackendName;
  valEvery?: number;
}

export type Slm16WorkerMessage = TrainerReport | { type: "exit"; step: number };

const send = (m: Slm16WorkerMessage): void => parentPort?.postMessage(m);

async function main(): Promise<void> {
  const wd = (workerData ?? {}) as Slm16WorkerData;
  let trainer: Slm16Trainer | null = null;
  try {
    const { tf, name } = await loadBackend(wd.backend);
    trainer = new Slm16Trainer({
      tf,
      backendName: name,
      hp: DEFAULT_HPARAMS,
      dataDir: wd.dataDir ?? DEFAULT_DATA_DIR,
      ckptDir: wd.ckptDir ?? DEFAULT_CKPT_DIR,
      valEvery: wd.valEvery,
      onReport: send,
    });

    parentPort?.on("message", (msg: { type: string }) => {
      if (msg?.type === "stop") trainer?.stop();
    });

    await trainer.run();
  } catch (err) {
    send({
      type: "error",
      step: 0,
      message: (err as Error).stack ?? String(err),
    });
  } finally {
    trainer?.dispose();
    send({ type: "exit", step: 0 });
  }
}

void main();
