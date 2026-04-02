#!/usr/bin/env bun
/**
 * SLM16 live smoke test — runs 100 real optimizer steps on this machine's
 * preferred backend (Metal on darwin), validating every 20 steps and
 * verifying the LKG rotation lands on disk.
 *
 * This is intentionally a *reduced-scale* model (128-dim × 4 layers, seq 128)
 * so the run completes in minutes, not hours. It exercises the full path:
 *   backend load → model build → fwd/bwd → Muon/Adam step → validation
 *   → int8 quantize → atomic write → versioned-LKG rotation → resume.
 *
 * Usage:
 *   bun plugins/plugin-avb/scripts/slm16-smoke.ts [--steps N] [--full]
 *     --steps N   override step count (default 100)
 *     --full      use full DEFAULT_HPARAMS (512×9) instead of reduced config
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import { loadBackend } from "../src/slm16/backend.ts";
import {
  DEFAULT_HPARAMS,
  LKG_FILENAME,
  LKG_META_FILENAME,
  SHARD_HEADER_INTS,
  SHARD_MAGIC,
  type Slm16Hyperparameters,
} from "../src/slm16/constants.ts";
import {
  type LkgMeta,
  Slm16Trainer,
  type TrainerReport,
  listVersionedLkg,
} from "../src/slm16/trainer.ts";

// ----------------------------------------------------------------------------

const argv = process.argv.slice(2);
const arg = (flag: string): string | undefined => {
  const i = argv.indexOf(flag);
  return i >= 0 ? argv[i + 1] : undefined;
};
const STEPS = Number(arg("--steps") ?? 100);
const FULL = argv.includes("--full");

// ----------------------------------------------------------------------------

function writeShard(file: string, tokens: Uint16Array): void {
  const header = new Int32Array(SHARD_HEADER_INTS);
  header[0] = SHARD_MAGIC;
  header[1] = 1;
  header[2] = tokens.length;
  fs.writeFileSync(
    file,
    Buffer.concat([Buffer.from(header.buffer), Buffer.from(tokens.buffer)]),
  );
}

/** xorshift32 → random token in [0, vocab). */
function makeTokenRng(seed: number, vocab: number): () => number {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) % vocab;
  };
}

function fmt(n: number, d = 4): string {
  return Number.isFinite(n) ? n.toFixed(d) : String(n);
}

// ----------------------------------------------------------------------------

async function main(): Promise<void> {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), "slm16-smoke-"));
  const dataDir = path.join(work, "data");
  const ckptDir = path.join(work, "ckpt");
  fs.mkdirSync(dataDir, { recursive: true });
  console.log(`[smoke] workdir: ${work}`);

  // -- Hyperparameters -------------------------------------------------------
  const hp: Slm16Hyperparameters = FULL
    ? DEFAULT_HPARAMS
    : {
        ...DEFAULT_HPARAMS,
        modelDim: 128,
        numLayers: 4,
        numHeads: 4,
        numKvHeads: 2,
        trainSeqLen: 128,
        trainBatchTokens: 512, // 4 seqs × 128
        warmdownIters: 60,
      };
  console.log(
    `[smoke] config: ${hp.modelDim}d × ${hp.numLayers}L, seq=${hp.trainSeqLen}, ` +
      `batch=${hp.trainBatchTokens} tok, vocab=${hp.vocabSize}` +
      (FULL ? " (FULL baseline)" : " (reduced)"),
  );

  // -- Synthetic FineWeb shards ---------------------------------------------
  // Enough train tokens for ~150 steps without wrapping more than a few times.
  const trainTokens = hp.trainBatchTokens * 40;
  const valTokens = hp.trainSeqLen * 64 + 64;
  const trng = makeTokenRng(0xc0ffee, hp.vocabSize);
  const vrng = makeTokenRng(0xfeed, hp.vocabSize);
  writeShard(
    path.join(dataDir, "fineweb_train_000.bin"),
    Uint16Array.from({ length: trainTokens }, () => trng()),
  );
  writeShard(
    path.join(dataDir, "fineweb_val_000.bin"),
    Uint16Array.from({ length: valTokens }, () => vrng()),
  );
  console.log(
    `[smoke] wrote synthetic shards: train=${trainTokens.toLocaleString()} tok, ` +
      `val=${valTokens.toLocaleString()} tok`,
  );

  // -- Backend ---------------------------------------------------------------
  const t0 = Date.now();
  const { tf, name: backendName, vramFraction } = await loadBackend();
  console.log(
    `[smoke] backend: ${backendName} (tfjs ${tf.version?.["tfjs-core"] ?? "?"}` +
      `, vramFraction=${vramFraction}) — loaded in ${Date.now() - t0}ms`,
  );
  if (os.platform() === "darwin" && backendName !== "metal") {
    console.warn(
      `[smoke] WARNING: expected Metal backend on darwin but got "${backendName}"`,
    );
  }

  // -- Trainer ---------------------------------------------------------------
  let lastStepWall = Date.now();
  const stepTimes: number[] = [];
  const valHistory: Array<{ step: number; loss: number }> = [];
  let checkpoints = 0;

  const onReport = (r: TrainerReport): void => {
    switch (r.type) {
      case "boot":
        console.log(
          `[boot] step=${r.step} resumed=${r.resumed} ` +
            `params=${r.paramCount.toLocaleString()} backend=${r.backend} ` +
            `bestValLoss=${fmt(r.bestValLoss)}`,
        );
        break;
      case "step": {
        const now = Date.now();
        const dt = now - lastStepWall;
        lastStepWall = now;
        stepTimes.push(dt);
        const tps = (hp.trainBatchTokens / (dt / 1000)).toFixed(0);
        console.log(
          `[step ${String(r.step).padStart(3)}] loss=${fmt(r.trainLoss)} ` +
            `lr×=${fmt(r.lrScale, 3)} ${dt}ms ${tps} tok/s`,
        );
        break;
      }
      case "validation":
        valHistory.push({ step: r.step, loss: r.valLoss });
        console.log(
          `[val  ${String(r.step).padStart(3)}] val_loss=${fmt(r.valLoss)} ` +
            `best=${fmt(r.bestValLoss)} ${r.improved ? "↑ improved" : ""}`,
        );
        break;
      case "checkpoint":
        checkpoints++;
        console.log(
          `[ckpt ${String(r.step).padStart(3)}] saved ${r.artifactBytes.toLocaleString()} B ` +
            `→ ${path.basename(r.path)}`,
        );
        break;
      case "error":
        console.error(`[error] ${r.message}`);
        break;
    }
  };

  const trainer = new Slm16Trainer({
    tf,
    backendName,
    hp,
    dataDir,
    ckptDir,
    valEvery: 20,
    reportEvery: 5,
    onReport,
  });

  console.log(`[smoke] running ${STEPS} steps (val every 20)...\n`);
  lastStepWall = Date.now();
  await trainer.run(STEPS);
  trainer.dispose();

  // -- Verify ----------------------------------------------------------------
  console.log("\n[smoke] ── results ────────────────────────────────────────");
  const wall = (Date.now() - t0) / 1000;
  const meanStep =
    stepTimes.length > 0
      ? stepTimes.reduce((a, b) => a + b, 0) / stepTimes.length
      : 0;
  console.log(
    `[smoke] wall=${wall.toFixed(1)}s  mean step=${meanStep.toFixed(0)}ms  ` +
      `validations=${valHistory.length}  checkpoints=${checkpoints}`,
  );

  const failures: string[] = [];
  const assert = (cond: boolean, msg: string): void => {
    console.log(`  ${cond ? "✓" : "✗"} ${msg}`);
    if (!cond) failures.push(msg);
  };

  const lkgFile = path.join(ckptDir, LKG_FILENAME);
  const metaFile = path.join(ckptDir, LKG_META_FILENAME);
  const versioned = listVersionedLkg(ckptDir);

  assert(valHistory.length >= STEPS / 20, `ran ≥${STEPS / 20} validations`);
  assert(checkpoints >= 1, "saved ≥1 LKG checkpoint");
  assert(fs.existsSync(lkgFile), `canonical LKG exists (${LKG_FILENAME})`);
  assert(fs.existsSync(metaFile), `canonical meta exists (${LKG_META_FILENAME})`);
  assert(versioned.length >= 1, `versioned rotation has ≥1 entry (${versioned.length})`);
  assert(versioned.length <= 3, `versioned rotation pruned to ≤3 (${versioned.length})`);

  if (fs.existsSync(metaFile)) {
    const meta = JSON.parse(fs.readFileSync(metaFile, "utf8")) as LkgMeta;
    console.log(
      `[smoke] LKG meta: step=${meta.step} val_loss=${fmt(meta.valLoss)} ` +
        `bytes=${meta.artifactBytes.toLocaleString()} ` +
        `history=[${meta.history.map((h) => h.step).join(",")}]`,
    );
    assert(
      Number.isFinite(meta.valLoss) && meta.valLoss > 0,
      "LKG val_loss is finite and positive",
    );
    if (versioned.length > 0) {
      const newestBytes = fs.readFileSync(versioned[0].file);
      const canonBytes = fs.readFileSync(lkgFile);
      assert(
        Buffer.compare(newestBytes, canonBytes) === 0,
        "canonical LKG is byte-identical to newest versioned",
      );
    }
  }

  if (valHistory.length >= 2) {
    const first = valHistory[0].loss;
    const last = valHistory[valHistory.length - 1].loss;
    console.log(
      `[smoke] val_loss trajectory: ${fmt(first)} → ${fmt(last)} ` +
        `(Δ=${fmt(last - first, 4)})`,
    );
  }

  console.log(
    `[smoke] ckptDir contents: ${fs.readdirSync(ckptDir).sort().join(", ")}`,
  );

  if (failures.length > 0) {
    console.error(`\n[smoke] FAILED (${failures.length}):`);
    for (const f of failures) console.error(`  - ${f}`);
    process.exitCode = 1;
  } else {
    console.log(`\n[smoke] PASSED — ${backendName} backend trained ${STEPS} steps.`);
  }
  console.log(`[smoke] artifacts left in ${work} for inspection.`);
}

main().catch((err) => {
  console.error("[smoke] fatal:", err);
  process.exit(1);
});
