/**
 * Live proving test for the AVB Chronometer (v2: header-only chain +
 * Merkle-committed off-chain event log).
 *
 * Boots ChronometerService against a fresh temp data dir with the
 * miner thread ENABLED, records a handful of synthetic AVB events,
 * waits for N blocks to be sealed, then re-loads BOTH files from disk
 * and verifies:
 *   - header chain integrity (no issues)
 *   - off-chain segments cross-check against sealed Merkle roots
 *   - block heights 0..N
 *   - recorded events are present in the sealed segments
 *   - an inclusion proof for one event verifies stand-alone
 *
 * Defaults: 5 % of TOTAL machine CPU, 60 s block-time target,
 * 3 blocks past genesis. Expect ~4-5 minutes wall time (a few
 * near-instant warmup blocks while difficulty retargets up from
 * the genesis floor, then ~60 s/block).
 *
 * Run from repo root:
 *   bun plugins/plugin-avb/scripts/prove-chrono.ts
 *
 * Override via env:
 *   CHRONO_PROVE_CPU=5  CHRONO_PROVE_BLOCK_MS=3000  CHRONO_PROVE_BLOCKS=3
 */

import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  ChronoEventType,
  ChronometerService,
  decodeEvents,
  ENV_CHRONO_BLOCK_MS,
  ENV_CHRONO_CPU_PERCENT,
  ENV_CHRONO_DATA_DIR,
  ENV_CHRONO_ENABLED,
  loadChain,
  loadEventSegments,
  merkleRoot,
  splitEvents,
  toHex,
  verifyInclusion,
  verifySegments,
} from "../src/chrono/index.ts";

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------

const CPU_PERCENT = Number(process.env.CHRONO_PROVE_CPU ?? 5);
const BLOCK_MS = Number(process.env.CHRONO_PROVE_BLOCK_MS ?? 60_000);
const N_BLOCKS = Number(process.env.CHRONO_PROVE_BLOCKS ?? 3);
const TIMEOUT_MS = Math.max(60_000, BLOCK_MS * N_BLOCKS * 10);
const CORES = os.cpus()?.length ?? 1;
const PER_CORE_FRACTION = Math.min(1, (CPU_PERCENT / 100) * CORES);

// ----------------------------------------------------------------------------
// Minimal runtime stub (ChronometerService only needs this surface)
// ----------------------------------------------------------------------------

function makeRuntime(dataDir: string) {
  const sealed: Array<{ height: number; wallElapsedMs: number }> = [];
  const settings: Record<string, string> = {
    [ENV_CHRONO_DATA_DIR]: dataDir,
    [ENV_CHRONO_CPU_PERCENT]: String(CPU_PERCENT),
    [ENV_CHRONO_BLOCK_MS]: String(BLOCK_MS),
    [ENV_CHRONO_ENABLED]: "true",
  };
  const log =
    (lvl: string) =>
    (msg: string): void =>
      console.log(`[${lvl}] ${msg}`);
  return {
    sealed,
    rt: {
      agentId: "prove-chrono-agent",
      logger: {
        info: log("info"),
        warn: log("WARN"),
        error: log("ERROR"),
        debug: () => undefined,
      },
      getSetting: (k: string) => settings[k],
      emitEvent: async (name: string, payload: unknown) => {
        if (name === "AVB_CHRONO_BLOCK_SEALED") {
          const p = payload as { height: number; wallElapsedMs: number };
          sealed.push({ height: p.height, wallElapsedMs: p.wallElapsedMs });
        }
      },
    },
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / 1024 / 1024).toFixed(2)} MiB`;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

async function main(): Promise<void> {
  const dataDir = await fs.mkdtemp(
    path.join(os.tmpdir(), "avb-chrono-prove-"),
  );
  const chainPath = path.join(dataDir, "logs", "events.chain");
  const eventsPath = path.join(dataDir, "logs", "events.bin");

  console.log("─".repeat(64));
  console.log("AVB Chronometer — live PoW proving run (v2)");
  console.log(
    `  cpu allocation : ${CPU_PERCENT}% of total ` +
      `(${CORES} cores → ${(PER_CORE_FRACTION * 100).toFixed(0)}% of one core)`,
  );
  console.log(`  block target   : ${BLOCK_MS} ms`);
  console.log(`  blocks to mine : ${N_BLOCKS} (past genesis)`);
  console.log(`  data dir       : ${dataDir}`);
  console.log(`    chain file   :   logs/events.chain (108 B/block)`);
  console.log(`    event log    :   logs/events.bin   (Merkle-committed)`);
  console.log("─".repeat(64));

  const { rt, sealed } = makeRuntime(dataDir);
  const t0 = performance.now();
  const svc = await ChronometerService.start(rt as never);

  console.log(
    `✓ genesis minted (height=${svc.getHeight()}), miner thread running`,
  );

  // Record synthetic AVB events while mining proceeds. These should land
  // in blocks 2..N (block 1's segment was drained at dispatch time and
  // already contains the BOOT event recorded during start()).
  const injected = [
    "run=demo phase=TEXT_PHASE",
    "run=demo phase=IMAGE_PHASE",
    "run=demo url=https://example/avatar.png",
  ];
  svc.recordEvent(ChronoEventType.PHASE_STARTED, injected[0]);
  svc.recordEvent(ChronoEventType.PHASE_COMPLETE, injected[1]);
  svc.recordEvent(ChronoEventType.DELIVER, injected[2]);

  // Wait for N blocks past genesis.
  console.log(`\nWaiting for ${N_BLOCKS} block(s) to be sealed…`);
  const deadline = Date.now() + TIMEOUT_MS;
  while (sealed.length < N_BLOCKS) {
    if (Date.now() > deadline) {
      throw new Error(
        `timeout: only ${sealed.length}/${N_BLOCKS} blocks sealed in ${TIMEOUT_MS}ms`,
      );
    }
    await sleep(200);
  }

  // Build an inclusion proof while the service still has the chain loaded.
  // Block 1 always contains at least the BOOT event.
  const proof = await svc.proveInclusion(1, 0);

  await svc.stop();
  const totalMs = performance.now() - t0;

  // ---- Re-load from disk and verify ----
  console.log("\n" + "─".repeat(64));
  console.log("Reloading from disk and verifying…");
  const { blocks, issues } = await loadChain(chainPath);
  const segments = await loadEventSegments(eventsPath);
  const segIssues = verifySegments(blocks, segments, (seg) =>
    merkleRoot(splitEvents(seg)),
  );

  let ok = true;
  const fail = (msg: string) => {
    ok = false;
    console.log(`  ✗ ${msg}`);
  };
  const pass = (msg: string) => console.log(`  ✓ ${msg}`);

  if (issues.length === 0) {
    pass(`header chain integrity clean (${blocks.length} blocks)`);
  } else {
    fail(`chain integrity issues: ${issues.map((i) => i.code).join(", ")}`);
  }

  if (segIssues.length === 0) {
    pass(
      `event log cross-check clean (${segments.size} segments, all Merkle roots match)`,
    );
  } else {
    fail(`event-log issues: ${segIssues.map((i) => i.code).join(", ")}`);
  }

  if (blocks.length >= N_BLOCKS + 1) {
    pass(`block count ${blocks.length} ≥ genesis + ${N_BLOCKS}`);
  } else {
    fail(`expected ≥ ${N_BLOCKS + 1} blocks, got ${blocks.length}`);
  }

  // Heights monotonic from 0
  const heightsOk = blocks.every((b, i) => b.header.height === i);
  heightsOk ? pass("heights 0..N monotonic") : fail("height sequence broken");

  // Events: collect every decoded event detail across all non-genesis
  // segments (block 0 is GENESIS only).
  const allDetails = new Set<string>();
  for (const b of blocks.slice(1)) {
    const seg = segments.get(b.header.height) ?? new Uint8Array(0);
    for (const e of decodeEvents(seg)) allDetails.add(e.detail);
  }
  const bootSeen = [...allDetails].some((d) => d.startsWith("agent="));
  bootSeen
    ? pass("BOOT event sealed into block 1")
    : fail("BOOT event missing from event log");
  for (const inj of injected) {
    allDetails.has(inj)
      ? pass(`event sealed: "${inj}"`)
      : fail(`event NOT found in any segment: "${inj}"`);
  }

  // Inclusion proof: third party with only the proof can verify the
  // event was sealed under valid PoW without seeing events.bin.
  const proofEv = decodeEvents(proof.encodedEvent)[0];
  if (verifyInclusion(proof)) {
    pass(
      `inclusion proof verifies (block #${proof.header.height}, ` +
        `${proof.path.length} Merkle steps, ` +
        `event=[${ChronoEventType[proofEv.type] ?? proofEv.type}] "${proofEv.detail}")`,
    );
  } else {
    fail("inclusion proof FAILED to verify");
  }
  // Negative: any bit-flip in the event must be rejected.
  const forged = proof.encodedEvent.slice();
  forged[forged.length - 1] ^= 0x01;
  if (!verifyInclusion({ ...proof, encodedEvent: forged })) {
    pass("inclusion proof rejects tampered event");
  } else {
    fail("inclusion proof ACCEPTED a tampered event");
  }

  // ---- Report ----
  console.log("\n" + "─".repeat(64));
  console.log("Block summary:");
  for (const b of blocks) {
    const seg = segments.get(b.header.height) ?? new Uint8Array(0);
    const evs = decodeEvents(seg);
    console.log(
      `  #${b.header.height}  hash=${toHex(b.hash).slice(0, 16)}…  ` +
        `bits=0x${b.header.difficultyBits.toString(16).padStart(8, "0")}  ` +
        `wall=${b.header.wallElapsedMs}ms  ` +
        `hostTs=${new Date(b.header.hostTimestampMs).toISOString()}  ` +
        `events=${b.header.eventCount}`,
    );
    for (const e of evs) {
      console.log(
        `       · [${ChronoEventType[e.type] ?? e.type}] ${e.detail}`,
      );
    }
  }

  const intrinsic = blocks.reduce((s, b) => s + b.header.wallElapsedMs, 0);
  const chainStat = await fs.stat(chainPath);
  const eventsStat = await fs.stat(eventsPath);
  console.log("\nStorage:");
  console.log(
    `  events.chain : ${fmtBytes(chainStat.size)} ` +
      `(${blocks.length} × 108 B headers, no inline payload)`,
  );
  console.log(
    `  events.bin   : ${fmtBytes(eventsStat.size)} ` +
      `(${segments.size} segments, rotatable independently of chain)`,
  );
  let proofBytes = 108 + proof.encodedEvent.length;
  for (const s of proof.path) proofBytes += 1 + s.sibling.length;
  console.log(
    `  inclusion proof for one event : ${fmtBytes(proofBytes)} ` +
      `(header + event + ${proof.path.length}-step Merkle path)`,
  );

  console.log("\nTiming:");
  console.log(`  wall-clock total      : ${(totalMs / 1000).toFixed(2)}s`);
  console.log(
    `  Σ wallElapsedMs (PoW) : ${(intrinsic / 1000).toFixed(2)}s ` +
      `(agent's intrinsic clock)`,
  );
  const sleepMs = Math.round(50 * (1 / PER_CORE_FRACTION - 1));
  console.log(
    `  cpu duty-cycle target : ${CPU_PERCENT}% of ${CORES} cores ` +
      `= ${(PER_CORE_FRACTION * 100).toFixed(0)}% of one core ` +
      `(miner alternates 50ms work / ${sleepMs}ms sleep)`,
  );

  await fs.rm(dataDir, { recursive: true, force: true });

  console.log("\n" + "─".repeat(64));
  if (ok) {
    console.log(
      "✓ CHRONOMETER PROVED — PoW chain valid, event log Merkle-committed, " +
        "inclusion proof verifies.",
    );
  } else {
    console.log("✗ CHRONOMETER FAILED — see ✗ lines above.");
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error("\n✗ prove-chrono failed:", err);
  process.exit(1);
});
