/**
 * Chronometer test suite.
 *
 * Coverage philosophy: the chronometer's whole point is detecting when
 * the world is lying to it. So most of these tests construct a valid
 * chain, mutilate it in a specific way, and assert the validator catches
 * exactly that mutilation with the right error code.
 *
 * We use mineSync() with GENESIS_BITS throughout — at that difficulty
 * blocks solve in microseconds, which keeps the suite fast. The async
 * worker miner is exercised separately in a smoke test that we run with
 * a generous timeout.
 *
 * Storage is split: chain file (header-only, fixed stride) + event log
 * (sparse, payload batches). Tamper tests cover both files.
 */

import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { appendBlock, loadChain, validateBlock } from "../chronometer/chain.ts";
import {
  GENESIS_BITS,
  initialDifficulty,
  retarget,
  scaleTarget,
  TARGET_BLOCK_MS,
} from "../chronometer/difficulty.ts";
import {
  bitsToTarget,
  buildGenesisHeader,
  computeEventsRoot,
  deserializeEventBatch,
  deserializeEvents,
  deserializeHeader,
  EMPTY_EVENTS_ROOT,
  headerHash,
  lte256,
  serializeEventBatch,
  serializeEvents,
  serializeHeader,
  sha256d,
  targetToBits,
  ZERO_HASH,
} from "../chronometer/encoding.ts";
import { mineSync, verifyPow } from "../chronometer/miner.ts";
import { digestRuntimeEvent } from "../chronometer/service.ts";
import {
  BLOCK_VERSION,
  type Block,
  CHAIN_MAGIC,
  type ChronoEvent,
  EVENTS_MAGIC,
  EventKind,
  HEADER_SIZE,
} from "../chronometer/types.ts";

// ----------------------------------------------------------------------------
// Test helpers
// ----------------------------------------------------------------------------

/** Mine a block chaining to `prev` (or genesis if undefined). */
function mineBlock(
  prev: Block | undefined,
  events: ChronoEvent[],
  hostTimestamp: number,
  bits: number = GENESIS_BITS,
): { block: Block; events: ChronoEvent[] } {
  const prevHash = prev ? prev.hash : ZERO_HASH;
  const height = prev ? prev.height + 1 : 0;
  const template = {
    version: BLOCK_VERSION,
    prevHash,
    eventsRoot: computeEventsRoot(events),
    hostTimestamp,
    bits,
    nonce: 0,
  };
  const { header, hash } = mineSync(template);
  return { block: { header, hash, height }, events };
}

function evt(data: string, at = 1000): ChronoEvent {
  return { kind: EventKind.GENERIC, at, data };
}

/** Append in one call — most tests don't care about ordering details. */
async function append(
  chainPath: string,
  eventLogPath: string,
  m: { block: Block; events: ChronoEvent[] },
): Promise<void> {
  await appendBlock(chainPath, eventLogPath, m.block, m.events);
}

// ----------------------------------------------------------------------------
// Encoding round-trips
// ----------------------------------------------------------------------------

describe("encoding", () => {
  it("header serializes to exactly 80 bytes", () => {
    const h = buildGenesisHeader(GENESIS_BITS, 1234);
    expect(serializeHeader(h).length).toBe(HEADER_SIZE);
  });

  it("header round-trips losslessly", () => {
    const h = {
      version: 7,
      prevHash: Buffer.from("aa".repeat(32), "hex"),
      eventsRoot: Buffer.from("bb".repeat(32), "hex"),
      hostTimestamp: 1_700_000_000,
      bits: 0x1d00ffff,
      nonce: 0xdeadbeef,
    };
    const buf = serializeHeader(h);
    const back = deserializeHeader(buf);
    expect(back.version).toBe(h.version);
    expect(back.prevHash.equals(h.prevHash)).toBe(true);
    expect(back.eventsRoot.equals(h.eventsRoot)).toBe(true);
    expect(back.hostTimestamp).toBe(h.hostTimestamp);
    expect(back.bits).toBe(h.bits);
    expect(back.nonce).toBe(h.nonce);
  });

  it("event round-trips with utf8 payload", () => {
    const events: ChronoEvent[] = [
      { kind: EventKind.PHASE_START, at: 1234.5, data: "hello" },
      { kind: EventKind.PHASE_COMPLETE, at: 5678, data: "wörld 🌍" },
      { kind: EventKind.GENERIC, at: 0, data: "" },
    ];
    const buf = serializeEvents(events);
    const { events: back, bytesRead } = deserializeEvents(buf, 0);
    expect(bytesRead).toBe(buf.length);
    expect(back).toHaveLength(3);
    expect(back[0]).toEqual(events[0]);
    expect(back[1]).toEqual(events[1]);
    expect(back[2]).toEqual(events[2]);
  });

  it("event batch round-trips through the height-tagged record format", () => {
    const events = [evt("a"), evt("b")];
    const buf = serializeEventBatch(42, events);
    const { height, events: back, bytesRead } = deserializeEventBatch(buf, 0);
    expect(bytesRead).toBe(buf.length);
    expect(height).toBe(42);
    expect(back).toHaveLength(2);
    expect(back[0].data).toBe("a");
    expect(back[1].data).toBe("b");
  });

  it("sha256d matches known vector", () => {
    // sha256d("hello") = sha256(sha256("hello"))
    // sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    // sha256(that) = 9595c9df90075148eb06860365df33584b75bff782a510c6cd4883a419833d50
    const out = sha256d(Buffer.from("hello"));
    expect(out.toString("hex")).toBe(
      "9595c9df90075148eb06860365df33584b75bff782a510c6cd4883a419833d50",
    );
  });

  it("eventsRoot is sensitive to event order", () => {
    const a = computeEventsRoot([evt("a"), evt("b")]);
    const b = computeEventsRoot([evt("b"), evt("a")]);
    expect(a.equals(b)).toBe(false);
  });

  it("eventsRoot is sensitive to data mutation", () => {
    const a = computeEventsRoot([evt("a")]);
    const b = computeEventsRoot([evt("A")]);
    expect(a.equals(b)).toBe(false);
  });

  it("EMPTY_EVENTS_ROOT matches computeEventsRoot([])", () => {
    expect(EMPTY_EVENTS_ROOT.equals(computeEventsRoot([]))).toBe(true);
  });
});

// ----------------------------------------------------------------------------
// nBits / target encoding
// ----------------------------------------------------------------------------

describe("nBits encoding", () => {
  it("decodes Bitcoin's max target correctly", () => {
    // Bitcoin block 0 nBits: 0x1d00ffff
    // → target = 0x00ffff * 2^(8*(0x1d-3)) = 00000000ffff0000...0000
    const target = bitsToTarget(0x1d00ffff);
    expect(target[0]).toBe(0);
    expect(target[1]).toBe(0);
    expect(target[2]).toBe(0);
    expect(target[3]).toBe(0);
    expect(target[4]).toBe(0xff);
    expect(target[5]).toBe(0xff);
    expect(target[6]).toBe(0);
  });

  it("round-trips bits → target → bits", () => {
    const cases = [0x1d00ffff, 0x1f00ffff, 0x1c0ae493, GENESIS_BITS];
    for (const bits of cases) {
      const target = bitsToTarget(bits);
      const back = targetToBits(target);
      // Round-trip may not be exact due to mantissa truncation but the
      // re-decoded target should match.
      const targetBack = bitsToTarget(back);
      expect(targetBack.equals(target)).toBe(true);
    }
  });

  it("lte256 compares big-endian correctly", () => {
    const a = Buffer.alloc(32);
    const b = Buffer.alloc(32);
    a[0] = 1;
    b[0] = 2;
    expect(lte256(a, b)).toBe(true);
    expect(lte256(b, a)).toBe(false);
    expect(lte256(a, a)).toBe(true);
  });

  it("lte256 distinguishes by least-significant differing byte", () => {
    const a = Buffer.alloc(32);
    const b = Buffer.alloc(32);
    a[31] = 1;
    b[31] = 2;
    expect(lte256(a, b)).toBe(true);
    expect(lte256(b, a)).toBe(false);
  });
});

// ----------------------------------------------------------------------------
// Difficulty adjustment
// ----------------------------------------------------------------------------

describe("difficulty", () => {
  it("scaleTarget halving makes the target smaller (harder)", () => {
    const before = bitsToTarget(GENESIS_BITS);
    const halvedBits = scaleTarget(GENESIS_BITS, 0.5);
    const after = bitsToTarget(halvedBits);
    expect(lte256(after, before)).toBe(true);
    expect(after.equals(before)).toBe(false);
  });

  it("scaleTarget doubling makes the target larger (easier)", () => {
    const before = bitsToTarget(GENESIS_BITS);
    const doubledBits = scaleTarget(GENESIS_BITS, 2.0);
    const after = bitsToTarget(doubledBits);
    expect(lte256(before, after)).toBe(true);
    expect(after.equals(before)).toBe(false);
  });

  it("retarget makes mining harder when blocks come too fast", () => {
    let state = initialDifficulty();
    const before = bitsToTarget(state.bits);
    // Simulate blocks solving in 1 second instead of 60.
    // EMA needs a few iterations to converge from initial TARGET_BLOCK_MS.
    for (let i = 0; i < 5; i++) state = retarget(state, 1000);
    const after = bitsToTarget(state.bits);
    // After should be smaller (harder) than before.
    expect(lte256(after, before)).toBe(true);
    expect(after.equals(before)).toBe(false);
  });

  it("retarget makes mining easier when blocks come too slow", () => {
    let state = initialDifficulty();
    const before = bitsToTarget(state.bits);
    // Simulate blocks solving in 5 minutes instead of 1.
    for (let i = 0; i < 5; i++) state = retarget(state, 300_000);
    const after = bitsToTarget(state.bits);
    expect(lte256(before, after)).toBe(true);
    expect(after.equals(before)).toBe(false);
  });

  it("retarget converges toward target block time", () => {
    let state = initialDifficulty();
    // Feed perfect block times — EMA should hold steady, bits unchanged.
    for (let i = 0; i < 10; i++) state = retarget(state, TARGET_BLOCK_MS);
    expect(state.emaBlockMs).toBeCloseTo(TARGET_BLOCK_MS, 0);
  });

  it("retarget clamps wild swings", () => {
    let state = initialDifficulty();
    // One absurdly fast block (1ms) shouldn't crater difficulty.
    const beforeBits = state.bits;
    state = retarget(state, 1);
    // Target should change but not by an absurd factor — the clamp is 4x
    // and the EMA dampens further. Verify it's still a sane non-zero target.
    const after = bitsToTarget(state.bits);
    let nonZero = false;
    for (let i = 0; i < 32; i++) if (after[i] !== 0) nonZero = true;
    expect(nonZero).toBe(true);
    expect(state.bits).not.toBe(beforeBits); // But it did move.
  });
});

// ----------------------------------------------------------------------------
// PoW
// ----------------------------------------------------------------------------

describe("proof of work", () => {
  it("mineSync finds a nonce satisfying the target", () => {
    const template = buildGenesisHeader(GENESIS_BITS, 1000);
    const { header, hash, attempts } = mineSync(template);
    const target = bitsToTarget(header.bits);
    expect(lte256(hash, target)).toBe(true);
    expect(attempts).toBeGreaterThan(0);
  });

  it("mined header passes verifyPow", () => {
    const template = buildGenesisHeader(GENESIS_BITS, 1000);
    const { header } = mineSync(template);
    expect(verifyPow(header)).toBe(true);
  });

  it("perturbing nonce by 1 invalidates PoW (with overwhelming probability)", () => {
    const template = buildGenesisHeader(GENESIS_BITS, 1000);
    const { header } = mineSync(template);
    // GENESIS_BITS gives ~1/256 chance of any nonce working. Try 10
    // perturbations; if ANY fails, we've shown what we wanted to show.
    let anyFailed = false;
    for (let delta = 1; delta <= 10; delta++) {
      const perturbed = { ...header, nonce: (header.nonce + delta) >>> 0 };
      if (!verifyPow(perturbed)) anyFailed = true;
    }
    expect(anyFailed).toBe(true);
  });

  it("headerHash is deterministic", () => {
    const h = buildGenesisHeader(GENESIS_BITS, 1000);
    const a = headerHash(h);
    const b = headerHash({ ...h });
    expect(a.equals(b)).toBe(true);
  });

  it("headerHash changes with any field", () => {
    const h = buildGenesisHeader(GENESIS_BITS, 1000);
    const base = headerHash(h);
    expect(headerHash({ ...h, version: h.version + 1 }).equals(base)).toBe(
      false,
    );
    expect(
      headerHash({ ...h, hostTimestamp: h.hostTimestamp + 1 }).equals(base),
    ).toBe(false);
    expect(headerHash({ ...h, nonce: h.nonce + 1 }).equals(base)).toBe(false);
    expect(headerHash({ ...h, bits: h.bits + 1 }).equals(base)).toBe(false);
  });
});

// ----------------------------------------------------------------------------
// Chain validation — the meat
// ----------------------------------------------------------------------------

describe("chain validation", () => {
  let dir: string;
  let chainPath: string;
  let eventLogPath: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "avb-chrono-"));
    chainPath = join(dir, "test.chain");
    eventLogPath = join(dir, "test.events");
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it("empty (no file) → ok with zero blocks", () => {
    const { blocks, report } = loadChain(chainPath, eventLogPath);
    expect(blocks).toHaveLength(0);
    expect(report.ok).toBe(true);
    expect(report.validBlocks).toBe(0);
  });

  it("clean three-block chain validates", async () => {
    const m0 = mineBlock(undefined, [evt("genesis")], 1000);
    const m1 = mineBlock(m0.block, [evt("event-a"), evt("event-b")], 1060);
    const m2 = mineBlock(m1.block, [], 1120);
    await append(chainPath, eventLogPath, m0);
    await append(chainPath, eventLogPath, m1);
    await append(chainPath, eventLogPath, m2);

    const { blocks, report } = loadChain(chainPath, eventLogPath);
    expect(report.ok).toBe(true);
    expect(report.issues).toHaveLength(0);
    expect(blocks).toHaveLength(3);
    expect(blocks[0].height).toBe(0);
    expect(blocks[2].height).toBe(2);
    // Blocks are header-only now — events live in the event log.
    expect(
      blocks[1].header.eventsRoot.equals(computeEventsRoot(m1.events)),
    ).toBe(true);
  });

  it("detects bad magic", () => {
    writeFileSync(chainPath, Buffer.from([0xde, 0xad, 0xbe, 0xef, 1, 0, 0, 0]));
    const { report } = loadChain(chainPath, eventLogPath);
    expect(report.ok).toBe(false);
    expect(report.issues[0].code).toBe("BAD_MAGIC");
  });

  it("detects truncated chain file (mid-header)", async () => {
    const m0 = mineBlock(undefined, [evt("x")], 1000);
    await append(chainPath, eventLogPath, m0);
    // Truncate to lose the last 10 bytes of the header.
    const raw = readFileSync(chainPath);
    writeFileSync(chainPath, raw.subarray(0, raw.length - 10));

    const { blocks, report } = loadChain(chainPath, eventLogPath);
    expect(report.ok).toBe(false);
    expect(report.issues.some((i) => i.code === "TRUNCATED")).toBe(true);
    expect(blocks).toHaveLength(0);
  });

  it("detects prevHash mismatch (chain break)", async () => {
    const m0 = mineBlock(undefined, [evt("genesis")], 1000);
    // m1 chains to m0 correctly.
    const m1 = mineBlock(m0.block, [evt("a")], 1060);
    // m2_bad chains to m0 instead of m1 — broken chain.
    const m2_bad = mineBlock(m0.block, [evt("b")], 1120);
    m2_bad.block.height = 2;
    await append(chainPath, eventLogPath, m0);
    await append(chainPath, eventLogPath, m1);
    await append(chainPath, eventLogPath, m2_bad);

    const { blocks, report } = loadChain(chainPath, eventLogPath);
    expect(report.ok).toBe(false);
    expect(report.issues.some((i) => i.code === "PREV_HASH_MISMATCH")).toBe(
      true,
    );
    const issue = report.issues.find((i) => i.code === "PREV_HASH_MISMATCH");
    expect(issue?.height).toBe(2);
    expect(blocks).toHaveLength(2); // Loaded up to the bad block.
  });

  it("detects PoW failure (chain header tampered post-mine)", async () => {
    const m0 = mineBlock(undefined, [evt("genesis")], 1000);
    const m1 = mineBlock(m0.block, [evt("a")], 1060);
    await append(chainPath, eventLogPath, m0);
    await append(chainPath, eventLogPath, m1);

    // Patch m1's hostTimestamp directly in the chain file. This invalidates
    // the PoW (hash no longer meets target). The chain link to m0 is
    // unaffected (prevHash unchanged), so PoW is the first failure.
    // Layout: [8 file header][80 header 0][80 header 1] — fixed stride.
    //   header: [4 ver][32 prev][32 root][4 ts][4 bits][4 nonce]
    const raw = readFileSync(chainPath);
    const tsOffset = 8 + 1 * HEADER_SIZE + 4 + 32 + 32;
    raw.writeUInt32LE(9999, tsOffset);
    writeFileSync(chainPath, raw);

    const { blocks, report } = loadChain(chainPath, eventLogPath);
    expect(report.ok).toBe(false);
    const issue = report.issues.find((i) => i.code === "POW_INSUFFICIENT");
    expect(issue?.height).toBe(1);
    expect(blocks).toHaveLength(1);
  });

  it("detects event log tampered (eventsRoot mismatch)", async () => {
    const m0 = mineBlock(undefined, [evt("genesis")], 1000);
    const m1 = mineBlock(m0.block, [evt("original")], 1060);
    await append(chainPath, eventLogPath, m0);
    // Append m1 with the WRONG event payload to the event log — the chain
    // header commits to "original" but the log says "tampered".
    await appendBlock(chainPath, eventLogPath, m1.block, [evt("tampered")]);

    const { blocks, report } = loadChain(chainPath, eventLogPath);
    expect(report.ok).toBe(false);
    const issue = report.issues.find((i) => i.code === "EVENTS_ROOT_MISMATCH");
    expect(issue?.height).toBe(1);
    expect(blocks).toHaveLength(1);
  });

  it("detects deleted event batch when header commits to non-empty", async () => {
    const m0 = mineBlock(undefined, [evt("genesis")], 1000);
    const m1 = mineBlock(m0.block, [evt("important")], 1060);
    await append(chainPath, eventLogPath, m0);
    await append(chainPath, eventLogPath, m1);

    // Truncate event log to drop m1's batch. m0's batch is still there;
    // the chain still commits to m1's events.
    const raw = readFileSync(eventLogPath);
    const m0BatchLen = serializeEventBatch(0, m0.events).length;
    writeFileSync(eventLogPath, raw.subarray(0, 8 + m0BatchLen));

    const { blocks, report } = loadChain(chainPath, eventLogPath);
    expect(report.ok).toBe(false);
    const issue = report.issues.find((i) => i.code === "EVENTS_ROOT_MISMATCH");
    expect(issue?.height).toBe(1);
    expect(issue?.detail).toContain("no batch found");
    expect(blocks).toHaveLength(1);
  });

  it("missing event log → warning, chain still validates", async () => {
    const m0 = mineBlock(undefined, [evt("genesis")], 1000);
    const m1 = mineBlock(m0.block, [evt("a")], 1060);
    await append(chainPath, eventLogPath, m0);
    await append(chainPath, eventLogPath, m1);

    // Delete the event log entirely. Chain integrity (linkage, PoW, ts)
    // still holds — just can't verify what was committed.
    rmSync(eventLogPath);

    const { blocks, report } = loadChain(chainPath, eventLogPath);
    expect(report.ok).toBe(true); // Chain itself is fine.
    expect(report.issues).toHaveLength(1);
    expect(report.issues[0].code).toBe("EVENTS_LOG_MISSING");
    expect(blocks).toHaveLength(2); // Both headers loaded.
  });

  it("orphan event batch (height beyond chain) is harmless", async () => {
    // Simulates a crash between event-log write and chain-header write.
    const m0 = mineBlock(undefined, [evt("genesis")], 1000);
    await append(chainPath, eventLogPath, m0);

    // Manually write a batch for height 1 without a corresponding chain
    // header. This is what an interrupted appendBlock leaves behind.
    const orphan = serializeEventBatch(1, [evt("never-committed")]);
    writeFileSync(
      eventLogPath,
      Buffer.concat([readFileSync(eventLogPath), orphan]),
    );

    const { blocks, report } = loadChain(chainPath, eventLogPath);
    expect(report.ok).toBe(true);
    expect(report.issues).toHaveLength(0);
    expect(blocks).toHaveLength(1); // Just m0. Orphan ignored.
  });

  it("flags timestamp regression as warning, not error", async () => {
    const m0 = mineBlock(undefined, [evt("a")], 1000);
    const m1 = mineBlock(m0.block, [evt("b")], 500); // Time went backwards!
    await append(chainPath, eventLogPath, m0);
    await append(chainPath, eventLogPath, m1);

    const { blocks, report } = loadChain(chainPath, eventLogPath);
    expect(report.ok).toBe(true); // Still ok — block is valid.
    expect(report.issues).toHaveLength(1);
    expect(report.issues[0].code).toBe("TIMESTAMP_REGRESSION");
    expect(blocks).toHaveLength(2); // Both blocks loaded.
  });

  it("validateBlock catches stale prevHash on freshly mined block", () => {
    const m0 = mineBlock(undefined, [evt("a")], 1000);
    const m1 = mineBlock(m0.block, [evt("b")], 1060);
    // Validate m1 against the WRONG predecessor.
    const wrongPrev = Buffer.from("ff".repeat(32), "hex");
    const issue = validateBlock(m1.block, wrongPrev, m1.events);
    expect(issue?.code).toBe("PREV_HASH_MISMATCH");
  });

  it("validateBlock passes on a freshly mined block", () => {
    const m0 = mineBlock(undefined, [evt("a")], 1000);
    const m1 = mineBlock(m0.block, [evt("b")], 1060);
    const issue = validateBlock(m1.block, m0.block.hash, m1.events);
    expect(issue).toBeNull();
  });

  it("validateBlock catches a lying worker (wrong cached hash)", () => {
    const m0 = mineBlock(undefined, [evt("a")], 1000);
    const liar: Block = {
      ...m0.block,
      hash: Buffer.from("00".repeat(32), "hex"), // Wrong hash.
    };
    const issue = validateBlock(liar, ZERO_HASH, m0.events);
    // The block's REAL hash is fine (header is valid), but the cached
    // one is a lie.
    expect(issue?.code).toBe("POW_INSUFFICIENT");
    expect(issue?.detail).toContain("cached hash");
  });

  it("validateBlock catches events that don't match the header commitment", () => {
    const m0 = mineBlock(undefined, [evt("a")], 1000);
    // Pass the wrong events for what the header committed to.
    const issue = validateBlock(m0.block, ZERO_HASH, [evt("wrong")]);
    expect(issue?.code).toBe("EVENTS_ROOT_MISMATCH");
  });
});

// ----------------------------------------------------------------------------
// File format internals
// ----------------------------------------------------------------------------

describe("file format", () => {
  let dir: string;
  let chainPath: string;
  let eventLogPath: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "avb-chrono-"));
    chainPath = join(dir, "test.chain");
    eventLogPath = join(dir, "test.events");
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it("chain file starts with magic + version", async () => {
    const m0 = mineBlock(undefined, [], 1000);
    await append(chainPath, eventLogPath, m0);
    const raw = readFileSync(chainPath);
    expect(raw.readUInt32BE(0)).toBe(CHAIN_MAGIC);
    expect(raw.readUInt32LE(4)).toBe(1); // FILE_VERSION
  });

  it("event log starts with its own magic", async () => {
    const m0 = mineBlock(undefined, [evt("x")], 1000);
    await append(chainPath, eventLogPath, m0);
    const raw = readFileSync(eventLogPath);
    expect(raw.readUInt32BE(0)).toBe(EVENTS_MAGIC);
    expect(raw.readUInt32LE(4)).toBe(1);
  });

  it("appendBlock creates parent dirs", async () => {
    const nestedChain = join(dir, "deep", "nested", "chain.bin");
    const nestedEvents = join(dir, "deep", "nested", "events.bin");
    const m0 = mineBlock(undefined, [], 1000);
    await append(nestedChain, nestedEvents, m0);
    const { blocks } = loadChain(nestedChain, nestedEvents);
    expect(blocks).toHaveLength(1);
  });

  it("chain file is fixed-stride (exactly HEADER_SIZE per block)", async () => {
    const m0 = mineBlock(undefined, [], 1000);
    await append(chainPath, eventLogPath, m0);
    const sizeAfter1 = readFileSync(chainPath).length;
    const m1 = mineBlock(m0.block, [evt("x"), evt("y"), evt("z")], 1060);
    await append(chainPath, eventLogPath, m1);
    const sizeAfter2 = readFileSync(chainPath).length;
    // Chain growth is independent of event count.
    expect(sizeAfter2 - sizeAfter1).toBe(HEADER_SIZE);
  });

  it("event log is sparse (empty blocks write nothing)", async () => {
    const m0 = mineBlock(undefined, [evt("genesis")], 1000); // non-empty
    const m1 = mineBlock(m0.block, [], 1060); // EMPTY
    const m2 = mineBlock(m1.block, [], 1120); // EMPTY
    await append(chainPath, eventLogPath, m0);
    await append(chainPath, eventLogPath, m1);
    await append(chainPath, eventLogPath, m2);

    // Event log should contain only m0's batch.
    const raw = readFileSync(eventLogPath);
    expect(raw.length).toBe(8 + serializeEventBatch(0, m0.events).length);

    // And the chain still validates — empty blocks check against EMPTY_EVENTS_ROOT.
    const { blocks, report } = loadChain(chainPath, eventLogPath);
    expect(report.ok).toBe(true);
    expect(blocks).toHaveLength(3);
  });

  it("event log not created if all blocks are empty", async () => {
    const m0 = mineBlock(undefined, [], 1000);
    const m1 = mineBlock(m0.block, [], 1060);
    await append(chainPath, eventLogPath, m0);
    await append(chainPath, eventLogPath, m1);

    expect(existsSync(chainPath)).toBe(true);
    expect(existsSync(eventLogPath)).toBe(false);

    // Loads as EVENTS_LOG_MISSING warning but ok=true; both headers
    // commit to EMPTY_EVENTS_ROOT so there's no actual mismatch even if
    // we COULD verify.
    const { blocks, report } = loadChain(chainPath, eventLogPath);
    expect(report.ok).toBe(true);
    expect(blocks).toHaveLength(2);
    // The warning is technically present but harmless here — every
    // header committed to empty events anyway.
    expect(report.issues[0].code).toBe("EVENTS_LOG_MISSING");
  });
});

// ----------------------------------------------------------------------------
// Runtime event bridge — payload digesting
// ----------------------------------------------------------------------------

describe("digestRuntimeEvent", () => {
  // The bridge sees raw runtime payloads — circular (`runtime` ref),
  // arbitrarily large (full Memory objects), and shape-varied per event
  // type. The digest must NOT throw on any of that; it just extracts what
  // it recognizes and drops the rest. These tests cover the payload shapes
  // that actually flow through @elizaos/core's emitEvent.

  it("always tags the event name even on garbage payloads", () => {
    expect(JSON.parse(digestRuntimeEvent("RUN_ENDED", undefined))).toEqual({
      event: "RUN_ENDED",
    });
    expect(JSON.parse(digestRuntimeEvent("RUN_ENDED", null))).toEqual({
      event: "RUN_ENDED",
    });
    expect(JSON.parse(digestRuntimeEvent("RUN_ENDED", "string"))).toEqual({
      event: "RUN_ENDED",
    });
    expect(JSON.parse(digestRuntimeEvent("RUN_ENDED", 42))).toEqual({
      event: "RUN_ENDED",
    });
  });

  it("survives circular runtime ref (every real payload has one)", () => {
    // emitEvent injects `runtime` into every payload before dispatch.
    // The runtime points back to its own event handler array which
    // closes over the runtime — JSON.stringify would throw here. The
    // digest must skip it.
    const runtime: Record<string, unknown> = { agentId: "agent-1" };
    runtime.self = runtime; // direct cycle for good measure
    const payload = { runtime, source: "test", runId: "run-abc" };

    const out = JSON.parse(digestRuntimeEvent("RUN_STARTED", payload));
    expect(out.event).toBe("RUN_STARTED");
    expect(out.source).toBe("test");
    expect(out.runId).toBe("run-abc");
    expect(out).not.toHaveProperty("runtime");
    expect(out).not.toHaveProperty("self");
  });

  it("extracts RunEventPayload identifying fields", () => {
    const out = JSON.parse(
      digestRuntimeEvent("RUN_ENDED", {
        runtime: {},
        source: "agent",
        runId: "11111111-1111-1111-1111-111111111111",
        messageId: "22222222-2222-2222-2222-222222222222",
        roomId: "33333333-3333-3333-3333-333333333333",
        entityId: "44444444-4444-4444-4444-444444444444",
        status: "completed",
        startTime: 1700000000000, // number, not string — should be skipped
        endTime: 1700000060000n, // bigint — definitely should be skipped
      }),
    );
    expect(out).toEqual({
      event: "RUN_ENDED",
      source: "agent",
      runId: "11111111-1111-1111-1111-111111111111",
      messageId: "22222222-2222-2222-2222-222222222222",
      roomId: "33333333-3333-3333-3333-333333333333",
      entityId: "44444444-4444-4444-4444-444444444444",
      status: "completed",
    });
  });

  it("unpacks nested Memory from MessagePayload (id, room, text length only)", () => {
    // The chronometer is a witness, not an archive. We prove a message
    // happened and how big it was — NOT what it said. PII stays out of
    // a file that's specifically designed to be hard to delete.
    const out = JSON.parse(
      digestRuntimeEvent("MESSAGE_SENT", {
        runtime: {},
        source: "discord",
        message: {
          id: "msg-001",
          roomId: "room-99",
          entityId: "user-7",
          content: {
            text: "the actual message body which we do NOT store",
            attachments: [{ id: "att-1", url: "https://huge.example/blob" }],
          },
          embedding: new Array(1536).fill(0.1), // huge, must not appear
        },
      }),
    );
    expect(out.event).toBe("MESSAGE_SENT");
    expect(out.source).toBe("discord");
    expect(out.messageId).toBe("msg-001");
    expect(out.roomId).toBe("room-99");
    expect(out.textLen).toBe(45);
    // Verify the privacy boundary held.
    const json = JSON.stringify(out);
    expect(json).not.toContain("actual message body");
    expect(json).not.toContain("huge.example");
    expect(json).not.toContain("0.1");
  });

  it("top-level roomId wins over nested message.roomId", () => {
    // Some payloads (ActionEventPayload) have roomId at top level AND
    // pass a message. Top-level is the authoritative one for the event;
    // the message's roomId might be where it was originally posted.
    const out = JSON.parse(
      digestRuntimeEvent("ACTION_STARTED", {
        roomId: "top-level-room",
        message: { id: "m1", roomId: "nested-room", content: {} },
      }),
    );
    expect(out.roomId).toBe("top-level-room");
    expect(out.messageId).toBe("m1");
  });

  it("extracts action names from content.actions", () => {
    const out = JSON.parse(
      digestRuntimeEvent("ACTION_COMPLETED", {
        runtime: {},
        roomId: "r1",
        content: {
          text: "ignored",
          actions: ["REPLY", "FOLLOW_UP", 123, null, "MUTE"], // mixed junk
        },
      }),
    );
    // Non-string entries silently dropped — defensive against plugin bugs.
    expect(out.actions).toEqual(["REPLY", "FOLLOW_UP", "MUTE"]);
  });

  it("caps output at 512 bytes and emits valid JSON when truncated", () => {
    // worldId is on the extraction allowlist, so an absurdly long one
    // would otherwise pass straight through. The cap is the last line
    // of defense.
    const out = digestRuntimeEvent("WORLD_JOINED", {
      worldId: "w".repeat(2000),
    });
    expect(out.length).toBeLessThanOrEqual(512);
    // Critically: the truncated string must still parse. Otherwise a
    // pathological payload could poison every downstream consumer of
    // the event log.
    const parsed = JSON.parse(out);
    expect(parsed.event).toBe("WORLD_JOINED");
  });

  it("ignores non-string values in ID-typed fields", () => {
    // The extractor's `typeof p[key] === "string"` guard exists because
    // payload shapes are duck-typed across plugins. A plugin that puts
    // an object where we expect a UUID shouldn't break the bridge.
    const out = JSON.parse(
      digestRuntimeEvent("MODEL_USED", {
        runId: { not: "a string" },
        roomId: 12345,
        entityId: ["array"],
        source: "valid", // this one's fine
      }),
    );
    expect(out).toEqual({ event: "MODEL_USED", source: "valid" });
  });
});
