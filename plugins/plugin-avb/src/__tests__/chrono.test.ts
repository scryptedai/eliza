/**
 * Chronometer unit tests.
 *
 * Covers:
 *  - block header round-trip serialization
 *  - event payload round-trip
 *  - compact difficulty (nBits) encode/decode + retarget clamping
 *  - chain parsing + integrity verification (tamper detection)
 *  - ChronometerService genesis + integrity-flag emission (miner disabled)
 */

import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { EventType } from "@elizaos/core";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  bitsToTarget,
  bytesEqual,
  decodeEvents,
  deserializeHeader,
  encodeEvent,
  encodeEvents,
  HEADER_SIZE,
  hashHeader,
  INITIAL_DIFFICULTY_BITS,
  meetsTarget,
  PROTOCOL_VERSION,
  retarget,
  serializeHeader,
  sha256,
  sha256d,
  splitEvents,
  targetToBits,
  ZERO_HASH,
} from "../chrono/block.ts";
import {
  CHRONO_BRIDGE_FORWARDED,
  CHRONO_BRIDGE_SKIPPED,
  chronoRuntimeEvents,
} from "../chrono/bridge.ts";
import {
  appendBlockBytes,
  appendEventSegment,
  parseChain,
  verifyBlocks,
  verifySegments,
} from "../chrono/chain.ts";
import { merkleProof, merkleRoot, rootFromProof } from "../chrono/merkle.ts";
import {
  CHRONO_INTEGRITY_EVENT,
  CHRONO_SERVICE_TYPE,
  ChronometerService,
  ENV_CHRONO_DATA_DIR,
  ENV_CHRONO_ENABLED,
  verifyInclusion,
} from "../chrono/service.ts";
import {
  type BlockHeader,
  type ChronoEvent,
  ChronoEventType,
} from "../chrono/types.ts";

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/** Build + mine a header at INITIAL_DIFFICULTY_BITS (trivial — usually nonce 0). */
function mineHeader(partial: Omit<BlockHeader, "version" | "nonce">): {
  headerBytes: Uint8Array;
  hash: Uint8Array;
} {
  const header: BlockHeader = {
    version: PROTOCOL_VERSION,
    height: partial.height,
    prevHash: partial.prevHash,
    eventsHash: partial.eventsHash,
    hostTimestampMs: partial.hostTimestampMs,
    wallElapsedMs: partial.wallElapsedMs,
    difficultyBits: partial.difficultyBits,
    nonce: 0n,
    eventCount: partial.eventCount,
    eventBytes: partial.eventBytes,
  };
  let bytes = serializeHeader(header);
  let hash = sha256d(bytes);
  const target = bitsToTarget(header.difficultyBits);
  let n = 0n;
  // INITIAL_DIFFICULTY_BITS accepts ~99.6% of hashes, so this terminates fast.
  while (hashGtTarget(hash, target)) {
    n++;
    header.nonce = n;
    bytes = serializeHeader(header);
    hash = sha256d(bytes);
  }
  return { headerBytes: bytes, hash };
}

function hashGtTarget(hash: Uint8Array, target: bigint): boolean {
  let v = 0n;
  for (const b of hash) v = (v << 8n) | BigInt(b);
  return v > target;
}

function concat(...parts: Uint8Array[]): Uint8Array {
  let len = 0;
  for (const p of parts) len += p.length;
  const out = new Uint8Array(len);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

// ----------------------------------------------------------------------------
// Header serialization
// ----------------------------------------------------------------------------

describe("chrono/block: header serialization", () => {
  it("round-trips all fields", () => {
    const h: BlockHeader = {
      version: 1,
      height: 42,
      prevHash: sha256(Uint8Array.of(1, 2, 3)),
      eventsHash: sha256(Uint8Array.of(4, 5, 6)),
      hostTimestampMs: 1_700_000_000_123,
      wallElapsedMs: 59_321,
      difficultyBits: 0x1d00ffff,
      nonce: 0xdeadbeefn,
      eventCount: 7,
      eventBytes: 99,
    };
    const bytes = serializeHeader(h);
    expect(bytes.length).toBe(HEADER_SIZE);
    expect(bytes[0]).toBe(0x41); // 'A'
    expect(bytes[3]).toBe(0x43); // 'C'
    const back = deserializeHeader(bytes);
    expect(back.version).toBe(h.version);
    expect(back.height).toBe(h.height);
    expect(bytesEqual(back.prevHash, h.prevHash)).toBe(true);
    expect(bytesEqual(back.eventsHash, h.eventsHash)).toBe(true);
    expect(back.hostTimestampMs).toBe(h.hostTimestampMs);
    expect(back.wallElapsedMs).toBe(h.wallElapsedMs);
    expect(back.difficultyBits).toBe(h.difficultyBits);
    expect(back.nonce).toBe(h.nonce);
    expect(back.eventCount).toBe(h.eventCount);
    expect(back.eventBytes).toBe(h.eventBytes);
  });

  it("hashHeader is deterministic and changes with nonce", () => {
    const base: BlockHeader = {
      version: 1,
      height: 0,
      prevHash: ZERO_HASH,
      eventsHash: ZERO_HASH,
      hostTimestampMs: 0,
      wallElapsedMs: 0,
      difficultyBits: INITIAL_DIFFICULTY_BITS,
      nonce: 0n,
      eventCount: 0,
      eventBytes: 0,
    };
    const a = hashHeader(base);
    const b = hashHeader({ ...base, nonce: 1n });
    expect(bytesEqual(a, hashHeader(base))).toBe(true);
    expect(bytesEqual(a, b)).toBe(false);
  });
});

// ----------------------------------------------------------------------------
// Event encoding
// ----------------------------------------------------------------------------

describe("chrono/block: event encoding", () => {
  it("round-trips events including UTF-8 details", () => {
    const events = [
      { type: ChronoEventType.BOOT, hostTimestampMs: 1000, detail: "boot" },
      {
        type: ChronoEventType.PHASE_COMPLETE,
        hostTimestampMs: 2000,
        detail: "run=αβγ phase=TEXT",
      },
      { type: ChronoEventType.OTHER, hostTimestampMs: 3000, detail: "" },
    ];
    const enc = encodeEvents(events);
    const dec = decodeEvents(enc);
    expect(dec).toHaveLength(3);
    expect(dec[0]).toEqual(events[0]);
    expect(dec[1]).toEqual(events[1]);
    expect(dec[2]).toEqual(events[2]);
  });
});

// ----------------------------------------------------------------------------
// Difficulty
// ----------------------------------------------------------------------------

describe("chrono/block: difficulty", () => {
  it("bitsToTarget / targetToBits round-trip on canonical values", () => {
    for (const bits of [0x1d00ffff, 0x1b0404cb, INITIAL_DIFFICULTY_BITS]) {
      const t = bitsToTarget(bits);
      expect(targetToBits(t)).toBe(bits >>> 0);
    }
  });

  it("retarget tightens when blocks are too fast and loosens when slow", () => {
    const base = 0x1d00ffff;
    const t = bitsToTarget(base);
    // Fast block (10s vs 60s target) → smaller target (harder)
    const tighter = bitsToTarget(retarget(base, 10_000, 60_000));
    expect(tighter < t).toBe(true);
    // Slow block (300s vs 60s) → larger target (easier), clamped to 4×
    const looser = bitsToTarget(retarget(base, 300_000, 60_000));
    expect(looser > t).toBe(true);
    expect(looser <= t * 4n).toBe(true);
  });

  it("retarget clamps adjustment to [1/4, 4]", () => {
    const base = 0x1d00ffff;
    const t = bitsToTarget(base);
    const veryFast = bitsToTarget(retarget(base, 1, 60_000));
    expect(veryFast >= t / 4n - 1n).toBe(true);
    const verySlow = bitsToTarget(retarget(base, 10_000_000, 60_000));
    expect(verySlow <= t * 4n).toBe(true);
  });

  it("meetsTarget agrees with arithmetic comparison", () => {
    expect(meetsTarget(new Uint8Array(32), 0x03000001)).toBe(true); // hash 0 ≤ 1
    const big = new Uint8Array(32).fill(0xff);
    expect(meetsTarget(big, 0x03000001)).toBe(false);
  });
});

// ----------------------------------------------------------------------------
// Chain parsing + integrity
// ----------------------------------------------------------------------------

describe("chrono/merkle", () => {
  function ev(detail: string): ChronoEvent {
    return { type: ChronoEventType.OTHER, hostTimestampMs: 1, detail };
  }

  it("root over a single leaf equals leafHash; empty → ZERO_HASH", () => {
    const e = encodeEvent(ev("only"));
    const root1 = merkleRoot([e]);
    expect(bytesEqual(root1, rootFromProof(e, []))).toBe(true);
    expect(bytesEqual(merkleRoot([]), ZERO_HASH)).toBe(true);
  });

  it("proof verifies for every leaf and rejects a tampered event", () => {
    const events = ["a", "bb", "ccc", "dddd", "eeeee"].map((d) =>
      encodeEvent(ev(d)),
    );
    const root = merkleRoot(events);
    for (let i = 0; i < events.length; i++) {
      const proof = merkleProof(events, i);
      expect(bytesEqual(rootFromProof(events[i], proof), root)).toBe(true);
    }
    const proof2 = merkleProof(events, 2);
    const tampered = encodeEvent(ev("CCC")); // different leaf
    expect(bytesEqual(rootFromProof(tampered, proof2), root)).toBe(false);
  });

  it("splitEvents recovers per-event encodings → identical root", () => {
    const events = ["x", "y", "z"].map((d) => encodeEvent(ev(d)));
    const seg = concat(...events);
    const back = splitEvents(seg);
    expect(back).toHaveLength(3);
    expect(bytesEqual(merkleRoot(back), merkleRoot(events))).toBe(true);
  });
});

// ----------------------------------------------------------------------------
// Chain parsing + integrity (v2: header-only chain, off-chain segments)
// ----------------------------------------------------------------------------

describe("chrono/chain: parse + verify", () => {
  /**
   * Build a v2 chain of `n` blocks: each block commits to two events
   * via Merkle root; chain bytes are headers only.
   */
  function buildChain(n: number): {
    raw: Uint8Array;
    headers: Uint8Array[];
    segments: Map<number, Uint8Array>;
  } {
    const headers: Uint8Array[] = [];
    const segments = new Map<number, Uint8Array>();
    let prev: Uint8Array = ZERO_HASH;
    let hostTs = 1_700_000_000_000;
    for (let h = 0; h < n; h++) {
      const evs: ChronoEvent[] = [
        {
          type: h === 0 ? ChronoEventType.GENESIS : ChronoEventType.OTHER,
          hostTimestampMs: hostTs,
          detail: `block ${h} a`,
        },
        {
          type: ChronoEventType.OTHER,
          hostTimestampMs: hostTs + 1,
          detail: `block ${h} b`,
        },
      ];
      const leaves = evs.map(encodeEvent);
      const seg = concat(...leaves);
      const { headerBytes, hash } = mineHeader({
        height: h,
        prevHash: prev,
        eventsHash: merkleRoot(leaves),
        hostTimestampMs: hostTs,
        wallElapsedMs: h === 0 ? 0 : 60_000,
        difficultyBits: INITIAL_DIFFICULTY_BITS,
        eventCount: evs.length,
        eventBytes: seg.length,
      });
      headers.push(headerBytes);
      segments.set(h, seg);
      prev = hash;
      hostTs += 60_000;
    }
    return { raw: concat(...headers), headers, segments };
  }

  const rootOf = (seg: Uint8Array) => merkleRoot(splitEvents(seg));

  it("parses a valid 3-block v2 chain with no issues", () => {
    const { raw, segments } = buildChain(3);
    const { blocks, issues } = parseChain(raw);
    expect(blocks).toHaveLength(3);
    expect(issues).toHaveLength(0);
    expect(blocks[0].payload.length).toBe(0); // v2: no inline payload
    expect(blocks[2].header.height).toBe(2);
    expect(bytesEqual(blocks[1].header.prevHash, blocks[0].hash)).toBe(true);
    // Off-chain segments cross-check cleanly.
    expect(verifySegments(blocks, segments, rootOf)).toHaveLength(0);
  });

  it("flags MERKLE_ROOT_MISMATCH when an off-chain segment is tampered", () => {
    const { raw, segments } = buildChain(3);
    const { blocks } = parseChain(raw);
    const seg1 = segments.get(1)!.slice();
    seg1[seg1.length - 1] ^= 0xff; // flip last detail byte
    const tampered = new Map(segments);
    tampered.set(1, seg1);
    const issues = verifySegments(blocks, tampered, rootOf);
    expect(issues.map((i) => i.code)).toEqual(["MERKLE_ROOT_MISMATCH"]);
    expect(issues[0].height).toBe(1);
  });

  it("flags MERKLE_ROOT_MISMATCH when an off-chain segment is missing", () => {
    const { raw, segments } = buildChain(2);
    const { blocks } = parseChain(raw);
    const partial = new Map(segments);
    partial.delete(1);
    const issues = verifySegments(blocks, partial, rootOf);
    expect(issues.map((i) => i.code)).toEqual(["MERKLE_ROOT_MISMATCH"]);
  });

  it("flags PREV_HASH_MISMATCH and HEIGHT_MISMATCH when a header is tampered", () => {
    const { raw } = buildChain(3);
    // Flip the height field of block 1 (offset = HEADER_SIZE + 8).
    const mutated = raw.slice();
    mutated[HEADER_SIZE + 8] = 99;
    const { issues } = parseChain(mutated);
    const codes = issues.map((i) => i.code);
    expect(codes).toContain("HEIGHT_MISMATCH");
    expect(codes).toContain("PREV_HASH_MISMATCH");
  });

  it("flags HOST_CLOCK_REGRESSION when timestamps go backward", () => {
    const b0 = mineHeader({
      height: 0,
      prevHash: ZERO_HASH,
      eventsHash: ZERO_HASH,
      hostTimestampMs: 2000,
      wallElapsedMs: 0,
      difficultyBits: INITIAL_DIFFICULTY_BITS,
      eventCount: 0,
      eventBytes: 0,
    });
    const b1 = mineHeader({
      height: 1,
      prevHash: b0.hash,
      eventsHash: ZERO_HASH,
      hostTimestampMs: 1000, // < 2000
      wallElapsedMs: 60_000,
      difficultyBits: INITIAL_DIFFICULTY_BITS,
      eventCount: 0,
      eventBytes: 0,
    });
    const { issues } = parseChain(concat(b0.headerBytes, b1.headerBytes));
    expect(issues.map((i) => i.code)).toContain("HOST_CLOCK_REGRESSION");
  });

  it("flags BAD_MAGIC and stops on lost framing", () => {
    const { raw } = buildChain(2);
    const mutated = raw.slice();
    mutated[0] = 0x00;
    const { blocks, issues } = parseChain(mutated);
    expect(blocks).toHaveLength(0);
    expect(issues[0].code).toBe("BAD_MAGIC");
  });

  it("flags TRUNCATED on partial trailing block", () => {
    const { raw } = buildChain(2);
    const cut = raw.slice(0, raw.length - 5);
    const { blocks, issues } = parseChain(cut);
    expect(blocks.length).toBeLessThan(2);
    expect(issues.some((i) => i.code === "TRUNCATED")).toBe(true);
  });

  it("verifyBlocks() on parsed valid chain returns no issues", () => {
    const { raw } = buildChain(4);
    const { blocks } = parseChain(raw);
    expect(verifyBlocks(blocks)).toHaveLength(0);
  });

  it("v1 blocks still parse with inline payload and EVENTS_HASH check", () => {
    // Hand-build one v1 block (version field forced to 1).
    const payload = encodeEvents([
      { type: ChronoEventType.GENESIS, hostTimestampMs: 1, detail: "v1" },
    ]);
    const header: BlockHeader = {
      version: 1,
      height: 0,
      prevHash: ZERO_HASH,
      eventsHash: sha256(payload),
      hostTimestampMs: 1,
      wallElapsedMs: 0,
      difficultyBits: INITIAL_DIFFICULTY_BITS,
      nonce: 0n,
      eventCount: 1,
      eventBytes: payload.length,
    };
    let bytes = serializeHeader(header);
    while (
      hashGtTarget(sha256d(bytes), bitsToTarget(INITIAL_DIFFICULTY_BITS))
    ) {
      header.nonce++;
      bytes = serializeHeader(header);
    }
    const ok = parseChain(concat(bytes, payload));
    expect(ok.blocks).toHaveLength(1);
    expect(ok.issues).toHaveLength(0);
    // Tamper payload → EVENTS_HASH_MISMATCH (v1 path).
    const bad = payload.slice();
    bad[0] ^= 0xff;
    const broken = parseChain(concat(bytes, bad));
    expect(broken.issues.map((i) => i.code)).toContain("EVENTS_HASH_MISMATCH");
  });
});

// ----------------------------------------------------------------------------
// ChronometerService (miner disabled)
// ----------------------------------------------------------------------------

describe("ChronometerService", () => {
  const tmpDirs: string[] = [];

  afterEach(async () => {
    while (tmpDirs.length) {
      const d = tmpDirs.pop();
      if (d) await fs.rm(d, { recursive: true, force: true });
    }
  });

  async function mkDataDir(): Promise<string> {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "avb-chrono-"));
    tmpDirs.push(dir);
    return dir;
  }

  function makeRuntime(dataDir: string) {
    const events: Array<{ name: string; payload: unknown }> = [];
    const settings: Record<string, string> = {
      [ENV_CHRONO_DATA_DIR]: dataDir,
      [ENV_CHRONO_ENABLED]: "false", // never spawn the worker in unit tests
    };
    const rt = {
      agentId: "agent-test",
      logger: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        debug: vi.fn(),
      },
      getSetting: (k: string) => settings[k],
      emitEvent: async (name: string, payload: unknown) => {
        events.push({ name, payload });
      },
    };
    return { rt, events, settings };
  }

  it("mints genesis on first start and persists chain + event log", async () => {
    const dir = await mkDataDir();
    const { rt } = makeRuntime(dir);

    const svc = await ChronometerService.start(rt as never);
    expect(svc.getHeight()).toBe(0);
    expect(svc.getIntegrityIssues()).toHaveLength(0);
    expect(svc.getDataDir()).toBe(dir);
    expect(svc.getChainPath()).toBe(path.join(dir, "logs", "events.chain"));
    expect(svc.getEventsPath()).toBe(path.join(dir, "logs", "events.bin"));

    // v2: chain file is exactly one 108-byte header, no inline payload.
    const chainStat = await fs.stat(svc.getChainPath());
    expect(chainStat.size).toBe(HEADER_SIZE);
    const evStat = await fs.stat(svc.getEventsPath());
    expect(evStat.size).toBeGreaterThan(8); // frame + ≥1 encoded event

    // Off-chain log cross-checks cleanly against the sealed Merkle root.
    expect(await svc.verifyEventLog()).toHaveLength(0);

    // Restart against same dir → loads tip, no second genesis.
    const { rt: rt2 } = makeRuntime(dir);
    const svc2 = await ChronometerService.start(rt2 as never);
    expect(svc2.getHeight()).toBe(0);
    expect(svc2.getIntegrityIssues()).toHaveLength(0);
    expect((await fs.stat(svc2.getChainPath())).size).toBe(HEADER_SIZE);
    await svc.stop();
    await svc2.stop();
  });

  it("flags MERKLE_ROOT_MISMATCH when the off-chain event log is tampered", async () => {
    const dir = await mkDataDir();

    // First instance mints genesis (1 event) → sealed Merkle root in header.
    {
      const { rt } = makeRuntime(dir);
      const svc = await ChronometerService.start(rt as never);
      await svc.stop();
    }

    // Flip the last byte of events.bin (inside the GENESIS event's detail
    // string — past the 8-byte segment frame and the 11-byte event header,
    // so framing/length stay intact and only the leaf hash changes).
    const eventsPath = path.join(dir, "logs", "events.bin");
    const raw = new Uint8Array(await fs.readFile(eventsPath));
    expect(raw.length).toBeGreaterThan(8 + 11);
    raw[raw.length - 1] ^= 0xff;
    await fs.writeFile(eventsPath, raw);

    // Fresh instance: header chain still verifies; on-demand event-log
    // verification catches the divergence and raises a warning flag.
    const { rt, events } = makeRuntime(dir);
    const svc = await ChronometerService.start(rt as never);
    expect(svc.getIntegrityIssues()).toHaveLength(0); // header-only checks ok

    const found = await svc.verifyEventLog();
    expect(found.map((i) => i.code)).toContain("MERKLE_ROOT_MISMATCH");
    expect(svc.getIntegrityIssues().map((i) => i.code)).toContain(
      "MERKLE_ROOT_MISMATCH",
    );
    expect(events.some((e) => e.name === CHRONO_INTEGRITY_EVENT)).toBe(true);
    expect(rt.logger.warn).toHaveBeenCalled();
    await svc.stop();
  });

  it("flags PREV_HASH_MISMATCH when the header chain itself is tampered", async () => {
    const dir = await mkDataDir();
    const chainPath = path.join(dir, "logs", "events.chain");
    const eventsPath = path.join(dir, "logs", "events.bin");

    // Hand-build a valid 2-block v2 chain + matching off-chain segments.
    const seg0 = encodeEvents([
      { type: ChronoEventType.GENESIS, hostTimestampMs: 1, detail: "g" },
    ]);
    const b0 = mineHeader({
      height: 0,
      prevHash: ZERO_HASH,
      eventsHash: merkleRoot(splitEvents(seg0)),
      hostTimestampMs: 1,
      wallElapsedMs: 0,
      difficultyBits: INITIAL_DIFFICULTY_BITS,
      eventCount: 1,
      eventBytes: seg0.length,
    });
    const seg1 = encodeEvents([
      { type: ChronoEventType.OTHER, hostTimestampMs: 60_001, detail: "ok" },
    ]);
    const b1 = mineHeader({
      height: 1,
      prevHash: b0.hash,
      eventsHash: merkleRoot(splitEvents(seg1)),
      hostTimestampMs: 60_001,
      wallElapsedMs: 60_000,
      difficultyBits: INITIAL_DIFFICULTY_BITS,
      eventCount: 1,
      eventBytes: seg1.length,
    });
    const empty = new Uint8Array(0);
    await appendBlockBytes(chainPath, b0.headerBytes, empty);
    await appendBlockBytes(chainPath, b1.headerBytes, empty);
    await appendEventSegment(eventsPath, 0, seg0);
    await appendEventSegment(eventsPath, 1, seg1);

    // Corrupt one byte of block 1's prevHash field (offset 12 in header).
    const raw = new Uint8Array(await fs.readFile(chainPath));
    raw[HEADER_SIZE + 12] ^= 0xff;
    await fs.writeFile(chainPath, raw);

    const { rt, events } = makeRuntime(dir);
    const svc = await ChronometerService.start(rt as never);

    const codes = svc.getIntegrityIssues().map((i) => i.code);
    expect(codes).toContain("PREV_HASH_MISMATCH");
    expect(events.some((e) => e.name === CHRONO_INTEGRITY_EVENT)).toBe(true);
    expect(rt.logger.warn).toHaveBeenCalled();
    // Off-chain log itself is untouched and still matches block 0's root;
    // block 1's header.eventsHash is unchanged so its segment still matches too.
    expect(await svc.verifyEventLog()).toHaveLength(0);
    await svc.stop();
  });

  it("proveInclusion → verifyInclusion round-trips and rejects tamper", async () => {
    const dir = await mkDataDir();
    const { rt } = makeRuntime(dir);
    const svc = await ChronometerService.start(rt as never);

    const proof = await svc.proveInclusion(0, 0);
    expect(proof.header.height).toBe(0);
    expect(decodeEvents(proof.encodedEvent)[0].type).toBe(
      ChronoEventType.GENESIS,
    );
    expect(verifyInclusion(proof)).toBe(true);

    // Mutated event no longer reproduces the sealed root.
    const forged = {
      ...proof,
      encodedEvent: (() => {
        const e = proof.encodedEvent.slice();
        e[e.length - 1] ^= 0x01;
        return e;
      })(),
    };
    expect(verifyInclusion(forged)).toBe(false);

    await expect(svc.proveInclusion(0, 99)).rejects.toThrow(RangeError);
    await expect(svc.proveInclusion(99, 0)).rejects.toThrow();
    await svc.stop();
  });

  it("recordEvent buffers events for the next block", async () => {
    const dir = await mkDataDir();
    const { rt } = makeRuntime(dir);
    const svc = await ChronometerService.start(rt as never);
    svc.recordEvent(ChronoEventType.PHASE_COMPLETE, "run=x phase=TEXT");
    // Pending events are private; verify indirectly via intrinsic time
    // (still 0 — nothing sealed yet) and that no error was thrown.
    expect(svc.getIntrinsicTimeMs()).toBe(0);
    await svc.stop();
  });
});

// ----------------------------------------------------------------------------
// Runtime → Chronometer event bridge
// ----------------------------------------------------------------------------

describe("chrono/bridge", () => {
  /** Fetch the (single) registered handler for an event type. */
  function handler<K extends keyof typeof chronoRuntimeEvents>(
    k: K,
  ): NonNullable<(typeof chronoRuntimeEvents)[K]>[number] {
    const hs = chronoRuntimeEvents[k];
    if (!hs || hs.length === 0) throw new Error(`no handler for ${String(k)}`);
    return hs[0];
  }

  function makeBridgeRuntime() {
    const recorded: Array<{ type: ChronoEventType; detail: string }> = [];
    const svc = {
      recordEvent: (type: ChronoEventType, detail: string) => {
        recorded.push({ type, detail });
      },
    };
    const runtime = {
      agentId: "agent-test",
      getService: (t: string) => (t === CHRONO_SERVICE_TYPE ? svc : null),
    };
    return { runtime, recorded };
  }

  it("forwards notable runtime events as RUNTIME chrono events", async () => {
    const { runtime, recorded } = makeBridgeRuntime();

    // MESSAGE_RECEIVED
    await handler(EventType.MESSAGE_RECEIVED)({
      runtime,
      message: {
        id: "msg-aaaaaaaa-bbbb",
        roomId: "room-1111-2222",
        entityId: "ent-9999-8888",
        content: { text: "hello world ".repeat(20), actions: ["REPLY"] },
      },
    } as never);

    // RUN_ENDED
    await handler(EventType.RUN_ENDED)({
      runtime,
      runId: "run-1234-5678",
      messageId: "m",
      roomId: "room-1111-2222",
      entityId: "e",
      startTime: 0,
      status: "completed",
      duration: 1234,
    } as never);

    // MODEL_USED
    await handler(EventType.MODEL_USED)({
      runtime,
      type: "TEXT_LARGE",
      tokens: { prompt: 100, completion: 50, total: 150 },
    } as never);

    expect(recorded).toHaveLength(3);
    for (const r of recorded) {
      expect(r.type).toBe(ChronoEventType.RUNTIME);
      expect(r.detail.length).toBeLessThanOrEqual(240);
    }
    expect(recorded[0].detail).toMatch(/^MESSAGE_RECEIVED /);
    expect(recorded[0].detail).toContain("room=room-111");
    expect(recorded[0].detail).toContain("from=ent-9999");
    expect(recorded[0].detail).toContain("actions=REPLY");
    // Long text was clipped, not stored verbatim.
    expect(recorded[0].detail).toContain("…");

    expect(recorded[1].detail).toMatch(/^RUN_ENDED /);
    expect(recorded[1].detail).toContain("run=run-1234");
    expect(recorded[1].detail).toContain("status=completed");
    expect(recorded[1].detail).toContain("dur=1234ms");

    expect(recorded[2].detail).toMatch(/^MODEL_USED /);
    expect(recorded[2].detail).toContain("type=TEXT_LARGE");
    expect(recorded[2].detail).toContain("tok=150");
  });

  it("is a silent no-op when ChronometerService is not registered", async () => {
    const runtime = {
      agentId: "agent-test",
      getService: () => null,
    };
    await expect(
      handler(EventType.MESSAGE_SENT)({
        runtime,
        message: { content: {} },
      } as never),
    ).resolves.toBeUndefined();
  });

  it("swallows summariser errors instead of throwing into the runtime", async () => {
    const { runtime, recorded } = makeBridgeRuntime();
    // WORLD_JOINED summariser dereferences p.world.id and p.rooms.length —
    // pass garbage and confirm it still records something.
    await handler(EventType.WORLD_JOINED)({
      runtime,
      world: undefined,
      rooms: undefined,
      entities: undefined,
    } as never);
    expect(recorded).toHaveLength(1);
    expect(recorded[0].detail).toMatch(/^WORLD_JOINED /);
  });

  it("partitions every core EventType into forwarded or skipped", () => {
    const all = Object.values(EventType) as EventType[];
    const fwd = new Set(CHRONO_BRIDGE_FORWARDED);
    const skip = new Set(CHRONO_BRIDGE_SKIPPED);
    // No overlap.
    for (const e of fwd) expect(skip.has(e)).toBe(false);
    // Every core event is accounted for one way or the other, so a
    // newly-added EventType in @elizaos/core fails this test until a
    // human decides whether it's notable or spam.
    for (const e of all) {
      expect(fwd.has(e) || skip.has(e)).toBe(true);
    }
  });
});
