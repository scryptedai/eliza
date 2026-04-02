/**
 * FfmService unit tests.
 *
 * Strategy (mirrors plugin-avb's service.test.ts):
 *   - build a fake runtime satisfying FfmRuntimeSurface
 *   - build a fake ScryptedAILike that captures listeners and lets the
 *     test fire terminal events manually
 *   - run FfmService.start() against the fake runtime, then drive the
 *     scryptedai listener to verify parse → record → fan-out
 *
 * No timers, no real I/O. The only async is the start()/start*() promise
 * machinery — all settled within microtask boundaries.
 */

import type { IAgentRuntime } from "@elizaos/core";
import type { NormalizedJobResult } from "@elizaos/plugin-scryptedai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { FfmService } from "../service.ts";
import type { ExpansionRecord, FfmProfile, ScryptedAILike } from "../types.ts";

// ----------------------------------------------------------------------------
// Fakes
// ----------------------------------------------------------------------------

type TerminalListener = (r: NormalizedJobResult) => void;

interface FakeScrypted extends ScryptedAILike {
  /** Test control surface — fire a terminal event into all subscribers. */
  __fire: (r: NormalizedJobResult) => void;
  /** Test control surface — captured payloads from startTextGeneration calls. */
  __payloads: Array<{
    inputData: Record<string, unknown>;
    opts?: { pollFallback?: boolean; metadata?: Record<string, unknown> };
  }>;
  __listeners: Set<TerminalListener>;
}

function makeFakeScrypted(): FakeScrypted {
  const listeners = new Set<TerminalListener>();
  const payloads: FakeScrypted["__payloads"] = [];
  let nextJobId = 1;

  return {
    __listeners: listeners,
    __payloads: payloads,
    __fire(r) {
      for (const l of listeners) l(r);
    },
    // biome-ignore lint/suspicious/useAwait: signature parity with the real service
    async startTextGeneration(inputData, opts) {
      payloads.push({ inputData, opts });
      return { jobId: `job-${nextJobId++}` };
    },
    onTerminal(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

interface FakeRuntime {
  agentId: string;
  logger: {
    info: ReturnType<typeof vi.fn>;
    warn: ReturnType<typeof vi.fn>;
    error: ReturnType<typeof vi.fn>;
    debug: ReturnType<typeof vi.fn>;
  };
  getServiceLoadPromise<T = unknown>(serviceType: string): Promise<T>;
  /** test control */
  __scrypted: FakeScrypted;
}

function makeFakeRuntime(opts?: { scryptedFails?: boolean }): FakeRuntime {
  const scrypted = makeFakeScrypted();
  return {
    agentId: "test-agent-0000",
    __scrypted: scrypted,
    logger: {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      debug: vi.fn(),
    },
    // biome-ignore lint/suspicious/useAwait: test fake
    async getServiceLoadPromise<T>(serviceType: string): Promise<T> {
      if (opts?.scryptedFails) {
        throw new Error("scryptedai not available");
      }
      if (serviceType === "scryptedai") {
        return scrypted as unknown as T;
      }
      throw new Error(`unexpected service type: ${serviceType}`);
    },
  };
}

// Cast helper — FfmService.start() takes IAgentRuntime; our fake satisfies
// FfmRuntimeSurface (the structural subset the service actually uses).
function asRuntime(rt: FakeRuntime): IAgentRuntime {
  return rt as unknown as IAgentRuntime;
}

// ----------------------------------------------------------------------------
// LLM output fixtures (valid JSON the parsers will accept)
// ----------------------------------------------------------------------------

const NARRATIVE_JSON = JSON.stringify({
  bio: ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"],
  beliefs: ["belief one", "belief two", "belief three", "belief four"],
  adjectives: ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"],
  topics: ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"],
});

const VOICE_JSON = JSON.stringify({
  messageExamples: Array.from({ length: 6 }, (_, i) => [
    { name: "user", content: { text: `user msg ${i}` } },
    {
      name: "Eliza",
      content: { text: `agent reply variant ${i} unique words` },
    },
  ]),
  postExamples: ["p1", "p2", "p3", "p4", "p5", "p6"],
  style: {
    all: ["s1", "s2", "s3"],
    chat: ["c1", "c2"],
    post: ["po1", "po2"],
  },
});

const SEED = "deadbeef".repeat(8);

// ----------------------------------------------------------------------------
// Lifecycle
// ----------------------------------------------------------------------------

describe("FfmService — lifecycle", () => {
  let rt: FakeRuntime;

  beforeEach(() => {
    rt = makeFakeRuntime();
  });

  it("start() subscribes to scryptedai onTerminal", async () => {
    expect(rt.__scrypted.__listeners.size).toBe(0);
    await FfmService.start(asRuntime(rt));
    expect(rt.__scrypted.__listeners.size).toBe(1);
    expect(rt.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("scryptedai linked"),
    );
  });

  it("start() survives scryptedai being unavailable", async () => {
    const failRt = makeFakeRuntime({ scryptedFails: true });
    const svc = await FfmService.start(asRuntime(failRt));
    expect(svc).toBeInstanceOf(FfmService);
    expect(failRt.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("scryptedai unavailable"),
    );
  });

  it("stop() unsubscribes from scryptedai and clears listeners", async () => {
    const svc = await FfmService.start(asRuntime(rt));
    const unsubscribe = svc.onExpansionTerminal(() => {});
    expect(rt.__scrypted.__listeners.size).toBe(1);

    await svc.stop();
    expect(rt.__scrypted.__listeners.size).toBe(0);

    // listener set should be cleared too — re-subscribe should still work
    // but the previous one is gone
    let fired = false;
    unsubscribe(); // no-op now, just shouldn't throw
    svc.onExpansionTerminal(() => {
      fired = true;
    });
    // Even if scrypted fires, we've unsubscribed, so handleScryptedTerminal
    // doesn't run. Verify by firing for an unknown job — nothing happens.
    rt.__scrypted.__fire({ jobId: "ghost", status: "completed", text: "{}" });
    expect(fired).toBe(false);
  });
});

// ----------------------------------------------------------------------------
// Class A — deriveProfile
// ----------------------------------------------------------------------------

describe("FfmService — deriveProfile", () => {
  let svc: FfmService;

  beforeEach(async () => {
    const rt = makeFakeRuntime();
    svc = await FfmService.start(asRuntime(rt));
  });

  it("generates a fresh seed when none provided", async () => {
    const p = await svc.deriveProfile();
    expect(p.seed).toMatch(/^[0-9a-f]{64}$/);
  });

  it("normalizes a provided seed (trims, lowercases)", async () => {
    const messy = "  " + SEED.toUpperCase() + "  ";
    const p = await svc.deriveProfile(messy);
    expect(p.seed).toBe(SEED);
  });

  it("same seed → identical profile every call", async () => {
    const a = await svc.deriveProfile(SEED);
    const b = await svc.deriveProfile(SEED);
    expect(a).toEqual(b);
    expect(a.traits).toEqual(b.traits);
    expect(a.archetype.code).toBe(b.archetype.code);
  });

  it("trait values are all in [0,1]", async () => {
    const p = await svc.deriveProfile(SEED);
    for (const v of Object.values(p.traits)) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("archetype code is in [0,31]", async () => {
    const p = await svc.deriveProfile(SEED);
    expect(p.archetype.code).toBeGreaterThanOrEqual(0);
    expect(p.archetype.code).toBeLessThanOrEqual(31);
    expect(p.archetype.sloan).toMatch(/^[A-Z]{5}$/);
  });

  it("returns frozen objects", async () => {
    const p = await svc.deriveProfile(SEED);
    expect(Object.isFrozen(p)).toBe(true);
    expect(Object.isFrozen(p.traits)).toBe(true);
  });

  it("works even when scryptedai is unavailable", async () => {
    const failRt = makeFakeRuntime({ scryptedFails: true });
    const failSvc = await FfmService.start(asRuntime(failRt));
    const p = await failSvc.deriveProfile(SEED);
    expect(p.seed).toBe(SEED);
  });
});

// ----------------------------------------------------------------------------
// Class B — expansion job tracking
// ----------------------------------------------------------------------------

describe("FfmService — narrative expansion", () => {
  let rt: FakeRuntime;
  let svc: FfmService;
  let profile: FfmProfile;

  beforeEach(async () => {
    rt = makeFakeRuntime();
    svc = await FfmService.start(asRuntime(rt));
    profile = await svc.deriveProfile(SEED);
  });

  it("startNarrativeExpansion registers a pending record and returns immediately", async () => {
    const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");
    expect(jobId).toBe("job-1");
    const rec = svc.getExpansion(jobId);
    expect(rec?.status).toBe("pending");
    expect(rec?.kind).toBe("narrative");
    expect(rec?.profile).toBe(profile);
  });

  it("sends a payload with system_prompt + user_prompt + max_tokens + auto_calculate_tokens=false", async () => {
    await svc.startNarrativeExpansion(profile, "Eliza");
    expect(rt.__scrypted.__payloads).toHaveLength(1);
    const sent = rt.__scrypted.__payloads[0].inputData;
    expect(typeof sent.system_prompt).toBe("string");
    expect(typeof sent.user_prompt).toBe("string");
    expect(typeof sent.max_tokens).toBe("number");
    // /generations/text/nova-pro defaults auto_calculate_tokens=true which
    // floors output at ~100 tokens and silently ignores max_tokens. Without
    // this flag, expansion JSON gets truncated mid-array.
    expect(sent.auto_calculate_tokens).toBe(false);
    // Capped below Nova Pro's 10k ceiling
    expect(sent.max_tokens).toBeLessThanOrEqual(4000);
    // metadata stamped
    expect(rt.__scrypted.__payloads[0].opts?.metadata?.source).toBe("ffm");
    expect(rt.__scrypted.__payloads[0].opts?.metadata?.kind).toBe("narrative");
    expect(rt.__scrypted.__payloads[0].opts?.pollFallback).toBe(true);
  });

  it("ignores terminal events for jobs it didn't start", async () => {
    const listener = vi.fn();
    svc.onExpansionTerminal(listener);
    rt.__scrypted.__fire({
      jobId: "not-mine",
      status: "completed",
      text: NARRATIVE_JSON,
    });
    expect(listener).not.toHaveBeenCalled();
  });

  it("parses successful terminal → record.completed → fan-out fires", async () => {
    const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");

    const captured: ExpansionRecord[] = [];
    svc.onExpansionTerminal((rec) => captured.push(rec));

    rt.__scrypted.__fire({
      jobId,
      status: "completed",
      text: NARRATIVE_JSON,
    });

    expect(captured).toHaveLength(1);
    expect(captured[0].status).toBe("completed");
    expect(captured[0].result?.kind).toBe("narrative");

    const result = captured[0].result;
    if (result?.kind !== "narrative") throw new Error("expected narrative");
    expect(result.bio).toHaveLength(8);
    expect(result.beliefs).toHaveLength(4);

    expect(svc.getExpansion(jobId)?.status).toBe("completed");
    expect(svc.getExpansion(jobId)?.rawText).toBe(NARRATIVE_JSON);
  });

  it("scryptedai failure → record.failed with error message", async () => {
    const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");

    const captured: ExpansionRecord[] = [];
    svc.onExpansionTerminal((rec) => captured.push(rec));

    rt.__scrypted.__fire({
      jobId,
      status: "failed",
      error: "upstream 500",
    });

    expect(captured).toHaveLength(1);
    expect(captured[0].status).toBe("failed");
    expect(captured[0].error).toBe("upstream 500");
    expect(captured[0].result).toBeUndefined();
    expect(rt.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("expansion failed"),
    );
  });

  it("completed-but-no-text → record.failed", async () => {
    const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");

    rt.__scrypted.__fire({
      jobId,
      status: "completed",
      // no text field
    });

    expect(svc.getExpansion(jobId)?.status).toBe("failed");
    expect(svc.getExpansion(jobId)?.error).toContain("no text output");
  });

  it("unparseable LLM output → record.failed (parse error)", async () => {
    const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");

    rt.__scrypted.__fire({
      jobId,
      status: "completed",
      text: "this is not json at all",
    });

    expect(svc.getExpansion(jobId)?.status).toBe("failed");
    expect(svc.getExpansion(jobId)?.error).toContain("[ffm]");
    expect(svc.getExpansion(jobId)?.rawText).toBe("this is not json at all");
    expect(rt.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("parse failed"),
    );
  });

  it("a throwing listener does not break fan-out to other listeners", async () => {
    const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");

    const good = vi.fn();
    svc.onExpansionTerminal(() => {
      throw new Error("listener boom");
    });
    svc.onExpansionTerminal(good);

    rt.__scrypted.__fire({ jobId, status: "completed", text: NARRATIVE_JSON });

    expect(good).toHaveBeenCalledOnce();
    expect(rt.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("listener threw"),
    );
  });

  it("onExpansionTerminal returns a working unsubscribe", async () => {
    const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");

    const listener = vi.fn();
    const unsubscribe = svc.onExpansionTerminal(listener);
    unsubscribe();

    rt.__scrypted.__fire({ jobId, status: "completed", text: NARRATIVE_JSON });
    expect(listener).not.toHaveBeenCalled();
  });

  it("throws when scryptedai is unavailable", async () => {
    const failRt = makeFakeRuntime({ scryptedFails: true });
    const failSvc = await FfmService.start(asRuntime(failRt));
    const p = await failSvc.deriveProfile(SEED);
    await expect(failSvc.startNarrativeExpansion(p, "Eliza")).rejects.toThrow(
      /scryptedai service unavailable/,
    );
  });
});

// ----------------------------------------------------------------------------
// Voice expansion (same machinery, different parser path)
// ----------------------------------------------------------------------------

describe("FfmService — voice expansion", () => {
  let rt: FakeRuntime;
  let svc: FfmService;
  let profile: FfmProfile;

  beforeEach(async () => {
    rt = makeFakeRuntime();
    svc = await FfmService.start(asRuntime(rt));
    profile = await svc.deriveProfile(SEED);
  });

  it("startVoiceExpansion registers a voice-kind record", async () => {
    const narrative = {
      bio: ["x"],
      beliefs: ["x"],
      adjectives: ["x"],
      topics: ["x"],
    };
    const { jobId } = await svc.startVoiceExpansion(
      profile,
      narrative,
      "Eliza",
    );
    expect(svc.getExpansion(jobId)?.kind).toBe("voice");
    expect(rt.__scrypted.__payloads[0].opts?.metadata?.kind).toBe("voice");
  });

  it("voice terminal routes through parseVoice", async () => {
    const narrative = {
      bio: ["x"],
      beliefs: ["x"],
      adjectives: ["x"],
      topics: ["x"],
    };
    const { jobId } = await svc.startVoiceExpansion(
      profile,
      narrative,
      "Eliza",
    );

    rt.__scrypted.__fire({ jobId, status: "completed", text: VOICE_JSON });

    const rec = svc.getExpansion(jobId);
    expect(rec?.status).toBe("completed");
    if (rec?.result?.kind !== "voice") throw new Error("expected voice");
    expect(rec.result.messageExamples).toHaveLength(6);
    expect(rec.result.postExamples).toHaveLength(6);
    expect(rec.result.style.all).toHaveLength(3);
  });
});

// ----------------------------------------------------------------------------
// awaitExpansion — optional blocking variant
// ----------------------------------------------------------------------------

describe("FfmService — awaitExpansion", () => {
  let rt: FakeRuntime;
  let svc: FfmService;
  let profile: FfmProfile;

  beforeEach(async () => {
    rt = makeFakeRuntime();
    svc = await FfmService.start(asRuntime(rt));
    profile = await svc.deriveProfile(SEED);
  });

  it("resolves when terminal arrives", async () => {
    const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");

    const promise = svc.awaitExpansion(jobId);
    rt.__scrypted.__fire({ jobId, status: "completed", text: NARRATIVE_JSON });

    const out = await promise;
    expect(out.kind).toBe("narrative");
    if (out.kind !== "narrative") throw new Error("narrowing");
    expect(out.bio).toHaveLength(8);
  });

  it("fast-path: resolves immediately if job already terminal", async () => {
    const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");
    rt.__scrypted.__fire({ jobId, status: "completed", text: NARRATIVE_JSON });

    // Job is already done — awaitExpansion should NOT register a new
    // listener, just return the cached result.
    const beforeCount = countListeners(svc);
    const out = await svc.awaitExpansion(jobId);
    const afterCount = countListeners(svc);

    expect(out.kind).toBe("narrative");
    expect(afterCount).toBe(beforeCount);
  });

  it("rejects on failed job", async () => {
    const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");

    const promise = svc.awaitExpansion(jobId);
    rt.__scrypted.__fire({ jobId, status: "failed", error: "boom" });

    await expect(promise).rejects.toThrow(/failed.*boom/);
  });

  it("rejects on parse failure", async () => {
    const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");

    const promise = svc.awaitExpansion(jobId);
    rt.__scrypted.__fire({ jobId, status: "completed", text: "garbage" });

    await expect(promise).rejects.toThrow(/failed/);
  });

  it("rejects on timeout and unsubscribes itself", async () => {
    vi.useFakeTimers();
    try {
      const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");

      const before = countListeners(svc);
      const promise = svc.awaitExpansion(jobId, 100);
      expect(countListeners(svc)).toBe(before + 1);

      vi.advanceTimersByTime(101);

      await expect(promise).rejects.toThrow(/timed out/);
      expect(countListeners(svc)).toBe(before); // listener removed on timeout

      // Late terminal arrival after timeout — should not throw, should not
      // resurrect the rejected promise.
      rt.__scrypted.__fire({
        jobId,
        status: "completed",
        text: NARRATIVE_JSON,
      });
      expect(svc.getExpansion(jobId)?.status).toBe("completed"); // record still updates
    } finally {
      vi.useRealTimers();
    }
  });

  it("clears its timeout when result arrives before deadline", async () => {
    vi.useFakeTimers();
    try {
      const { jobId } = await svc.startNarrativeExpansion(profile, "Eliza");

      const promise = svc.awaitExpansion(jobId, 60_000);
      rt.__scrypted.__fire({
        jobId,
        status: "completed",
        text: NARRATIVE_JSON,
      });

      // Result arrives → timeout should be cleared. Advancing the clock
      // past the deadline should NOT cause a late rejection.
      const out = await promise;
      vi.advanceTimersByTime(70_000);
      expect(out.kind).toBe("narrative");
      // No pending timers (timeout was cleared)
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("two awaiters on different jobs each get their own result", async () => {
    const { jobId: j1 } = await svc.startNarrativeExpansion(profile, "Eliza");
    const { jobId: j2 } = await svc.startNarrativeExpansion(profile, "Eliza");

    const p1 = svc.awaitExpansion(j1);
    const p2 = svc.awaitExpansion(j2);

    // Fire j2 first — p1 should NOT resolve yet
    rt.__scrypted.__fire({
      jobId: j2,
      status: "completed",
      text: NARRATIVE_JSON,
    });
    rt.__scrypted.__fire({
      jobId: j1,
      status: "completed",
      text: NARRATIVE_JSON,
    });

    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1.kind).toBe("narrative");
    expect(r2.kind).toBe("narrative");
    // Both awaiters cleaned up after themselves
    expect(countListeners(svc)).toBe(0);
  });
});

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/** Reach into the service to count internal listeners (for cleanup assertions). */
function countListeners(svc: FfmService): number {
  return (svc as unknown as { listeners: Set<unknown> }).listeners.size;
}
