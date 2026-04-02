import { afterEach, describe, expect, it, vi } from "vitest";
import { MAX_INFLIGHT_JOBS } from "../constants.ts";
import { ScryptedAIService } from "../service.ts";
import type { NormalizedJobResult, RecipeExecutionResponse } from "../types.ts";

/**
 * Minimal mock runtime for testing the service's processTerminal logic
 * without the full ElizaOS runtime.
 */
function createMockRuntime() {
  return {
    logger: {
      info: vi.fn(),
      debug: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    },
    getSetting: vi.fn(),
    getService: vi.fn(),
  };
}

/**
 * Create a service instance with internals wired for testing.
 * Bypasses start() since we're only testing the terminal processor.
 */
function createTestService(): ScryptedAIService {
  // @ts-expect-error — intentionally constructing with mock runtime
  return new ScryptedAIService(createMockRuntime());
}

describe("service: processTerminal idempotency", () => {
  it("processes a completed job once", () => {
    const svc = createTestService();
    const result: NormalizedJobResult = {
      jobId: "abc",
      status: "completed",
      result: { text: "hi" },
    };

    const r1 = svc.processTerminal(result);
    expect(r1.idempotent).toBe(false);
    expect(r1.terminal).toBe(true);

    const job = svc.getJob("abc");
    expect(job?.status).toBe("completed");
    expect(job?.result).toEqual(result);
  });

  it("is idempotent on (jobId, status) — second call is no-op", () => {
    const svc = createTestService();
    const result: NormalizedJobResult = {
      jobId: "abc",
      status: "completed",
      result: { text: "hi" },
    };

    svc.processTerminal(result);
    const r2 = svc.processTerminal(result);
    expect(r2.idempotent).toBe(true);
    expect(r2.terminal).toBe(true);
  });

  it("allows distinct statuses for the same job before terminal", () => {
    const svc = createTestService();

    const r1 = svc.processTerminal({ jobId: "x", status: "pending" });
    expect(r1.idempotent).toBe(false);
    expect(r1.terminal).toBe(false);

    const r2 = svc.processTerminal({ jobId: "x", status: "processing" });
    expect(r2.idempotent).toBe(false);
    expect(r2.terminal).toBe(false);

    const r3 = svc.processTerminal({ jobId: "x", status: "completed" });
    expect(r3.idempotent).toBe(false);
    expect(r3.terminal).toBe(true);

    expect(svc.getJob("x")?.status).toBe("completed");
  });

  it("ignores non-terminal updates after a terminal state", () => {
    const svc = createTestService();

    svc.processTerminal({ jobId: "y", status: "completed" });
    const r = svc.processTerminal({ jobId: "y", status: "processing" });

    expect(r.idempotent).toBe(true);
    expect(r.terminal).toBe(false);
    expect(svc.getJob("y")?.status).toBe("completed");
  });

  it("handles webhook + polling race (first terminal wins)", () => {
    const svc = createTestService();

    // Simulate poll arriving first
    const pollResult = svc.processTerminal({
      jobId: "race-1",
      status: "completed",
      result: { text: "from poll" },
    });
    expect(pollResult.idempotent).toBe(false);

    // Webhook arrives second with same (jobId, status)
    const webhookResult = svc.ingestWebhook({
      jobId: "race-1",
      status: "completed",
      result: { text: "from webhook" },
    });
    expect(webhookResult.idempotent).toBe(true);

    // First writer wins — poll's result is preserved
    expect(svc.getJob("race-1")?.result?.result).toEqual({ text: "from poll" });
  });

  it("fires terminal listeners exactly once per (jobId, terminal-status)", () => {
    const svc = createTestService();
    const listener = vi.fn();
    svc.onTerminal(listener);

    svc.processTerminal({ jobId: "z", status: "completed" });
    svc.processTerminal({ jobId: "z", status: "completed" }); // duplicate
    svc.ingestWebhook({ jobId: "z", status: "completed" }); // also duplicate

    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener.mock.calls[0][0].jobId).toBe("z");
    expect(listener.mock.calls[0][0].status).toBe("completed");
  });

  it("does not fire listeners for non-terminal updates", () => {
    const svc = createTestService();
    const listener = vi.fn();
    svc.onTerminal(listener);

    svc.processTerminal({ jobId: "p", status: "pending" });
    svc.processTerminal({ jobId: "p", status: "processing" });

    expect(listener).not.toHaveBeenCalled();
  });

  it("creates a record for unknown jobs (webhook-only case)", () => {
    const svc = createTestService();
    expect(svc.getJob("unknown")).toBeUndefined();

    svc.ingestWebhook({
      jobId: "unknown",
      status: "failed",
      error: "provider error",
    });

    const record = svc.getJob("unknown");
    expect(record).toBeDefined();
    expect(record?.status).toBe("failed");
    expect(record?.rawError).toBe("provider error");
    expect(record?.jobType).toBe("unknown");
  });

  it("listener unsubscribe works", () => {
    const svc = createTestService();
    const listener = vi.fn();
    const unsubscribe = svc.onTerminal(listener);

    unsubscribe();
    svc.processTerminal({ jobId: "u", status: "completed" });

    expect(listener).not.toHaveBeenCalled();
  });

  it("listener exceptions do not halt processing", () => {
    const svc = createTestService();
    const throwingListener = vi.fn().mockImplementation(() => {
      throw new Error("listener boom");
    });
    const normalListener = vi.fn();

    svc.onTerminal(throwingListener);
    svc.onTerminal(normalListener);

    expect(() =>
      svc.processTerminal({ jobId: "e", status: "completed" }),
    ).not.toThrow();

    expect(throwingListener).toHaveBeenCalledTimes(1);
    expect(normalListener).toHaveBeenCalledTimes(1);
    expect(svc.getJob("e")?.status).toBe("completed");
  });
});

describe("service: listJobs", () => {
  it("returns all tracked jobs", () => {
    const svc = createTestService();
    svc.processTerminal({ jobId: "a", status: "pending" });
    svc.processTerminal({ jobId: "b", status: "completed" });

    const jobs = svc.listJobs();
    expect(jobs).toHaveLength(2);
    expect(jobs.map((j) => j.jobId).sort()).toEqual(["a", "b"]);
  });
});

// ----------------------------------------------------------------------------
// Memory bounds
// ----------------------------------------------------------------------------

describe("service: pruneTerminalJobs (TTL eviction)", () => {
  afterEach(() => vi.useRealTimers());

  it("evicts terminal jobs older than maxAge and their processed keys", () => {
    vi.useFakeTimers();
    vi.setSystemTime(1_000_000);

    const svc = createTestService();
    svc.processTerminal({ jobId: "old", status: "completed" });
    svc.processTerminal({ jobId: "fresh", status: "completed" });

    // Age "old" past the threshold by moving the clock forward, then
    // touch "fresh" so its terminalAt is recent. (Both are terminal so
    // re-touching is idempotent; instead, advance time between them.)
    // Simpler: advance time, add a third job, then prune with maxAge=50ms.
    vi.setSystemTime(1_000_000 + 100);
    const evicted = svc.pruneTerminalJobs(50);

    expect(evicted).toBe(2); // both "old" and "fresh" are >50ms old now
    expect(svc.getJob("old")).toBeUndefined();
    expect(svc.getJob("fresh")).toBeUndefined();

    // Re-ingesting the same (jobId, status) is NOT idempotent anymore —
    // proves the processed-Set keys were also pruned.
    const r = svc.processTerminal({ jobId: "old", status: "completed" });
    expect(r.idempotent).toBe(false);
  });

  it("does not evict non-terminal jobs regardless of age", () => {
    vi.useFakeTimers();
    vi.setSystemTime(1_000_000);

    const svc = createTestService();
    svc.processTerminal({ jobId: "running", status: "processing" });

    vi.setSystemTime(1_000_000 + 10 * 60_000); // 10 min later
    const evicted = svc.pruneTerminalJobs(1_000);

    expect(evicted).toBe(0);
    expect(svc.getJob("running")?.status).toBe("processing");
  });

  it("retains terminal jobs within the grace window", () => {
    vi.useFakeTimers();
    vi.setSystemTime(1_000_000);

    const svc = createTestService();
    svc.processTerminal({ jobId: "recent", status: "failed", error: "x" });

    vi.setSystemTime(1_000_000 + 10);
    const evicted = svc.pruneTerminalJobs(1_000);

    expect(evicted).toBe(0);
    expect(svc.getJob("recent")?.status).toBe("failed");
  });
});

describe("service: in-flight slot semaphore", () => {
  /**
   * Inject a fake client into the (private) `client` field. The fake
   * exposes only the methods exercised by these tests; the cast is
   * intentional and scoped to test setup.
   */
  function withFakeClient(
    svc: ScryptedAIService,
    client: Partial<{
      invokeTextGeneration: (
        input: Record<string, unknown>,
        opts: unknown,
      ) => Promise<RecipeExecutionResponse>;
    }>,
  ): void {
    (svc as unknown as { client: unknown }).client = client;
  }

  it("increments on start*() and decrements on terminal", async () => {
    const svc = createTestService();
    withFakeClient(svc, {
      invokeTextGeneration: async () => ({
        job_id: "job-1",
        status: "pending",
      }),
    });

    expect(svc.inflightCount()).toBe(0);
    await svc.startTextGeneration({ prompt: "hi" }, { pollFallback: false });
    expect(svc.inflightCount()).toBe(1);

    svc.processTerminal({ jobId: "job-1", status: "completed" });
    expect(svc.inflightCount()).toBe(0);
  });

  it("releases the slot when the invoke response is already terminal", async () => {
    const svc = createTestService();
    withFakeClient(svc, {
      invokeTextGeneration: async () => ({
        job_id: "fast",
        status: "completed",
        result: { text: "instant" },
      }),
    });

    await svc.startTextGeneration({ prompt: "hi" }, { pollFallback: false });
    // trackJob() routed the sync-terminal response through processTerminal,
    // which must have found and released the held slot.
    expect(svc.inflightCount()).toBe(0);
    expect(svc.getJob("fast")?.status).toBe("completed");
  });

  it("releases the slot when the upstream invoke throws", async () => {
    const svc = createTestService();
    withFakeClient(svc, {
      invokeTextGeneration: async () => {
        throw new Error("upstream 500");
      },
    });

    await expect(
      svc.startTextGeneration({ prompt: "hi" }, { pollFallback: false }),
    ).rejects.toThrow("upstream 500");
    expect(svc.inflightCount()).toBe(0);
  });

  it("does not release a slot for webhook-only jobs (no underflow)", () => {
    const svc = createTestService();
    expect(svc.inflightCount()).toBe(0);

    // Job arrives via webhook only — never went through start*(), holds no slot.
    svc.ingestWebhook({ jobId: "external", status: "completed" });
    expect(svc.inflightCount()).toBe(0);
  });

  it("applies backpressure at MAX_INFLIGHT_JOBS and resumes when a slot frees", async () => {
    const svc = createTestService();
    let nextId = 0;
    withFakeClient(svc, {
      invokeTextGeneration: async () => ({
        job_id: `job-${nextId++}`,
        status: "pending",
      }),
    });

    // Fill all slots.
    for (let i = 0; i < MAX_INFLIGHT_JOBS; i++) {
      await svc.startTextGeneration({ prompt: "p" }, { pollFallback: false });
    }
    expect(svc.inflightCount()).toBe(MAX_INFLIGHT_JOBS);

    // Next start should block until a slot is released.
    let resolved = false;
    const pending = svc
      .startTextGeneration({ prompt: "p" }, { pollFallback: false })
      .then(() => {
        resolved = true;
      });

    // Yield a microtask — should NOT have resolved yet.
    await Promise.resolve();
    expect(resolved).toBe(false);
    expect(svc.inflightCount()).toBe(MAX_INFLIGHT_JOBS);

    // Free one slot → waiter wakes, invokes, and re-occupies the slot.
    svc.processTerminal({ jobId: "job-0", status: "completed" });
    await pending;
    expect(resolved).toBe(true);
    expect(svc.inflightCount()).toBe(MAX_INFLIGHT_JOBS);
  });
});
