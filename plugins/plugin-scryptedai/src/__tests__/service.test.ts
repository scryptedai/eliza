import { describe, expect, it, vi } from "vitest";
import { ScryptedAIService } from "../service.ts";
import type { NormalizedJobResult } from "../types.ts";

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
