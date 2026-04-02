import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ScryptedClient } from "../client.ts";
import { ScryptedAPIError, ScryptedNetworkError } from "../exceptions.ts";
import { pollJobToCompletion } from "../polling.ts";
import type { JobStatusResponse } from "../types.ts";

// Test seam: a hand-built object matching the structural surface
// pollJobToCompletion consumes. No vi.mock() — @elizaos/core resolves to
// real source via tsconfig paths. Do not add __mocks__/.
function mockClient(
  responses: Array<JobStatusResponse | Error>,
): ScryptedClient {
  let idx = 0;
  return {
    getJobStatus: vi.fn().mockImplementation(async () => {
      const r = responses[Math.min(idx, responses.length - 1)];
      idx++;
      if (r instanceof Error) throw r;
      return r;
    }),
  } as unknown as ScryptedClient;
}

describe("polling: pollJobToCompletion", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("resolves when job reaches completed status", async () => {
    const client = mockClient([
      { job_id: "j1", status: "processing" },
      { job_id: "j1", status: "completed", result: { text: "done" } },
    ]);

    const promise = pollJobToCompletion(client, "j1", {
      jobType: "text",
      intervalsSeconds: [0.01],
      maxWaitSeconds: 10,
    });
    await vi.runAllTimersAsync();
    const result = await promise;

    expect(result.completed).toBe(true);
    expect(result.timedOut).toBe(false);
    expect(result.result.status).toBe("completed");
    expect(result.result.text).toBe("done");
    expect(result.attempts).toBe(2);
  });

  it("resolves when job reaches failed status", async () => {
    const client = mockClient([
      { job_id: "j2", status: "FAILED", error_message: "boom" },
    ]);

    const promise = pollJobToCompletion(client, "j2", {
      jobType: "text",
      intervalsSeconds: [0.01],
      maxWaitSeconds: 10,
    });
    await vi.runAllTimersAsync();
    const result = await promise;

    expect(result.completed).toBe(true);
    expect(result.result.status).toBe("failed");
    expect(result.result.error).toBe("boom");
  });

  it("normalizes mixed-case statuses", async () => {
    const client = mockClient([
      { job_id: "j3", status: "Processing" },
      { job_id: "j3", status: "COMPLETED" },
    ]);

    const promise = pollJobToCompletion(client, "j3", {
      jobType: "text",
      intervalsSeconds: [0.01],
      maxWaitSeconds: 10,
    });
    await vi.runAllTimersAsync();
    const result = await promise;

    expect(result.result.status).toBe("completed");
  });

  it("treats 502/503/504 as transient and continues polling", async () => {
    const client = mockClient([
      new ScryptedAPIError("bad gateway", 502),
      new ScryptedAPIError("service unavailable", 503),
      new ScryptedAPIError("gateway timeout", 504),
      { job_id: "j4", status: "completed" },
    ]);

    const promise = pollJobToCompletion(client, "j4", {
      jobType: "text",
      intervalsSeconds: [0.01],
      maxWaitSeconds: 30,
    });
    await vi.runAllTimersAsync();
    const result = await promise;

    expect(result.completed).toBe(true);
    expect(result.result.status).toBe("completed");
    expect(result.attempts).toBe(4);
  });

  it("treats network errors as transient", async () => {
    const client = mockClient([
      new ScryptedNetworkError("connection reset"),
      { job_id: "j5", status: "completed" },
    ]);

    const promise = pollJobToCompletion(client, "j5", {
      jobType: "text",
      intervalsSeconds: [0.01],
      maxWaitSeconds: 10,
    });
    await vi.runAllTimersAsync();
    const result = await promise;

    expect(result.completed).toBe(true);
    expect(result.attempts).toBe(2);
  });

  it("rethrows non-transient errors (e.g. 404)", async () => {
    const client = mockClient([new ScryptedAPIError("not found", 404)]);

    const promise = pollJobToCompletion(client, "j6", {
      jobType: "text",
      intervalsSeconds: [0.01],
      maxWaitSeconds: 10,
    });
    const expectation = expect(promise).rejects.toThrow(ScryptedAPIError);
    await vi.runAllTimersAsync();
    await expectation;
  });

  it("times out when max-wait elapses", async () => {
    const client = mockClient([{ job_id: "j7", status: "processing" }]);

    const promise = pollJobToCompletion(client, "j7", {
      jobType: "text",
      intervalsSeconds: [0.05],
      maxWaitSeconds: 0.2,
    });
    await vi.runAllTimersAsync();
    const result = await promise;

    expect(result.timedOut).toBe(true);
    expect(result.completed).toBe(false);
    expect(result.result.status).toBe("processing");
  });

  it("invokes onPoll callback for each attempt", async () => {
    const onPoll = vi.fn();
    const client = mockClient([
      { job_id: "j8", status: "processing" },
      { job_id: "j8", status: "completed" },
    ]);

    const promise = pollJobToCompletion(client, "j8", {
      jobType: "text",
      intervalsSeconds: [0.01],
      maxWaitSeconds: 10,
      onPoll,
    });
    await vi.runAllTimersAsync();
    await promise;

    expect(onPoll).toHaveBeenCalledTimes(2);
    expect(onPoll.mock.calls[0][0].status).toBe("processing");
    expect(onPoll.mock.calls[1][0].status).toBe("completed");
  });

  it("respects AbortSignal", async () => {
    const client = mockClient([{ job_id: "j9", status: "processing" }]);
    const controller = new AbortController();

    const promise = pollJobToCompletion(client, "j9", {
      jobType: "text",
      intervalsSeconds: [0.05],
      maxWaitSeconds: 100,
      signal: controller.signal,
    });

    // Advance past minAge + first poll, then abort
    await vi.advanceTimersByTimeAsync(3000);
    controller.abort();
    await vi.runAllTimersAsync();

    const result = await promise;
    expect(result.completed).toBe(false);
    expect(result.timedOut).toBe(false); // aborted, not timed out
  });
});
