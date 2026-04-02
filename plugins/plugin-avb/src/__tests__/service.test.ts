/**
 * AvbService unit tests.
 *
 * Strategy: build a fake runtime object satisfying AvbRuntimeSurface with
 * an in-memory task store. Run AvbService.start() against it, then drive
 * workers manually (simulating TaskService ticks) to verify idempotence
 * and pipeline transitions.
 */

import type { Task } from "@elizaos/core";
import type {
  JobRecord,
  JobTerminalListener,
  NormalizedJobResult,
} from "@elizaos/plugin-scryptedai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { tagForJob, tagForRun, WORKER_NAMES } from "../constants.ts";
import { AvbService } from "../service.ts";
import type { AvbPhaseMetadata } from "../types.ts";

// ----------------------------------------------------------------------------
// Fake runtime builder
// ----------------------------------------------------------------------------

type WorkerMap = Map<
  string,
  {
    name: string;
    execute: (
      rt: unknown,
      opts: Record<string, unknown>,
      task: Task,
    ) => Promise<void>;
  }
>;

interface FakeScrypted {
  onTerminal: (listener: JobTerminalListener) => () => void;
  getJob: (jobId: string) => JobRecord | undefined;
  fetchJobStatus: (jobId: string) => Promise<NormalizedJobResult>;
  resumePolling: (jobId: string, jobType?: string) => void;
  startTextGeneration: (
    input: Record<string, unknown>,
  ) => Promise<{ jobId: string }>;
  startImageGeneration: (
    method: string,
    input: Record<string, unknown>,
  ) => Promise<{ jobId: string }>;
  // Test control surface
  __jobs: Map<string, JobRecord>;
  __listeners: Set<JobTerminalListener>;
  __setTerminal: (jobId: string, result: NormalizedJobResult) => void;
}

function makeFakeScrypted(): FakeScrypted {
  const jobs = new Map<string, JobRecord>();
  const listeners = new Set<JobTerminalListener>();

  const ensureRecord = (
    jobId: string,
    jobType: "text" | "image",
  ): JobRecord => {
    let rec = jobs.get(jobId);
    if (!rec) {
      rec = {
        jobId,
        jobType,
        status: "processing",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        polling: true,
      };
      jobs.set(jobId, rec);
    }
    return rec;
  };

  return {
    __jobs: jobs,
    __listeners: listeners,
    __setTerminal(jobId, result) {
      const rec = jobs.get(jobId);
      if (rec) {
        rec.status = result.status;
        rec.result = result;
        rec.updatedAt = Date.now();
      }
      for (const l of listeners) l(result);
    },
    onTerminal(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    getJob(jobId) {
      return jobs.get(jobId);
    },
    async fetchJobStatus(jobId) {
      const rec = jobs.get(jobId);
      if (rec?.result) return rec.result;
      return { jobId, status: rec?.status ?? "processing" };
    },
    resumePolling(jobId) {
      if (!jobs.has(jobId)) {
        jobs.set(jobId, {
          jobId,
          jobType: "unknown",
          status: "processing",
          createdAt: Date.now(),
          updatedAt: Date.now(),
          polling: true,
        });
      }
    },
    startTextGeneration: vi.fn(async (_input) => {
      const jobId = `text-job-${Math.random().toString(36).slice(2, 8)}`;
      ensureRecord(jobId, "text");
      return { jobId };
    }),
    startImageGeneration: vi.fn(async (_method, _input) => {
      const jobId = `image-job-${Math.random().toString(36).slice(2, 8)}`;
      ensureRecord(jobId, "image");
      return { jobId };
    }),
  };
}

interface FakeRuntime {
  // structural surface
  agentId: string;
  character: Record<string, unknown>;
  logger: { info: unknown; warn: unknown; error: unknown; debug: unknown };
  getService<T>(type: string): T | undefined;
  getServiceLoadPromise(type: string): Promise<unknown>;
  getSetting(key: string): unknown;
  createTask(task: Task): Promise<string>;
  getTask(id: string): Promise<Task | null>;
  getTasks(params: { tags?: string[] }): Promise<Task[]>;
  updateTask(id: string, partial: Partial<Task>): Promise<void>;
  deleteTask(id: string): Promise<void>;
  createMemory(mem: Record<string, unknown>, table: string): Promise<string>;
  getMemories(params: {
    roomId: string;
    tableName: string;
    count?: number;
  }): Promise<Array<{ content?: Record<string, unknown> }>>;
  emitEvent(name: string, payload: unknown): Promise<void>;
  registerTaskWorker(w: {
    name: string;
    execute: (
      rt: unknown,
      opts: Record<string, unknown>,
      t: Task,
    ) => Promise<void>;
  }): void;
  // test control surface
  __tasks: Map<string, Task>;
  __workers: WorkerMap;
  __memories: Array<Record<string, unknown>>;
  __events: Array<{ name: string; payload: unknown }>;
  __scrypted: FakeScrypted;
  __settings: Map<string, unknown>;
}

function makeFakeRuntime(settings: Record<string, unknown> = {}): FakeRuntime {
  const tasks = new Map<string, Task>();
  const workers: WorkerMap = new Map();
  const memories: Array<Record<string, unknown>> = [];
  const events: Array<{ name: string; payload: unknown }> = [];
  const scrypted = makeFakeScrypted();
  // Default autogen OFF in tests so existing assertions (task counts etc.)
  // remain deterministic. Individual tests override via the settings param.
  const settingsMap = new Map<string, unknown>(
    Object.entries({ AVB_AUTOGEN_ON_BOOT: "false", ...settings }),
  );
  let nextId = 1;

  return {
    agentId: "agent-123",
    character: {
      name: "TestAgent",
      bio: ["A test agent for unit testing."],
      adjectives: ["precise", "deterministic"],
      topics: ["testing", "validation"],
      // Pre-seed FFM so bootstrapFfmPersonality() is a no-op in unit tests
      // (otherwise it would call startTextGeneration during start() and
      // skew per-test call-count assertions).
      settings: { AVB_FFM_SEED: "00".repeat(32) },
    },
    logger: {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      debug: vi.fn(),
    },
    getService<T>(type: string): T | undefined {
      if (type === "scryptedai") return scrypted as unknown as T;
      return undefined;
    },
    async getServiceLoadPromise(type: string): Promise<unknown> {
      if (type === "scryptedai") return scrypted;
      throw new Error(`Service ${type} not found`);
    },
    getSetting(key: string) {
      return settingsMap.get(key);
    },
    async createTask(task: Task): Promise<string> {
      const id = `task-${nextId++}`;
      tasks.set(id, { ...task, id });
      return id;
    },
    async getTask(id: string): Promise<Task | null> {
      return tasks.get(id) ?? null;
    },
    async getTasks(params: { tags?: string[] }): Promise<Task[]> {
      const wanted = params.tags;
      if (!wanted || wanted.length === 0) return [...tasks.values()];
      return [...tasks.values()].filter((t) =>
        wanted.every((tag) => (t.tags ?? []).includes(tag)),
      );
    },
    async updateTask(id: string, partial: Partial<Task>): Promise<void> {
      const existing = tasks.get(id);
      if (!existing) return;
      tasks.set(id, { ...existing, ...partial });
    },
    async deleteTask(id: string): Promise<void> {
      tasks.delete(id);
    },
    async createMemory(
      mem: Record<string, unknown>,
      _table: string,
    ): Promise<string> {
      const id = `mem-${memories.length}`;
      memories.push({ ...mem, id });
      return id;
    },
    async getMemories(params: {
      roomId: string;
      tableName: string;
      count?: number;
    }): Promise<Array<{ content?: Record<string, unknown> }>> {
      const matching = memories.filter((m) => m.roomId === params.roomId);
      const limited = params.count ? matching.slice(0, params.count) : matching;
      return limited as Array<{ content?: Record<string, unknown> }>;
    },
    async emitEvent(name: string, payload: unknown): Promise<void> {
      events.push({ name, payload });
    },
    registerTaskWorker(w) {
      workers.set(w.name, w);
    },
    __tasks: tasks,
    __workers: workers,
    __memories: memories,
    __events: events,
    __scrypted: scrypted,
    __settings: settingsMap,
  };
}

/**
 * Flush microtasks so fire-and-forget promises (autoStartIfNeeded) settle
 * before assertions run. Two awaits cover the promise-chain depth
 * (getTasks → getMemories → createRun).
 */
async function flush(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

/** Simulate a TaskService tick: execute every registered worker for its matching tasks. */
async function tick(rt: FakeRuntime): Promise<void> {
  for (const task of [...rt.__tasks.values()]) {
    const worker = rt.__workers.get(task.name);
    if (worker) {
      await worker.execute(rt, {}, task);
    }
  }
}

function metaOf(task: Task): AvbPhaseMetadata {
  return task.metadata as unknown as AvbPhaseMetadata;
}

/** Test helper: assert-and-extract a submitted jobId from a task. */
function jobIdOf(task: Task): string {
  const id = metaOf(task).scryptedJobId;
  if (typeof id !== "string") throw new Error("expected scryptedJobId on task");
  return id;
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

describe("AvbService: start() + worker registration", () => {
  it("registers all three phase workers", async () => {
    const rt = makeFakeRuntime();
    await AvbService.start(rt as never);

    expect(rt.__workers.has(WORKER_NAMES.TEXT_PHASE)).toBe(true);
    expect(rt.__workers.has(WORKER_NAMES.IMAGE_PHASE)).toBe(true);
    expect(rt.__workers.has(WORKER_NAMES.DELIVER)).toBe(true);
  });

  it("subscribes to scryptedai onTerminal", async () => {
    const rt = makeFakeRuntime();
    await AvbService.start(rt as never);
    expect(rt.__scrypted.__listeners.size).toBe(1);
  });

  it("unsubscribes on stop()", async () => {
    const rt = makeFakeRuntime();
    const svc = await AvbService.start(rt as never);
    expect(rt.__scrypted.__listeners.size).toBe(1);
    await svc.stop();
    expect(rt.__scrypted.__listeners.size).toBe(0);
  });
});

describe("AvbService: autonomous avatar generation on boot", () => {
  it("self-triggers createRun when no avatar exists and autogen enabled", async () => {
    const rt = makeFakeRuntime({ AVB_AUTOGEN_ON_BOOT: "true" });
    await AvbService.start(rt as never);
    await flush();

    // A TEXT_PHASE task was spawned for the agent's own room
    expect(rt.__tasks.size).toBe(1);
    const task = [...rt.__tasks.values()][0];
    expect(task.name).toBe(WORKER_NAMES.TEXT_PHASE);
    const meta = metaOf(task);
    expect(meta.runContext.roomId).toBe("agent-123");
    // Character digest was frozen at boot
    expect(meta.runContext.characterDigest).toContain("TestAgent");
  });

  it("skips when a run is already in flight (idempotent across restarts)", async () => {
    const rt = makeFakeRuntime({ AVB_AUTOGEN_ON_BOOT: "true" });
    // Seed an existing AVB task row (simulating a restart mid-run)
    await rt.createTask({
      name: WORKER_NAMES.TEXT_PHASE,
      tags: ["queue", "repeat", "avb", tagForRun("prior-run")],
      metadata: {} as never,
    } as Task);

    await AvbService.start(rt as never);
    await flush();

    // Still exactly the one pre-existing task — no duplicate run spawned
    expect(rt.__tasks.size).toBe(1);
  });

  it("skips when an avatar was already delivered to the agent's room", async () => {
    const rt = makeFakeRuntime({ AVB_AUTOGEN_ON_BOOT: "true" });
    // Seed a prior delivered avatar memory
    rt.__memories.push({
      roomId: "agent-123",
      content: {
        source: "avb",
        attachments: [{ url: "https://cdn.example/prior.png" }],
      },
    });

    await AvbService.start(rt as never);
    await flush();

    expect(rt.__tasks.size).toBe(0);
  });

  it("skips when AVB_AUTOGEN_ON_BOOT is explicitly false", async () => {
    const rt = makeFakeRuntime({ AVB_AUTOGEN_ON_BOOT: "false" });
    await AvbService.start(rt as never);
    await flush();

    expect(rt.__tasks.size).toBe(0);
  });

  it("autogen default-on: fires when setting is absent", async () => {
    const rt = makeFakeRuntime();
    // Remove the test-fixture default-off so we exercise the real default
    rt.__settings.delete("AVB_AUTOGEN_ON_BOOT");

    await AvbService.start(rt as never);
    await flush();

    // Default-on: a run was triggered
    expect(rt.__tasks.size).toBe(1);
    expect([...rt.__tasks.values()][0].name).toBe(WORKER_NAMES.TEXT_PHASE);
  });
});

describe("AvbService: createRun() spawns TEXT_PHASE task", () => {
  it("creates a TEXT_PHASE task with correct tags + metadata", async () => {
    const rt = makeFakeRuntime();
    const svc = await AvbService.start(rt as never);

    const runId = await svc.createRun("room-1", "msg-1");

    expect(rt.__tasks.size).toBe(1);
    const task = [...rt.__tasks.values()][0];
    expect(task.name).toBe(WORKER_NAMES.TEXT_PHASE);
    expect(task.tags).toContain("queue");
    expect(task.tags).toContain("repeat");
    expect(task.tags).toContain("avb");
    expect(task.tags).toContain(tagForRun(runId));

    const md = metaOf(task);
    expect(md.phase).toBe("TEXT_PHASE");
    expect(md.runContext.runId).toBe(runId);
    expect(md.runContext.roomId).toBe("room-1");
    expect(md.runContext.triggeringMessageId).toBe("msg-1");
    expect(md.runContext.characterDigest).toContain("TestAgent");
    expect(md.scryptedJobId).toBeUndefined();
    expect(md.deadlineAt).toBeGreaterThan(Date.now());
  });

  it("getRunTasks() finds the task by run tag", async () => {
    const rt = makeFakeRuntime();
    const svc = await AvbService.start(rt as never);

    const runId = await svc.createRun("room-1");
    const found = await svc.getRunTasks(runId);

    expect(found.length).toBe(1);
    expect(metaOf(found[0]).runContext.runId).toBe(runId);
  });
});

describe("TEXT_PHASE worker: idempotent lifecycle", () => {
  let rt: FakeRuntime;
  let svc: AvbService;
  let runId: string;

  beforeEach(async () => {
    rt = makeFakeRuntime();
    svc = await AvbService.start(rt as never);
    runId = await svc.createRun("room-1");
  });

  it("tick 1: submits text gen with PromptSet-rendered payload and persists jobId + tag", async () => {
    await tick(rt);

    expect(rt.__scrypted.startTextGeneration).toHaveBeenCalledTimes(1);
    // Verify the PromptSet → toScryptedPayload wire shape
    const call = (rt.__scrypted.startTextGeneration as ReturnType<typeof vi.fn>)
      .mock.calls[0];
    const payload = call[0] as Record<string, unknown>;
    expect(typeof payload.system_prompt).toBe("string");
    expect(typeof payload.user_prompt).toBe("string");
    expect(payload.system_prompt).toContain("visual prompt engineer");
    expect(payload.user_prompt).toContain("TestAgent");
    // Server-side auto-calc pins output to 100 tokens; must be disabled
    // with an explicit ceiling so the model can finish (verified live).
    expect(payload.auto_calculate_tokens).toBe(false);
    expect(typeof payload.max_tokens).toBe("number");
    expect(payload.max_tokens as number).toBeGreaterThan(100);
    // System and user are distinct (PromptSet keeps them separate)
    expect(payload.system_prompt).not.toContain("TestAgent");

    const task = [...rt.__tasks.values()][0];
    const jobId = jobIdOf(task);
    expect(jobId).toMatch(/^text-job-/);
    expect(task.tags).toContain(tagForJob(jobId));
  });

  it("tick 2+ (non-terminal): does NOT resubmit", async () => {
    await tick(rt); // submit
    await tick(rt); // check, still processing
    await tick(rt); // check, still processing

    expect(rt.__scrypted.startTextGeneration).toHaveBeenCalledTimes(1);
    // Task still exists (not terminal yet)
    expect(rt.__tasks.size).toBe(1);
  });

  it("terminal completed: transitions to IMAGE_PHASE", async () => {
    await tick(rt); // submit

    const textTask = [...rt.__tasks.values()][0];
    const jobId = jobIdOf(textTask);

    // Simulate scryptedai job completing
    rt.__scrypted.__setTerminal(jobId, {
      jobId,
      status: "completed",
      text: "A portrait of a precise, deterministic test entity.",
    });

    await tick(rt); // should detect terminal → complete → spawn IMAGE_PHASE

    // Old TEXT_PHASE task deleted, new IMAGE_PHASE task spawned
    expect(rt.__tasks.size).toBe(1);
    const imgTask = [...rt.__tasks.values()][0];
    expect(imgTask.name).toBe(WORKER_NAMES.IMAGE_PHASE);

    const md = metaOf(imgTask);
    expect(md.phase).toBe("IMAGE_PHASE");
    expect(md.runContext.runId).toBe(runId); // same run
    expect(md.runContext.imagePrompt).toBe(
      "A portrait of a precise, deterministic test entity.",
    );
    expect(md.runContext.characterDigest).toContain("TestAgent"); // carried forward
    expect(md.scryptedJobId).toBeUndefined(); // new phase, no job yet
  });

  it("terminal failed: delivers error message and deletes task", async () => {
    await tick(rt); // submit

    const textTask = [...rt.__tasks.values()][0];
    const jobId = jobIdOf(textTask);

    rt.__scrypted.__setTerminal(jobId, {
      jobId,
      status: "failed",
      error: "model overloaded",
    });

    await tick(rt);

    // Task deleted, no next phase spawned
    expect(rt.__tasks.size).toBe(0);
    // Error message delivered
    expect(rt.__memories.length).toBe(1);
    const mem = rt.__memories[0];
    expect((mem.content as Record<string, unknown>).text).toContain("failed");
    expect((mem.content as Record<string, unknown>).text).toContain(
      "model overloaded",
    );
  });

  it("terminal completed but empty text: fails the phase", async () => {
    await tick(rt);

    const jobId = jobIdOf([...rt.__tasks.values()][0]);
    rt.__scrypted.__setTerminal(jobId, {
      jobId,
      status: "completed",
      text: "   ", // whitespace only
    });

    await tick(rt);

    expect(rt.__tasks.size).toBe(0);
    expect(rt.__memories.length).toBe(1);
    expect(
      (rt.__memories[0].content as Record<string, unknown>).text,
    ).toContain("empty text");
  });

  it("deadline exceeded: fails the phase without resubmitting", async () => {
    const task = [...rt.__tasks.values()][0];
    // Force deadline into the past
    (task.metadata as unknown as AvbPhaseMetadata).deadlineAt =
      Date.now() - 1000;

    await tick(rt);

    expect(rt.__scrypted.startTextGeneration).not.toHaveBeenCalled();
    expect(rt.__tasks.size).toBe(0);
    expect(rt.__memories.length).toBe(1);
    expect(
      (rt.__memories[0].content as Record<string, unknown>).text,
    ).toContain("deadline");
  });
});

describe("IMAGE_PHASE → DELIVER → done (full pipeline)", () => {
  it("runs TEXT → IMAGE → DELIVER and delivers image", async () => {
    const rt = makeFakeRuntime();
    const svc = await AvbService.start(rt as never);
    const runId = await svc.createRun("room-42", "msg-42");

    // --- TEXT_PHASE: submit → complete ---
    await tick(rt);
    const textJob = jobIdOf([...rt.__tasks.values()][0]);
    rt.__scrypted.__setTerminal(textJob, {
      jobId: textJob,
      status: "completed",
      text: "A glowing geometric avatar.",
    });
    await tick(rt);

    // --- IMAGE_PHASE: submit → complete ---
    expect(rt.__tasks.size).toBe(1);
    expect([...rt.__tasks.values()][0].name).toBe(WORKER_NAMES.IMAGE_PHASE);

    await tick(rt); // submit image gen
    expect(rt.__scrypted.startImageGeneration).toHaveBeenCalledTimes(1);
    const imgCall = (
      rt.__scrypted.startImageGeneration as ReturnType<typeof vi.fn>
    ).mock.calls[0];
    // Seedream is the default method
    expect(imgCall[0]).toBe("invokeSeedream4Generation");
    expect(imgCall[1]).toMatchObject({ prompt: "A glowing geometric avatar." });

    const imgJob = jobIdOf([...rt.__tasks.values()][0]);
    rt.__scrypted.__setTerminal(imgJob, {
      jobId: imgJob,
      status: "completed",
      imageUrl: "https://cdn.example.com/avatar-abc.png",
    });
    await tick(rt);

    // --- DELIVER: spawned ---
    expect(rt.__tasks.size).toBe(1);
    expect([...rt.__tasks.values()][0].name).toBe(WORKER_NAMES.DELIVER);
    const deliverMd = metaOf([...rt.__tasks.values()][0]);
    expect(deliverMd.runContext.imageUrl).toBe(
      "https://cdn.example.com/avatar-abc.png",
    );
    expect(deliverMd.runContext.imagePrompt).toBe(
      "A glowing geometric avatar.",
    );

    await tick(rt);

    // --- Pipeline done ---
    expect(rt.__tasks.size).toBe(0);

    // Verify delivery
    expect(rt.__memories.length).toBe(1);
    const mem = rt.__memories[0];
    expect(mem.roomId).toBe("room-42");
    const content = mem.content as Record<string, unknown>;
    expect(content.text).toBe("A glowing geometric avatar.");
    const attachments = content.attachments as Array<Record<string, unknown>>;
    expect(attachments[0].url).toBe("https://cdn.example.com/avatar-abc.png");
    expect(attachments[0].contentType).toBe("image");
    expect(content.inReplyTo).toBe("msg-42");

    // Verify MESSAGE_SENT event
    expect(rt.__events.length).toBe(1);
    expect(rt.__events[0].name).toBe("MESSAGE_SENT");

    // getRunTasks now empty
    const remaining = await svc.getRunTasks(runId);
    expect(remaining.length).toBe(0);
  });
});

describe("restart recovery: resumePolling() path", () => {
  it("calls resumePolling + fetchJobStatus when in-memory record is gone", async () => {
    const rt = makeFakeRuntime();
    await AvbService.start(rt as never);

    // Simulate a TEXT_PHASE task that already has a jobId persisted
    // but no in-memory record in the scryptedai service (post-restart).
    const persistedJobId = "resurrected-job-123";
    const taskId = await rt.createTask({
      name: WORKER_NAMES.TEXT_PHASE,
      tags: [
        "queue",
        "repeat",
        "avb",
        tagForRun("run-x"),
        tagForJob(persistedJobId),
      ],
      metadata: {
        phase: "TEXT_PHASE",
        runContext: {
          runId: "run-x",
          roomId: "room-1",
          characterDigest: "Name: X.",
        },
        scryptedJobId: persistedJobId,
        deadlineAt: Date.now() + 60_000,
        updateInterval: 5000,
        updatedAt: Date.now(),
      } as unknown as Task["metadata"],
    });
    expect(taskId).toBeTruthy();

    const resumeSpy = vi.spyOn(rt.__scrypted, "resumePolling");
    const fetchSpy = vi.spyOn(rt.__scrypted, "fetchJobStatus");

    await tick(rt);

    expect(resumeSpy).toHaveBeenCalledWith(persistedJobId, "text");
    expect(fetchSpy).toHaveBeenCalledWith(persistedJobId);
    // Still processing → task remains
    expect(rt.__tasks.size).toBe(1);
  });
});

describe("webhook fast-path: eager execute on terminal", () => {
  it("terminal event triggers worker without waiting for tick()", async () => {
    const rt = makeFakeRuntime();
    const svc = await AvbService.start(rt as never);
    await svc.createRun("room-9");

    await tick(rt); // submit text gen
    const jobId = jobIdOf([...rt.__tasks.values()][0]);

    // Fire terminal through scryptedai (simulates webhook arrival).
    // This should eager-execute the TEXT_PHASE worker.
    rt.__scrypted.__setTerminal(jobId, {
      jobId,
      status: "completed",
      text: "Fast-path avatar prompt.",
    });

    // Give the eager-execute microtask a turn to run
    await new Promise((r) => setImmediate(r));

    // TEXT_PHASE should have transitioned without an explicit tick()
    expect(rt.__tasks.size).toBe(1);
    expect([...rt.__tasks.values()][0].name).toBe(WORKER_NAMES.IMAGE_PHASE);
  });
});
