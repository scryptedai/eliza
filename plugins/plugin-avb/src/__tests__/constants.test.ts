import { describe, expect, it } from "vitest";
import {
  BASE_TAGS,
  DEFAULT_IMAGE_METHOD,
  IMAGE_METHODS,
  isImageMethodName,
  PIPELINE,
  tagForJob,
  tagForRun,
  WORKER_NAMES,
} from "../constants.ts";

describe("constants: PIPELINE graph", () => {
  it("TEXT_PHASE → IMAGE_PHASE → DELIVER → null", () => {
    expect(PIPELINE.TEXT_PHASE.next).toBe("IMAGE_PHASE");
    expect(PIPELINE.IMAGE_PHASE.next).toBe("DELIVER");
    expect(PIPELINE.DELIVER.next).toBe(null);
  });

  it("every phase has a positive deadline", () => {
    for (const [name, spec] of Object.entries(PIPELINE)) {
      expect(spec.deadlineMs).toBeGreaterThan(0);
      expect(spec.workerName).toBe(
        WORKER_NAMES[name as keyof typeof WORKER_NAMES],
      );
    }
  });

  it("phases with scryptedai jobs have matching jobType windows", () => {
    expect(PIPELINE.TEXT_PHASE.jobType).toBe("text");
    expect(PIPELINE.IMAGE_PHASE.jobType).toBe("image");
    // DELIVER doesn't submit → "unknown"
    expect(PIPELINE.DELIVER.jobType).toBe("unknown");
  });

  it("deadlines exceed scryptedai polling windows (headroom)", () => {
    // scryptedai: text=60s, image=300s
    expect(PIPELINE.TEXT_PHASE.deadlineMs).toBeGreaterThanOrEqual(60_000);
    expect(PIPELINE.IMAGE_PHASE.deadlineMs).toBeGreaterThanOrEqual(300_000);
  });
});

describe("constants: image method default", () => {
  it("defaults to Seedream 4", () => {
    expect(DEFAULT_IMAGE_METHOD).toBe("invokeSeedream4Generation");
  });

  it("default is in the allow-list", () => {
    expect(IMAGE_METHODS).toContain(DEFAULT_IMAGE_METHOD);
  });
});

describe("constants: isImageMethodName", () => {
  it("accepts every entry in IMAGE_METHODS", () => {
    for (const m of IMAGE_METHODS) {
      expect(isImageMethodName(m)).toBe(true);
    }
  });

  it("rejects typos and unknown values", () => {
    expect(isImageMethodName("invokeSeadream4Generation")).toBe(false);
    expect(isImageMethodName("invokeBogus")).toBe(false);
    expect(isImageMethodName("")).toBe(false);
  });

  it("rejects non-string input", () => {
    expect(isImageMethodName(undefined)).toBe(false);
    expect(isImageMethodName(null)).toBe(false);
    expect(isImageMethodName(123)).toBe(false);
  });
});

describe("constants: tags", () => {
  it("BASE_TAGS include queue + repeat for TaskService pickup", () => {
    expect(BASE_TAGS).toContain("queue");
    expect(BASE_TAGS).toContain("repeat");
    expect(BASE_TAGS).toContain("avb");
  });

  it("tagForRun namespaces runId", () => {
    expect(tagForRun("abc-123")).toBe("avb:run:abc-123");
  });

  it("tagForJob namespaces jobId", () => {
    expect(tagForJob("job-xyz")).toBe("avb:job:job-xyz");
  });
});
