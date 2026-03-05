import { describe, expect, it } from "vitest";
import {
  extractError,
  extractImageUrl,
  extractText,
  extractVideoUrl,
  isTerminalStatus,
  mergeResult,
  normalizeJobPayload,
  normalizeStatus,
} from "../adapter.ts";

describe("adapter: normalizeStatus", () => {
  it("lowercases and normalizes known terminal statuses", () => {
    expect(normalizeStatus("COMPLETED")).toBe("completed");
    expect(normalizeStatus("Completed")).toBe("completed");
    expect(normalizeStatus("FAILED")).toBe("failed");
    expect(normalizeStatus("Failed")).toBe("failed");
    expect(normalizeStatus("CANCELLED")).toBe("cancelled");
    expect(normalizeStatus("canceled")).toBe("cancelled");
  });

  it("maps aliases to canonical values", () => {
    expect(normalizeStatus("success")).toBe("completed");
    expect(normalizeStatus("succeeded")).toBe("completed");
    expect(normalizeStatus("complete")).toBe("completed");
    expect(normalizeStatus("error")).toBe("failed");
    expect(normalizeStatus("failure")).toBe("failed");
    expect(normalizeStatus("running")).toBe("processing");
    expect(normalizeStatus("in_progress")).toBe("processing");
  });

  it("falls back to pending for unknown/empty", () => {
    expect(normalizeStatus("")).toBe("pending");
    expect(normalizeStatus(undefined)).toBe("pending");
    expect(normalizeStatus(null)).toBe("pending");
    expect(normalizeStatus("mystery")).toBe("pending");
  });

  it("isTerminalStatus correctly identifies terminal states", () => {
    expect(isTerminalStatus("completed")).toBe(true);
    expect(isTerminalStatus("failed")).toBe(true);
    expect(isTerminalStatus("cancelled")).toBe(true);
    expect(isTerminalStatus("pending")).toBe(false);
    expect(isTerminalStatus("processing")).toBe(false);
  });
});

describe("adapter: extractError", () => {
  it("extracts error_message first", () => {
    expect(
      extractError({
        error_message: "primary",
        error: "secondary",
      }),
    ).toBe("primary");
  });

  it("extracts error.message from object", () => {
    expect(
      extractError({
        error: { message: "nested", code: 42 },
      }),
    ).toBe("nested");
  });

  it("uses error string directly", () => {
    expect(extractError({ error: "plain string" })).toBe("plain string");
  });

  it("stringifies opaque error objects", () => {
    const result = extractError({ error: { code: 500, detail: "boom" } });
    expect(result).toContain("500");
    expect(result).toContain("boom");
  });

  it("returns undefined for missing error", () => {
    expect(extractError({})).toBeUndefined();
    expect(extractError({ error: null })).toBeUndefined();
    expect(extractError({ error: "" })).toBeUndefined();
  });
});

describe("adapter: mergeResult", () => {
  it("merges result_data > result > output with precedence", () => {
    const merged = mergeResult({
      result_data: { a: 1, shared: "from_data" },
      result: { b: 2, shared: "from_result" },
      output: { c: 3, shared: "from_output" },
    });
    expect(merged).toEqual({ a: 1, b: 2, c: 3, shared: "from_data" });
  });

  it("returns undefined when no containers present", () => {
    expect(mergeResult({})).toBeUndefined();
    expect(mergeResult({ result: null })).toBeUndefined();
  });

  it("handles single container", () => {
    expect(mergeResult({ result: { x: 1 } })).toEqual({ x: 1 });
  });
});

describe("adapter: extractImageUrl", () => {
  it("extracts from result_data.images[0].asset_url (highest priority)", () => {
    const url = extractImageUrl({
      result_data: {
        images: [{ asset_url: "https://cdn.example/img1.png" }],
      },
      result: { image_url: "https://legacy.example/old.png" },
    });
    expect(url).toBe("https://cdn.example/img1.png");
  });

  it("probes cdn_url / cloudfront_url / url in order", () => {
    expect(
      extractImageUrl({
        result_data: { images: [{ cdn_url: "https://cdn/x" }] },
      }),
    ).toBe("https://cdn/x");
    expect(
      extractImageUrl({
        result_data: { images: [{ cloudfront_url: "https://cf/x" }] },
      }),
    ).toBe("https://cf/x");
    expect(
      extractImageUrl({
        result_data: { images: [{ url: "https://plain/x" }] },
      }),
    ).toBe("https://plain/x");
  });

  it("falls back to result.images[0]", () => {
    expect(
      extractImageUrl({
        result: { images: [{ asset_url: "https://r/x" }] },
      }),
    ).toBe("https://r/x");
  });

  it("handles legacy result.image.url", () => {
    expect(
      extractImageUrl({ result: { image: { url: "https://legacy/img" } } }),
    ).toBe("https://legacy/img");
  });

  it("handles legacy result.image_url and result.imageUrl", () => {
    expect(extractImageUrl({ result: { image_url: "https://a" } })).toBe(
      "https://a",
    );
    expect(extractImageUrl({ result: { imageUrl: "https://b" } })).toBe(
      "https://b",
    );
  });

  it("handles top-level images array (pre-merged)", () => {
    expect(extractImageUrl({ images: [{ asset_url: "https://top/x" }] })).toBe(
      "https://top/x",
    );
  });

  it("returns undefined when no image present", () => {
    expect(extractImageUrl({})).toBeUndefined();
    expect(extractImageUrl({ result: {} })).toBeUndefined();
    expect(extractImageUrl({ result_data: { images: [] } })).toBeUndefined();
  });

  it("accepts string items in images array", () => {
    expect(
      extractImageUrl({ result_data: { images: ["https://direct/str"] } }),
    ).toBe("https://direct/str");
  });
});

describe("adapter: extractVideoUrl", () => {
  it("extracts from result_data.video (singular)", () => {
    expect(
      extractVideoUrl({
        result_data: { video: { asset_url: "https://v/single" } },
      }),
    ).toBe("https://v/single");
  });

  it("extracts from result_data.videos[0]", () => {
    expect(
      extractVideoUrl({
        result_data: { videos: [{ cdn_url: "https://v/arr" }] },
      }),
    ).toBe("https://v/arr");
  });

  it("falls back to result.video and result.videos[0]", () => {
    expect(extractVideoUrl({ result: { video: { url: "https://r/v" } } })).toBe(
      "https://r/v",
    );
    expect(
      extractVideoUrl({ result: { videos: [{ url: "https://r/v0" }] } }),
    ).toBe("https://r/v0");
  });

  it("handles legacy result.video_url and result.videoUrl", () => {
    expect(extractVideoUrl({ result: { video_url: "https://legacy/v" } })).toBe(
      "https://legacy/v",
    );
    expect(extractVideoUrl({ result: { videoUrl: "https://legacy/v2" } })).toBe(
      "https://legacy/v2",
    );
  });

  it("singular .video takes priority over .videos[]", () => {
    expect(
      extractVideoUrl({
        result_data: {
          video: { url: "https://single" },
          videos: [{ url: "https://array" }],
        },
      }),
    ).toBe("https://single");
  });

  it("returns undefined when no video present", () => {
    expect(extractVideoUrl({})).toBeUndefined();
    expect(extractVideoUrl({ result: {} })).toBeUndefined();
  });
});

describe("adapter: extractText", () => {
  it("extracts from result_data.text, result.text, output.text", () => {
    expect(extractText({ result_data: { text: "hello" } })).toBe("hello");
    expect(extractText({ result: { text: "world" } })).toBe("world");
    expect(extractText({ output: { text: "out" } })).toBe("out");
  });

  it("prefers result_data over result", () => {
    expect(
      extractText({
        result_data: { text: "primary" },
        result: { text: "secondary" },
      }),
    ).toBe("primary");
  });

  it("falls back to top-level .text", () => {
    expect(extractText({ text: "top" })).toBe("top");
  });

  it("returns undefined for missing text", () => {
    expect(extractText({})).toBeUndefined();
    expect(extractText({ result: {} })).toBeUndefined();
  });
});

describe("adapter: normalizeJobPayload", () => {
  it("normalizes a full completed image job", () => {
    const result = normalizeJobPayload({
      job_id: "abc-123",
      status: "COMPLETED",
      result_data: {
        images: [{ asset_url: "https://cdn/img.png" }],
      },
    });
    expect(result.jobId).toBe("abc-123");
    expect(result.status).toBe("completed");
    expect(result.imageUrl).toBe("https://cdn/img.png");
    expect(result.result).toEqual({
      images: [{ asset_url: "https://cdn/img.png" }],
    });
    expect(result.error).toBeUndefined();
  });

  it("normalizes a failed job with error_message", () => {
    const result = normalizeJobPayload({
      job_id: "def-456",
      status: "Failed",
      error_message: "Provider rejected: content policy",
    });
    expect(result.jobId).toBe("def-456");
    expect(result.status).toBe("failed");
    expect(result.error).toBe("Provider rejected: content policy");
    expect(result.result).toBeUndefined();
  });

  it("accepts camelCase jobId", () => {
    const result = normalizeJobPayload({ jobId: "xyz", status: "pending" });
    expect(result.jobId).toBe("xyz");
  });

  it("handles pending job with no result", () => {
    const result = normalizeJobPayload({
      job_id: "pending-1",
      status: "processing",
      estimated_seconds: 30,
    });
    expect(result.status).toBe("processing");
    expect(result.result).toBeUndefined();
    expect(result.imageUrl).toBeUndefined();
  });

  it("normalizes a video job with legacy shape", () => {
    const result = normalizeJobPayload({
      job_id: "vid-1",
      status: "completed",
      result: { video_url: "https://legacy/out.mp4" },
    });
    expect(result.videoUrl).toBe("https://legacy/out.mp4");
  });

  it("normalizes a text job", () => {
    const result = normalizeJobPayload({
      job_id: "txt-1",
      status: "completed",
      result: { text: "Generated text output." },
    });
    expect(result.text).toBe("Generated text output.");
  });
});
