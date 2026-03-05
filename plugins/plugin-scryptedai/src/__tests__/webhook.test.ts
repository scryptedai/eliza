import { createHmac } from "node:crypto";
import { describe, expect, it } from "vitest";
import { parseWebhookPayload, verifyWebhookSignature } from "../webhook.ts";

function sign(body: string, secret: string): string {
  return createHmac("sha256", secret).update(body).digest("hex");
}

describe("webhook: verifyWebhookSignature", () => {
  const secret = "test-secret-abc123";

  it("accepts a valid signature (string body)", () => {
    const body = '{"job_id":"abc","status":"completed"}';
    const sig = sign(body, secret);
    expect(verifyWebhookSignature(body, sig, secret)).toBe(true);
  });

  it("accepts a valid signature (Buffer body)", () => {
    const body = '{"job_id":"abc","status":"completed"}';
    const sig = sign(body, secret);
    expect(verifyWebhookSignature(Buffer.from(body), sig, secret)).toBe(true);
  });

  it("rejects an invalid signature", () => {
    const body = '{"job_id":"abc","status":"completed"}';
    expect(verifyWebhookSignature(body, "deadbeef".repeat(8), secret)).toBe(
      false,
    );
  });

  it("rejects a tampered body", () => {
    const original = '{"job_id":"abc","status":"completed"}';
    const tampered = '{"job_id":"abc","status":"failed"}';
    const sig = sign(original, secret);
    expect(verifyWebhookSignature(tampered, sig, secret)).toBe(false);
  });

  it("rejects signature with wrong secret", () => {
    const body = '{"job_id":"abc","status":"completed"}';
    const sig = sign(body, "wrong-secret");
    expect(verifyWebhookSignature(body, sig, secret)).toBe(false);
  });

  it("rejects missing signature", () => {
    expect(verifyWebhookSignature("{}", undefined, secret)).toBe(false);
    expect(verifyWebhookSignature("{}", "", secret)).toBe(false);
  });

  it("rejects missing secret", () => {
    const body = "{}";
    const sig = sign(body, secret);
    expect(verifyWebhookSignature(body, sig, undefined)).toBe(false);
    expect(verifyWebhookSignature(body, sig, "")).toBe(false);
  });

  it("rejects length-mismatched signature without throwing", () => {
    const body = '{"job_id":"x"}';
    expect(verifyWebhookSignature(body, "short", secret)).toBe(false);
  });

  it("is sensitive to body whitespace (proving raw-body requirement)", () => {
    // If a JSON parser re-serialized the body, whitespace would change and
    // the signature would no longer match. This test demonstrates why the
    // raw body MUST be preserved.
    const original = '{"job_id":"abc","status":"completed"}';
    const reserialized = JSON.stringify(JSON.parse(original), null, 2);
    const sig = sign(original, secret);
    expect(verifyWebhookSignature(original, sig, secret)).toBe(true);
    expect(verifyWebhookSignature(reserialized, sig, secret)).toBe(false);
  });
});

describe("webhook: parseWebhookPayload", () => {
  it("parses top-level payload", () => {
    const body = JSON.stringify({
      job_id: "abc",
      status: "COMPLETED",
      result: { text: "hi" },
    });
    const { normalized, payload } = parseWebhookPayload(body);
    expect(normalized.jobId).toBe("abc");
    expect(normalized.status).toBe("completed");
    expect(normalized.text).toBe("hi");
    expect(payload.job_id).toBe("abc");
  });

  it("unwraps payload wrapper", () => {
    const body = JSON.stringify({
      event: "job.completed",
      payload: {
        job_id: "wrapped-123",
        status: "completed",
        result_data: { images: [{ asset_url: "https://x/img" }] },
      },
    });
    const { normalized, payload } = parseWebhookPayload(body);
    expect(normalized.jobId).toBe("wrapped-123");
    expect(normalized.imageUrl).toBe("https://x/img");
    expect(payload.job_id).toBe("wrapped-123");
  });

  it("does not unwrap when payload field is not an object", () => {
    const body = JSON.stringify({
      job_id: "toplevel",
      status: "failed",
      payload: "this is a string, not an object",
    });
    const { normalized } = parseWebhookPayload(body);
    expect(normalized.jobId).toBe("toplevel");
    expect(normalized.status).toBe("failed");
  });

  it("normalizes status casing in parsed result", () => {
    const body = JSON.stringify({ job_id: "x", status: "FAILED" });
    const { normalized } = parseWebhookPayload(body);
    expect(normalized.status).toBe("failed");
  });

  it("extracts error from nested object", () => {
    const body = JSON.stringify({
      job_id: "err-1",
      status: "failed",
      error: { message: "upstream timeout", code: 504 },
    });
    const { normalized } = parseWebhookPayload(body);
    expect(normalized.error).toBe("upstream timeout");
  });

  it("accepts Buffer input", () => {
    const body = Buffer.from(
      JSON.stringify({ job_id: "buf-1", status: "completed" }),
    );
    const { normalized } = parseWebhookPayload(body);
    expect(normalized.jobId).toBe("buf-1");
  });

  it("throws on invalid JSON", () => {
    expect(() => parseWebhookPayload("not json")).toThrow(
      /Invalid webhook JSON/,
    );
  });

  it("throws on non-object root", () => {
    expect(() => parseWebhookPayload('"just a string"')).toThrow(
      /must be a JSON object/,
    );
    expect(() => parseWebhookPayload("[]")).toThrow(/must be a JSON object/);
  });

  it("throws on missing job_id", () => {
    expect(() => parseWebhookPayload('{"status":"completed"}')).toThrow(
      /job_id/,
    );
  });

  it("throws on missing job_id inside wrapper", () => {
    expect(() =>
      parseWebhookPayload('{"payload":{"status":"completed"}}'),
    ).toThrow(/job_id/);
  });
});
