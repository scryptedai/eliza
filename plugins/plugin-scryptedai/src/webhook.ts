/**
 * ScryptedAI webhook handling.
 *
 * Exposes two layers:
 * 1. Standalone pure utilities — `verifyWebhookSignature`, `parseWebhookPayload`
 *    — reusable independently of the ElizaOS route system.
 * 2. The plugin route handler — wires those utilities into a POST /webhook
 *    endpoint that dispatches to the ScryptedAIService.
 *
 * SECURITY INVARIANTS (per integration guide §3.2):
 * - HMAC-SHA256 computed over the RAW request body bytes, hex-encoded
 * - Constant-time comparison via crypto.timingSafeEqual
 * - Never log the raw body, the signature, or the secret
 * - Fail closed if we can't obtain a raw body (pre-parsed JSON defeats HMAC)
 */

import { createHmac, timingSafeEqual } from "node:crypto";
import type { IAgentRuntime, Route } from "@elizaos/core";
import { normalizeJobPayload } from "./adapter.ts";
import {
  ENV_WEBHOOK_SECRET,
  SCRYPTEDAI_SERVICE_TYPE,
  WEBHOOK_ROUTE_PATH,
} from "./constants.ts";
import type { ScryptedAIService } from "./service.ts";
import type { NormalizedJobResult } from "./types.ts";

// ----------------------------------------------------------------------------
// Standalone: signature verification
// ----------------------------------------------------------------------------

/**
 * Verify an HMAC-SHA256 webhook signature against the raw body.
 *
 * @param rawBody - The raw request body as a string or Buffer. Must be the
 *   exact bytes ScryptedAI signed — NOT a re-serialized JSON object.
 * @param signature - The `X-Webhook-Signature` header value (hex string).
 * @param secret - The shared secret passed as `webhookSecret` on invoke.
 * @returns true iff the signature is valid.
 */
export function verifyWebhookSignature(
  rawBody: string | Buffer,
  signature: string | undefined,
  secret: string | undefined,
): boolean {
  if (!signature || !secret) return false;

  const expected = createHmac("sha256", secret)
    .update(
      typeof rawBody === "string" ? Buffer.from(rawBody, "utf8") : rawBody,
    )
    .digest("hex");

  // timingSafeEqual requires equal-length buffers; if lengths differ, fail fast
  // (this is safe: length mismatch leaks no useful timing information beyond
  // "wrong length", which an attacker already knows from the hex digest length)
  if (signature.length !== expected.length) return false;

  try {
    return timingSafeEqual(
      Buffer.from(signature, "utf8"),
      Buffer.from(expected, "utf8"),
    );
  } catch {
    return false;
  }
}

// ----------------------------------------------------------------------------
// Standalone: payload parsing
// ----------------------------------------------------------------------------

export interface ParsedWebhookPayload {
  /** The normalized job result extracted from the body. */
  normalized: NormalizedJobResult;
  /** The unwrapped payload object (after `payload` wrapper removal). */
  payload: Record<string, unknown>;
}

/**
 * Parse a ScryptedAI webhook body.
 *
 * Handles the `payload` wrapper variant (per guide §3.1):
 *   { payload: { job_id, status, result, error } }
 * vs top-level:
 *   { job_id, status, result, error }
 *
 * @param rawBody - Raw body bytes.
 * @returns Parsed and normalized payload, or throws on invalid JSON / missing job_id.
 */
export function parseWebhookPayload(
  rawBody: string | Buffer,
): ParsedWebhookPayload {
  const text = typeof rawBody === "string" ? rawBody : rawBody.toString("utf8");

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(
      `Invalid webhook JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("Webhook body must be a JSON object");
  }

  const outer = parsed as Record<string, unknown>;
  // Unwrap `payload` wrapper if present and it's an object
  const payload =
    typeof outer.payload === "object" &&
    outer.payload !== null &&
    !Array.isArray(outer.payload)
      ? (outer.payload as Record<string, unknown>)
      : outer;

  if (typeof payload.job_id !== "string" || payload.job_id.length === 0) {
    throw new Error("Webhook payload missing required field: job_id");
  }

  const normalized = normalizeJobPayload(payload);
  return { normalized, payload };
}

// ----------------------------------------------------------------------------
// Internal: coerce RouteRequest.body into raw bytes (or fail)
// ----------------------------------------------------------------------------

/**
 * Best-effort extraction of raw body bytes from an ElizaOS RouteRequest.
 *
 * The RouteRequest.body type is `Record<string, RouteBodyValue>` (parsed JSON),
 * but at runtime the actual Express req.body may be a Buffer or string if the
 * server is configured with express.raw() for this path.
 *
 * @returns Raw body as Buffer/string, or null if only a parsed object is available.
 */
function coerceRawBody(body: unknown): string | Buffer | null {
  if (Buffer.isBuffer(body)) return body;
  if (typeof body === "string") return body;
  // Parsed object → cannot verify signature safely
  return null;
}

// ----------------------------------------------------------------------------
// Route handler
// ----------------------------------------------------------------------------

/**
 * The ElizaOS route for receiving ScryptedAI webhooks.
 *
 * Mounted path: `/scryptedai/webhook` (runtime prefixes plugin name).
 *
 * Flow:
 * 1. Extract raw body → if only parsed object available, fail 400 with guidance.
 * 2. Verify signature (constant-time, fail-closed).
 * 3. Parse + unwrap payload.
 * 4. Hand normalized result to ScryptedAIService.ingestWebhook() — that method
 *    handles idempotency and terminal processing.
 * 5. Return 200 immediately (even if idempotent/duplicate) so ScryptedAI
 *    doesn't retry unnecessarily.
 */
export const scryptedaiWebhookRoute: Route = {
  path: WEBHOOK_ROUTE_PATH,
  type: "POST",
  public: true,
  name: "scryptedai-webhook",
  handler: async (req, res, runtime: IAgentRuntime): Promise<void> => {
    // --- 1. Raw body extraction ---
    const rawBody = coerceRawBody(req.body);
    if (rawBody == null) {
      runtime.logger.error(
        "[scryptedai] Webhook received pre-parsed body — cannot verify HMAC signature. " +
          "Configure express.raw({ type: 'application/json' }) for the /scryptedai/webhook path.",
      );
      res.status(400).json({
        success: false,
        error: "raw_body_required",
        message:
          "Signature verification requires the raw request body. " +
          "Server middleware must be configured with express.raw() for this path.",
      });
      return;
    }

    // --- 2. Signature verification ---
    const headers = req.headers ?? {};
    const sigHeader =
      headers["x-webhook-signature"] ?? headers["X-Webhook-Signature"];
    const signature = Array.isArray(sigHeader) ? sigHeader[0] : sigHeader;

    // Resolve secret: env fallback (service-level invocation may override per-job,
    // but we verify against the configured default here; per-job secrets require
    // a lookup by job_id which means parsing first — we use env as the primary).
    const secret = runtime.getSetting(ENV_WEBHOOK_SECRET);
    if (typeof secret !== "string" || !secret) {
      runtime.logger.error(
        `[scryptedai] Webhook received but ${ENV_WEBHOOK_SECRET} is not configured — rejecting`,
      );
      res.status(500).json({
        success: false,
        error: "webhook_secret_not_configured",
      });
      return;
    }

    if (!verifyWebhookSignature(rawBody, signature, secret)) {
      runtime.logger.warn(
        "[scryptedai] Webhook signature verification failed — rejecting",
      );
      res.status(401).json({ success: false, error: "invalid_signature" });
      return;
    }

    // --- 3. Parse + unwrap ---
    let parsed: ParsedWebhookPayload;
    try {
      parsed = parseWebhookPayload(rawBody);
    } catch (error) {
      runtime.logger.warn(
        `[scryptedai] Webhook payload parse error: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
      res.status(400).json({
        success: false,
        error: "invalid_payload",
        message: error instanceof Error ? error.message : String(error),
      });
      return;
    }

    // --- 4. Dispatch to service ---
    const service = runtime.getService<ScryptedAIService>(
      SCRYPTEDAI_SERVICE_TYPE,
    );
    if (!service) {
      runtime.logger.error(
        "[scryptedai] Webhook received but service not available",
      );
      res.status(503).json({
        success: false,
        error: "service_unavailable",
      });
      return;
    }

    const ingestResult = service.ingestWebhook(parsed.normalized);

    // --- 5. Always 200 on successful ingest (idempotent or not) ---
    res.status(200).json({
      success: true,
      idempotent: ingestResult.idempotent,
      jobId: parsed.normalized.jobId,
      status: parsed.normalized.status,
    });
  },
};
