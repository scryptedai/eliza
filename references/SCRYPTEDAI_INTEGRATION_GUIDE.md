# ScryptedAI SDK Integration Guide

This guide explains how to integrate the ScryptedAI SDK in another project: client setup, recipe and direct-generation invocations, webhooks, polling fallback, response normalization, and error handling. It reflects patterns and gotchas from a production integration.

---

## 1. SDK Overview

### 1.1 What the SDK Provides

- **ScryptedClient** – HTTP client for the ScryptedAI API (bearer token auth, retries, timeouts).
- **Recipe execution** – `invokeRecipe(recipeId, inputData, idempotencyKey?, webhookUrl?, webhookSecret?)` → returns `{ job_id, status, result?, estimated_seconds? }`.
- **Job status** – `getJobStatus(jobId)` → `{ job_id, status, result?, error?, estimated_seconds?, created_at?, updated_at? }`.
- **Direct generation endpoints** – Dedicated methods for specific models (e.g. `/generations/images/grok-imagine-image`, `/generations/videos/grok-imagine-i2v`) with typed inputs; same async job pattern (job_id + optional webhook).
- **Recipe interface** – `getRecipeInterface(recipeId)` → input/output JSON schemas for a recipe.
- **Exceptions** – `ScryptedError`, `ScryptedAPIError`, `ScryptedAuthenticationError`, `ScryptedPaymentError`, `ScryptedValidationError`, `ScryptedNetworkError`, `ScryptedTimeoutError`, `ScryptedRateLimitError`.
- **x402** – Optional payment handler for 402 Payment Required (permissionless flows).

### 1.2 Initialization

```typescript
import { ScryptedClient } from './vendor/scryptedai'; // or your path

const bearerToken = process.env.SCRYPTEDAI_BEARER_TOKEN;
if (!bearerToken) {
  throw new Error('SCRYPTEDAI_BEARER_TOKEN is required');
}

const client = new ScryptedClient(
  bearerToken,
  'https://api.scrypted.ai',  // baseUrl (default)
  30,                         // timeout seconds (default)
  3,                          // maxRetries (default)
  { keepAlive: true }         // optional transport options
);

// When shutting down (e.g. scripts): client.close({ force: true });
```

**Bearer token format:** Must start with `scrypted_` and meet minimum length (see `vendor/scryptedai/constants.ts`: `TOKEN_PREFIX`, `TOKEN_MIN_LENGTH`). Invalid format throws at construction.

---

## 2. Invoking Work

### 2.1 Recipe Execution (Generic)

- **Recipe ID format:** Opaque string; often includes colons (e.g. `scrypted:converter:eli5-title-cleaner`). Always **URL-encode** when building paths (SDK does this for `invokeRecipe` and `getRecipeInterface`).
- **Input:** Must match the recipe’s input schema. Use `getRecipeInterface(recipeId)` to inspect.
- **Idempotency:** Pass an idempotency key (e.g. UUID or `IdempotencyKey.generate().toString()`) to avoid duplicate work on retries.
- **Webhook:** Provide `webhookUrl` and `webhookSecret` so ScryptedAI can POST completion/failure; without them you must rely only on polling.

```typescript
const response = await client.invokeRecipe(
  'scrypted:converter:eli5-title-cleaner',
  { input_text: '...' },
  idempotencyKey,
  webhookUrl,
  webhookSecret
);
// response: { job_id: string, status: 'pending'|'processing'|'completed'|'failed', result?, estimated_seconds? }
```

### 2.2 Direct Generation Endpoints

The SDK and API expose **direct endpoints** for specific models (images/videos), e.g.:

- Images: `/generations/images/aws-canvas`, `/generations/images/seedream-4`, `/generations/images/flux-2-pro`, `/generations/images/grok-imagine-image`, Nano-Banana (and edit) variants, etc.
- Videos: `/generations/videos/hailuo-2-3-i2v`, `/generations/videos/veo-3-i2v`, `/generations/videos/sora2-pro-i2v`, `/generations/videos/grok-imagine-i2v`, Topaz upscale, etc.

These are invoked via dedicated client methods (e.g. `invokeGrokImagineImage`, `invokeHailuo23I2V`). Each returns the same shape: `{ job_id, status, result?, estimated_seconds? }`. Always pass webhook URL and secret when you want callbacks.

**Capabilities:** Available endpoints and parameters are defined in the SDK (e.g. `vendor/scryptedai/constants.ts` and client methods). For new models, add the endpoint constant and a corresponding invoke method following existing patterns.

---

## 3. Webhooks

### 3.1 Endpoint and Body

- **URL:** You must expose a **POST** endpoint (e.g. `https://your-app.com/api/scryptedai/webhook`).
- **Body:** ScryptedAI sends JSON. The **payload may be nested** under a `payload` key or at top level; handle both:

  ```typescript
  const parsed = JSON.parse(rawBody);
  const payload = parsed.payload ?? parsed;
  const { job_id, status: rawStatus, result, error } = payload;
  ```

- **Required fields:** `job_id` (string), `status` (string). Optional: `result` (object), `error` (string or object with `message`).

### 3.2 Signature Verification

- **Header:** `X-Webhook-Signature` (or `x-webhook-signature`).
- **Algorithm:** HMAC-SHA256 of the **raw request body** (string), hex-encoded.
- **Secret:** Same value you passed as `webhookSecret` when invoking the recipe/generation.

```typescript
const crypto = require('crypto');
function verifyWebhookSignature(payload: string, signature: string | undefined, secret: string | undefined): boolean {
  if (!signature || !secret) return false;
  const expected = crypto.createHmac('sha256', secret).update(payload).digest('hex');
  return crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expected));
}
```

- **Critical:** Use the **raw body** (e.g. `express.raw({ type: 'application/json' })`) before any JSON parsing so the signature matches what ScryptedAI signed.

### 3.3 Idempotency

- Webhooks can be delivered more than once. Before updating state or side effects, check whether this `(job_id, status)` has already been processed (e.g. by DB lookup).
- If already processed, return 200 with a body like `{ success: true, idempotent: true }` so ScryptedAI does not retry unnecessarily.

### 3.4 Status Values

- **Normalize status** to lowercase before branching: `const status = (rawStatus || '').toLowerCase();`
- Treat `completed` and `failed` as terminal; handle `pending`/`processing` only if you need to track them (e.g. logging). Completion handling should use `result`; failure handling should use `error`.

---

## 4. Polling (Fallback and Recovery)

### 4.1 When to Poll

- **Primary:** Rely on webhooks for completion/failure.
- **Fallback:** Poll when:
  - Webhooks are not configured, or
  - Webhook delivery fails (e.g. network, your server down), or
  - You are **recovering** after a restart (in-flight jobs have no guarantee of webhook delivery).

### 4.2 Job Status API

- **Endpoint:** `GET /jobs/{job_id}` (see `ENDPOINTS.jobs` in the SDK).
- **Response:** `JobStatusResponse`: `job_id`, `status`, optional `result`, `error`, `estimated_seconds`, timestamps. Status values: `pending`, `processing`, `completed`, `failed`.

**Gotcha – Status casing:** The API may return status in different casings (e.g. `FAILED`, `Completed`). Always normalize to lowercase before comparing: `(response.status || '').toLowerCase()`.

### 4.3 Polling Strategy

- **Type-based windows:** Different job types (text, image, video) have different typical runtimes. Start polling after a “minimum age” (e.g. don’t treat as stuck in the first N seconds). Give up after a “max wait” (e.g. 1 min text, 5 min image, 30 min video) and mark as failed/refund.
- **Backoff:** Use a backoff between polls (e.g. 15s, 30s, 60s) to avoid hammering the API; sinusoidal or exponential patterns both work.
- **Retriable errors:** On **5xx** or gateway-style errors (502, 503, 504, “bad gateway”, “gateway timeout”, “service unavailable”), treat the job as still in progress and retry polling later; do not immediately mark as failed.

### 4.4 Webhook vs Polling Race

- The same job can complete via **webhook** and **polling** (e.g. webhook delayed, poll returns first). Design for **one writer wins**:
  - Before applying a completion or failure, check again if the job is already in a terminal state (e.g. in DB).
  - Use a single “process completion” path (e.g. one function that updates DB, transfers media, broadcasts events) and call it from both webhook and polling; that function should be idempotent for the same generation/job.

---

## 5. Response Formats and Normalization

### 5.1 API vs Webhook Shape

- **Invoke response:** `{ job_id, status, result?, estimated_seconds? }`. `result` may be present only when status is `completed` (e.g. fast sync completion).
- **Job status (polling):** Same shape; `result` and `error` populated when terminal.
- **Webhook payload:** Often `{ job_id, status, result?, error? }`. The **result** structure varies by recipe/endpoint (see below). Your webhook handler may receive `result` at top level or inside a wrapper; normalize before passing to business logic.

### 5.2 Result Structure by Output Type

- **Text:** Often `result.text` or `result_data.text` or nested under `output`. Parse once and support multiple locations.
- **Image:** Multiple possible shapes. Prefer a single adapter that checks in order:
  - `result_data.images[0].asset_url` (or `cdn_url`, `cloudfront_url`, `url`)
  - `result.images[0]....`
  - Legacy: `result.image.url`, `result.image_url`, `result.imageUrl`
- **Video:** Similarly:
  - `result_data.video` (singular object) or `result_data.videos[0]`
  - `result.video`, `result.videos[0]`
  - Legacy: `result.video_url`, `result.videoUrl`

**Recommendation:** Implement a small **response normalizer** (e.g. “adapter”) that maps `job_id` → `jobId`, `result_data` / `result` / `output` → a single `result`, and `error_message` / `error.message` / `error` (string) → a single `error` string. Use this for both polling and webhook paths so the rest of your app sees one shape.

### 5.3 Snake_case vs CamelCase

- The API uses **snake_case** (`job_id`, `result_data`, `estimated_seconds`, `error_message`). Your normalizer can expose camelCase internally if your codebase prefers it.

---

## 6. Errors and Retries

### 6.1 SDK Exceptions

- **ScryptedAuthenticationError** – Invalid or missing bearer token. Do not retry; fix config.
- **ScryptedValidationError** – Bad input (e.g. invalid token format). Do not retry; fix request.
- **ScryptedPaymentError** – 402 Payment Required. Handle via x402 flow if you use permissionless; otherwise do not retry blindly.
- **ScryptedRateLimitError** – 429. Use `retryAfter` (seconds) if present; back off and retry.
- **ScryptedAPIError** – Other HTTP errors (4xx/5xx). Check `statusCode` and `responseData`; for 5xx or gateway errors, retry with backoff.
- **ScryptedTimeoutError** / **ScryptedNetworkError** – Timeout or network failure. Retry with backoff.

The SDK’s `retryWithBackoff` does **not** retry auth, validation, or payment errors; it does retry others. For **job status** polling, treat 502/503/504 and gateway-like errors as transient and retry later rather than marking the job failed.

### 6.2 Validation Errors (422)

- The SDK logs 422 response bodies. Use them to fix input (e.g. required field missing, schema mismatch). Do not retry the same payload without changing input.

### 6.3 Job-Level Failures

- When `status === 'failed'`, the `error` field (or `error_message`) contains the provider message. Store it for support/debugging. Do not surface raw provider messages to end users; map to safe, user-facing messages and log the full error server-side.

---

## 7. Checklist and Gotchas Summary

| Topic | Recommendation / Gotcha |
|-------|--------------------------|
| **Auth** | Set `SCRYPTEDAI_BEARER_TOKEN`; token must start with `scrypted_` and meet min length. |
| **Recipe IDs** | URL-encode when building paths; SDK encodes in `invokeRecipe` / `getRecipeInterface`. |
| **Webhook URL** | Must be HTTPS in production; in dev use a tunnel (e.g. ngrok) if you want callbacks. |
| **Webhook body** | Parse raw body for signature; then support both `payload` wrapper and top-level fields. |
| **Webhook secret** | Verify HMAC-SHA256 on raw body; use same secret as in invoke. |
| **Idempotency** | Webhook and polling can both see completion; make completion handling idempotent. |
| **Status** | Always normalize status to lowercase; API may return `FAILED`, `Completed`, etc. |
| **Result location** | Result may be in `result`, `result_data`, or `output`; images/videos in various keys (see §5.2). |
| **Polling** | Use type-based min/max wait and backoff; treat 502/503/504 as retriable. |
| **Recovery** | After restarts, poll any in-flight jobs you stored by `job_id`; don’t rely only on webhooks. |
| **Errors** | Don’t retry auth/validation/402; do retry 5xx and network/timeout; map job errors to user-safe messages. |

---

## 8. References in This Repo

- **SDK:** `vendor/scryptedai/` (`client.ts`, `models.ts`, `exceptions.ts`, `constants.ts`, `utils.ts`)
- **Service wrapper:** `server/scryptedai-service.ts` (webhook URL generation, invoke helpers)
- **Webhook handler:** `server/scryptedai-webhook-router.ts` (signature, idempotency, status normalization, completion/failure handling)
- **Response normalization:** `server/adapters/scryptedai-adapter.ts` (status, result, error, image/video URL extraction)
- **Recovery and polling:** `server/job-recovery-service.ts`, `server/config/recovery-config.ts` (type-based windows, backoff, 5xx retriable)
- **Job completion (unified):** `server/job-processor.ts` (`processJobCompletion` used by both webhook and polling)

This guide reflects the Delula integration as of the last update; for the latest API contract and endpoint list, refer to ScryptedAI’s official API documentation and the SDK constants/methods.
