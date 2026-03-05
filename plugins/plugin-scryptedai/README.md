# @elizaos/plugin-scryptedai

ScryptedAI multimodal generation provider for ElizaOS — images, videos, and text via `api.scrypted.ai`. Handles async job lifecycle with webhook callbacks and polling fallback.

## Features

- **Per-endpoint client** — 18+ named invoke methods for all ScryptedAI generation endpoints (Nano Banana, Flux, Seedream, Sora 2, Veo 3, Hailuo, Nova Reel, Topaz upscale, text generation)
- **Recipe execution** — invoke user-defined recipes by ID
- **Unified job lifecycle** — single terminal processor handles both webhook and polling completion paths; idempotent on `(jobId, status)`
- **Webhook handler** — HMAC-SHA256 signature verification (constant-time), `payload` wrapper unwrapping, fail-closed raw-body requirement
- **Polling fallback** — type-based backoff windows; 502/503/504 and network errors treated as transient
- **Response normalization** — canonical `{jobId, status, result, error, imageUrl, videoUrl, text}` extracted from all variant shapes

## Installation

Add the plugin to your character's plugin list and set environment variables:

```typescript
import { scryptedaiPlugin } from "@elizaos/plugin-scryptedai";

export const character = {
  name: "...",
  plugins: [scryptedaiPlugin],
  // ...
};
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `SCRYPTEDAI_BEARER_TOKEN` | **Yes** | API token (must start with `scrypted_`, min 20 chars). Service fails to start without it. |
| `SCRYPTEDAI_WEBHOOK_SECRET` | For webhooks | Shared secret for HMAC-SHA256 webhook signature verification. If unset, webhook route returns 500. |
| `SCRYPTEDAI_BASE_URL` | No | Override the API base URL (defaults to `https://api.scrypted.ai`). |

## Webhook Setup

The plugin registers `POST /scryptedai/webhook` (public route). To use it:

### 1. Configure raw-body middleware

HMAC signature verification requires the **exact raw bytes** ScryptedAI signed. If your server pre-parses JSON, the signature check cannot work — the route will return `400 raw_body_required`.

For Express-based servers, register `express.raw()` **before** your JSON body parser for this path:

```typescript
app.use("/scryptedai/webhook", express.raw({ type: "application/json" }));
app.use(express.json()); // global JSON parser — must come AFTER
```

### 2. Expose the webhook URL

For local development, use a tunnel (e.g. `ngrok http 3000`). Pass the public URL as `webhookUrl` when invoking jobs:

```typescript
const svc = runtime.getService<ScryptedAIService>("scryptedai");
await svc.startImageGeneration("invokeNanoBananaGeneration", {
  prompt: "a fox in a spacesuit",
}, {
  webhookUrl: "https://abc123.ngrok.io/scryptedai/webhook",
});
```

If `webhookUrl` is omitted, the service falls back to polling automatically.

## Usage

### Via the service (recommended — tracks jobs + polling fallback)

```typescript
import type { ScryptedAIService } from "@elizaos/plugin-scryptedai";

const svc = runtime.getService<ScryptedAIService>("scryptedai");

// Start an image generation (returns immediately with job_id)
const { jobId } = await svc.startImageGeneration(
  "invokeNanoBananaGeneration",
  { prompt: "a fox in a spacesuit" },
);

// Listen for terminal events (fires for webhook OR polling completion)
const unsubscribe = svc.onTerminal((result) => {
  if (result.jobId === jobId && result.status === "completed") {
    console.log("Image ready:", result.imageUrl);
  }
});

// Or inspect the job record directly
const record = svc.getJob(jobId);
```

### Via the client directly (no tracking)

```typescript
import { ScryptedClient } from "@elizaos/plugin-scryptedai";

const client = new ScryptedClient({
  bearerToken: process.env.SCRYPTEDAI_BEARER_TOKEN!,
  // baseUrl defaults to https://api.scrypted.ai
});

const response = await client.invokeNanoBananaGeneration({
  prompt: "a fox in a spacesuit",
});
const status = await client.getJobStatus(response.job_id);
```

### Permissionless bootstrap (no token yet)

The client can be constructed **without** a token, then `createAccount()` will obtain and auto-adopt one:

```typescript
import { ScryptedClient } from "@elizaos/plugin-scryptedai";

// Construct tokenless — uses default https://api.scrypted.ai
const client = new ScryptedClient();

// POST /accounts — no auth required, returns a fresh bearer token
const account = await client.createAccount();
console.log("New token:", account.bearer_token);
console.log("User ID:", account.user_id);

// Token is now auto-adopted — all subsequent calls are authenticated
const info = await client.getAccountInfo(); // GET /accounts/me
const response = await client.invokeTextGeneration({ user_prompt: "hello" });
```

Save `account.bearer_token` — it's only returned once.

### Standalone utilities (exported independently)

```typescript
import {
  verifyWebhookSignature,
  parseWebhookPayload,
  pollJobToCompletion,
  normalizeJobPayload,
} from "@elizaos/plugin-scryptedai";

// Verify a webhook anywhere (e.g. custom server)
const valid = verifyWebhookSignature(rawBody, sigHeader, secret);

// Block until a job finishes (for scripts)
const { result, timedOut } = await pollJobToCompletion(client, jobId, {
  jobType: "image",
});
```

## Available Endpoints

### Images
- `invokeImageGeneration` (AWS Canvas)
- `invokeNanoBananaGeneration` / `invokeNanoBananaProGeneration`
- `invokeNanoBananaEditGeneration` / `invokeNanoBananaProEditGeneration` (requires `image_urls[]`)
- `invokeSeedream4Generation`
- `invokeFlux2ProGeneration`
- `invokeGrokImagineImageGeneration`

### Videos
- `invokeVideoGeneration` (generic)
- `invokeNovaReelGeneration`
- `invokeHailuo23Generation` / `invokeVeo3Generation` / `invokeVeo31Generation` / `invokeSora2Generation` / `invokeSora2I2VGeneration` / `invokeGrokImagineI2VGeneration` (all require `image_url`)
- `invokeTopazVideoUpscale` (requires `video_url`)

### Text
- `invokeTextGeneration` (Nova Pro, requires `user_prompt`)

### Recipes & Jobs
- `invokeRecipe(recipeId, inputData)`
- `getRecipeInterface(recipeId)`
- `getJobStatus(jobId)`
- `cancelJob(jobId)`

### Account
- `createAccount()` — permissionless; obtains + auto-adopts a bearer token
- `getAccountInfo()` — requires token
- `getBearerToken()` — current adopted token (or `undefined`)

## Operational Notes

- **Polling windows** — text: ~1min max, image: ~5min, video: ~30min. Transient 502/503/504 during polling is absorbed; the job is **not** marked failed.
- **Retry exclusions** — auth, validation, and payment (402) errors are never retried.
- **Idempotency** — duplicate webhook deliveries and webhook/polling races are safe; first terminal writer wins.
- **Job store** — in-memory, lost on restart. Persist `jobId` yourself if you need recovery; call `client.getJobStatus(jobId)` after restart.

## Development

```bash
bun install
bun run typecheck    # TypeScript — note: @elizaos/core source has pre-existing errors (missing generated proto)
bun run lint
bun run test         # 97 tests
```
