/**
 * Live test: Seedream 4 image generation.
 *
 * Reads SCRYPTEDAI_BEARER_TOKEN from env (never logged).
 * Invokes /generations/images/seedream-4, polls to completion, prints result.
 *
 * Run from plugin dir:
 *   bun --env-file=../../.env scripts/test-seedream.ts
 */
import { ScryptedClient } from "../src/client.ts";
import { pollJobToCompletion } from "../src/polling.ts";
import type { NormalizedJobResult } from "../src/types.ts";

const PROMPT =
  "a single red fox sitting in a snowy forest clearing, soft morning light, photorealistic";

// --- token check (value never printed) ---
const token = process.env.SCRYPTEDAI_BEARER_TOKEN;
if (!token) {
  console.error(
    "✗ SCRYPTEDAI_BEARER_TOKEN not set. Run with --env-file=../../.env",
  );
  process.exit(1);
}

const client = new ScryptedClient({ bearerToken: token });

// --- invoke ---
console.log("POST /generations/images/seedream-4");
console.log("  prompt:", JSON.stringify(PROMPT));

const start = Date.now();
const invoke = await client.invokeSeedream4Generation({
  prompt: PROMPT,
  num_images: 1,
});

console.log("\n→ job_id:", invoke.job_id);
console.log("→ status:", invoke.status);
if (invoke.estimated_seconds) {
  console.log("→ estimated:", invoke.estimated_seconds, "seconds");
}

// --- poll to completion ---
console.log("\nPolling (jobType=image, max 5 min, backoff 5→10→20→30s)...");

let attempts = 0;
const { result, timedOut, completed } = await pollJobToCompletion(
  client,
  invoke.job_id,
  {
    jobType: "image",
    onPoll: (normalized: NormalizedJobResult) => {
      attempts++;
      const elapsed = ((Date.now() - start) / 1000).toFixed(1);
      process.stdout.write(
        `  [${elapsed}s] attempt=${attempts} status=${normalized.status}\n`,
      );
    },
  },
);

const elapsed = ((Date.now() - start) / 1000).toFixed(1);

// --- report ---
console.log("\n" + "─".repeat(60));
if (timedOut) {
  console.log(`✗ TIMED OUT after ${elapsed}s (${attempts} polls)`);
  console.log("  Last status:", result.status);
  process.exit(1);
}

if (!completed || result.status !== "completed") {
  console.log(`✗ FAILED after ${elapsed}s`);
  console.log("  status:", result.status);
  console.log("  error: ", result.error ?? "(none)");
  if (result.result) {
    console.log("  raw result:", JSON.stringify(result.result, null, 2));
  }
  process.exit(1);
}

console.log(`✓ COMPLETED in ${elapsed}s (${attempts} polls)`);
console.log("  jobId:   ", result.jobId);
console.log("  status:  ", result.status);
console.log("  imageUrl:", result.imageUrl ?? "(not extracted — see raw below)");

if (!result.imageUrl && result.result) {
  // Normalization didn't find an imageUrl in known shapes — dump raw for inspection
  console.log("\n  Raw result payload:");
  console.log(JSON.stringify(result.result, null, 2));
}
