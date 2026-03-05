/**
 * Live runner: boot ElizaOS with a default agent + scryptedai plugin,
 * then have the agent generate its own avatar via Seedream 4.
 *
 * Exercises the full service path:
 *   AgentRuntime → registerPlugin → ScryptedAIService.start()
 *   → startImageGeneration → polling fallback → onTerminal listener
 *
 * Run from plugin dir:
 *   bun --env-file=../../.env scripts/runner-avatar.ts
 */
import { AgentRuntime, type Character } from "@elizaos/core";
import { scryptedaiPlugin, type ScryptedAIService } from "../src/index.ts";
import type { NormalizedJobResult } from "../src/types.ts";

// --- character: a minimal default agent with enough flavor for an avatar prompt ---
const character: Character = {
  name: "Eliza",
  bio: [
    "A curious and resourceful AI agent.",
    "Enjoys solving problems, exploring ideas, and building things.",
  ],
  adjectives: ["thoughtful", "precise", "warm", "analytical"],
  topics: ["software", "design", "systems"],
  system:
    "You are Eliza, a helpful AI agent. You think clearly, act carefully, and explain yourself well.",
  templates: {},
  messageExamples: [],
  postExamples: [],
  knowledge: [],
  plugins: [],
  // Secrets: token is read here by ScryptedAIService.start() via runtime.getSetting()
  secrets: {
    SCRYPTEDAI_BEARER_TOKEN: process.env.SCRYPTEDAI_BEARER_TOKEN ?? "",
  },
  settings: {},
};

if (!character.secrets?.SCRYPTEDAI_BEARER_TOKEN) {
  console.error(
    "✗ SCRYPTEDAI_BEARER_TOKEN not set. Run with --env-file=../../.env",
  );
  process.exit(1);
}

// --- build avatar prompt from character traits ---
function buildAvatarPrompt(c: Character): string {
  const name = c.name ?? "an AI agent";
  const bio = Array.isArray(c.bio) ? c.bio.join(" ") : (c.bio ?? "");
  const adjectives = (c.adjectives ?? []).slice(0, 4).join(", ");
  const topics = (c.topics ?? []).slice(0, 3).join(", ");

  return (
    `A character portrait representing ${name}, an AI agent. ` +
    `Personality: ${adjectives}. Interests: ${topics}. ` +
    `${bio} ` +
    `Style: clean digital illustration, friendly expression, soft lighting, ` +
    `modern minimal aesthetic, suitable as a profile avatar.`
  );
}

// --- main ---
async function main() {
  console.log("─".repeat(60));
  console.log("Booting ElizaOS runtime...");
  console.log("  character:", character.name);
  console.log("  plugins:  ", [scryptedaiPlugin.name]);

  const runtime = new AgentRuntime({
    character,
    plugins: [scryptedaiPlugin],
    logLevel: "info",
  });

  await runtime.initialize({ allowNoDatabase: true });
  console.log("✓ Runtime initialized");

  // --- wait for service registration (async, happens post-initialize) ---
  const svc = (await runtime.getServiceLoadPromise(
    "scryptedai",
  )) as ScryptedAIService;
  console.log("✓ ScryptedAIService available");

  // --- build prompt from agent's own character ---
  const prompt = buildAvatarPrompt(runtime.character);
  console.log("\nAvatar prompt:");
  console.log(" ", JSON.stringify(prompt));

  // --- invoke via service (polling fallback, no webhook) ---
  console.log("\nInvoking Seedream 4 via service.startImageGeneration...");
  const { jobId, response } = await svc.startImageGeneration(
    "invokeSeedream4Generation",
    { prompt, num_images: 1 },
  );

  console.log("→ job_id:", jobId);
  console.log("→ status:", response.status);

  // --- wait for terminal via listener (polling fallback drives this) ---
  console.log("\nWaiting for terminal event (polling fallback active)...\n");
  const start = Date.now();
  const result = await new Promise<NormalizedJobResult>((resolve) => {
    const unsubscribe = svc.onTerminal((r: NormalizedJobResult) => {
      if (r.jobId !== jobId) return;
      const elapsed = ((Date.now() - start) / 1000).toFixed(1);
      console.log(`  [${elapsed}s] terminal → ${r.status}`);
      unsubscribe();
      resolve(r);
    });
  });

  // --- report ---
  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  console.log("\n" + "─".repeat(60));
  if (result.status === "completed") {
    console.log(`✓ AVATAR GENERATED in ${elapsed}s`);
    console.log("  imageUrl:", result.imageUrl);
  } else {
    console.log(`✗ GENERATION FAILED after ${elapsed}s`);
    console.log("  status:", result.status);
    console.log("  error: ", result.error ?? "(none)");
    process.exitCode = 1;
  }

  // --- cleanup ---
  await runtime.stop();
  console.log("✓ Runtime stopped");
}

main().catch((err) => {
  console.error("\n✗ Runner failed:", err);
  process.exit(1);
});
