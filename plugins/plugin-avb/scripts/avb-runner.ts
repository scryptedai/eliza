/**
 * avb-runner — boot vanilla ElizaOS with the AVB plugin active.
 *
 * No custom character is inlined here. Character discovery is delegated
 * entirely to core's loadCharacter(), which follows the standard search
 * order (./character.ts, ./agent.ts, ./character.json, ./agent.json,
 * ~/.eliza/character.json) and falls back to core's own minimal Eliza
 * if nothing is found. Whatever character core resolves is what the
 * AVB pipeline sees — no invented attributes.
 *
 * The AVB service self-triggers avatar generation on boot
 * (AVB_AUTOGEN_ON_BOOT default-on). This runner only boots and observes.
 *
 * Usage (from plugin dir):
 *   bun --env-file=../../.env scripts/avb-runner.ts
 *
 * Opt-out of auto-gen: AVB_AUTOGEN_ON_BOOT=false
 * Override character:  place a character.{ts,json} in cwd
 */
import {
  AgentRuntime,
  loadCharacter,
  type Memory,
  type Task,
} from "@elizaos/core";
import { ffmPlugin } from "@elizaos/plugin-ffm";
import { scryptedaiPlugin } from "@elizaos/plugin-scryptedai";
import { type AvbPhaseMetadata, avbPlugin } from "../src/index.ts";

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function phaseOf(task: Task): string {
  const md = task.metadata as unknown as AvbPhaseMetadata | undefined;
  return md?.phase ?? "?";
}

function jobIdOf(task: Task): string {
  const md = task.metadata as unknown as AvbPhaseMetadata | undefined;
  return md?.scryptedJobId ?? "(pending)";
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

async function main() {
  // --- Character: delegate to core's discovery path ---
  // loadCharacter() handles: file search, validation, env-secret import,
  // encryption salt, and the minimal-Eliza fallback. We add nothing.
  const { character, filePath, fromDefault } = await loadCharacter();

  console.log("─".repeat(60));
  console.log("Booting ElizaOS runtime...");
  console.log("  character:", character.name);
  console.log(
    "  source:   ",
    fromDefault
      ? "core default (no character file found)"
      : `${filePath}`,
  );
  console.log("  plugins:  ", [
    scryptedaiPlugin.name,
    ffmPlugin.name,
    avbPlugin.name,
  ]);

  // --- Preflight: scryptedai bearer token must be present in env ---
  // We do NOT inject it into the character. Instead, process.env is
  // passed as runtime `settings`, and runtime.getSetting() falls through
  // to it after character.secrets / character.settings miss.
  if (!process.env.SCRYPTEDAI_BEARER_TOKEN) {
    console.error(
      "✗ SCRYPTEDAI_BEARER_TOKEN not set. Run with --env-file=../../.env",
    );
    process.exit(1);
  }

  // --- Boot ---
  // Character is untouched. `settings: process.env` gives getSetting()
  // its env fallback without modifying any character attributes.
  const runtime = new AgentRuntime({
    character,
    // Order matters only for the dependencies graph; AVB awaits both
    // scryptedai and ffm via getServiceLoadPromise inside start(), so the
    // personality bootstrap runs after both providers are ready.
    plugins: [scryptedaiPlugin, ffmPlugin, avbPlugin],
    settings: process.env as Record<string, string | undefined>,
    logLevel: "info",
  });

  await runtime.initialize({ allowNoDatabase: true });
  console.log("✓ Runtime initialized (InMemoryDatabaseAdapter)");

  await runtime.getServiceLoadPromise("scryptedai");
  await runtime.getServiceLoadPromise("ffm");
  await runtime.getServiceLoadPromise("avb");
  console.log("✓ scryptedai + ffm + avb services available");
  console.log(
    "✓ NOT calling createRun() — waiting for autonomous trigger...\n",
  );

  // --- Observe: poll by base "avb" tag (runId is internal to the service) ---
  const roomId = runtime.agentId;
  const start = Date.now();
  const MAX_WAIT_MS = 6 * 60 * 1000;
  let lastPhase = "";
  let sawAnyTask = false;

  console.log("Observing pipeline progress (poll every 3s)...\n");
  while (Date.now() - start < MAX_WAIT_MS) {
    const tasks = await runtime.getTasks({ tags: ["avb"] });
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);

    if (tasks.length === 0) {
      if (sawAnyTask) {
        console.log(
          `  [${elapsed}s] no phase tasks remaining → pipeline terminal`,
        );
        break;
      }
      console.log(`  [${elapsed}s] waiting for autonomous trigger...`);
      await sleep(1000);
      continue;
    }

    sawAnyTask = true;
    const t = tasks[0];
    const phase = phaseOf(t);
    const jobId = jobIdOf(t);
    if (phase !== lastPhase) {
      console.log(`  [${elapsed}s] phase=${phase} job=${jobId}`);
      lastPhase = phase;
    } else {
      console.log(`  [${elapsed}s] ${phase} (job=${jobId}, waiting...)`);
    }

    await sleep(3000);
  }

  // --- Fetch result ---
  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  console.log("\n" + "─".repeat(60));

  const memories = await runtime.getMemories({
    roomId,
    tableName: "messages",
    count: 10,
  });
  const delivered = memories.find(
    (m: Memory) =>
      Array.isArray(m.content.actions) &&
      m.content.actions.includes("GENERATE_AVATAR"),
  );

  if (!delivered) {
    console.log(`✗ PIPELINE TIMED OUT after ${elapsed}s (no delivered memory)`);
    process.exitCode = 1;
  } else if (delivered.content.attachments?.[0]?.url) {
    const url = delivered.content.attachments[0].url;
    console.log(`✓ AVATAR GENERATED in ${elapsed}s`);
    console.log("  character: ", character.name, fromDefault ? "(default)" : "");
    console.log("  imagePrompt:", JSON.stringify(delivered.content.text));
    console.log("  imageUrl:   ", url);
  } else {
    console.log(`✗ PIPELINE FAILED after ${elapsed}s`);
    console.log("  message:", delivered.content.text);
    process.exitCode = 1;
  }

  await runtime.stop();
  console.log("✓ Runtime stopped");
}

main().catch((err) => {
  console.error("\n✗ avb-runner failed:", err);
  process.exit(1);
});
