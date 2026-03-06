/**
 * Live proving test for the AVB pipeline.
 *
 * Boots ElizaOS with both plugins (scryptedai + avb), starts an avatar run,
 * then polls the run's Task rows until the pipeline reaches terminal.
 *
 * Exercises the full db-persisted state machine:
 *   createRun() → TEXT_PHASE task → IMAGE_PHASE task → DELIVER task → done
 *
 * Unlike plugin-scryptedai/scripts/runner-avatar.ts (which calls the service
 * directly and blocks), this script demonstrates the NON-BLOCKING flow:
 * we kick off the run and observe progress externally via getRunTasks().
 *
 * Run from plugin dir:
 *   bun --env-file=../../.env scripts/prove.ts
 */
import {
  AgentRuntime,
  type Character,
  type Memory,
  type Task,
} from "@elizaos/core";
import { scryptedaiPlugin } from "@elizaos/plugin-scryptedai";
import { avbPlugin, type AvbPhaseMetadata, AvbService } from "../src/index.ts";

// ----------------------------------------------------------------------------
// Character: default eliza agent (enough identity for a meaningful digest)
// ----------------------------------------------------------------------------

const character: Character = {
  name: "Eliza",
  bio: [
    "A curious and resourceful AI agent.",
    "Enjoys solving problems, exploring ideas, and building things.",
    "Equal parts philosopher and engineer.",
  ],
  adjectives: ["thoughtful", "precise", "warm", "analytical", "witty"],
  topics: ["software", "design", "systems", "philosophy"],
  style: {
    all: ["concise", "direct", "playful"],
  },
  system:
    "You are Eliza, a helpful AI agent. You think clearly, act carefully, and explain yourself well.",
  templates: {},
  messageExamples: [],
  postExamples: [],
  knowledge: [],
  plugins: [],
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
  console.log("─".repeat(60));
  console.log("Booting ElizaOS runtime (in-memory db)...");
  console.log("  character:", character.name);
  console.log("  plugins:  ", [scryptedaiPlugin.name, avbPlugin.name]);

  const runtime = new AgentRuntime({
    character,
    plugins: [scryptedaiPlugin, avbPlugin],
    logLevel: "info",
  });

  await runtime.initialize({ allowNoDatabase: true });
  console.log("✓ Runtime initialized (InMemoryDatabaseAdapter)");

  // Wait for both services
  await runtime.getServiceLoadPromise("scryptedai");
  const avb = (await runtime.getServiceLoadPromise("avb")) as AvbService;
  console.log("✓ scryptedai + avb services available");

  // --- Start a run (non-blocking) ---
  // Use the agent's own agentId as the roomId for the prove script.
  const roomId = runtime.agentId;
  console.log("\nStarting avatar run (room=" + roomId + ")...");
  const runId = await avb.createRun(roomId);
  console.log("→ runId:", runId);

  // --- Observe progress by polling getRunTasks() ---
  // The pipeline advances itself via TaskService ticks; we just watch.
  const start = Date.now();
  const MAX_WAIT_MS = 6 * 60 * 1000; // text (90s) + image (360s) headroom
  let lastPhase = "";

  console.log("\nObserving pipeline progress (poll every 3s)...\n");
  while (Date.now() - start < MAX_WAIT_MS) {
    const tasks = await avb.getRunTasks(runId);
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);

    if (tasks.length === 0) {
      console.log(`  [${elapsed}s] no phase tasks remaining → pipeline terminal`);
      break;
    }

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

  // --- Fetch the delivered memory ---
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
  console.error("\n✗ prove.ts failed:", err);
  process.exit(1);
});
