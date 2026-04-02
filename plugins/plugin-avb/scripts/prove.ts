/**
 * Live proving test for the AVB pipeline — AUTONOMOUS mode.
 *
 * Boots ElizaOS with both plugins (scryptedai + avb) using an INLINE
 * character whose `secrets` carry SCRYPTEDAI_BEARER_TOKEN. This exercises
 * the character.secrets → runtime.getSetting() path specifically (vs.
 * avb-runner.ts which exercises the env→settings path with a discovered
 * character).
 *
 * The AVB service self-triggers avatar generation on boot
 * (AVB_AUTOGEN_ON_BOOT default-on). This script NEVER calls createRun() —
 * it only boots the runtime and observes. The pipeline fires itself.
 *
 * Run from plugin dir:
 *   bun --env-file=../../.env scripts/prove.ts
 *
 * To disable auto-start and drive manually, set AVB_AUTOGEN_ON_BOOT=false.
 */
import { AgentRuntime, type Character } from "@elizaos/core";
import { scryptedaiPlugin } from "@elizaos/plugin-scryptedai";
import { avbPlugin } from "../src/index.ts";
import { observeAvbPipeline, reportAvbResult } from "./_observe.ts";

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
  // `style` omitted: StyleGuides is a protobuf message type requiring
  // `$typeName`/create(); bio + adjectives + topics already give the
  // digest enough identity for a proof run.
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

  // Wait for both services to finish loading.
  // AvbService.start() self-triggers createRun() via autoStartIfNeeded()
  // because no avatar exists yet and AVB_AUTOGEN_ON_BOOT defaults to on.
  await runtime.getServiceLoadPromise("scryptedai");
  await runtime.getServiceLoadPromise("avb");
  console.log("✓ scryptedai + avb services available");
  console.log(
    "✓ NOT calling createRun() — waiting for autonomous trigger...\n",
  );

  const result = await observeAvbPipeline(runtime);
  reportAvbResult(result);

  await runtime.stop();
  console.log("✓ Runtime stopped");
}

main().catch((err) => {
  console.error("\n✗ prove.ts failed:", err);
  process.exit(1);
});
