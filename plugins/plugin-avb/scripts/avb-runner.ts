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
import { AgentRuntime, loadCharacter } from "@elizaos/core";
import { scryptedaiPlugin } from "@elizaos/plugin-scryptedai";
import { avbPlugin } from "../src/index.ts";
import { observeAvbPipeline, reportAvbResult } from "./_observe.ts";

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
    fromDefault ? "core default (no character file found)" : `${filePath}`,
  );
  console.log("  plugins:  ", [scryptedaiPlugin.name, avbPlugin.name]);

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
    plugins: [scryptedaiPlugin, avbPlugin],
    settings: process.env as Record<string, string | undefined>,
    logLevel: "info",
  });

  await runtime.initialize({ allowNoDatabase: true });
  console.log("✓ Runtime initialized (InMemoryDatabaseAdapter)");

  await runtime.getServiceLoadPromise("scryptedai");
  await runtime.getServiceLoadPromise("avb");
  console.log("✓ scryptedai + avb services available");
  console.log(
    "✓ NOT calling createRun() — waiting for autonomous trigger...\n",
  );

  const result = await observeAvbPipeline(runtime);
  reportAvbResult(result, [
    `  character:  ${character.name}${fromDefault ? " (default)" : ""}`,
  ]);

  await runtime.stop();
  console.log("✓ Runtime stopped");
}

main().catch((err) => {
  console.error("\n✗ avb-runner failed:", err);
  process.exit(1);
});
