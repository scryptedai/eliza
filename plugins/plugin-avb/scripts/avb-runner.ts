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
 * Two phases:
 *
 *   1. Avatar (foreground, ~minutes)
 *      The AVB service self-triggers avatar generation on boot
 *      (AVB_AUTOGEN_ON_BOOT default-on). This runner observes the phase
 *      transitions and reports the final image URL.
 *
 *   2. SLM16 trainer (background, indefinite)
 *      The Slm16Service spawns its worker thread during runtime.initialize()
 *      and trains continuously regardless of what the avatar pipeline is
 *      doing. After the avatar phase concludes (success, failure, or skip),
 *      this runner stays alive to keep the worker running and surfaces its
 *      progress (step, val_loss, intelligence) until SIGINT.
 *
 *      SIGINT triggers a graceful shutdown: the worker is sent {type:'stop'},
 *      checkpoints its current state to disk (up to 30s grace), and exits
 *      cleanly. Training resumes from that checkpoint on the next boot.
 *
 * Usage (from plugin dir):
 *   bun --env-file=../../.env scripts/avb-runner.ts
 *
 * Env knobs:
 *   AVB_AUTOGEN_ON_BOOT=false  — skip avatar gen, go straight to trainer watch
 *   AVB_RUNNER_PERSIST=false   — old behaviour: exit after avatar (kills trainer)
 *   SLM16_AUTOTRAIN=false      — boot without spawning the trainer worker
 *   SLM16_DATA_DIR=…           — where FineWeb .bin shards live
 *   SLM16_CHECKPOINT_DIR=…     — where lkg.int8.bin / trainer_state.bin land
 *
 * Override character: place a character.{ts,json} in cwd.
 */
import {
  AgentRuntime,
  loadCharacter,
  type Memory,
  type Task,
} from "@elizaos/core";
import { scryptedaiPlugin } from "@elizaos/plugin-scryptedai";
import {
  type AvbPhaseMetadata,
  avbPlugin,
  SLM16_SERVICE_TYPE,
  SLM16_SETTING,
  type Slm16Event,
  type Slm16Service,
} from "../src/index.ts";

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

/** "false" / "0" / false → false. Everything else (including undefined) → true. */
function envFlag(name: string, deflt: boolean): boolean {
  const v = process.env[name];
  if (v === undefined) return deflt;
  return !(v === "false" || v === "0" || v === "");
}

/**
 * Render an Slm16Event into a single log line. The service already logs
 * these to runtime.logger, but those go through eliza's pino transport;
 * we want something terser and aligned for the runner's own stdout stream.
 */
function fmtSlm16Event(e: Slm16Event): string {
  switch (e.type) {
    case "ready":
      return (
        `ready  backend=${e.backend} params=${e.paramCount.toLocaleString()} ` +
        `resumed_step=${e.resumedAtStep}`
      );
    case "step": {
      const rate = e.trainSeconds > 0 ? Math.round(e.tokensSeen / e.trainSeconds) : 0;
      return `step ${e.step}  train_loss=${e.trainLoss.toFixed(4)}  tok/s≈${rate}`;
    }
    case "val":
      return `val  ${e.step}  val_loss=${e.valLoss.toFixed(4)}  bpb=${e.valBpb.toFixed(4)}`;
    case "lkg": {
      const mb = (e.artifactBytes / 1e6).toFixed(2);
      return `LKG  ${e.step}  val_loss=${e.valLoss.toFixed(4)}  ${mb}MB ${e.underCap ? "✓" : "OVER CAP"}`;
    }
    case "error":
      return `${e.fatal ? "FATAL" : "warn "} ${e.message}`;
    case "idle":
      return `idle  reason=${e.reason}`;
  }
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
  // Slm16Service.start() runs synchronously inside this await — by the time
  // the promise resolves, spawnWorker() has already fired (or AUTOTRAIN was
  // off and it deliberately didn't). Either way the service handle is live.
  const slm16 = (await runtime.getServiceLoadPromise(
    SLM16_SERVICE_TYPE,
  )) as Slm16Service;
  console.log(
    `✓ scryptedai + avb + slm16 services available ` +
      `(trainer ${slm16.isTraining() ? "spawned" : "not spawned"})`,
  );

  // --- Decide what we're doing this run ---
  const persist = envFlag("AVB_RUNNER_PERSIST", true);
  const autogen = envFlag("AVB_AUTOGEN_ON_BOOT", true);

  // --- SIGINT → graceful shutdown ---
  // The worker checkpoints on stop, which can take up to 30s (it has to
  // finish the current optimizer step, then write trainer_state.bin +
  // meta.json). A second Ctrl+C while that's in flight aborts hard.
  let shuttingDown = false;
  const onSigint = () => {
    if (shuttingDown) {
      console.error("\n✗ Forced exit (checkpoint may be incomplete)");
      process.exit(130);
    }
    shuttingDown = true;
    console.log(
      "\n⏻ SIGINT — checkpointing trainer and shutting down (Ctrl+C again to force)…",
    );
    void runtime.stop().then(
      () => {
        console.log("✓ Runtime stopped (trainer checkpointed)");
        process.exit(0);
      },
      (e) => {
        console.error(`✗ Shutdown error: ${(e as Error).message}`);
        process.exit(1);
      },
    );
  };
  process.on("SIGINT", onSigint);
  process.on("SIGTERM", onSigint);

  // --- Phase 1: avatar pipeline observation ---
  const roomId = runtime.agentId;
  const start = Date.now();

  if (autogen) {
    console.log(
      "✓ NOT calling createRun() — waiting for autonomous trigger...\n",
    );

    const MAX_WAIT_MS = 6 * 60 * 1000;
    let lastPhase = "";
    let sawAnyTask = false;

    console.log("Observing avatar pipeline (poll every 3s)...\n");
    while (Date.now() - start < MAX_WAIT_MS && !shuttingDown) {
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
  } else {
    console.log("  AVB_AUTOGEN_ON_BOOT=false — skipping avatar phase\n");
  }
  if (shuttingDown) return;

  // --- Avatar result (only if we ran that phase) ---
  let avatarOk = !autogen; // skipping counts as success for persist purposes
  if (autogen) {
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
      console.log(`✗ AVATAR TIMED OUT after ${elapsed}s (no delivered memory)`);
    } else if (delivered.content.attachments?.[0]?.url) {
      const url = delivered.content.attachments[0].url;
      console.log(`✓ AVATAR GENERATED in ${elapsed}s`);
      console.log("  character: ", character.name, fromDefault ? "(default)" : "");
      console.log("  imagePrompt:", JSON.stringify(delivered.content.text));
      console.log("  imageUrl:   ", url);
      avatarOk = true;
    } else {
      console.log(`✗ AVATAR PIPELINE FAILED after ${elapsed}s`);
      console.log("  message:", delivered.content.text);
    }
  }

  // --- Decide: exit or persist? ---
  if (!persist) {
    // Old behaviour. The trainer worker, if it was running, gets a clean
    // stop signal here and checkpoints whatever it managed to do.
    if (!avatarOk) process.exitCode = 1;
    process.off("SIGINT", onSigint);
    process.off("SIGTERM", onSigint);
    await runtime.stop();
    console.log("✓ Runtime stopped (AVB_RUNNER_PERSIST=false)");
    return;
  }

  // --- Phase 2: SLM16 trainer watch (indefinite) ---
  console.log("\n" + "─".repeat(60));
  console.log("Persisting for SLM16 background trainer.");
  console.log(`  data dir:       ${slm16.getStatus().config.dataDir}`);
  console.log(`  checkpoint dir: ${slm16.getStatus().config.checkpointDir}`);
  console.log("  Ctrl+C to checkpoint and exit.\n");

  // We poll the service's recentEvents ring rather than tapping the worker
  // message stream directly — the Service is the single owner of that
  // stream (it forwards into runtime settings) and we don't want a second
  // listener competing. The ring holds the last 20 events; we track which
  // ones we've already printed by reference identity.
  const seen = new WeakSet<Slm16Event>();
  let lastSummaryAt = 0;
  const SUMMARY_EVERY_MS = 60_000;

  while (!shuttingDown) {
    const status = slm16.getStatus();

    // Drain new events.
    for (const ev of status.recentEvents) {
      if (seen.has(ev)) continue;
      seen.add(ev);
      console.log(`  [slm16] ${fmtSlm16Event(ev)}`);
    }

    // Periodic settings snapshot. This is what the personality sees.
    // getStatus().settings is Record<string, unknown> (it forwards
    // runtime.getSetting which is untyped); coerce for display.
    const now = Date.now();
    if (now - lastSummaryAt >= SUMMARY_EVERY_MS) {
      lastSummaryAt = now;
      const s = status.settings;
      const get = (k: string) => (s[k] === undefined ? "—" : String(s[k]));
      const intel = s[SLM16_SETTING.INTELLIGENCE];
      console.log(
        `  ── ${SLM16_SETTING.STATUS}=${get(SLM16_SETTING.STATUS)} ` +
          `${SLM16_SETTING.STEP}=${get(SLM16_SETTING.STEP)} ` +
          `${SLM16_SETTING.VAL_LOSS}=${get(SLM16_SETTING.VAL_LOSS)}` +
          (intel !== undefined ? ` ${SLM16_SETTING.INTELLIGENCE}=${String(intel)}` : "") +
          ` ${SLM16_SETTING.TRAIN_TIME}=${get(SLM16_SETTING.TRAIN_TIME)}s ──`,
      );
    }

    // The worker may have exited (no_data, max_tokens, fatal error). The
    // process stays up regardless — the user can drop new shards into
    // dataDir and call slm16.spawnWorker() in a future iteration of this
    // runner, or just restart. For now we surface the state and keep
    // polling; nothing crashes.
    if (!status.isTraining) {
      const why = String(status.settings[SLM16_SETTING.STATUS] ?? "unknown");
      console.log(
        `  [slm16] worker not running (status=${why}). ` +
          `Process stays alive; restart to respawn.`,
      );
      // Slow the poll way down — nothing's changing.
      await sleep(30_000);
      continue;
    }

    await sleep(2000);
  }
}

main().catch((err) => {
  console.error("\n✗ avb-runner failed:", err);
  process.exit(1);
});
