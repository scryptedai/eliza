/**
 * Shared observation loop for live-proof scripts.
 *
 * Both prove.ts (inline character → secrets path) and avb-runner.ts
 * (loadCharacter() → settings path) boot a runtime, then need to:
 *   1. wait for the autonomous trigger to create the first phase task,
 *   2. poll the `avb` tag until no phase tasks remain,
 *   3. fetch the delivered memory and report success/failure.
 *
 * That ~80-line block was previously duplicated verbatim. Live-proof
 * scripts are not unit tests (see plugins/TESTING.md) but they are still
 * code we maintain; keeping the loop in one place means a change to the
 * pipeline's task tags or delivery shape only needs updating once.
 */

import type { AgentRuntime, Memory, Task } from "@elizaos/core";
import type { AvbPhaseMetadata } from "../src/index.ts";

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

export interface ObserveResult {
  /** The GENERATE_AVATAR memory delivered to the agent's room, if any. */
  delivered: Memory | undefined;
  /** Wall-clock seconds from observe start to terminal, formatted to 1dp. */
  elapsedSeconds: string;
  /** True iff at least one phase task was ever seen. */
  sawAnyTask: boolean;
}

/**
 * Poll the runtime's task store for `avb`-tagged phase tasks until none
 * remain (pipeline terminal) or `maxWaitMs` elapses, then fetch the
 * delivered avatar memory.
 */
export async function observeAvbPipeline(
  runtime: AgentRuntime,
  maxWaitMs = 6 * 60 * 1000, // text (≤90s) + image (≤360s) headroom
): Promise<ObserveResult> {
  const roomId = runtime.agentId;
  const start = Date.now();
  let lastPhase = "";
  let sawAnyTask = false;

  console.log("Observing pipeline progress (poll every 3s)...\n");
  while (Date.now() - start < maxWaitMs) {
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

  const elapsedSeconds = ((Date.now() - start) / 1000).toFixed(1);

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

  return { delivered, elapsedSeconds, sawAnyTask };
}

/**
 * Print a standard pass/fail summary for an ObserveResult and set
 * `process.exitCode` on failure. Returns true on success.
 */
export function reportAvbResult(
  { delivered, elapsedSeconds }: ObserveResult,
  extraSuccessLines: string[] = [],
): boolean {
  console.log("\n" + "─".repeat(60));

  if (!delivered) {
    console.log(
      `✗ PIPELINE TIMED OUT after ${elapsedSeconds}s (no delivered memory)`,
    );
    process.exitCode = 1;
    return false;
  }

  const url = delivered.content.attachments?.[0]?.url;
  if (url) {
    console.log(`✓ AVATAR GENERATED in ${elapsedSeconds}s`);
    for (const line of extraSuccessLines) console.log(line);
    console.log("  imagePrompt:", JSON.stringify(delivered.content.text));
    console.log("  imageUrl:   ", url);
    return true;
  }

  console.log(`✗ PIPELINE FAILED after ${elapsedSeconds}s`);
  console.log("  message:", delivered.content.text);
  process.exitCode = 1;
  return false;
}
