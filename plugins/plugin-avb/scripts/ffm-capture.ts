/**
 * ffm-capture — exercise the FFM personality expansion against the LIVE
 * ScryptedAI text endpoint and persist the responses as test fixtures.
 *
 * Purpose: prove the PromptSet → ScryptedAI → parse → Character pipeline
 * works against the real API (not just mocks), and cache the real response
 * shapes so the unit-test suite can replay them deterministically without
 * spamming the endpoint.
 *
 * Output: src/__tests__/fixtures/ffm-live.json
 *   { capturedAt, model, samples: [{ seed, profile, rawText, fields, character }] }
 *
 * Usage (from repo root):
 *   bun --cwd plugins/plugin-avb \
 *       --env-file=/abs/path/to/.env \
 *       scripts/ffm-capture.ts [seedHex...]
 *
 * If no seeds are passed, two fixed seeds are used (one low-N, one high-N
 * archetype) so fixtures cover both calm and volatile personalities.
 */

import fs from "node:fs";
import path from "node:path";
import { AgentRuntime, validateCharacter } from "@elizaos/core";
import {
  SCRYPTEDAI_SERVICE_TYPE,
  type ScryptedAIService,
  scryptedaiPlugin,
} from "@elizaos/plugin-scryptedai";
import {
  buildFfmCharacter,
  deriveFfmProfile,
  FFM_TEXT_MODEL,
  type FfmProfile,
  formatFfmScores,
  parseFfmCharacterFields,
  renderFfmPrompt,
} from "../src/ffm.ts";
import { toScryptedPayload } from "@elizaos/core";

// Two reference seeds chosen so their archetypes differ on the N axis.
// (Discovered by brute-search; stable because rollFfmTraits is pure.)
const DEFAULT_SEEDS = [
  // Low-neuroticism exemplar
  "4a7d1ed414474e4033ac29ccb8653d9b4a7d1ed414474e4033ac29ccb8653d9b",
  // High-neuroticism exemplar
  "ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100",
];

const FIXTURE_PATH = path.resolve(
  import.meta.dirname,
  "../src/__tests__/fixtures/ffm-live.json",
);

interface CaptureSample {
  seed: string;
  profile: FfmProfile;
  /** The exact payload sent to ScryptedAI (system_prompt/user_prompt/...). */
  request: Record<string, unknown>;
  /** Verbatim text returned by the model. */
  rawText: string;
  /** Full result_data container from the API (usage, cost, finish_reason, …). */
  rawResult: Record<string, unknown> | undefined;
  /** Parsed FfmCharacterFields. */
  fields: ReturnType<typeof parseFfmCharacterFields>;
  /** Fully assembled ElizaOS Character. */
  character: ReturnType<typeof buildFfmCharacter>;
  /** Whether core's validateCharacter() accepted it. */
  valid: boolean;
  validationError?: string;
  elapsedMs: number;
}

async function main() {
  if (!process.env.SCRYPTEDAI_BEARER_TOKEN) {
    console.error(
      "✗ SCRYPTEDAI_BEARER_TOKEN not set. Run with --env-file pointing at the project .env",
    );
    process.exit(1);
  }

  const seeds = process.argv.slice(2).length
    ? process.argv.slice(2)
    : DEFAULT_SEEDS;

  // Minimal runtime: scryptedai only, in-memory db, no avb plugin (we call
  // the FFM helpers directly so we can intercept raw responses).
  const runtime = new AgentRuntime({
    character: { name: "FfmCapture", bio: ["fixture capture agent"] },
    plugins: [scryptedaiPlugin],
    settings: process.env as Record<string, string | undefined>,
    logLevel: "warn",
  });
  await runtime.initialize({ allowNoDatabase: true });
  const scrypted = (await runtime.getServiceLoadPromise(
    SCRYPTEDAI_SERVICE_TYPE,
  )) as ScryptedAIService;
  console.log("✓ ScryptedAI service ready\n");

  const samples: CaptureSample[] = [];

  for (const seed of seeds) {
    const profile = deriveFfmProfile(seed);
    console.log(
      `→ seed ${seed.slice(0, 8)}…  ${profile.archetype.name} ` +
        `(${profile.archetype.code})  ${formatFfmScores(profile.scores)}`,
    );

    const rendered = renderFfmPrompt(profile, profile.archetype.name);
    const request = toScryptedPayload(rendered);

    const t0 = Date.now();
    const { jobId } = await scrypted.startTextGeneration(request);
    console.log(`  job ${jobId} submitted; awaiting…`);
    const result = await scrypted.awaitJob(jobId, 120_000);
    const elapsedMs = Date.now() - t0;

    const usage = (result.result as Record<string, unknown> | undefined)
      ?.usage as Record<string, number> | undefined;
    const finishReason = (result.result as Record<string, unknown> | undefined)
      ?.finish_reason as string | undefined;

    if (result.status !== "completed" || !result.text) {
      console.error(
        `  ✗ ${result.status}${result.error ? ` — ${result.error}` : ""} (${elapsedMs}ms)`,
      );
      throw new Error(`FFM capture failed for seed ${seed.slice(0, 8)}`);
    }
    console.log(
      `  ✓ completed in ${elapsedMs}ms (${result.text.length} chars, ` +
        `in=${usage?.input_tokens} out=${usage?.output_tokens} ` +
        `finish=${finishReason})`,
    );
    console.log("  ── raw text ──────────────────────────────────────────");
    console.log(
      result.text
        .split("\n")
        .map((l) => `  │ ${l}`)
        .join("\n"),
    );
    console.log("  ──────────────────────────────────────────────────────");

    const fields = parseFfmCharacterFields(result.text);
    const character = buildFfmCharacter(
      { name: profile.archetype.name.replace(/^The\s+/i, "") },
      profile,
      fields,
    );
    const v = validateCharacter(character);

    samples.push({
      seed,
      profile,
      request,
      rawText: result.text,
      rawResult: result.result,
      fields,
      character,
      valid: v.success,
      validationError: v.success ? undefined : v.error?.message,
      elapsedMs,
    });

    console.log(
      `  parsed: bio×${fields.bio?.length ?? 0} adj×${fields.adjectives?.length ?? 0} ` +
        `topics×${fields.topics?.length ?? 0} style.all×${fields.style?.all?.length ?? 0} ` +
        `posts×${fields.postExamples?.length ?? 0} beliefs×${fields.beliefs?.length ?? 0}`,
    );
    console.log(
      `  validateCharacter: ${v.success ? "✓ valid" : `✗ ${v.error?.message}`}\n`,
    );
  }

  fs.mkdirSync(path.dirname(FIXTURE_PATH), { recursive: true });
  fs.writeFileSync(
    FIXTURE_PATH,
    JSON.stringify(
      { capturedAt: new Date().toISOString(), model: FFM_TEXT_MODEL, samples },
      null,
      2,
    ),
  );
  console.log(`✓ Fixtures written → ${FIXTURE_PATH}`);

  await runtime.stop();
}

main().catch((err) => {
  console.error("\n✗ ffm-capture failed:", err);
  process.exit(1);
});
