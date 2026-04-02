#!/usr/bin/env bun
/**
 * ffm-demo — full FFM derivation chain runner with quality gates.
 *
 * Two operation classes mirroring the plugin:
 *   Class A (deterministic, instant, no I/O):
 *     seed → 8 lanes → 5 trait scores → archetype → both rendered prompts
 *     ALWAYS available. No network, no token.
 *
 *   Class B (slow, networked):
 *     ScryptedClient.invokeTextGeneration → pollJobToCompletion → parse
 *     Skipped (exit 2) if SCRYPTEDAI_BEARER_TOKEN unset. Sections 1–4 + 6
 *     still print so the prompt design is reviewable offline.
 *
 * Why a standalone client instead of booting AgentRuntime: the runtime
 * would just call the same ScryptedClient under the hood, but with a 60-line
 * boot ceremony in the way. ScryptedClient + pollJobToCompletion are exported
 * specifically for this kind of script use (per polling.ts header docs).
 *
 * Usage (from REPO ROOT — never `cd` into the plugin dir):
 *   bun --env-file=.env plugins/plugin-ffm/scripts/ffm-demo.ts
 *   bun --env-file=.env plugins/plugin-ffm/scripts/ffm-demo.ts --seed deadbeef...
 *   bun --env-file=.env plugins/plugin-ffm/scripts/ffm-demo.ts --population 1000
 *   bun --env-file=.env plugins/plugin-ffm/scripts/ffm-demo.ts --out /tmp/ffm-character.demo.json
 *   bun plugins/plugin-ffm/scripts/ffm-demo.ts --offline   # sections 1–4+6 only
 *
 * Exit codes:
 *   0  — full chain succeeded, all quality gates pass
 *   1  — runtime error (parse failure, API error, gate failure)
 *   2  — offline mode (no token); deterministic sections still printed
 */

import { writeFileSync } from "node:fs";
import { toScryptedPayload } from "@elizaos/core";
import {
  pollJobToCompletion,
  ScryptedClient,
} from "@elizaos/plugin-scryptedai";
import {
  ARCHETYPES,
  classifyTraits,
  deriveTraits,
  FfmParseError,
  generateSeed,
  laneToUniforms,
  MAX_REPLY_TOKEN_OVERLAP,
  type MergeableCharacter,
  MIN_BIO_LINES,
  MIN_MESSAGE_EXAMPLES,
  MIN_POST_EXAMPLES,
  maxReplyOverlap,
  mergePersonality,
  type NarrativeExpansion,
  normalizeSeed,
  parseNarrative,
  parseVoice,
  renderNarrativePrompt,
  renderTraitDigest,
  renderVoicePrompt,
  seedToLanes,
  TRAIT_KEYS,
  TRAIT_PARAMS,
  type VoiceExpansion,
} from "../src/index.ts";

// ============================================================================
// CLI parsing
// ============================================================================

interface Args {
  seed?: string;
  population?: number;
  out?: string;
  offline: boolean;
  name: string;
}

function parseArgs(argv: string[]): Args {
  const out: Args = { offline: false, name: "Eliza" };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--seed") out.seed = argv[++i];
    else if (a === "--population") out.population = Number(argv[++i]);
    else if (a === "--out") out.out = argv[++i];
    else if (a === "--name") out.name = argv[++i];
    else if (a === "--offline") out.offline = true;
    else if (a === "--help" || a === "-h") {
      console.log(
        "Usage: bun plugins/plugin-ffm/scripts/ffm-demo.ts [--seed HEX64] [--population N] [--out FILE] [--name NAME] [--offline]",
      );
      process.exit(0);
    } else {
      console.error(`Unknown arg: ${a}`);
      process.exit(1);
    }
  }
  return out;
}

// ============================================================================
// Output formatting
// ============================================================================

const HR = "─".repeat(72);

function head(n: number, title: string): void {
  console.log(`\n${HR}`);
  console.log(`§${n}  ${title}`);
  console.log(HR);
}

/** ASCII trait bar: [████████░░░░░░░░] 0.51 (high) */
function traitBar(score: number, median: number): string {
  const W = 20;
  const filled = Math.round(score * W);
  const bar = "█".repeat(filled) + "░".repeat(W - filled);
  const pole = score >= median ? "high" : "low";
  // mark the median tick on a parallel scale line below
  return `[${bar}] ${score.toFixed(3)} (${pole}, median=${median.toFixed(2)})`;
}

function gate(label: string, ok: boolean, detail: string): boolean {
  const mark = ok ? "✓" : "✗";
  console.log(`  ${mark} ${label}: ${detail}`);
  return ok;
}

// ============================================================================
// Population mode — distribution validation
// ============================================================================

function populationMode(n: number): never {
  head(0, `Population mode — rolling ${n} fresh seeds`);

  const traitSamples: Record<string, number[]> = {
    O: [],
    C: [],
    E: [],
    A: [],
    N: [],
  };
  const archetypeHist = new Array(32).fill(0);

  for (let i = 0; i < n; i++) {
    const seed = generateSeed();
    const traits = deriveTraits(seed);
    for (const k of TRAIT_KEYS) traitSamples[k].push(traits[k]);
    archetypeHist[classifyTraits(traits).code]++;
  }

  console.log("\nPer-trait distribution stats vs target:");
  console.log(
    "  trait | empirical mean | empirical median | empirical sd | empirical skew | target median | target sd | target skew",
  );
  for (const k of TRAIT_KEYS) {
    const arr = traitSamples[k];
    const sorted = [...arr].sort((a, b) => a - b);
    const mean = arr.reduce((s, x) => s + x, 0) / arr.length;
    const median = sorted[Math.floor(arr.length / 2)];
    const variance = arr.reduce((s, x) => s + (x - mean) ** 2, 0) / arr.length;
    const sd = Math.sqrt(variance);
    const m3 = arr.reduce((s, x) => s + (x - mean) ** 3, 0) / arr.length;
    const skew = m3 / sd ** 3;
    const t = TRAIT_PARAMS[k];
    console.log(
      `    ${k}   |     ${mean.toFixed(4)}     |      ${median.toFixed(4)}      |    ${sd.toFixed(4)}    |     ${skew >= 0 ? "+" : ""}${skew.toFixed(3)}     |     ${t.mean.toFixed(2)}      |   ${t.sd.toFixed(2)}   |    ${t.skew >= 0 ? "+" : ""}${t.skew.toFixed(1)}`,
    );
  }

  console.log("\nArchetype histogram (32 bins):");
  const max = Math.max(...archetypeHist);
  const W = 40;
  for (let code = 0; code < 32; code++) {
    const count = archetypeHist[code];
    const a = ARCHETYPES[code];
    const bar = "█".repeat(Math.round((count / max) * W));
    console.log(
      `  ${code.toString().padStart(2)} ${a.sloan} ${bar.padEnd(W)} ${count.toString().padStart(5)} (${((count / n) * 100).toFixed(1)}%) ${a.label}`,
    );
  }

  // Quick sanity gates
  console.log("\nDistribution sanity gates:");
  const nMedian = [...traitSamples.N].sort((a, b) => a - b)[Math.floor(n / 2)];
  const nBelow = traitSamples.N.filter((v) => v < 0.5).length / n;
  const aMedian = [...traitSamples.A].sort((a, b) => a - b)[Math.floor(n / 2)];
  const aAbove = traitSamples.A.filter((v) => v > 0.5).length / n;

  let ok = true;
  ok =
    gate(
      "N median ≈ 0.42",
      Math.abs(nMedian - 0.42) < 0.03,
      nMedian.toFixed(4),
    ) && ok;
  ok =
    gate(
      "N: more below 0.5 than above (right-skewed)",
      nBelow > 0.55,
      `${(nBelow * 100).toFixed(1)}% below`,
    ) && ok;
  ok =
    gate(
      "A median ≈ 0.58",
      Math.abs(aMedian - 0.58) < 0.03,
      aMedian.toFixed(4),
    ) && ok;
  ok =
    gate(
      "A: more above 0.5 than below (left-skewed)",
      aAbove > 0.55,
      `${(aAbove * 100).toFixed(1)}% above`,
    ) && ok;
  ok =
    gate(
      "All 32 archetypes hit",
      archetypeHist.every((c) => c > 0),
      `${archetypeHist.filter((c) => c > 0).length}/32`,
    ) && ok;

  process.exit(ok ? 0 : 1);
}

// ============================================================================
// Live LLM call — ScryptedClient direct, no runtime
// ============================================================================

async function callTextGen(
  client: ScryptedClient,
  payload: { system_prompt: string; user_prompt: string; max_tokens: number },
  label: string,
): Promise<string> {
  const startedAt = Date.now();
  console.log(`  → invoking text generation (${label})...`);
  // /generations/text/nova-pro defaults auto_calculate_tokens=true which
  // floors output at ~100 tokens and ignores max_tokens. Disable it. Cap the
  // request below Nova Pro's 10k ceiling — narrative ~600 tok, voice ~1500.
  const resp = await client.invokeTextGeneration({
    ...payload,
    max_tokens: Math.min(payload.max_tokens, 4000),
    auto_calculate_tokens: false,
  });
  const jobId = resp.job_id;
  console.log(`  → job ${jobId} accepted, polling...`);

  let pollNum = 0;
  const { result, completed, timedOut, attempts } = await pollJobToCompletion(
    client,
    jobId,
    {
      jobType: "text",
      onPoll: (r) =>
        console.log(
          `    [${((Date.now() - startedAt) / 1000).toFixed(1)}s] poll #${++pollNum}: ${r.status}`,
        ),
    },
  );

  const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
  if (timedOut) {
    throw new Error(
      `${label}: polling timed out after ${elapsed}s (${attempts} attempts)`,
    );
  }
  if (!completed || result.status === "failed") {
    throw new Error(
      `${label}: job failed — ${result.error ?? "unknown error"}`,
    );
  }
  if (!result.text) {
    throw new Error(`${label}: job completed but no text in result`);
  }

  console.log(`  ✓ ${label} done in ${elapsed}s (${result.text.length} chars)`);
  return result.text;
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const args = parseArgs(process.argv);

  if (args.population !== undefined) {
    if (!Number.isInteger(args.population) || args.population < 1) {
      console.error("--population must be a positive integer");
      process.exit(1);
    }
    populationMode(args.population);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // §1  Seed + lane decomposition
  // ──────────────────────────────────────────────────────────────────────────
  head(1, "Seed + lane decomposition");
  const seedHex = args.seed ? normalizeSeed(args.seed) : generateSeed();
  const seedSource = args.seed ? "provided" : "fresh random";
  console.log(`  seed (${seedSource}): ${seedHex}`);

  const lanes = seedToLanes(seedHex);
  console.log("\n  8 × uint32 lanes (big-endian):");
  const traitNames = ["O", "C", "E", "A", "N"];
  for (let i = 0; i < 8; i++) {
    const purpose =
      i < 5
        ? `→ ${traitNames[i]} (${["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"][i]})`
        : "(reserved)";
    const [u1, u2] = laneToUniforms(lanes[i]);
    console.log(
      `    lane[${i}] = 0x${lanes[i].toString(16).padStart(8, "0")} = ${lanes[i].toString().padStart(10)}  uniforms=(${u1.toFixed(4)}, ${u2.toFixed(4)})  ${purpose}`,
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // §2  Trait roll
  // ──────────────────────────────────────────────────────────────────────────
  head(2, "Trait derivation");
  const traits = deriveTraits(seedHex);
  const labels: Record<string, string> = {
    O: "Openness",
    C: "Conscientiousness",
    E: "Extraversion",
    A: "Agreeableness",
    N: "Neuroticism",
  };
  for (const k of TRAIT_KEYS) {
    const p = TRAIT_PARAMS[k];
    const z = (traits[k] - p.mean) / p.sd;
    const poleLabel = traits[k] >= p.mean ? p.hi : p.lo;
    console.log(`  ${labels[k].padEnd(18)} ${traitBar(traits[k], p.mean)}`);
    console.log(
      `  ${"".padEnd(18)} z=${z >= 0 ? "+" : ""}${z.toFixed(2)}  ${poleLabel}`,
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // §3  Archetype
  // ──────────────────────────────────────────────────────────────────────────
  head(3, "Archetype classification");
  const archetype = classifyTraits(traits);
  const bits = TRAIT_KEYS.map((k) =>
    traits[k] >= TRAIT_PARAMS[k].mean ? "1" : "0",
  ).join("");
  console.log(`  bit pattern (OCEAN MSB→LSB): 0b${bits} = ${archetype.code}`);
  console.log(`  SLOAN code:    ${archetype.sloan}`);
  console.log(`  label:         ${archetype.label}`);
  console.log(`  summary:       ${archetype.summary}`);

  const profile = Object.freeze({ seed: seedHex, traits, archetype });

  // ──────────────────────────────────────────────────────────────────────────
  // §4  Narrative prompt (rendered)
  // ──────────────────────────────────────────────────────────────────────────
  head(4, "Narrative prompt (rendered) — ScryptedAI call #1 payload");
  const narrPrompt = renderNarrativePrompt(profile, args.name);
  const narrPayload = toScryptedPayload(narrPrompt);
  console.log("  Trait digest (spliced into both prompts):");
  for (const line of renderTraitDigest(profile).split("\n")) {
    console.log(`    ${line}`);
  }
  console.log(
    `\n  ── system prompt (${narrPayload.system_prompt.length} chars) ──`,
  );
  console.log(narrPayload.system_prompt);
  console.log(
    `\n  ── user prompt (${narrPayload.user_prompt.length} chars) ──`,
  );
  console.log(narrPayload.user_prompt);
  console.log(`\n  max_tokens: ${narrPayload.max_tokens}`);

  // ──────────────────────────────────────────────────────────────────────────
  // Token check — gate Class B
  // ──────────────────────────────────────────────────────────────────────────
  const token = process.env.SCRYPTEDAI_BEARER_TOKEN;
  if (!token || args.offline) {
    head(5, "Narrative result");
    console.log(
      args.offline
        ? "  [skipped — --offline flag set]"
        : "  [skipped — SCRYPTEDAI_BEARER_TOKEN not set]",
    );
    head(
      6,
      "Voice prompt (rendered) — ScryptedAI call #2 payload [WITHOUT narrative splice]",
    );
    console.log(
      "  Voice prompt depends on narrative output for {{BIO}} and {{BELIEFS}} tags.",
    );
    console.log(
      "  Showing the TEMPLATE so the structure is reviewable (tags will be empty):",
    );
    const stub: NarrativeExpansion = {
      bio: ["<bio line from call #1>"],
      beliefs: ["<belief from call #1>"],
      adjectives: [],
      topics: [],
    };
    const voicePrompt = renderVoicePrompt(profile, stub, args.name);
    const voicePayload = toScryptedPayload(voicePrompt);
    console.log(
      `\n  ── system prompt (${voicePayload.system_prompt.length} chars) ──`,
    );
    console.log(voicePayload.system_prompt);
    console.log(
      `\n  ── user prompt (${voicePayload.user_prompt.length} chars) ──`,
    );
    console.log(voicePayload.user_prompt);

    head(7, "Voice result");
    console.log("  [skipped — no token]");
    head(8, "Merged character.json");
    console.log("  [skipped — no LLM output to merge]");
    console.log(
      `\n${HR}\nDeterministic chain complete. Run with SCRYPTEDAI_BEARER_TOKEN for live LLM expansion.\n`,
    );
    process.exit(2);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // §5  Live narrative call
  // ──────────────────────────────────────────────────────────────────────────
  head(5, "Narrative result (live ScryptedAI call)");
  const client = new ScryptedClient({ bearerToken: token });

  let narrative: NarrativeExpansion;
  let narrativeRaw: string;
  try {
    narrativeRaw = await callTextGen(client, narrPayload, "narrative");
    narrative = parseNarrative(narrativeRaw);
  } catch (e) {
    if (e instanceof FfmParseError) {
      console.error(`\n✗ Narrative parse failed: ${e.message}`);
      console.error("Raw LLM output (first 800 chars):");
      console.error(e.rawText.slice(0, 800));
    } else {
      console.error(
        `\n✗ Narrative call failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
    process.exit(1);
  }

  console.log(`\n  bio (${narrative.bio.length} lines):`);
  narrative.bio.forEach((l, i) => console.log(`    [${i}] ${l}`));
  console.log(`\n  beliefs (${narrative.beliefs.length}):`);
  narrative.beliefs.forEach((b, i) => console.log(`    [${i}] "${b}"`));
  console.log(
    `\n  adjectives (${narrative.adjectives.length}): ${narrative.adjectives.join(", ")}`,
  );
  console.log(`\n  topics (${narrative.topics.length}):`);
  narrative.topics.forEach((t, i) => console.log(`    [${i}] ${t}`));

  // ──────────────────────────────────────────────────────────────────────────
  // §6  Voice prompt (rendered, WITH narrative spliced in)
  // ──────────────────────────────────────────────────────────────────────────
  head(6, "Voice prompt (rendered) — ScryptedAI call #2 payload");
  const voicePrompt = renderVoicePrompt(profile, narrative, args.name);
  const voicePayload = toScryptedPayload(voicePrompt);
  console.log(
    "  Note the {{BIO}} and {{BELIEFS}} from §5 spliced into the user prompt:",
  );
  console.log(
    `\n  ── system prompt (${voicePayload.system_prompt.length} chars) ──`,
  );
  console.log(voicePayload.system_prompt);
  console.log(
    `\n  ── user prompt (${voicePayload.user_prompt.length} chars) ──`,
  );
  console.log(voicePayload.user_prompt);
  console.log(`\n  max_tokens: ${voicePayload.max_tokens}`);

  // ──────────────────────────────────────────────────────────────────────────
  // §7  Live voice call
  // ──────────────────────────────────────────────────────────────────────────
  head(7, "Voice result (live ScryptedAI call)");

  let voice: VoiceExpansion;
  let voiceRaw: string;
  try {
    voiceRaw = await callTextGen(client, voicePayload, "voice");
    voice = parseVoice(voiceRaw);
  } catch (e) {
    if (e instanceof FfmParseError) {
      console.error(`\n✗ Voice parse failed: ${e.message}`);
      console.error("Raw LLM output (first 800 chars):");
      console.error(e.rawText.slice(0, 800));
    } else {
      console.error(
        `\n✗ Voice call failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
    process.exit(1);
  }

  const scenarios = [
    "help request → Agreeableness",
    "disagreement → Agreeableness + Neuroticism",
    "small talk → Extraversion",
    "bad news → Neuroticism + Agreeableness",
    "weird question → Openness",
    "commitment ask → Conscientiousness",
  ];
  console.log(
    `\n  messageExamples (${voice.messageExamples.length} exchanges):`,
  );
  voice.messageExamples.forEach((ex, i) => {
    const tag = scenarios[i] ?? "(extra)";
    console.log(`\n    ── exchange ${i + 1}: ${tag} ──`);
    console.log(`      ${ex[0].name}: ${ex[0].content.text}`);
    console.log(`      ${ex[1].name}: ${ex[1].content.text}`);
  });

  console.log(`\n  postExamples (${voice.postExamples.length}):`);
  voice.postExamples.forEach((p, i) => console.log(`    [${i}] ${p}`));

  console.log(`\n  style.all (${voice.style.all.length}):`);
  voice.style.all.forEach((s) => console.log(`    - ${s}`));
  console.log(`  style.chat (${voice.style.chat.length}):`);
  voice.style.chat.forEach((s) => console.log(`    - ${s}`));
  console.log(`  style.post (${voice.style.post.length}):`);
  voice.style.post.forEach((s) => console.log(`    - ${s}`));

  // ──────────────────────────────────────────────────────────────────────────
  // §8  Merged character.json
  // ──────────────────────────────────────────────────────────────────────────
  head(8, "Merged character.json (full output)");
  const blank: MergeableCharacter = { name: args.name };
  const merged = mergePersonality(blank, profile, narrative, voice);
  const json = JSON.stringify(merged, null, 2);
  console.log(json);

  if (args.out) {
    writeFileSync(args.out, `${json}\n`, { mode: 0o600 });
    console.log(`\n  → written to ${args.out}`);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // §9  Quality gates
  // ──────────────────────────────────────────────────────────────────────────
  head(9, "Quality gates");
  const overlap = maxReplyOverlap(voice.messageExamples);
  let ok = true;

  ok =
    gate(
      `bio.length >= ${MIN_BIO_LINES}`,
      narrative.bio.length >= MIN_BIO_LINES,
      `${narrative.bio.length}`,
    ) && ok;
  ok =
    gate(
      `messageExamples.length >= ${MIN_MESSAGE_EXAMPLES}`,
      voice.messageExamples.length >= MIN_MESSAGE_EXAMPLES,
      `${voice.messageExamples.length}`,
    ) && ok;
  ok =
    gate(
      "every exchange has exactly 2 turns with {name, content.text}",
      voice.messageExamples.every(
        (ex) => ex.length === 2 && ex.every((t) => t.name && t.content?.text),
      ),
      voice.messageExamples.every((ex) => ex.length === 2)
        ? "all 2-turn"
        : "structure violation",
    ) && ok;
  ok =
    gate(
      `postExamples.length >= ${MIN_POST_EXAMPLES}`,
      voice.postExamples.length >= MIN_POST_EXAMPLES,
      `${voice.postExamples.length}`,
    ) && ok;
  ok =
    gate(
      `max reply token overlap < ${MAX_REPLY_TOKEN_OVERLAP} (no repetition collapse)`,
      overlap < MAX_REPLY_TOKEN_OVERLAP,
      overlap.toFixed(3),
    ) && ok;
  ok =
    gate(
      "settings.ffm.seed matches input seed",
      merged.settings?.ffm?.seed === seedHex,
      merged.settings?.ffm?.seed?.slice(0, 16) + "...",
    ) && ok;
  ok =
    gate(
      "merged bio includes beliefs (folded in)",
      Array.isArray(merged.bio) &&
        merged.bio.length === narrative.bio.length + narrative.beliefs.length,
      `${Array.isArray(merged.bio) ? merged.bio.length : 0} = ${narrative.bio.length} + ${narrative.beliefs.length}`,
    ) && ok;

  console.log(`\n${HR}`);
  if (ok) {
    console.log("✓ ALL QUALITY GATES PASS");
    process.exit(0);
  } else {
    console.log("✗ ONE OR MORE QUALITY GATES FAILED");
    process.exit(1);
  }
}

main().catch((e) => {
  console.error("\n✗ Unhandled error:", e instanceof Error ? e.stack : e);
  process.exit(1);
});
