/**
 * Fixture-backed replay tests for the FFM → ScryptedAI → Character pipeline.
 *
 * These tests do NOT hit the network. They replay real ScryptedAI nova-pro
 * responses captured by `scripts/ffm-capture.ts` and stored at
 * `__tests__/fixtures/ffm-live.json`. The fixture is the source of truth for
 * what the live API actually returns — every assertion here is grounded in
 * an observed wire payload, not a hand-written mock.
 *
 * Regenerate the fixture (requires SCRYPTEDAI_BEARER_TOKEN):
 *   bun --cwd plugins/plugin-avb --env-file=<abs>/.env scripts/ffm-capture.ts
 *
 * Two regression guards in here exist specifically because of a real bug
 * found via live testing: the nova-pro recipe's `auto_calculate_tokens`
 * default silently caps output at 100 tokens. The "request shape" and
 * "finish_reason" tests below ensure that fix never regresses.
 */

import { validateCharacter } from "@elizaos/core";
import { describe, expect, it } from "vitest";
import {
  buildFfmCharacter,
  deriveFfmProfile,
  expandFfmToCharacterFields,
  FFM_SEED_SETTING,
  FFM_TEXT_MODEL,
  type FfmCharacterFields,
  type FfmProfile,
  parseFfmCharacterFields,
} from "../ffm.ts";

import fixtureJson from "./fixtures/ffm-live.json";

// ----------------------------------------------------------------------------
// Fixture shape (mirrors scripts/ffm-capture.ts CaptureSample)
// ----------------------------------------------------------------------------

interface FixtureSample {
  seed: string;
  profile: FfmProfile;
  request: {
    system_prompt: string;
    user_prompt: string;
    max_tokens?: number;
    auto_calculate_tokens?: boolean;
  };
  rawText: string;
  rawResult?: {
    finish_reason?: string;
    usage?: { input_tokens: number; output_tokens: number };
    metadata?: { max_tokens_requested?: number };
  };
  fields: FfmCharacterFields;
  character: Record<string, unknown>;
  valid: boolean;
  elapsedMs: number;
}

interface Fixture {
  capturedAt: string;
  model: string;
  samples: FixtureSample[];
}

const fixture = fixtureJson as Fixture;

// ----------------------------------------------------------------------------
// 0. Fixture sanity
// ----------------------------------------------------------------------------

describe("ffm-live: fixture sanity", () => {
  it("fixture loaded with ≥1 sample for the expected model", () => {
    expect(fixture.samples.length).toBeGreaterThanOrEqual(1);
    expect(fixture.model).toBe(FFM_TEXT_MODEL);
    expect(fixture.capturedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
  });

  it("every captured sample was marked valid at capture time", () => {
    for (const s of fixture.samples) {
      expect(s.valid).toBe(true);
    }
  });
});

// ----------------------------------------------------------------------------
// 1. Wire-shape regression guards (the 100-token-truncation bug)
// ----------------------------------------------------------------------------

describe("ffm-live: wire-shape regression guards", () => {
  it("request: auto_calculate_tokens=false and explicit max_tokens were sent", () => {
    // If either of these is missing the live API silently caps output at
    // 100 tokens (verified 2026-04). This guard fails fast if toScryptedPayload
    // ever drops them — re-capture the fixture after fixing.
    for (const s of fixture.samples) {
      expect(s.request.auto_calculate_tokens).toBe(false);
      expect(typeof s.request.max_tokens).toBe("number");
      expect(s.request.max_tokens).toBeGreaterThan(100);
      expect(s.request.system_prompt.length).toBeGreaterThan(0);
      expect(s.request.user_prompt.length).toBeGreaterThan(0);
    }
  });

  it("response: model finished naturally (end_turn), not by token cap", () => {
    // finish_reason === "max_tokens" means the JSON was cut mid-stream and
    // parseFfmCharacterFields will throw. The fixture must prove the fix
    // actually worked against the live endpoint.
    for (const s of fixture.samples) {
      expect(s.rawResult?.finish_reason).toBe("end_turn");
      const out = s.rawResult?.usage?.output_tokens ?? 0;
      expect(out).toBeGreaterThan(100);
      // Server echoed back what we asked for, not 100.
      expect(s.rawResult?.metadata?.max_tokens_requested).toBe(
        s.request.max_tokens,
      );
    }
  });
});

// ----------------------------------------------------------------------------
// 2. Determinism: re-deriving the profile from the seed must match the capture
// ----------------------------------------------------------------------------

describe("ffm-live: profile determinism vs. capture", () => {
  it("deriveFfmProfile(seed) reproduces the captured scores + archetype", () => {
    for (const s of fixture.samples) {
      const p = deriveFfmProfile(s.seed);
      expect(p.seed).toBe(s.profile.seed);
      expect(p.archetype.id).toBe(s.profile.archetype.id);
      expect(p.archetype.code).toBe(s.profile.archetype.code);
      // Float equality: sfc32 + Cheng's BB are pure, so values are bit-exact.
      expect(p.scores).toEqual(s.profile.scores);
    }
  });
});

// ----------------------------------------------------------------------------
// 3. Parse the REAL model output (no synthetic JSON)
// ----------------------------------------------------------------------------

describe("ffm-live: parse real nova-pro output", () => {
  for (const s of fixture.samples) {
    const label = `${s.seed.slice(0, 8)} (${s.profile.archetype.name})`;

    it(`${label}: parseFfmCharacterFields() succeeds on raw response`, () => {
      const fields = parseFfmCharacterFields(s.rawText);
      // All schema-required arrays present and non-empty.
      expect(fields.bio?.length).toBeGreaterThanOrEqual(3);
      expect(fields.adjectives?.length).toBeGreaterThanOrEqual(6);
      expect(fields.topics?.length).toBeGreaterThanOrEqual(5);
      expect(fields.style?.all?.length).toBeGreaterThanOrEqual(4);
      expect(fields.style?.chat?.length).toBeGreaterThanOrEqual(3);
      expect(fields.style?.post?.length).toBeGreaterThanOrEqual(3);
      expect(fields.postExamples?.length).toBeGreaterThanOrEqual(3);
      expect(fields.beliefs?.length).toBeGreaterThanOrEqual(3);
      // Re-parse must equal the parse captured at fixture time.
      expect(fields).toEqual(s.fields);
    });

    it(`${label}: buildFfmCharacter() → validateCharacter() passes`, () => {
      const fields = parseFfmCharacterFields(s.rawText);
      const profile = deriveFfmProfile(s.seed);
      const character = buildFfmCharacter({ name: "Replay" }, profile, fields);

      const v = validateCharacter(character);
      expect(v.success).toBe(true);

      // FFM provenance is embedded in settings.
      const settings = character.settings as Record<string, unknown>;
      expect(settings[FFM_SEED_SETTING]).toBe(s.seed);
      expect(settings.AVB_FFM_SCORES).toEqual(profile.scores);
      expect(
        (settings.AVB_FFM_ARCHETYPE as Record<string, unknown>).code,
      ).toBe(profile.archetype.code);
      expect(Array.isArray(settings.AVB_FFM_BELIEFS)).toBe(true);
    });
  }
});

// ----------------------------------------------------------------------------
// 4. expandFfmToCharacterFields(): replay through a fake ScryptedAI service
//    that returns the captured rawText — proves the runtime integration path
//    handles the real response shape end-to-end without the network.
// ----------------------------------------------------------------------------

describe("ffm-live: expandFfmToCharacterFields() replay", () => {
  function fakeRuntime(rawText: string) {
    let resolveTerminal!: (v: unknown) => void;
    const terminal = new Promise((r) => {
      resolveTerminal = r;
    });
    const scrypted = {
      startTextGeneration: async (payload: Record<string, unknown>) => {
        // Same wire-shape guard as above, but exercised through the actual
        // expandFfmToCharacterFields → toScryptedPayload code path.
        expect(payload.auto_calculate_tokens).toBe(false);
        expect(typeof payload.max_tokens).toBe("number");
        // Resolve on next tick so awaitJob actually has to await.
        queueMicrotask(() =>
          resolveTerminal({
            jobId: "replay-job",
            status: "completed",
            text: rawText,
          }),
        );
        return { jobId: "replay-job", response: {}, record: {} };
      },
      awaitJob: async () => terminal,
    };
    return {
      getService: <T,>() => scrypted as T,
    };
  }

  for (const s of fixture.samples) {
    const label = `${s.seed.slice(0, 8)} (${s.profile.archetype.name})`;

    it(`${label}: round-trips through the runtime helper`, async () => {
      const profile = deriveFfmProfile(s.seed);
      const fields = await expandFfmToCharacterFields(
        fakeRuntime(s.rawText),
        profile,
        { name: "Replay", timeoutMs: 1000 },
      );
      expect(fields).toEqual(s.fields);
    });
  }

  it("propagates a non-completed status as an error", async () => {
    const rt = {
      getService: <T,>() =>
        ({
          startTextGeneration: async () => ({ jobId: "j" }),
          awaitJob: async () => ({
            jobId: "j",
            status: "failed",
            error: "boom",
          }),
        }) as T,
    };
    await expect(
      expandFfmToCharacterFields(rt, deriveFfmProfile(fixture.samples[0].seed)),
    ).rejects.toThrow(/failed.*boom/);
  });
});
