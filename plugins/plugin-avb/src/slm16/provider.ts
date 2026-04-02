/**
 * SLM16 status provider — surfaces the trainer's state into the agent's
 * personality so it can give regular, in-character updates ("I've been
 * training my own little brain in the background — step 4 200, val_loss
 * 3.21, intelligence 38/100 vs Nova Pro").
 *
 * `dynamic: true` so it's only included when the message handler asks for
 * it (status changes every few seconds; no point baking it into every
 * prompt by default).
 */

import type { IAgentRuntime, Memory, Provider, State } from "@elizaos/core";

import { SLM16_SERVICE_TYPE } from "./constants.ts";
import type { Slm16Service } from "./service.ts";

function fmt(n: number | null, digits = 4): string {
  return n == null || !Number.isFinite(n) ? "—" : n.toFixed(digits);
}

export const slm16StatusProvider: Provider = {
  name: "SLM16_STATUS",
  description:
    "Live status of the SLM16 background trainer: step, train/val loss, " +
    "LKG checkpoint size, and the cosine-similarity intelligence score " +
    "against Amazon Nova Pro. Use this when the user asks how the local " +
    "model is doing, or when giving an autonomous progress update.",
  dynamic: true,

  get: async (runtime: IAgentRuntime, _message: Memory, _state: State) => {
    const svc = runtime.getService<Slm16Service>(SLM16_SERVICE_TYPE);
    if (!svc) {
      return {
        text: "# SLM16\nThe SLM16 trainer service is not loaded.",
        values: { slm16_running: false },
      };
    }

    // The agent is responsible for keeping its own training alive.
    if (!svc.isRunning()) svc.ensureTraining();

    const s = svc.getStatus();
    const sizeMiB =
      s.artifactBytes != null
        ? `${(s.artifactBytes / 1024 / 1024).toFixed(2)} MiB / 16 MiB`
        : "no checkpoint yet";

    const lines = [
      "# SLM16 (local 16 MB language model)",
      `state: ${s.running ? "training" : "idle"} on ${s.backend ?? "unresolved backend"}`,
      `step: ${s.step}`,
      `train_loss: ${fmt(s.trainLoss)}  val_loss: ${fmt(s.valLoss)}  best_val_loss: ${fmt(s.bestValLoss)}`,
      `lkg_checkpoint: ${sizeMiB}${s.lastCheckpointAt ? ` (saved ${s.lastCheckpointAt})` : ""}`,
      `intelligence_score: ${s.intelligence != null ? `${s.intelligence}/100 (cosine vs Nova Pro @ T=0.1)` : "not yet evaluated"}`,
    ];
    if (s.lastError) lines.push(`last_error: ${s.lastError}`);

    return {
      text: lines.join("\n"),
      values: {
        slm16_running: s.running,
        slm16_backend: s.backend,
        slm16_step: s.step,
        slm16_train_loss: s.trainLoss,
        slm16_val_loss: s.valLoss,
        slm16_best_val_loss: s.bestValLoss,
        slm16_intelligence: s.intelligence,
        slm16_param_count: s.paramCount,
      },
      data: { slm16: { ...s } },
    };
  },
};
