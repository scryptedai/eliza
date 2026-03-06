/**
 * GENERATE_AVATAR action — thin trigger for the AVB pipeline.
 *
 * Handler returns immediately after scheduling the first phase Task.
 * All actual work (text gen → image gen → delivery) happens in the
 * background via TaskService-driven worker ticks.
 */

import type {
  Action,
  ActionResult,
  IAgentRuntime,
  Memory,
} from "@elizaos/core";
import { SCRYPTEDAI_SERVICE_TYPE } from "@elizaos/plugin-scryptedai";

import { AVB_SERVICE_TYPE } from "./constants.ts";
import type { AvbService } from "./service.ts";

// ----------------------------------------------------------------------------
// Internal: typed service lookup (mirrors the runtime surface cast pattern)
// ----------------------------------------------------------------------------

function getAvb(runtime: IAgentRuntime): AvbService | undefined {
  return (
    runtime as unknown as { getService<T>(type: string): T | undefined }
  ).getService<AvbService>(AVB_SERVICE_TYPE);
}

function hasScrypted(runtime: IAgentRuntime): boolean {
  return (
    (
      runtime as unknown as { getService(type: string): unknown | undefined }
    ).getService(SCRYPTEDAI_SERVICE_TYPE) !== undefined
  );
}

// ----------------------------------------------------------------------------
// Action
// ----------------------------------------------------------------------------

export const generateAvatarAction: Action = {
  name: "GENERATE_AVATAR",
  description:
    "Generate a visual avatar that personifies the agent's character. " +
    "Runs asynchronously: schedules a background pipeline (character digest " +
    "→ text prompt → image generation) and delivers the result to this room " +
    "when ready.",

  similes: [
    "CREATE_AVATAR",
    "DRAW_YOURSELF",
    "MAKE_PROFILE_PICTURE",
    "SELF_PORTRAIT",
  ],

  examples: [
    [
      {
        name: "{{user}}",
        content: { text: "What do you look like?" },
      },
      {
        name: "{{agent}}",
        content: {
          text: "Let me generate a visual for you — one moment.",
          actions: ["GENERATE_AVATAR"],
        },
      },
    ],
    [
      {
        name: "{{user}}",
        content: { text: "Can you make an avatar of yourself?" },
      },
      {
        name: "{{agent}}",
        content: {
          text: "Generating an avatar now.",
          actions: ["GENERATE_AVATAR"],
        },
      },
    ],
  ],

  /**
   * Valid if both services are running. We check eagerly so the
   * action doesn't show up in the agent's repertoire if the
   * dependency isn't configured.
   */
  validate: async (runtime: IAgentRuntime): Promise<boolean> => {
    return getAvb(runtime) !== undefined && hasScrypted(runtime);
  },

  handler: async (
    runtime: IAgentRuntime,
    message: Memory,
  ): Promise<ActionResult> => {
    const avb = getAvb(runtime);
    if (!avb) {
      return {
        success: false,
        error: "AVB service not available",
      };
    }

    const runId = await avb.createRun(message.roomId, message.id);

    return {
      success: true,
      text: "Avatar generation started — I'll post it here when ready.",
      data: { runId, actionName: "GENERATE_AVATAR" },
    };
  },
};
