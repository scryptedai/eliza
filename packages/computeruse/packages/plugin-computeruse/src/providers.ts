/**
 * Providers — inject desktop context into the agent's prompt.
 *
 * `DESKTOP_STATE` is dynamic (only fetched when the model asks for it or a
 * relevance keyword fires) because enumerating applications touches the
 * accessibility API and we don't want that on every turn.
 *
 * `COMPUTERUSE_HEALTH` is cheap (cached at service boot) and always runs
 * so the model knows up-front whether desktop actions are even possible.
 */

import type {
  IAgentRuntime,
  Memory,
  Provider,
  ProviderResult,
  State,
} from "@elizaos/core";
import { getComputerUse } from "./service.ts";

export const desktopStateProvider: Provider = {
  name: "DESKTOP_STATE",
  description:
    "Current focused element and list of running applications on the local desktop.",
  dynamic: true,
  relevanceKeywords: [
    "screen",
    "desktop",
    "window",
    "click",
    "type",
    "application",
    "screenshot",
  ],
  get: async (
    runtime: IAgentRuntime,
    _msg: Memory,
    _state: State,
  ): Promise<ProviderResult> => {
    const svc = getComputerUse(runtime);
    if (!svc?.isAvailable()) {
      return { text: "Desktop automation: unavailable on this host." };
    }
    try {
      const d = svc.requireDesktop();
      const focused = d.focusedElement();
      const apps = d
        .applications()
        .map((a) => a.name() ?? a.role())
        .filter(Boolean)
        .slice(0, 30);
      const text =
        `Desktop state — focused: ${focused.role()} "${focused.name() ?? ""}". ` +
        `Running apps (${apps.length}): ${apps.join(", ")}.`;
      return {
        text,
        data: {
          focused: { role: focused.role(), name: focused.name() },
          applications: apps,
        },
      };
    } catch (e) {
      return { text: `Desktop state unavailable: ${(e as Error).message}` };
    }
  },
};

export const healthProvider: Provider = {
  name: "COMPUTERUSE_HEALTH",
  description:
    "Whether the local UI-automation engine is usable, plus the exec-safety mode in force.",
  alwaysRun: true,
  position: -50,
  get: async (runtime): Promise<ProviderResult> => {
    const svc = getComputerUse(runtime);
    if (!svc) return { text: "" };
    const h = svc.getHealth();
    const mode = svc.getExecMode();
    if (h.available) {
      return {
        text:
          `Desktop automation: available (${h.platform}, ${h.applicationCount ?? "?"} apps, ` +
          `exec-mode=${mode}). Use GET_WINDOW_TREE → CLICK_ELEMENT/TYPE_TEXT to drive the UI.`,
        values: { computeruseAvailable: true, computeruseExecMode: mode },
        data: { health: h, execMode: mode },
      };
    }
    return {
      text: `Desktop automation: UNAVAILABLE on this host (${h.reason}). Desktop actions will be skipped.`,
      values: { computeruseAvailable: false },
      data: { health: h, execMode: mode },
    };
  },
};

export const computerUseProviders: Provider[] = [
  healthProvider,
  desktopStateProvider,
];
