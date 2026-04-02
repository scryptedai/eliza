/**
 * @elizaos/plugin-computeruse
 *
 * Surfaces the `computeruse` Rust engine (NAPI) as a first-class ElizaOS
 * plugin so any character can drive the local desktop without hand-wiring
 * an MCP transport.
 *
 * Provides:
 *   - ComputerUseService          long-lived `Desktop` handle + health probe
 *   - Actions                      SCREENSHOT, GET_WINDOW_TREE, CLICK_ELEMENT,
 *                                  TYPE_TEXT, OPEN_APPLICATION, RUN_COMMAND
 *   - Providers                    COMPUTERUSE_HEALTH (always), DESKTOP_STATE (dynamic)
 *
 * Safety:
 *   - RUN_COMMAND / OPEN_APPLICATION respect COMPUTERUSE_EXEC_MODE (Issue #1)
 *   - All side-effecting actions accept `idempotencyKey` (Issue #2)
 *   - Native-load failure is non-fatal: actions validate() to false instead
 *     of crashing the agent
 */

import type { Plugin } from "@elizaos/core";
import { computerUseActions } from "./actions.ts";
import { computerUseProviders } from "./providers.ts";
import { ComputerUseService } from "./service.ts";

export const computerUsePlugin: Plugin = {
  name: "computeruse",
  description:
    "Native desktop automation: see (screenshot/window-tree) and act " +
    "(click/type/open/run) on the local machine via the accessibility tree.",
  services: [ComputerUseService],
  actions: computerUseActions,
  providers: computerUseProviders,
};

export default computerUsePlugin;

// ----------------------------------------------------------------------------
// Public API re-exports
// ----------------------------------------------------------------------------

export {
  clickElementAction,
  computerUseActions,
  getWindowTreeAction,
  openApplicationAction,
  runCommandAction,
  screenshotAction,
  typeTextAction,
} from "./actions.ts";
export {
  computerUseProviders,
  desktopStateProvider,
  healthProvider,
} from "./providers.ts";
export {
  COMPUTERUSE_SERVICE_TYPE,
  type ComputerUseHealth,
  ComputerUseService,
  type ExecMode,
  getComputerUse,
} from "./service.ts";
