/**
 * ElizaOS Actions backed by the `computeruse` NAPI engine.
 *
 * Each action follows the same shape:
 *   validate() → service available (and, for exec actions, policy permits)
 *   handler()  → pull params, call Desktop, wrap as ActionResult
 *
 * All side-effecting handlers accept an optional `idempotencyKey`
 * parameter (Issue #2) which is forwarded into the result `data` so
 * downstream evaluators / sequence runners can dedupe; the heavy-weight
 * cache lives on the Rust side and is reached when these same operations
 * go through the MCP path.
 */

import type {
  Action,
  ActionResult,
  HandlerOptions,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import { getComputerUse } from "./service.ts";

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

type Params = Record<string, unknown>;

function param<T>(opts: unknown, key: string): T | undefined {
  const p = (opts as HandlerOptions | undefined)?.parameters as
    | Params
    | undefined;
  return p?.[key] as T | undefined;
}

function fail(msg: string): ActionResult {
  return { success: false, error: msg, text: msg };
}

function ok(text: string, data?: Record<string, unknown>): ActionResult {
  return { success: true, text, data: data as ActionResult["data"] };
}

const requireService = (runtime: IAgentRuntime) => {
  const svc = getComputerUse(runtime);
  if (!svc) throw new Error("computeruse service not registered");
  return svc;
};

const validateAvailable = async (runtime: IAgentRuntime): Promise<boolean> =>
  getComputerUse(runtime)?.isAvailable() ?? false;

// ----------------------------------------------------------------------------
// SCREENSHOT
// ----------------------------------------------------------------------------

export const screenshotAction: Action = {
  name: "SCREENSHOT",
  description:
    "Capture a screenshot of a running application window (or the whole " +
    "monitor if `entireMonitor` is true). Returns the PNG as base64 plus " +
    "dimensions. Use this to *see* the current screen before deciding on " +
    "the next CLICK_ELEMENT / TYPE_TEXT step.",
  similes: ["TAKE_SCREENSHOT", "CAPTURE_SCREEN", "LOOK_AT_SCREEN"],
  parameters: [
    {
      name: "process",
      description:
        "Process / application name whose front window to capture (e.g. 'Safari', 'Code').",
      required: true,
      schema: { type: "string" },
    },
    {
      name: "entireMonitor",
      description: "Capture the whole monitor instead of just the window.",
      required: false,
      schema: { type: "boolean", default: false },
    },
  ],
  validate: validateAvailable,
  handler: async (
    runtime: IAgentRuntime,
    _msg: Memory,
    _state?: State,
    opts?: unknown,
  ): Promise<ActionResult> => {
    const process = param<string>(opts, "process");
    if (!process) return fail("SCREENSHOT requires `process`");
    const entire = param<boolean>(opts, "entireMonitor") ?? false;
    try {
      const desktop = requireService(runtime).requireDesktop();
      const shot = await desktop.captureScreenshot(process, null, entire);
      const b64 = Buffer.from(shot.imageData).toString("base64");
      return ok(
        `Captured ${shot.width}×${shot.height} screenshot of ${process}.`,
        {
          width: shot.width,
          height: shot.height,
          mimeType: "image/png",
          imageBase64: b64,
        },
      );
    } catch (e) {
      return fail(`screenshot failed: ${(e as Error).message}`);
    }
  },
};

// ----------------------------------------------------------------------------
// GET_WINDOW_TREE
// ----------------------------------------------------------------------------

export const getWindowTreeAction: Action = {
  name: "GET_WINDOW_TREE",
  description:
    "Dump the accessibility tree of an application window as structured " +
    "JSON (roles, names, bounds, indices). Use the returned `selector` " +
    "strings or `index` values with CLICK_ELEMENT / TYPE_TEXT. Prefer this " +
    "over SCREENSHOT when you need to *act* rather than just *look* — the " +
    "tree gives stable handles, the screenshot only gives pixels.",
  similes: ["INSPECT_WINDOW", "LIST_UI_ELEMENTS", "READ_SCREEN_STRUCTURE"],
  parameters: [
    {
      name: "process",
      description: "Process / application name to inspect.",
      required: true,
      schema: { type: "string" },
    },
    {
      name: "title",
      description: "Optional window-title substring to disambiguate.",
      required: false,
      schema: { type: "string" },
    },
  ],
  validate: validateAvailable,
  handler: async (runtime, _msg, _state, opts): Promise<ActionResult> => {
    const process = param<string>(opts, "process");
    if (!process) return fail("GET_WINDOW_TREE requires `process`");
    const title = param<string>(opts, "title");
    try {
      const desktop = requireService(runtime).requireDesktop();
      const tree = desktop.getWindowTree(process, title ?? null);
      return ok(`Window tree for ${process} captured.`, {
        process,
        title,
        tree: tree as unknown as Record<string, unknown>,
      });
    } catch (e) {
      return fail(`get_window_tree failed: ${(e as Error).message}`);
    }
  },
};

// ----------------------------------------------------------------------------
// CLICK_ELEMENT
// ----------------------------------------------------------------------------

export const clickElementAction: Action = {
  name: "CLICK_ELEMENT",
  description:
    "Click a UI element identified by an accessibility selector " +
    "(e.g. `role:Button|name:Submit`, `text:Save`, `id:close-btn`). " +
    "Resolve selectors via GET_WINDOW_TREE first if unsure.",
  similes: ["CLICK", "PRESS_BUTTON", "TAP_ELEMENT"],
  parameters: [
    {
      name: "selector",
      description:
        "Accessibility selector string. Supported prefixes: role:, name:, text:, id:, nativeid:, classname:, path:.",
      required: true,
      schema: { type: "string" },
      examples: ["role:Button|name:Submit", "text:Sign in"],
    },
    {
      name: "process",
      description:
        "Application to scope the search to. Strongly recommended — without it the whole desktop is searched.",
      required: false,
      schema: { type: "string" },
    },
    {
      name: "idempotencyKey",
      description:
        "Opaque key making this click safe to retry — repeated calls with the same key inside one sequence are deduped.",
      required: false,
      schema: { type: "string" },
    },
  ],
  validate: validateAvailable,
  handler: async (runtime, _msg, _state, opts): Promise<ActionResult> => {
    const selector = param<string>(opts, "selector");
    if (!selector) return fail("CLICK_ELEMENT requires `selector`");
    const process = param<string>(opts, "process");
    const idem = param<string>(opts, "idempotencyKey");
    try {
      const desktop = requireService(runtime).requireDesktop();
      const root = process ? desktop.application(process) : desktop.root();
      const el = await root.locator(selector).first(10_000);
      const result = await el.click();
      return ok(
        `Clicked ${el.role()} "${el.name() ?? selector}" in ${process ?? "desktop"}.`,
        {
          selector,
          process,
          idempotencyKey: idem,
          click: result as unknown as Record<string, unknown>,
        },
      );
    } catch (e) {
      return fail(`click failed: ${(e as Error).message}`);
    }
  },
};

// ----------------------------------------------------------------------------
// TYPE_TEXT
// ----------------------------------------------------------------------------

export const typeTextAction: Action = {
  name: "TYPE_TEXT",
  description:
    "Type literal text into an input element identified by selector. " +
    "Focuses the element first. Use `clear:true` to wipe existing content.",
  similes: ["ENTER_TEXT", "FILL_FIELD", "INPUT_TEXT"],
  parameters: [
    {
      name: "selector",
      description: "Accessibility selector of the target input.",
      required: true,
      schema: { type: "string" },
    },
    {
      name: "text",
      description: "Text to type.",
      required: true,
      schema: { type: "string" },
    },
    {
      name: "process",
      description: "Application to scope the search to.",
      required: false,
      schema: { type: "string" },
    },
    {
      name: "idempotencyKey",
      description: "Dedupe key for safe retry.",
      required: false,
      schema: { type: "string" },
    },
  ],
  validate: validateAvailable,
  handler: async (runtime, _msg, _state, opts): Promise<ActionResult> => {
    const selector = param<string>(opts, "selector");
    const text = param<string>(opts, "text");
    if (!selector || text === undefined)
      return fail("TYPE_TEXT requires `selector` and `text`");
    const process = param<string>(opts, "process");
    try {
      const desktop = requireService(runtime).requireDesktop();
      const root = process ? desktop.application(process) : desktop.root();
      const el = await root.locator(selector).first(10_000);
      el.typeText(text);
      return ok(`Typed ${text.length} chars into ${selector}.`, {
        selector,
        process,
        length: text.length,
        idempotencyKey: param<string>(opts, "idempotencyKey"),
      });
    } catch (e) {
      return fail(`type_text failed: ${(e as Error).message}`);
    }
  },
};

// ----------------------------------------------------------------------------
// OPEN_APPLICATION  (exec-gated)
// ----------------------------------------------------------------------------

export const openApplicationAction: Action = {
  name: "OPEN_APPLICATION",
  description:
    "Launch (or foreground) a desktop application by name. Subject to " +
    "COMPUTERUSE_EXEC_MODE — refused outright when mode is `deny`.",
  similes: ["LAUNCH_APP", "START_APPLICATION"],
  parameters: [
    {
      name: "name",
      description:
        "Application name as the OS knows it (e.g. 'Safari', 'Notepad').",
      required: true,
      schema: { type: "string" },
    },
  ],
  validate: async (runtime) => {
    const svc = getComputerUse(runtime);
    return (svc?.isAvailable() ?? false) && svc!.getExecMode() !== "deny";
  },
  handler: async (runtime, _msg, _state, opts): Promise<ActionResult> => {
    const name = param<string>(opts, "name");
    if (!name) return fail("OPEN_APPLICATION requires `name`");
    const svc = requireService(runtime);
    const denial = svc.checkExecPolicy("open_application", name);
    if (denial) return fail(denial);
    try {
      const el = svc.requireDesktop().openApplication(name, false, false);
      return ok(`Opened ${name}.`, {
        name,
        role: el.role(),
        title: el.name(),
      });
    } catch (e) {
      return fail(`open_application failed: ${(e as Error).message}`);
    }
  },
};

// ----------------------------------------------------------------------------
// RUN_COMMAND  (exec-gated)
// ----------------------------------------------------------------------------

export const runCommandAction: Action = {
  name: "RUN_COMMAND",
  description:
    "Execute a shell command on the host and return stdout/stderr. " +
    "Hard-gated by COMPUTERUSE_EXEC_MODE: refused when `deny`, and the " +
    "direct-NAPI path treats `allowlist` as deny (the allowlist parser " +
    "lives in the MCP server). Prefer narrower actions (CLICK_ELEMENT, " +
    "OPEN_APPLICATION) when they suffice.",
  similes: ["EXEC", "SHELL", "EXECUTE_COMMAND"],
  parameters: [
    {
      name: "command",
      description: "Command line to execute via the platform shell.",
      required: true,
      schema: { type: "string" },
    },
    {
      name: "idempotencyKey",
      description: "Dedupe key for safe retry.",
      required: false,
      schema: { type: "string" },
    },
  ],
  validate: async (runtime) => {
    const svc = getComputerUse(runtime);
    // Only advertise this action when policy would actually allow it.
    return (svc?.isAvailable() ?? false) && svc!.getExecMode() === "full";
  },
  handler: async (runtime, _msg, _state, opts): Promise<ActionResult> => {
    const cmd = param<string>(opts, "command");
    if (!cmd) return fail("RUN_COMMAND requires `command`");
    const svc = requireService(runtime);
    const denial = svc.checkExecPolicy("run_command", cmd);
    if (denial) return fail(denial);
    try {
      const isWin = process.platform === "win32";
      const out = await svc
        .requireDesktop()
        .runCommand(isWin ? cmd : null, isWin ? null : cmd);
      return ok(`Command exited ${out.exitStatus ?? 0}.`, {
        command: cmd,
        exitStatus: out.exitStatus,
        stdout: out.stdout,
        stderr: out.stderr,
        idempotencyKey: param<string>(opts, "idempotencyKey"),
      });
    } catch (e) {
      return fail(`run_command failed: ${(e as Error).message}`);
    }
  },
};

// ----------------------------------------------------------------------------

export const computerUseActions: Action[] = [
  screenshotAction,
  getWindowTreeAction,
  clickElementAction,
  typeTextAction,
  openApplicationAction,
  runCommandAction,
];
