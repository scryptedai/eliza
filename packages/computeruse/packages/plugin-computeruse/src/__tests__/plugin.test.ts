/**
 * @elizaos/plugin-computeruse — unit tests.
 *
 * Strategy: mock the native `@elizaos/computeruse` module with a fake
 * `Desktop` class so tests run on hosts without the .node binary or
 * accessibility permission. Build a minimal fake runtime that returns
 * the started service from `getService`. Drive each action's handler
 * directly with `parameters` and assert on the ActionResult.
 */

import type { IAgentRuntime, Memory } from "@elizaos/core";
import { beforeEach, describe, expect, it, vi } from "vitest";

// ---------------------------------------------------------------------------
// Native module mock
// ---------------------------------------------------------------------------

const fakeElement = (name = "OK", role = "Button") => ({
  name: () => name,
  role: () => role,
  click: vi.fn(async () => ({ method: "ax", x: 1, y: 2 })),
  typeText: vi.fn(() => ({ ok: true })),
  locator: (sel: string) => ({
    first: async () => fakeElement(sel, "Button"),
  }),
});

const calls: { runCommand: Array<[unknown, unknown]> } = { runCommand: [] };

class FakeDesktop {
  applications() {
    return [fakeElement("Safari", "Application")];
  }
  application(_name: string) {
    return fakeElement(_name, "Application");
  }
  root() {
    return fakeElement("root", "Desktop");
  }
  focusedElement() {
    return fakeElement("Search", "TextField");
  }
  openApplication(name: string) {
    return fakeElement(name, "Application");
  }
  getWindowTree(process: string) {
    return { role: "Window", name: process, children: [] };
  }
  async captureScreenshot() {
    return { width: 4, height: 2, imageData: [1, 2, 3, 4] };
  }
  async runCommand(win: unknown, unix: unknown) {
    calls.runCommand.push([win, unix]);
    return { exitStatus: 0, stdout: "ok", stderr: "" };
  }
}

class PermissionDeniedError extends Error {}

vi.mock("@elizaos/computeruse", () => ({
  Desktop: FakeDesktop,
  PermissionDeniedError,
}));

// Imports AFTER vi.mock so the dynamic import inside service.ts resolves
// to the mock.
const {
  ComputerUseService,
  COMPUTERUSE_SERVICE_TYPE,
  computerUsePlugin,
  runCommandAction,
  clickElementAction,
  openApplicationAction,
  screenshotAction,
  healthProvider,
} = await import("../index.ts");

// ---------------------------------------------------------------------------
// Fake runtime
// ---------------------------------------------------------------------------

function makeRuntime(settings: Record<string, string> = {}): {
  runtime: IAgentRuntime;
  setService: (svc: unknown) => void;
} {
  let service: unknown;
  const runtime = {
    getSetting: (k: string) => settings[k] ?? null,
    getService: (type: string) =>
      type === COMPUTERUSE_SERVICE_TYPE ? service : undefined,
  } as unknown as IAgentRuntime;
  return { runtime, setService: (s) => (service = s) };
}

const dummyMsg = { roomId: "r", content: { text: "" } } as unknown as Memory;
// Handler 4th arg is `HandlerOptions | Record<string, JsonValue|undefined>`;
// the test fixture only needs `.parameters`, so we cast through the union.
const opts = (parameters: Record<string, unknown>) =>
  ({ parameters }) as unknown as import("@elizaos/core").HandlerOptions;

// ---------------------------------------------------------------------------

describe("computerUsePlugin shape", () => {
  it("registers service, six actions, and two providers", () => {
    expect(computerUsePlugin.name).toBe("computeruse");
    expect(computerUsePlugin.services).toEqual([ComputerUseService]);
    expect(computerUsePlugin.actions?.map((a) => a.name)).toEqual([
      "SCREENSHOT",
      "GET_WINDOW_TREE",
      "CLICK_ELEMENT",
      "TYPE_TEXT",
      "OPEN_APPLICATION",
      "RUN_COMMAND",
    ]);
    expect(computerUsePlugin.providers?.map((p) => p.name)).toEqual([
      "COMPUTERUSE_HEALTH",
      "DESKTOP_STATE",
    ]);
  });
});

describe("ComputerUseService", () => {
  beforeEach(() => {
    calls.runCommand.length = 0;
  });

  it("starts, probes health, and reports available", async () => {
    const { runtime, setService } = makeRuntime();
    const svc = await ComputerUseService.start(runtime);
    setService(svc);
    expect(svc.isAvailable()).toBe(true);
    const h = svc.getHealth();
    expect(h.available).toBe(true);
    expect(h.applicationCount).toBe(1);
    expect(svc.getExecMode()).toBe("full");
  });

  it("exec policy: deny blocks run_command and open_application", async () => {
    const { runtime, setService } = makeRuntime({
      COMPUTERUSE_EXEC_MODE: "deny",
    });
    const svc = await ComputerUseService.start(runtime);
    setService(svc);

    expect(svc.checkExecPolicy("run_command", "rm -rf /")).toMatch(/deny/);

    // RUN_COMMAND validate() must be false so the action isn't even offered
    expect(await runCommandAction.validate(runtime, dummyMsg)).toBe(false);
    // OPEN_APPLICATION also gated
    expect(await openApplicationAction.validate(runtime, dummyMsg)).toBe(false);

    // Handler still refuses if somehow invoked
    const res = await runCommandAction.handler(
      runtime,
      dummyMsg,
      undefined,
      opts({ command: "echo hi" }),
    );
    expect(res?.success).toBe(false);
    expect(calls.runCommand.length).toBe(0);
  });

  it("exec policy: full allows run_command and routes by platform", async () => {
    const { runtime, setService } = makeRuntime({
      COMPUTERUSE_EXEC_MODE: "full",
    });
    const svc = await ComputerUseService.start(runtime);
    setService(svc);

    expect(await runCommandAction.validate(runtime, dummyMsg)).toBe(true);
    const res = await runCommandAction.handler(
      runtime,
      dummyMsg,
      undefined,
      opts({ command: "echo hi", idempotencyKey: "k1" }),
    );
    expect(res?.success).toBe(true);
    expect(res?.data?.stdout).toBe("ok");
    expect(res?.data?.idempotencyKey).toBe("k1");
    expect(calls.runCommand.length).toBe(1);
    const [win, unix] = calls.runCommand[0];
    if (process.platform === "win32") {
      expect(win).toBe("echo hi");
      expect(unix).toBeNull();
    } else {
      expect(win).toBeNull();
      expect(unix).toBe("echo hi");
    }
  });

  it("CLICK_ELEMENT resolves selector via locator and reports success", async () => {
    const { runtime, setService } = makeRuntime();
    setService(await ComputerUseService.start(runtime));
    const res = await clickElementAction.handler(
      runtime,
      dummyMsg,
      undefined,
      opts({ selector: "role:Button|name:OK", process: "Safari" }),
    );
    expect(res?.success).toBe(true);
    expect(res?.text).toMatch(/Clicked/);
  });

  it("SCREENSHOT returns base64-encoded image data", async () => {
    const { runtime, setService } = makeRuntime();
    setService(await ComputerUseService.start(runtime));
    const res = await screenshotAction.handler(
      runtime,
      dummyMsg,
      undefined,
      opts({ process: "Safari" }),
    );
    expect(res?.success).toBe(true);
    expect(res?.data?.width).toBe(4);
    expect(typeof res?.data?.imageBase64).toBe("string");
    const decoded = Buffer.from(res!.data!.imageBase64 as string, "base64");
    expect(Array.from(decoded)).toEqual([1, 2, 3, 4]);
  });

  it("COMPUTERUSE_HEALTH provider reflects service availability", async () => {
    const { runtime, setService } = makeRuntime();
    setService(await ComputerUseService.start(runtime));
    const out = await healthProvider.get(runtime, dummyMsg, {} as never);
    expect(out.values?.computeruseAvailable).toBe(true);
    expect(out.text).toMatch(/available/);
  });

  it("actions validate() false when service missing", async () => {
    const { runtime } = makeRuntime(); // never setService
    expect(await clickElementAction.validate(runtime, dummyMsg)).toBe(false);
    expect(await screenshotAction.validate(runtime, dummyMsg)).toBe(false);
  });
});
