/**
 * ComputerUseService — owns a single `Desktop` handle (the NAPI bridge
 * into the Rust `computeruse` engine) for the lifetime of the agent.
 *
 * Why a Service and not just per-action `new Desktop()`:
 *  - Constructing `Desktop` initialises the platform accessibility engine
 *    (UIAutomation COM on Windows, AX on macOS). Doing that per call is
 *    expensive and on some platforms leaks observers.
 *  - Centralising the handle lets us run the health probe once at boot
 *    and surface a single, actionable error if permissions are missing,
 *    instead of every action failing opaquely.
 *  - Exec-safety policy (Issue #1) is enforced here so RUN_COMMAND /
 *    OPEN_APPLICATION cannot bypass it by going around the service.
 *
 * The native module is loaded lazily inside `start()` so that merely
 * *listing* this plugin in a character file does not crash hosts that
 * lack the prebuilt `.node` binary — the agent boots, the service marks
 * itself unavailable, and actions `validate()` to false.
 */

// Type-only import: pulls the .d.ts without executing the native loader.
import type * as ComputerUse from "@elizaos/computeruse";
import { type IAgentRuntime, logger, Service } from "@elizaos/core";

export const COMPUTERUSE_SERVICE_TYPE = "computeruse";

export type ExecMode = "deny" | "allowlist" | "full";

/** Snapshot of the host's automation readiness, refreshed at boot. */
export interface ComputerUseHealth {
  available: boolean;
  platform: string;
  /** Human-readable reason when `available` is false. */
  reason?: string;
  /** Number of running applications visible to the AX/UIA layer. */
  applicationCount?: number;
}

export class ComputerUseService extends Service {
  static serviceType = COMPUTERUSE_SERVICE_TYPE;

  capabilityDescription =
    "Drives the local desktop via the native accessibility tree: screenshot, " +
    "click, type, read window structure, open apps, and run shell commands.";

  /** Lazily-loaded native module (undefined ⇒ load failed). */
  private native?: typeof ComputerUse;
  /** Long-lived engine handle. */
  private desktop?: ComputerUse.Desktop;
  /** Cached health from the boot probe. */
  private health: ComputerUseHealth = {
    available: false,
    platform: process.platform,
    reason: "service not started",
  };
  private execMode: ExecMode = "full";

  // --------------------------------------------------------------------
  // Lifecycle
  // --------------------------------------------------------------------

  static async start(runtime: IAgentRuntime): Promise<ComputerUseService> {
    const svc = new ComputerUseService(runtime);
    await svc.initialize();
    return svc;
  }

  private async initialize(): Promise<void> {
    const setting = (
      this.runtime as { getSetting?(k: string): unknown } | undefined
    )?.getSetting?.("COMPUTERUSE_EXEC_MODE");
    this.execMode = readExecMode(setting ?? process.env.COMPUTERUSE_EXEC_MODE);

    // 1. Load the native addon. Failure here is non-fatal: we record why
    //    and let actions opt out via validate().
    try {
      this.native = (await import(
        "@elizaos/computeruse"
      )) as typeof ComputerUse;
    } catch (e) {
      this.health = {
        available: false,
        platform: process.platform,
        reason: `native module failed to load: ${(e as Error).message}`,
      };
      logger.warn(
        `[computeruse] native addon unavailable — actions disabled (${this.health.reason})`,
      );
      return;
    }

    // 2. Construct the engine and probe.
    try {
      this.desktop = new this.native.Desktop(false, false);
      const apps = this.desktop.applications();
      this.health = {
        available: true,
        platform: process.platform,
        applicationCount: apps.length,
      };
      logger.info(
        `[computeruse] ready (platform=${process.platform}, apps=${apps.length}, execMode=${this.execMode})`,
      );
    } catch (e) {
      const err = e as Error;
      const isPerm =
        err.name === "PermissionDeniedError" || /permission/i.test(err.message);
      this.health = {
        available: false,
        platform: process.platform,
        reason: isPerm
          ? `accessibility permission denied: ${err.message}`
          : `engine init failed: ${err.message}`,
      };
      logger.warn(`[computeruse] ${this.health.reason}`);
    }
  }

  async stop(): Promise<void> {
    // The NAPI handle has no explicit close; dropping the reference lets
    // the Rust side `Drop` impl release platform observers on GC.
    this.desktop = undefined;
    this.native = undefined;
  }

  // --------------------------------------------------------------------
  // Accessors
  // --------------------------------------------------------------------

  isAvailable(): boolean {
    return this.health.available && this.desktop !== undefined;
  }

  getHealth(): ComputerUseHealth {
    return { ...this.health };
  }

  getExecMode(): ExecMode {
    return this.execMode;
  }

  /**
   * Exec-policy gate — TS-side mirror of the Rust `exec_policy` module
   * (Issue #1). The Rust layer is authoritative when going through MCP;
   * this guard covers the direct-NAPI path used by this plugin.
   *
   * `subject` is the command line for RUN_COMMAND or the app name for
   * OPEN_APPLICATION. Returns `null` when allowed, otherwise a denial
   * message suitable for surfacing to the user.
   */
  checkExecPolicy(
    tool: "run_command" | "open_application",
    subject: string,
  ): string | null {
    if (this.execMode === "full") return null;
    if (this.execMode === "deny") {
      return (
        `${tool} blocked: COMPUTERUSE_EXEC_MODE=deny. ` +
        `Set COMPUTERUSE_EXEC_MODE=full (or allowlist with a policy file) to permit '${subject}'.`
      );
    }
    // allowlist: the TS side has no policy-file parser — defer to deny
    // with a pointer to the MCP path which does evaluate the allowlist.
    return (
      `${tool} blocked: COMPUTERUSE_EXEC_MODE=allowlist is enforced by the ` +
      `MCP server (computeruse-mcp-agent), not the direct NAPI bridge. ` +
      `Either route through MCP or set COMPUTERUSE_EXEC_MODE=full for '${subject}'.`
    );
  }

  /**
   * Returns the live `Desktop` handle or throws with the cached health
   * reason. Actions call this after `validate()` has already returned
   * true, so a throw here indicates a race (e.g. permission revoked
   * mid-session) and is surfaced as `ActionResult.error`.
   */
  requireDesktop(): ComputerUse.Desktop {
    if (!this.desktop) {
      throw new Error(
        this.health.reason ?? "computeruse desktop handle not initialised",
      );
    }
    return this.desktop;
  }
}

// ----------------------------------------------------------------------------

function readExecMode(raw: unknown): ExecMode {
  const v = String(raw ?? "full")
    .trim()
    .toLowerCase();
  return v === "deny" || v === "allowlist" ? v : "full";
}

/** Typed runtime lookup used by actions/providers. */
export function getComputerUse(
  runtime: IAgentRuntime,
): ComputerUseService | undefined {
  return (
    runtime as unknown as { getService<T>(type: string): T | undefined }
  ).getService<ComputerUseService>(COMPUTERUSE_SERVICE_TYPE);
}
