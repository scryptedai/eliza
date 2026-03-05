/**
 * Minimal @elizaos/core mock for vitest.
 * The real package requires generated protobuf files that are not present
 * in the source tree; tests only need the abstract Service base.
 */

export abstract class Service {
  protected runtime!: unknown;
  constructor(runtime?: unknown) {
    if (runtime) this.runtime = runtime;
  }
  abstract stop(): Promise<void>;
  static serviceType: string;
  abstract capabilityDescription: string;
}

// Type-only exports — never evaluated at runtime
export type IAgentRuntime = unknown;
export type Route = unknown;
export type Plugin = unknown;
