/**
 * Test-time mock for @elizaos/core.
 *
 * Trimmed copy of plugin-avb's mock — only the surface plugin-ffm imports.
 * We need a working PromptSet (it's used in promptset.ts) so we provide a
 * functional minimal implementation, not a stub.
 */

// ----------------------------------------------------------------------------
// Service abstract base
// ----------------------------------------------------------------------------

export abstract class Service {
  protected runtime!: unknown;
  constructor(runtime?: unknown) {
    if (runtime) this.runtime = runtime;
  }
  abstract stop(): Promise<void>;
  static serviceType: string;
  abstract capabilityDescription: string;
}

// ----------------------------------------------------------------------------
// Opaque types
// ----------------------------------------------------------------------------

export type IAgentRuntime = unknown;
export type Plugin = unknown;
export type UUID = string;

// ----------------------------------------------------------------------------
// PromptSet — functional minimal implementation
// ----------------------------------------------------------------------------

export interface ModelLimits {
  maxInputTokens: number;
  maxOutputTokens: number;
}

const DEFAULT_LIMITS: ModelLimits = {
  maxInputTokens: 32_000,
  maxOutputTokens: 4_096,
};

export interface RenderedPromptSet {
  readonly system: string;
  readonly user: string;
  readonly limits: ModelLimits;
  readonly truncated: boolean;
}

const TAG = /\{\{([A-Za-z0-9_.-]+)\}\}/g;

export class PromptSet {
  readonly systemTemplate: string;
  readonly userTemplate: string;
  readonly limits: ModelLimits;

  constructor(opts: {
    system: string;
    user: string;
    model?: string;
    limits?: ModelLimits;
  }) {
    this.systemTemplate = opts.system;
    this.userTemplate = opts.user;
    this.limits = opts.limits ?? DEFAULT_LIMITS;
  }

  render(values: Record<string, unknown> = {}): RenderedPromptSet {
    const sub = (tpl: string) =>
      tpl.replace(TAG, (_m, key) => {
        const v = values[key];
        return v == null ? "" : String(v);
      });
    return Object.freeze({
      system: sub(this.systemTemplate),
      user: sub(this.userTemplate),
      limits: this.limits,
      truncated: false,
    });
  }
}

export function toScryptedPayload(r: RenderedPromptSet): {
  system_prompt: string;
  user_prompt: string;
  max_tokens: number;
} {
  return {
    system_prompt: r.system,
    user_prompt: r.user,
    max_tokens: r.limits.maxOutputTokens,
  };
}
