/**
 * Test-time mock for @elizaos/core.
 *
 * Only the structural surface that plugin-avb actually imports/uses.
 * Runtime typings that are consumed via casts (IAgentRuntime method calls)
 * are left as `unknown` — tests provide concrete fake objects.
 */

// ----------------------------------------------------------------------------
// Service (abstract base)
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
// Primitive types
// ----------------------------------------------------------------------------

export type UUID = string;
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

// ----------------------------------------------------------------------------
// Content / Memory (structural subset)
// ----------------------------------------------------------------------------

export interface Content {
  text?: string;
  source?: string;
  attachments?: Array<{
    id: string;
    url: string;
    title?: string;
    contentType?: string;
  }>;
  thought?: string;
  actions?: string[];
  [key: string]: unknown;
}

export interface Memory {
  id?: UUID;
  entityId?: UUID;
  agentId?: UUID;
  roomId: UUID;
  content: Content;
  createdAt?: number;
  [key: string]: unknown;
}

// ----------------------------------------------------------------------------
// Action / ActionResult (structural subset)
// ----------------------------------------------------------------------------

export interface ActionResult {
  success: boolean;
  text?: string;
  data?: Record<string, unknown>;
  error?: string | Error;
}

export interface Action {
  name: string;
  description: string;
  similes?: string[];
  examples?: unknown[][];
  validate: (runtime: unknown, message: Memory) => Promise<boolean>;
  handler: (
    runtime: unknown,
    message: Memory,
    ...rest: unknown[]
  ) => Promise<ActionResult | undefined>;
}

// ----------------------------------------------------------------------------
// Task / TaskWorker (structural subset)
// ----------------------------------------------------------------------------

export interface TaskMetadata {
  updateInterval?: number;
  updatedAt?: number;
  blocking?: boolean;
  [key: string]: unknown;
}

export interface Task {
  id?: UUID;
  name: string;
  description?: string;
  roomId?: UUID;
  worldId?: UUID;
  entityId?: UUID;
  tags?: string[];
  metadata?: TaskMetadata;
  createdAt?: number | bigint;
  updatedAt?: number | bigint;
}

export interface TaskWorker {
  name: string;
  execute: (
    runtime: unknown,
    options: Record<string, unknown>,
    task: Task,
  ) => Promise<void>;
  validate?: (
    runtime: unknown,
    message: Memory,
    state: unknown,
  ) => Promise<boolean>;
}

// ----------------------------------------------------------------------------
// Enums (string-valued)
// ----------------------------------------------------------------------------

export const EventType = {
  MESSAGE_SENT: "MESSAGE_SENT",
} as const;

export const ContentType = {
  IMAGE: "image",
  VIDEO: "video",
} as const;

// ----------------------------------------------------------------------------
// Opaque placeholders (consumed only via structural casts in source)
// ----------------------------------------------------------------------------

export type IAgentRuntime = unknown;
export type Plugin = unknown;
export type State = unknown;
export type Character = {
  name: string;
  bio?: string[] | string;
  topics?: string[];
  adjectives?: string[];
  system?: string;
  style?: {
    all?: string[];
    chat?: string[];
    post?: string[];
  };
  [key: string]: unknown;
};
