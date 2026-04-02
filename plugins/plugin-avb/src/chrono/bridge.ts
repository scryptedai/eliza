/**
 * Runtime → Chronometer event bridge.
 *
 * The ElizaOS runtime emits a stream of `EventType` notifications
 * (`runtime.emitEvent`) describing what the agent is doing — messages,
 * action runs, model calls, world/entity joins, session lifecycle.
 * This module subscribes the AVB plugin to a curated subset of those
 * and forwards each to `ChronometerService.recordEvent(RUNTIME, …)`,
 * so the agent's tamper-evident PoW log automatically captures its
 * own behavioural history without every call site having to remember
 * to call `recordEvent` itself.
 *
 * Selection policy: notable, not spammy.
 *   - Per-tool-call hooks, embedding requests and voice frames are
 *     SKIPPED — they fire many times per user turn and would dominate
 *     the log without adding forensic value.
 *   - Lifecycle, message, action, run, model and topology events are
 *     KEPT — at ~100/min worst-case the v2 off-chain log handles them
 *     comfortably (see DESIGN-growth.md).
 *
 * Detail strings are deliberately short (event name + a handful of
 * IDs / counters); free-text content is truncated to a 64-char head
 * so the log records *that* something was said, not the full transcript.
 */

import {
  type ActionEventPayload,
  type EntityPayload,
  type EvaluatorEventPayload,
  type EventPayload,
  EventType,
  type HookAgentLifecyclePayload,
  type HookCommandPayload,
  type HookCompactionPayload,
  type HookGatewayPayload,
  type HookSessionPayload,
  type IAgentRuntime,
  type InvokePayload,
  type MessagePayload,
  type ModelEventPayload,
  type PluginEvents,
  type RunEventPayload,
  type WorldPayload,
} from "@elizaos/core";

import { CHRONO_SERVICE_TYPE, type ChronometerService } from "./service.ts";
import { ChronoEventType } from "./types.ts";

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/** Hard cap on detail length so a single event can't bloat a block. */
const MAX_DETAIL = 240;
/** How much of free-text content (message bodies etc.) to retain. */
const TEXT_HEAD = 64;

function clip(s: string, n: number): string {
  if (s.length <= n) return s;
  return `${s.slice(0, n - 1)}…`;
}

function id(v: unknown): string {
  return typeof v === "string" && v.length > 0 ? v.slice(0, 8) : "-";
}

function record(
  runtime: IAgentRuntime,
  name: EventType,
  summary: string,
): void {
  const svc = runtime.getService<ChronometerService>(CHRONO_SERVICE_TYPE);
  if (!svc) return; // chronometer not (yet) registered — drop silently
  svc.recordEvent(
    ChronoEventType.RUNTIME,
    clip(`${name} ${summary}`.trimEnd(), MAX_DETAIL),
  );
}

/**
 * Wrap a summariser into the `EventHandler` shape the plugin system
 * expects. All payloads extend `EventPayload` so `runtime` is always
 * present; the wrapper swallows summariser errors so a malformed
 * payload can never break the runtime's event dispatch.
 */
function on<P extends EventPayload>(
  name: EventType,
  summarise: (p: P) => string,
): (p: P) => Promise<void> {
  return async (p: P): Promise<void> => {
    let summary = "";
    try {
      summary = summarise(p);
    } catch {
      summary = "(payload unreadable)";
    }
    record(p.runtime, name, summary);
  };
}

// ----------------------------------------------------------------------------
// Per-shape summarisers
// ----------------------------------------------------------------------------

function sumMessage(p: MessagePayload): string {
  const m = p.message as {
    id?: string;
    roomId?: string;
    entityId?: string;
    content?: { text?: string; actions?: string[] };
  };
  const text = m.content?.text;
  const acts = m.content?.actions;
  return (
    `room=${id(m.roomId)} from=${id(m.entityId)} msg=${id(m.id)}` +
    (acts?.length ? ` actions=${acts.join(",")}` : "") +
    (text ? ` text="${clip(text.replace(/\s+/g, " "), TEXT_HEAD)}"` : "")
  );
}

function sumWorld(p: WorldPayload): string {
  const w = p.world as { id?: string; name?: string };
  return `world=${id(w.id)} name=${w.name ?? "-"} rooms=${p.rooms.length} entities=${p.entities.length}`;
}

function sumEntity(p: EntityPayload): string {
  return (
    `entity=${id(p.entityId)} room=${id(p.roomId)} world=${id(p.worldId)}` +
    (p.metadata?.username ? ` user=${p.metadata.username}` : "")
  );
}

function sumRun(p: RunEventPayload): string {
  const dur = p.duration !== undefined ? ` dur=${Number(p.duration)}ms` : "";
  const err = p.error
    ? ` err="${clip(p.error instanceof Error ? p.error.message : String(p.error), 60)}"`
    : "";
  return `run=${id(p.runId)} room=${id(p.roomId)} status=${p.status}${dur}${err}`;
}

function sumAction(p: ActionEventPayload): string {
  const acts = (p.content as { actions?: string[] }).actions;
  return `room=${id(p.roomId)} msg=${id(p.messageId)} action=${acts?.join(",") ?? "-"}`;
}

function sumEvaluator(p: EvaluatorEventPayload): string {
  return (
    `evaluator=${p.evaluatorName}` +
    (p.completed === false ? " completed=false" : "") +
    (p.error ? ` err="${clip(p.error.message, 60)}"` : "")
  );
}

function sumModel(p: ModelEventPayload): string {
  const t = p.tokens;
  return (
    `type=${p.type}` +
    (t
      ? ` tok=${t.total}(p${t.prompt}/c${t.completion}` +
        (t.cacheRead ? `/r${t.cacheRead}` : "") +
        (t.cacheWrite ? `/w${t.cacheWrite}` : "") +
        ")"
      : "")
  );
}

function sumInvoke(p: InvokePayload): string {
  return `room=${id(p.roomId)} world=${id(p.worldId)} src=${p.source ?? "-"}`;
}

function sumSession(p: HookSessionPayload): string {
  return `session=${p.sessionKey} chan=${p.channelId ?? "-"}`;
}

function sumCommand(p: HookCommandPayload): string {
  return `cmd=${p.command} session=${p.sessionKey} src=${p.commandSource ?? "-"}`;
}

function sumAgentLife(p: HookAgentLifecyclePayload): string {
  return (
    `session=${p.sessionKey}` +
    (p.success !== undefined ? ` ok=${p.success}` : "") +
    (p.durationMs !== undefined ? ` dur=${p.durationMs}ms` : "") +
    (p.error ? ` err="${clip(p.error, 60)}"` : "")
  );
}

function sumGateway(p: HookGatewayPayload): string {
  return `host=${p.host ?? "-"} port=${p.port ?? "-"} chans=${p.channels?.length ?? 0}`;
}

function sumCompaction(p: HookCompactionPayload): string {
  return `msgs=${p.messageCount}${p.compactedCount !== undefined ? ` compacted=${p.compactedCount}` : ""}${p.tokenCount !== undefined ? ` tok=${p.tokenCount}` : ""}`;
}

// ----------------------------------------------------------------------------
// Curated event map
// ----------------------------------------------------------------------------

/**
 * Runtime EventTypes deliberately NOT forwarded — high-volume / low
 * forensic value. Exported for visibility (and so the README can list
 * them).
 */
export const CHRONO_BRIDGE_SKIPPED: readonly EventType[] = [
  EventType.VOICE_MESSAGE_RECEIVED,
  EventType.VOICE_MESSAGE_SENT,
  EventType.EMBEDDING_GENERATION_REQUESTED,
  EventType.EMBEDDING_GENERATION_COMPLETED,
  EventType.EMBEDDING_GENERATION_FAILED,
  EventType.HOOK_TOOL_BEFORE,
  EventType.HOOK_TOOL_AFTER,
  EventType.HOOK_TOOL_PERSIST,
  EventType.HOOK_MESSAGE_SENDING,
  EventType.CONTROL_MESSAGE,
  EventType.MESSAGE_DELETED,
  EventType.CHANNEL_CLEARED,
  EventType.ENTITY_UPDATED,
  EventType.EVALUATOR_STARTED,
  EventType.FORM_FIELD_CONFIRMED,
  EventType.FORM_FIELD_CANCELLED,
  EventType.HOOK_AGENT_BOOTSTRAP,
  // No EventPayloadMap entry → PluginEvents can't subscribe; world/entity
  // joins already cover the interesting topology changes.
  EventType.ROOM_JOINED,
  EventType.ROOM_LEFT,
];

/**
 * Plugin event handlers that mirror notable runtime events into the
 * Chronometer's binary log. Spread/merge into the AVB plugin's
 * `events` field.
 */
export const chronoRuntimeEvents: PluginEvents = {
  // Messaging
  [EventType.MESSAGE_RECEIVED]: [on(EventType.MESSAGE_RECEIVED, sumMessage)],
  [EventType.MESSAGE_SENT]: [on(EventType.MESSAGE_SENT, sumMessage)],
  [EventType.REACTION_RECEIVED]: [on(EventType.REACTION_RECEIVED, sumMessage)],
  [EventType.INTERACTION_RECEIVED]: [
    on(EventType.INTERACTION_RECEIVED, sumMessage),
  ],
  [EventType.POST_GENERATED]: [on(EventType.POST_GENERATED, sumInvoke)],

  // Topology
  [EventType.WORLD_JOINED]: [on(EventType.WORLD_JOINED, sumWorld)],
  [EventType.WORLD_CONNECTED]: [on(EventType.WORLD_CONNECTED, sumWorld)],
  [EventType.WORLD_LEFT]: [on(EventType.WORLD_LEFT, sumWorld)],
  [EventType.ENTITY_JOINED]: [on(EventType.ENTITY_JOINED, sumEntity)],
  [EventType.ENTITY_LEFT]: [on(EventType.ENTITY_LEFT, sumEntity)],

  // Reasoning loop
  [EventType.RUN_STARTED]: [on(EventType.RUN_STARTED, sumRun)],
  [EventType.RUN_ENDED]: [on(EventType.RUN_ENDED, sumRun)],
  [EventType.RUN_TIMEOUT]: [on(EventType.RUN_TIMEOUT, sumRun)],
  [EventType.ACTION_STARTED]: [on(EventType.ACTION_STARTED, sumAction)],
  [EventType.ACTION_COMPLETED]: [on(EventType.ACTION_COMPLETED, sumAction)],
  [EventType.EVALUATOR_COMPLETED]: [
    on(EventType.EVALUATOR_COMPLETED, sumEvaluator),
  ],
  [EventType.MODEL_USED]: [on(EventType.MODEL_USED, sumModel)],

  // Process lifecycle (rare, always notable)
  [EventType.HOOK_SESSION_START]: [
    on(EventType.HOOK_SESSION_START, sumSession),
  ],
  [EventType.HOOK_SESSION_END]: [on(EventType.HOOK_SESSION_END, sumSession)],
  [EventType.HOOK_AGENT_START]: [on(EventType.HOOK_AGENT_START, sumAgentLife)],
  [EventType.HOOK_AGENT_END]: [on(EventType.HOOK_AGENT_END, sumAgentLife)],
  [EventType.HOOK_GATEWAY_START]: [
    on(EventType.HOOK_GATEWAY_START, sumGateway),
  ],
  [EventType.HOOK_GATEWAY_STOP]: [on(EventType.HOOK_GATEWAY_STOP, sumGateway)],
  [EventType.HOOK_COMMAND_NEW]: [on(EventType.HOOK_COMMAND_NEW, sumCommand)],
  [EventType.HOOK_COMMAND_RESET]: [
    on(EventType.HOOK_COMMAND_RESET, sumCommand),
  ],
  [EventType.HOOK_COMMAND_STOP]: [on(EventType.HOOK_COMMAND_STOP, sumCommand)],
  [EventType.HOOK_COMPACTION_BEFORE]: [
    on(EventType.HOOK_COMPACTION_BEFORE, sumCompaction),
  ],
  [EventType.HOOK_COMPACTION_AFTER]: [
    on(EventType.HOOK_COMPACTION_AFTER, sumCompaction),
  ],
};

/** Event types this bridge actively forwards. */
export const CHRONO_BRIDGE_FORWARDED: readonly EventType[] = Object.keys(
  chronoRuntimeEvents,
) as EventType[];
