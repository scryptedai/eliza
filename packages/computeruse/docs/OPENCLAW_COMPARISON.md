# ElizaOS `computeruse` vs OpenClaw — Capability Comparison

> Research snapshot: 2026-04-02
> OpenClaw ref: `github.com/openclaw/openclaw` @ HEAD (shallow clone, /tmp/openclaw-research)
> computeruse ref: `packages/computeruse` @ `3852c14e` (develop)

---

## 1. Executive Summary

`computeruse` and OpenClaw solve **adjacent but different** problems:

| | **computeruse** | **OpenClaw** |
|---|---|---|
| **Category** | Desktop accessibility automation SDK | Personal AI-assistant framework |
| **Primary surface** | OS accessibility tree (UIA / AX / AT-SPI) | Gateway WebSocket + multi-channel inbox |
| **Execution model** | Deterministic, local, single-host | Distributed Gateway ↔ Node, approval-gated |
| **Language** | Rust core + NAPI/PyO3 bindings | TypeScript core + native Swift/Kotlin nodes |
| **AI role** | "AI for recovery only" (>95% deterministic) | AI-first agent loop with tool use |

**Verdict:** `computeruse` is **stronger at low-level desktop control** (rich selectors, accessibility-tree actions, workflow recording). OpenClaw is **stronger at orchestration, safety, and multi-device reach**. The "underpowered" perception comes from `computeruse` lacking OpenClaw's *operational layer* — approvals, idempotency, remote nodes, visual workspace, skill registry — not from its actuation primitives, which are actually more capable.

---

## 2. Architecture Side-by-Side

### 2.1 Control Plane

| Aspect | computeruse | OpenClaw |
|---|---|---|
| Transport | MCP over stdio (`computeruse-mcp-agent`) | WebSocket Gateway `ws://127.0.0.1:18789` |
| Protocol typing | `rmcp` JSON-RPC | TypeBox JSON-Schema (`src/gateway/protocol/schema/`) |
| Multi-client | ❌ single MCP client | ✅ operators + nodes + UI on one Gateway |
| Remote devices | ❌ local only | ✅ Node pairing (macOS/iOS/Android) over `node.invoke` |
| Idempotency | ❌ none | ✅ required `idempotencyKey` on side-effecting calls |
| Event stream | ❌ request/response only | ✅ `{type:"event", seq, stateVersion}` push |

**Refs:** `crates/computeruse-mcp-agent/src/server.rs:9770`; OpenClaw `src/gateway/protocol/schema/nodes.ts:66-75`, `docs/concepts/architecture.md:59-102`

### 2.2 Actuation (UI / OS control)

| Capability | computeruse | OpenClaw |
|---|---|---|
| Accessibility-tree selectors | ✅ 23+ types incl. spatial (`RightOf`, `Near`), logical (`And`/`Or`/`Not`), chain | ❌ none (browser DOM only) |
| Native click/type/drag/scroll | ✅ full (`element.rs:913-1479`) | ⚠️ via `system.run` shell, no first-class API |
| Window/app management | ✅ launch, focus, min/max, foreground restore | ⚠️ macOS-only via node commands |
| Browser automation | ✅ Chrome extension bridge (WS :17373) + `execute_browser_script` | ✅ dedicated Chromium, snapshot/act/tabs/console, profile mgmt |
| Screenshot / OCR | ✅ multi-monitor, element capture, `uni-ocr`, Omniparser, Gemini Vision | ✅ `canvas.snapshot`, `screen_record` |
| Workflow recording | ✅ Windows-only (`computeruse-workflow-recorder`) | ❌ none |
| Camera / mic / GPS | ❌ | ✅ `camera_snap`, `camera_clip`, `location_get` |
| Notifications | ❌ | ✅ `system.notify`, `notifications_list/action` |

**Refs:** `crates/computeruse/src/selector.rs:4-56`; OpenClaw `extensions/browser/src/browser-tool.ts`, `src/agents/tools/nodes-tool-media.ts`

### 2.3 Safety & Approval

| Aspect | computeruse | OpenClaw |
|---|---|---|
| Command exec | `run_command` — **unrestricted** shell/Python/JS/Node | `system.run` — graduated `deny / allowlist / full` |
| Per-agent allowlists | ❌ | ✅ pattern-based, persisted |
| Human approval gate | ❌ (only `ask_user` behind feature flag) | ✅ `ExecApprovals` store + `on-miss / always` ask modes |
| Tamper detection | ❌ | ✅ SHA-256 snapshot of mutable file operands |
| Untrusted-content wrapping | ❌ | ✅ `wrapExternalContent()` before LLM ingestion |
| Output caps | ⚠️ ad-hoc | ✅ `OUTPUT_CAP` ≈ 200 KB |
| Audit trail | ⚠️ Sentry/OTel telemetry only | ✅ session-bound, agentId, turnSource logged |

**Refs:** `crates/computeruse-mcp-agent/src/server.rs` (`run_command`); OpenClaw `src/infra/exec-approvals.ts:12-143`, `src/node-host/invoke-system-run.ts`

### 2.4 Workflows

| Aspect | computeruse | OpenClaw (Lobster) |
|---|---|---|
| Definition format | YAML / TypeScript (`@mediar-ai/workflow`, Zod) | YAML `.lobster` files, typed pipelines |
| Step data flow | `execute_sequence` tool + KV store (Redis) | `$step.stdout` references, JSON envelope |
| Approval gates | ❌ | ✅ `approval: required` per step |
| Resumability | ❌ (cancellation only) | ✅ `resumeToken` after pause |
| Conditional steps | ⚠️ via `expression_eval.rs` | ✅ `condition: $approve.approved` |
| Scheduling | ✅ `tokio-cron-scheduler` in CLI | ✅ cron tools |

**Refs:** `crates/computeruse-mcp-agent/src/expression_eval.rs`, `packages/workflow/`; OpenClaw `docs/tools/lobster.md`

### 2.5 Visual / Agent-to-User Surface

| | computeruse | OpenClaw |
|---|---|---|
| Agent-rendered UI | ❌ (only `highlight_element` overlay) | ✅ Canvas + A2UI v0.8 (`canvas.present/navigate/eval/snapshot`) |
| Live reload | ❌ | ✅ WS-injected reload |
| Native host | n/a | WKWebView (macOS/iOS), WebView (Android) |

**Refs:** OpenClaw `src/canvas-host/a2ui.ts`, `docs/platforms/mac/canvas.md`

### 2.6 Extensibility / Distribution

| | computeruse | OpenClaw |
|---|---|---|
| Plugin model | none — fixed 35-tool MCP surface | `api.registerTool/registerCli/registerGatewayMethod` |
| Skill registry | none | ClawHub (`clawhub.ai`, ~13.7k skills) |
| Install UX | `npx -y computeruse-mcp-agent` | `openclaw skills install <slug>` |

### 2.7 Platform Parity

| | computeruse | OpenClaw |
|---|---|---|
| Windows | ✅ full (best-supported) | ⚠️ via WSL2 |
| macOS | ✅ full actuation; ⚠️ health checks stubbed (`health.rs:176-199`) | ✅ first-class native app |
| Linux | ✅ AT-SPI; ⚠️ experimental, health stubbed | ✅ Gateway/CLI |
| iOS / Android | ❌ | ✅ native nodes |

---

## 3. Where computeruse Wins

1. **Accessibility-tree actuation** — no equivalent in OpenClaw; spatial/relational selectors are state-of-the-art.
2. **Deterministic replay** — recorded workflows run at CPU speed without an LLM in the loop.
3. **Vision stack** — OCR + Omniparser + Gemini Vision fallback already wired into `get_window_tree`.
4. **Native performance** — Rust core, LTO `fat`, single binary distribution.
5. **Workflow recorder** — captures human demonstrations → JSON (Windows).

## 4. Where OpenClaw Wins (the "underpowered" gaps)

1. **Safety/approval model** — `computeruse` will run arbitrary shell with zero gating; OpenClaw has graduated trust + durable approval store + tamper detection.
2. **Idempotency & recovery** — `computeruse` has no retry semantics; a dropped MCP connection mid-`click_element` is unrecoverable. OpenClaw dedupes by `idempotencyKey` and supports `resumeToken`.
3. **Multi-device orchestration** — `computeruse` is single-host. OpenClaw's Gateway lets one agent drive a Mac, an iPhone camera, and a remote Linux box in the same session.
4. **Agent-rendered UI (Canvas/A2UI)** — `computeruse` can *read* UIs but can't *render* one back to the user; AVB-style agents have no surface to show the avatar/state they generate.
5. **Extensibility & discovery** — `computeruse` tools are compiled-in; no runtime registration, no skill registry. OpenClaw has 94 extensions + ClawHub.
6. **Cross-platform parity** — workflow recording is Windows-only; macOS/Linux health checks are stubs; threading-safety comment in `lib.rs:97` is unresolved.
7. **Untrusted-content hygiene** — browser DOM / OCR text is fed to the LLM unwrapped; OpenClaw marks it explicitly.

---

## 5. Integration Note (ElizaOS context)

`@elizaos/computeruse` is consumed as a NAPI library by the ElizaOS runtime, but there is **no `Plugin` adapter** exposing it as ElizaOS actions/providers/services. The MCP surface (35 tools) is the only agent-facing API today, which means ElizaOS agents must shell out to the MCP server rather than call `Desktop` in-process. OpenClaw's `api.registerTool` pattern maps almost 1:1 onto ElizaOS's `Plugin.actions` — adopting it would let `computeruse` capabilities appear natively in the AVB pipeline alongside ScryptedAI.

---

## 6. Source Index

| Topic | computeruse | OpenClaw |
|---|---|---|
| Selectors | `crates/computeruse/src/selector.rs:4-56` | — |
| UI actions | `crates/computeruse/src/element.rs:913-1479` | `extensions/browser/src/browser-tool.actions.ts` |
| MCP tools | `crates/computeruse-mcp-agent/src/server.rs:9879-10258` | `src/agents/tools/nodes-tool.ts:87-128` |
| Exec safety | — | `src/infra/exec-approvals.ts:12-143` |
| Gateway | — | `docs/concepts/architecture.md`, `src/gateway/protocol/schema/` |
| Workflows | `packages/workflow/`, `expression_eval.rs` | `docs/tools/lobster.md` |
| Canvas | — | `src/canvas-host/a2ui.ts`, `docs/platforms/mac/canvas.md` |
| Health gaps | `crates/computeruse/src/health.rs:176-199` | — |
