# `computeruse` Upgrade Plan — Closing the Gap with OpenClaw

> Companion to `OPENCLAW_COMPARISON.md`. Issues ranked by **risk-reduction × ElizaOS leverage**.
> Each issue lists: problem, OpenClaw reference design, proposed change, touch points, acceptance.

---

## Issue #1 — Execution Safety & Approval Gates

### Problem
`run_command` (server.rs) and `execute_browser_script` execute arbitrary shell/JS with **no allowlist, no approval, no audit binding**. An ElizaOS agent given the MCP server has unrestricted host access. This is the single largest blocker to shipping `computeruse` inside autonomous agents like AVB.

### OpenClaw reference
`src/infra/exec-approvals.ts` — `ExecSecurity = deny | allowlist | full`, `ExecAsk = off | on-miss | always`, durable `SystemRunApprovalPlan` with argv/cwd/SHA-256 file snapshots.

### Proposed change
1. New crate module `crates/computeruse-mcp-agent/src/exec_policy.rs`:
   - `ExecPolicy { mode: Deny|Allowlist|Full, ask: Off|OnMiss|Always, patterns: Vec<GlobPattern> }`
   - Loaded from `~/.computeruse/policy.toml` + env `COMPUTERUSE_EXEC_MODE` (see `.env.example`).
   - `evaluate(argv, cwd) -> Decision { Allow, Deny, AskUser(ApprovalRequest) }`.
2. Wire into `dispatch_tool()` for `run_command`, `execute_browser_script`, `open_application`, `write_file`, `edit_file`.
3. Surface `AskUser` via existing MCP elicitation (`src/elicitation/`) — already scaffolded, currently feature-gated.
4. Persist approvals to `~/.computeruse/approvals.json` keyed by `(agent_id, argv_hash)`.
5. Wrap all externally-sourced text (OCR, browser DOM, `get_window_tree` content) in a tagged envelope before returning to the LLM (port of OpenClaw `wrapExternalContent`).

### Touch points
`crates/computeruse-mcp-agent/src/{server.rs, elicitation/, helpers.rs}`, new `exec_policy.rs`; `.env.example` (+`COMPUTERUSE_EXEC_MODE`, `COMPUTERUSE_POLICY_PATH`).

### Acceptance
- Default mode = `allowlist` with empty patterns ⇒ all `run_command` denied unless approved.
- Approval round-trips through MCP elicitation and is cached for the session.
- Unit tests in `src/tests/` cover deny/allow/ask paths.

---

## Issue #2 — Idempotency, Retry & Resumable Workflows

### Problem
MCP tool calls are fire-and-forget. A network blip or agent restart mid-`execute_sequence` leaves the desktop in an unknown state with no way to resume. `CancellationToken` exists but there is no dedupe or replay.

### OpenClaw reference
Gateway requires `idempotencyKey` on side-effecting methods; short-lived response cache returns the prior result on retry. Lobster issues `resumeToken` when a pipeline pauses on approval.

### Proposed change
1. Extend MCP tool input schemas with optional `idempotency_key: String` (server.rs `mcp_types.rs`).
2. Add `IdempotencyCache` (LRU, TTL 5 min) in `server.rs` keyed by `(tool_name, key)` → `CallToolResult`. On hit, return cached result without re-execution.
3. `execute_sequence`: emit a `sequence_id` + per-step `checkpoint` into the existing `execution_logger.rs`; add `resume_sequence(sequence_id, from_step)` tool that replays from the last checkpoint using the logged step outputs.
4. `@mediar-ai/workflow`: add `resumeToken` to step result type; CLI `computeruse run --resume <token>`.

### Touch points
`crates/computeruse-mcp-agent/src/{server.rs, mcp_types.rs, execution_logger.rs, server_sequence.rs}`; `crates/computeruse-cli/src/workflow_result.rs`; `packages/workflow/src/`.

### Acceptance
- Calling `click_element` twice with the same `idempotency_key` performs one click, returns identical results.
- Killing the MCP process mid-sequence and re-running with `--resume` continues from the last completed step.

---

## Issue #3 — Native ElizaOS Plugin Adapter (replace MCP-only surface)

### Problem
ElizaOS agents (incl. `@elizaos/plugin-avb`) cannot call `Desktop` in-process; they must spawn the MCP server and speak JSON-RPC. There is no `Plugin { actions, providers, services }` export, so `computeruse` is invisible to the runtime's action planner. OpenClaw's `api.registerTool` maps directly onto ElizaOS `Action`.

### Proposed change
New package `packages/computeruse/packages/plugin-computeruse/` (`@elizaos/plugin-computeruse`):
- `ComputerUseService extends Service` — owns a singleton `Desktop` (from `@elizaos/computeruse` NAPI), lifecycle-managed by `AgentRuntime`.
- **Actions** (thin wrappers over NAPI, schema via Zod, mirroring the 35 MCP tools but grouped):
  `SCREENSHOT`, `CLICK_ELEMENT`, `TYPE_TEXT`, `GET_WINDOW_TREE`, `OPEN_APPLICATION`, `RUN_COMMAND` (gated by Issue #1 policy), `EXECUTE_BROWSER_SCRIPT`, `RECORD_WORKFLOW`.
- **Providers**: `desktopState` (active window, focused element, monitor layout) for prompt context.
- **Evaluator**: `desktopActionVerifier` — post-action, re-query the tree to confirm the intended state change (closes the loop the way OpenClaw's snapshot-after-act does).

### Touch points
New `packages/computeruse/packages/plugin-computeruse/{package.json, src/index.ts, src/service.ts, src/actions/*.ts, src/providers/*.ts}`; register in root `turbo.json` workspace; depend on `@elizaos/core` + `@elizaos/computeruse`.

### Acceptance
- `elizaos start` with `plugins: ["@elizaos/plugin-computeruse"]` exposes `CLICK_ELEMENT` etc. in the agent's action list.
- AVB pipeline can call `SCREENSHOT` → feed to ScryptedAI image model without spawning a subprocess.

---

## Issue #4 — Cross-Platform Parity (macOS/Linux health + recorder)

### Problem
`health.rs:176-199` returns hard-coded "healthy" on macOS/Linux — agents cannot detect missing Accessibility/AT-SPI permissions and fail opaquely. Workflow recorder is Windows-only. Threading-safety caveat at `lib.rs:97` is undocumented.

### OpenClaw reference
macOS node performs TCC permission checks and surfaces `device_permissions` via `node.describe`; degrades gracefully with `CANVAS_DISABLED`-style codes.

### Proposed change
1. **macOS health**: implement `AXIsProcessTrustedWithOptions` check in `crates/computeruse/src/platforms/macos/health.rs`; return `PermissionDenied` with remediation hint.
2. **Linux health**: probe AT-SPI bus via `zbus` ping + check `wmctrl`/`xdotool` on PATH.
3. **Recorder parity (macOS first)**: new `crates/computeruse-workflow-recorder/src/macos.rs` using `CGEventTap` for mouse/keyboard + AX for element-under-cursor. Linux deferred (evdev + AT-SPI) — tracked separately.
4. Expose health as MCP tool `get_health` and as `Provider` in the new plugin (Issue #3) so the LLM sees "Accessibility permission missing — ask user to grant in System Settings" instead of a stack trace.
5. Resolve `lib.rs:97` by wrapping Windows COM init in `CoInitializeEx(COINIT_MULTITHREADED)` guard + document thread model.

### Touch points
`crates/computeruse/src/health.rs`, `crates/computeruse/src/platforms/{macos,linux}/`, `crates/computeruse-workflow-recorder/src/`, `crates/computeruse-mcp-agent/src/server.rs` (+`get_health` tool).

### Acceptance
- On macOS without Accessibility permission, `Desktop::new()` returns `Err(PermissionDenied)` with actionable message; `get_health` reports it.
- macOS recorder produces the same JSON schema as Windows recorder for a 5-click Calculator workflow.

---

## Issue #5 — Agent-Rendered Surface (Canvas-lite) for AVB

### Problem
AVB generates avatars/images/video via ScryptedAI but `computeruse` has no way to **present** them — `highlight_element` is the only visual output. OpenClaw's Canvas/A2UI gives agents a controllable WebView to push generated media, status, and approval prompts.

### Proposed change (minimal, ElizaOS-native — not a full A2UI port)
1. New `crates/computeruse-mcp-agent/src/canvas.rs`: spin up a tiny `axum` HTTP server (reuse port range, default `:17374`) serving a single-page WebView host from `~/.computeruse/canvas/`.
2. MCP tools: `canvas_present { html | url | image_path }`, `canvas_eval { js }`, `canvas_snapshot`, `canvas_hide`.
3. On `canvas_present`, open the URL via existing `open_url()` (system browser) or, on macOS, a `WKWebView` panel via a small Swift helper (stretch goal).
4. Live-reload: WebSocket on the same server pushes `reload` on file change (mirrors OpenClaw `a2ui.ts:81-110`).
5. Plugin (Issue #3) exposes `CANVAS_PRESENT` action so AVB can do: generate avatar → `CANVAS_PRESENT(image_path)` → user sees it.

### Touch points
`crates/computeruse-mcp-agent/src/{canvas.rs, server.rs}`; `crates/computeruse-mcp-agent/Cargo.toml` (+`axum`, `tokio-tungstenite` already present); `packages/computeruse/packages/plugin-computeruse/src/actions/canvas.ts`.

### Acceptance
- `canvas_present { image_path: "./avatar.png" }` opens a local page showing the image within 500 ms.
- `canvas_snapshot` returns a PNG of the rendered canvas.

---

## Sequencing & Dependencies

```
#1 Exec Safety ──┐
                 ├──► #3 ElizaOS Plugin ──► #5 Canvas-lite
#2 Idempotency ──┘
#4 Platform Parity (parallel, independent)
```

- **#1** and **#2** are prerequisites for **#3** (plugin must not expose unsafe/non-idempotent actions to autonomous agents).
- **#5** depends on **#3** for the ElizaOS action surface.
- **#4** is orthogonal; can ship independently.

## Out of Scope (tracked, not in top-5)

- ClawHub-style skill registry (ElizaOS already has its own plugin registry).
- Multi-device Gateway (large; would duplicate ElizaOS `@elizaos/daemon`).
- iOS/Android nodes.

## Environment Note

Full `bun run build` currently fails on hosts without the Rust toolchain (`@elizaos/rust`, `@elizaos/computeruse`, `@elizaos/sweagent-root` require `cargo`). All TS packages build clean (11/11 via `turbo run build --filter='!@elizaos/rust' --filter='!@elizaos/computeruse*' --filter='!@elizaos/sweagent-root'`). Implementing #1/#2/#4 requires `cargo`; #3/#5 (TS plugin + axum already a dep) can be developed against published `@elizaos/computeruse` binaries.
