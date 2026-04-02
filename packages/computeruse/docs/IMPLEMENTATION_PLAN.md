# Plan: `computeruse` → OpenClaw Parity Upgrade

## Context

Research deliverables already written (pre-plan):
- `packages/computeruse/docs/OPENCLAW_COMPARISON.md` — side-by-side capability matrix
- `packages/computeruse/docs/UPGRADE_PLAN.md` — Top-5 ranked issues with refs, touch points, acceptance criteria

**Finding:** `computeruse` has stronger low-level actuation (accessibility-tree selectors, deterministic replay) but lacks OpenClaw's operational layer: exec safety gates, idempotency, native ElizaOS plugin surface, cross-platform health, and an agent-rendered canvas.

**Blocker found during pre-work:** Rust toolchain (`cargo`) is not installed on this host. `@elizaos/rust`, `@elizaos/computeruse`, `@elizaos/sweagent-root` cannot build. User has authorized global install.

---

## Step 0 — Environment: Install Rust Toolchain (PREREQUISITE)

**User-authorized.** Required for Issues #1, #2, #4 and for a clean full `bun run build`.

1. Install rustup (official installer, default stable toolchain):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile default
   ```
2. Make available in **this session** (shell state doesn't persist between Bash calls, so prefix subsequent cargo-dependent commands):
   ```bash
   export PATH="$HOME/.cargo/bin:$PATH" && cargo --version && rustc --version
   ```
   All later `bun run build` / `cargo` invocations will be run as:
   ```bash
   PATH="$HOME/.cargo/bin:$PATH" bun run build
   ```
3. Add components needed by the workspace:
   ```bash
   PATH="$HOME/.cargo/bin:$PATH" rustup component add rustfmt clippy
   ```
4. Verify clean full build:
   ```bash
   PATH="$HOME/.cargo/bin:$PATH" bun run build 2>&1 | tail -150
   ```
5. If `@elizaos/computeruse` NAPI build needs additional targets (e.g. `napi-rs` CLI), install on demand — do not preemptively add cross-compile targets.

**Acceptance:** `bun run build` exits 0 for all 25 packages (or fails only on genuinely broken code, not missing tooling).

---

## Scope

**All five issues are in-scope deliverables.** Sequenced by dependency graph; success = all five implemented, building, and tested.

```
Step 0  Rust toolchain ────────────────────────────────► clean `bun run build`
Step 1  #1 Exec Safety (Rust) ──┐
Step 2  #2 Idempotency (Rust) ──┼─► Step 4  #3 ElizaOS Plugin (TS) ─► Step 6  #5 Canvas-lite
Step 3  #4 Platform Health ─────┘                                      (Rust + TS)
Step 5  Integration tests across #1–#4
Step 7  Full verification (build + test + lint + typecheck, all 25 pkgs)
```

---

## Step 1 — Issue #1: Exec Safety (Rust)

Files:
- **New** `crates/computeruse-mcp-agent/src/exec_policy.rs` — `ExecPolicy`, `Decision`, `evaluate()`, TOML loader
- **Edit** `crates/computeruse-mcp-agent/src/server.rs` — gate `run_command`, `execute_browser_script`, `write_file`, `edit_file`, `open_application` through `ExecPolicy::evaluate`; on `Decision::AskUser` route to existing `elicitation::request_approval`
- **Edit** `crates/computeruse-mcp-agent/src/elicitation/schemas.rs` — add `ExecApprovalRequest` schema
- **Edit** `crates/computeruse-mcp-agent/src/lib.rs` — `pub mod exec_policy;`
- **Edit** `crates/computeruse-mcp-agent/src/helpers.rs` — add `wrap_external_content(text, source) -> String` and apply in `get_window_tree` / `execute_browser_script` return paths
- **Edit** root `.env.example` — add `COMPUTERUSE_EXEC_MODE=allowlist`, `COMPUTERUSE_POLICY_PATH=~/.computeruse/policy.toml`
- **New** `crates/computeruse-mcp-agent/src/tests/test_exec_policy.rs`

Reuse: `src/elicitation/helpers.rs` (already has request/response plumbing), `glob` crate already in deps for pattern matching.

---

## Step 2 — Issue #2: Idempotency & Resumable Sequences (Rust)

Files:
- **Edit** `crates/computeruse-mcp-agent/src/mcp_types.rs` — add `idempotency_key: Option<String>` to tool-arg base
- **New** `crates/computeruse-mcp-agent/src/idempotency.rs` — `IdempotencyCache` (LRU via `lru` crate, TTL 5 min) keyed `(tool_name, key)` → `CallToolResult`
- **Edit** `crates/computeruse-mcp-agent/src/server.rs` — check cache at top of `dispatch_tool()`; store result before return
- **Edit** `crates/computeruse-mcp-agent/src/server_sequence.rs` — emit `sequence_id` + per-step checkpoint via `execution_logger`; add `resume_sequence` tool
- **Edit** `crates/computeruse-mcp-agent/src/execution_logger.rs` — add `read_checkpoints(sequence_id)`
- **Edit** `crates/computeruse-cli/src/workflow_result.rs` + `src/main.rs` — `--resume <sequence_id>` flag
- **Edit** `packages/workflow/src/` — surface `resumeToken` on step result type
- **New** `crates/computeruse-mcp-agent/src/tests/test_idempotency.rs`

Reuse: `execution_logger.rs` already persists JSONL per run; `lru` crate to be added to `Cargo.toml`.

---

## Step 3 — Issue #4: Cross-Platform Health & macOS Recorder (Rust)

Files:
- **Edit** `crates/computeruse/src/health.rs` — replace macOS/Linux stubs
- **New** `crates/computeruse/src/platforms/macos/health.rs` — `AXIsProcessTrustedWithOptions` via `accessibility-sys`
- **New** `crates/computeruse/src/platforms/linux/health.rs` — `zbus` ping to `org.a11y.Bus` + `which::which("wmctrl")`
- **Edit** `crates/computeruse-mcp-agent/src/server.rs` — register `get_health` tool returning `HealthReport { platform, accessibility: Ok|MissingPermission|MissingDependency, details }`
- **New** `crates/computeruse-workflow-recorder/src/macos.rs` — `CGEventTap` (mouse/keyboard) + AX element-under-cursor; emit identical JSON schema to `windows.rs`
- **Edit** `crates/computeruse-workflow-recorder/src/lib.rs` — `#[cfg(target_os="macos")]` dispatch
- **Edit** `crates/computeruse/src/lib.rs:97` — wrap Windows engine init in explicit `CoInitializeEx` guard; remove uncertainty comment
- **New** `crates/computeruse/src/tests/test_health.rs` (mocked permission states)

---

## Step 4 — Issue #3: `@elizaos/plugin-computeruse` (TypeScript)

Files (all **new** under `packages/computeruse/packages/plugin-computeruse/`):
- `package.json` — name `@elizaos/plugin-computeruse`, deps `@elizaos/core` + `@elizaos/computeruse` (workspace), peerDep on Node 23.3+
- `tsconfig.json`, `tsup.config.ts` (mirror `plugins/plugin-avb` build setup)
- `src/index.ts` — `export const computerUsePlugin: Plugin = { name, description, services, actions, providers, evaluators }`
- `src/service.ts` — `class ComputerUseService extends Service` wrapping `Desktop` from `@elizaos/computeruse`; `static serviceType = 'computeruse'`; `initialize()` runs health check, `stop()` releases handles
- `src/actions/` — `screenshot.ts`, `clickElement.ts`, `typeText.ts`, `getWindowTree.ts`, `openApplication.ts`, `runCommand.ts` (delegates to service; `runCommand` checks `COMPUTERUSE_EXEC_MODE` and refuses if policy would deny — TS-side mirror of #1 until MCP path is used)
- `src/providers/desktopState.ts` — returns `{ activeWindow, focusedElement, monitors }` for prompt context
- `src/evaluators/verifyAction.ts` — post-action re-query to confirm state change
- `src/__tests__/plugin.test.ts` — vitest, mock `Desktop`

Edits:
- Root `package.json` workspaces already glob `packages/computeruse/packages/*` — verify, else add
- `turbo.json` — no change needed (inherits `build`/`test` pipeline)

Reference patterns: `plugins/plugin-avb/src/index.ts`, `plugins/plugin-scryptedai/src/service.ts`, `packages/typescript/src/types/plugin.ts` for `Plugin`/`Action`/`Service` interfaces.

---

Additionally exposes from #1/#2/#4:
- `src/providers/health.ts` — wraps `get_health` so the LLM sees permission state in context
- All actions accept optional `idempotencyKey` and forward to the Rust layer

---

## Step 5 — Issue #5: Canvas-lite (Rust + TS)

Files:
- **New** `crates/computeruse-mcp-agent/src/canvas.rs` — `axum` server on `:17374` serving `~/.computeruse/canvas/`; WS endpoint `/__canvas_ws` for live-reload push
- **Edit** `crates/computeruse-mcp-agent/src/server.rs` — register tools `canvas_present { html?|url?|image_path? }`, `canvas_eval { js }`, `canvas_snapshot`, `canvas_hide`
- **Edit** `crates/computeruse-mcp-agent/Cargo.toml` — add `axum` (`tokio-tungstenite` already present)
- **Edit** `crates/computeruse-mcp-agent/src/lib.rs` — `pub mod canvas;`
- **New** `packages/computeruse/packages/plugin-computeruse/src/actions/canvas.ts` — `CANVAS_PRESENT`, `CANVAS_SNAPSHOT` actions calling MCP or NAPI bridge
- **New** `crates/computeruse-mcp-agent/assets/canvas-host.html` — minimal host page (img/iframe slot + WS reload listener)

Reuse: `crates/computeruse/src/lib.rs::open_url()` to launch system browser at canvas URL.

---

## Step 6 — Integration Tests

- **New** `crates/computeruse-mcp-agent/tests/integration_policy_idempotency.rs` — spin MCP server in-proc, assert: denied `run_command` without policy; duplicate `idempotency_key` returns cached result; `get_health` returns structured report
- **New** `packages/computeruse/packages/plugin-computeruse/src/__tests__/integration.test.ts` — mock `Desktop`, assert plugin registers all actions/providers, `RUN_COMMAND` respects `COMPUTERUSE_EXEC_MODE=deny`
- **New** `crates/computeruse-mcp-agent/tests/integration_canvas.rs` — start canvas server, `canvas_present` an inline HTML, HTTP GET it back

---

## Step 7 — Final Verification (Definition of Done)

```bash
export PATH="$HOME/.cargo/bin:$PATH"

# 1. Full monorepo build — all 25 packages
bun run build

# 2. Rust tests (mcp-agent + core + recorder)
cargo test -p computeruse-mcp-agent
cargo test -p computeruse-rs
cargo test -p computeruse-workflow-recorder

# 3. TS tests
bunx turbo run test --filter=@elizaos/plugin-computeruse
bunx turbo run test --filter=@mediar-ai/workflow

# 4. Lint / format / typecheck
bun run lint && bun run typecheck
cargo fmt --all -- --check && cargo clippy --workspace -- -D warnings

# 5. Smoke: load plugin in default agent (manual)
#    add "@elizaos/plugin-computeruse" to character.plugins[], `elizaos start`,
#    confirm CLICK_ELEMENT / CANVAS_PRESENT / health provider appear
```

**Success criteria:** all of the above exit 0; `OPENCLAW_COMPARISON.md` gaps #1–#5 each have a passing test demonstrating closure.

---

## Out of Scope (genuinely external)

- OpenClaw Gateway / multi-device node pairing — overlaps `@elizaos/daemon`, separate RFC.
- ClawHub-style skill registry — ElizaOS has its own plugin registry.
- iOS/Android native nodes.
- Linux workflow recorder (evdev) — tracked as follow-up after macOS recorder lands.

---

## Risks

- NAPI binary for `@elizaos/computeruse` may need rebuild for darwin-arm64 after cargo install; if `napi build` fails, fall back to published npm binary for plugin development.
- `server.rs` is ~10k LOC; exec-policy gating must be surgical — insert at `dispatch_tool()` entry (line ~9770), not per-tool.
- macOS Accessibility permission required to actually exercise `Desktop` at runtime — tests must mock.
