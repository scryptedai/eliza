# ComputerUse vs OpenClaw — Comparative Analysis

**Status:** Draft for review
**Date:** 2026-04-02
**Scope:** ElizaOS `packages/computeruse` (v2.0.0) vs OpenClaw (`github.com/openclaw/openclaw`, HEAD@2026-04-02)
**Purpose:** Identify gaps in computeruse that block ElizaOS agents from achieving OpenClaw-level automation reliability.

---

## TL;DR

ComputerUse and OpenClaw occupy **opposite quadrants** of the automation space:

|                       | Native Desktop GUI | Browser / Web      |
|-----------------------|--------------------|--------------------|
| **ComputerUse**       | Strong (Win only)  | Weak (fragile ext) |
| **OpenClaw**          | None (by design)   | Excellent (PW+CDP) |

The "underpowered" perception is real **where they overlap** (browser automation, agent-facing snapshots, cross-platform). ComputerUse's core advantage — native UIAutomation on Windows — is a moat OpenClaw doesn't even attempt to enter. But on macOS/Linux that moat is empty: **76 stubbed methods on macOS, 88 on Linux** return `UnsupportedOperation`, leaving agents with screenshot-and-pray fallbacks via remote vision APIs.

The Top-5 upgrades (detailed in the companion plan) target the overlap zone:
1. Replace the browser extension bridge with Playwright/CDP (steal OpenClaw's stack)
2. Ship agent-facing accessibility snapshots with stable refs (steal OpenClaw's `aria-ref` model)
3. Implement macOS AX tree traversal (un-stub `children()`, `parent()`, `set_value()`)
4. Bound the vision-cache memory leak in the MCP server
5. Add a `BrowserBackend` trait so agents can choose extension/CDP/remote-CDP per profile

---

## 1. System Overview

### 1.1 ComputerUse (`packages/computeruse`)

**Stack:** Rust workspace (5 crates) + napi-rs Node bindings + PyO3 stubs.
**Delivery:** MCP server (`computeruse-mcp-agent`), CLI, TS SDK (`@elizaos/computeruse`).
**Philosophy:** Pre-trained deterministic workflows; AI invoked only on recovery. No cursor hijacking — uses accessibility APIs to act in background.

**Crate map:**
```
crates/
├── computeruse/                  # core: Selector, Locator, AccessibilityEngine trait
│   └── src/platforms/{windows,macos,linux}/
├── computeruse-mcp-agent/        # MCP server, 15 tools, vision integration
├── computeruse-workflow-recorder/# OS event hooks → YAML
├── computeruse-cli/
└── computeruse-computer-use/     # Gemini Computer Use bridge types
packages/
├── computeruse-ts/               # napi-rs bindings (cdylib)
├── workflow/                     # TS WorkflowBuilder
└── kv/
```

### 1.2 OpenClaw (`/tmp/openclaw-research/openclaw`)

**Stack:** TypeScript (pnpm workspace, 11,308 files) + Swift (Swabble voice daemon, macOS-only).
**Delivery:** Local gateway daemon (`openclaw.mjs`), CLI, plugin SDK, **91 bundled extensions**.
**Philosophy:** Local-first AI hub. Channels in (Discord/Slack/iMessage/voice) → gateway → tools (browser, shell, coding agents) → channels out. Browser is the "computer use" surface; native GUI deliberately out of scope.

**Layout:**
```
src/                  # gateway, plugin-sdk, agents/, channels/, flows/
extensions/           # 91 packages: browser, anthropic, openai, discord, …
  └── browser/        # 276 TS files — Playwright + CDP browser tool
skills/               # ~55 SKILL.md files (coding-agent, clawflow, …)
Swabble/              # Swift 6 — macOS Speech.framework wake-word daemon
Dockerfile.sandbox-browser  # Chromium + Xvfb + noVNC headless container
```

---

## 2. Capability Matrix

| Capability                        | ComputerUse | OpenClaw | Winner | Notes |
|-----------------------------------|:-----------:|:--------:|:------:|-------|
| **Native desktop element finding (Win)** | ✅ Full    | ❌       | CU     | CU: UIAutomation deep tree, 12,000 LOC |
| **Native desktop element finding (mac)** | 🟡 Partial | ❌       | CU*    | CU: AX read-only, **76 stubbed methods** |
| **Native desktop element finding (Linux)** | 🟡 Partial | ❌       | CU*    | CU: AT-SPI window-level, **88 stubbed methods** |
| **Native click/type/scroll**      | ✅ Win / 🟡  | ❌       | CU     | OC has none — by design |
| **Window minimize/maximize**      | ✅ Win only  | ❌       | CU     | CU stubbed on mac/Linux |
| **Browser navigate/click/type**   | 🟡          | ✅✅      | **OC** | CU: custom WS ext; OC: Playwright + CDP |
| **Browser accessibility snapshot** | ❌          | ✅✅      | **OC** | OC: aria-ref tree, role refs, `--efficient` |
| **Browser multi-profile**         | ❌          | ✅       | **OC** | OC: isolated/user/remote profiles |
| **Browser remote/hosted**         | ❌          | ✅       | **OC** | OC: Browserless, Browserbase, custom CDP |
| **Browser PDF/download/upload**   | ❌          | ✅       | **OC** | OC: Playwright handles all three |
| **Browser cookies/storage**       | ❌          | ✅       | **OC** | OC: CDP `Storage.*` |
| **Browser tab management**        | 🟡          | ✅       | **OC** | CU: close-only; OC: full lifecycle |
| **Selector expressiveness**       | ✅✅         | ✅       | CU     | CU: spatial+boolean (`RightOf && role:Button`) |
| **Workflow recording**            | ✅ Win / 🟡  | ❌       | CU     | CU: rdev hooks → YAML → TS replay |
| **Workflow replay**               | ✅          | 🟡       | CU     | OC: skills are static prompts, not recorded |
| **Vision: OCR**                   | 🟡 Win only  | ❌       | CU     | CU: uni-ocr (Windows OCR API) |
| **Vision: VLM element detection** | 🟡 Remote   | 🟡       | tie    | CU: Omniparser/Gemini (network); OC: model-passthrough |
| **Vision: a11y-tree-as-text**     | 🟡          | ✅✅      | **OC** | OC's snapshot IS the agent-facing view |
| **Sandbox / isolation**           | ❌          | ✅       | **OC** | OC: Docker browser + loopback control + auth |
| **Plugin / extension model**      | ❌          | ✅✅      | **OC** | CU: monolithic crate; OC: 91 extensions |
| **Voice I/O**                     | ❌          | ✅ mac   | OC     | OC: Swabble (Speech.framework) |
| **Agent-facing tool ergonomics**  | 🟡          | ✅       | **OC** | CU: 15 tools, raw YAML tree; OC: ref-based actions |
| **Long-running stability**        | ⚠️          | ✅       | **OC** | CU: unbounded vision caches (memory leak) |

`*` CU "wins" by default — OpenClaw doesn't compete here. But CU's mac/Linux story is so stubbed that the win is hollow.

---

## 3. Architecture Deep-Dive

### 3.1 Element targeting model

**ComputerUse — selector calculus** ([selector.rs](../crates/computeruse/src/selector.rs))

A full boolean algebra over accessibility attributes:

```rust
Selector::And(vec![
    Selector::Role { role: "Button", name: None },
    Selector::RightOf(Box::new(Selector::Name("Username".into()))),
    Selector::Visible(true),
])
```

Parsed via Shunting-Yard with precedence `OR > AND > NOT`, plus spatial combinators (`RightOf`, `Above`, `Near`), structural (`Has`, `Parent`, `Chain >>`), and `Nth(i)`. This is **richer than anything OpenClaw exposes** — but the matching backend only fully exists on Windows. On macOS/Linux, spatial selectors fall through to no-match.

**OpenClaw — snapshot + ref** ([extensions/browser/src/browser/pw-role-snapshot.ts](https://github.com/openclaw/openclaw))

Inverted model: don't ask the agent to construct selectors. Instead:
1. Take Playwright accessibility snapshot of the page
2. Annotate each interactive element with a stable numeric ref (`aria-ref="12"`) or role ref (`e12`)
3. Hand the agent a **text representation** of the tree
4. Agent says `click 12` — Playwright resolves the ref

This is **dramatically more agent-friendly**. The LLM never has to guess `name:` vs `role:` — it reads the snapshot, picks a ref, done. CSS selectors are intentionally *not* exposed to agents (deterministic, no brittleness).

**Verdict:** ComputerUse's selector system is more powerful for **scripted** automation. OpenClaw's snapshot model is more reliable for **agentic** automation. ElizaOS needs both: keep the selector calculus, **add** a snapshot-with-refs path.

### 3.2 Browser automation

**ComputerUse — extension bridge** ([extension_bridge.rs](../crates/computeruse/src/extension_bridge.rs))

```rust
const DEFAULT_WS_ADDR: &str = "127.0.0.1:17373";  // hardcoded, no env override
```

Custom WebSocket server. The Chrome extension connects, ComputerUse sends `EvalRequest{ code, await_promise }`, extension `eval()`s in the active tab. Capabilities:
- ✅ `eval_js(code)` — arbitrary JS in active tab
- ✅ `close_tab(id|url|title)`
- ✅ DOM element capture (serialized)
- ❌ No navigation primitive (must `eval("location.href = ...")`)
- ❌ No screenshot-via-CDP (uses native xcap instead → DPI mismatch with DOM coords)
- ❌ No file upload, download interception, PDF, cookie/storage access
- ❌ No iframe scoping
- ❌ No auth on the WS port (any local process can connect)
- ❌ Extension must be manually installed; no auto-discovery

**OpenClaw — Playwright + CDP** ([extensions/browser/src/browser/pw-session.ts](https://github.com/openclaw/openclaw))

`playwright-core@1.58.2` + raw CDP (`Target.setAutoAttach`, `Storage.*`, `Runtime.evaluate`). Capabilities:
- ✅ `page.goto()`, `getByRole().click()`, `fill()`, `type()`
- ✅ `page.screenshot()` and element-scoped screenshots
- ✅ `page.pdf()`, download interception, `fileChooser.setFiles()`
- ✅ Cookie/localStorage read/write via CDP
- ✅ iframe scoping (`--frame "iframe#main"`)
- ✅ Multi-profile: isolated `openclaw` profile, attach-to-user via Chrome MCP, remote CDP (Browserless/Browserbase)
- ✅ Loopback HTTP control plane with optional bearer auth
- ✅ Auto-managed Chromium download/launch

**Gap quantification:** OpenClaw's browser extension is **276 TS files**. ComputerUse's browser bridge is **one Rust file** (~600 LOC). This is a 50× delta in surface area.

### 3.3 Cross-platform native automation

Hard line counts ([crates/computeruse/src/platforms/](../crates/computeruse/src/platforms/)):

| Platform | LOC    | Files | `UnsupportedOperation` | Stub density |
|----------|-------:|------:|----------------------:|-------------:|
| Windows  | 12,000 | 14    | 0                     | 0%           |
| macOS    | 2,089  | 1     | **76**                | ~36%         |
| Linux    | 2,315  | 1     | **88**                | ~38%         |

**What's stubbed on macOS** (sample, [macos/mod.rs](../crates/computeruse/src/platforms/macos/mod.rs)):
- L420, L451, L468: `set_value()`, `invoke()`, `perform_action()`
- L512–527: `minimize_window()`, `maximize_window()`, window state
- L603–671: parent/child tree navigation, value getters
- L697–937: ~30 element trait methods returning `UnsupportedOperation`
- L1186–1246: engine-level locator creation paths

What this means in practice: an agent on macOS can **see** the accessibility tree (top-level), can **click coordinates** (via enigo), but cannot **set a text field's value programmatically**, cannot **walk children of a window**, cannot **resize a window**. Every selector resolves to "element not found" because `children()` returns `Ok(Vec::new())`.

**OpenClaw doesn't have this problem because OpenClaw doesn't try.** Their bet: most automation that matters happens in a browser, and Playwright covers that on every OS. They're not wrong — but it leaves ElizaOS without a story for "click the Slack desktop app button" on macOS.

### 3.4 Agent-facing tool ergonomics

**ComputerUse MCP tools** (15 total, [server.rs](../crates/computeruse-mcp-agent/src/server.rs)):

| Tool | Line | Issue |
|------|------|-------|
| `get_window_tree` | :1338 | Returns raw YAML — no stable refs, agent must construct selectors |
| `click_element` | :2492 | Requires `process` + `selector` — agent guesses selector syntax |
| `type_into_element` | :2160 | Same — selector-string fragility |
| `execute_sequence` | :7529 | Workflow YAML — useful for replay, awkward for ad-hoc |
| `read_file`, `write_file`, `edit_file`, `glob_files`, `grep_files` | :8708+ | Out of scope (file ops, not automation) — but fine |

The agent loop is: call `get_window_tree` → **parse YAML in-prompt** → guess a selector string → `click_element(process, "role:Button name:Submit")` → if `ElementNotFound`, try a different selector. Each retry is a full LLM roundtrip.

**OpenClaw browser tool**:

Agent loop: call `snapshot` → reads `[12] button "Submit"` → `click 12`. **No selector synthesis.** The ref is stable across the snapshot's lifetime. If the page changes, take a new snapshot; refs renumber but the model holds.

This is the single biggest ergonomic gap. ComputerUse already serializes the tree to YAML — adding stable refs is one extra column.

### 3.5 Vision pipeline

**ComputerUse** ([vision.rs](../crates/computeruse-mcp-agent/src/vision.rs), [omniparser.rs](../crates/computeruse-mcp-agent/src/omniparser.rs)):

Three backends, all coordinate-emitting:
1. `uni-ocr` — local, Windows OCR API only
2. Omniparser — `POST $OMNIPARSER_BACKEND_URL` (default `https://app.mediar.ai/api/omniparser/parse`), 5 min timeout, returns icon bboxes
3. Gemini Vision — `POST $GEMINI_VISION_BACKEND_URL`, returns semantic element descriptions

Results cached in `Arc<Mutex<HashMap>>` ([server.rs:~1705](../crates/computeruse-mcp-agent/src/server.rs)) — **never evicted**. Long-running MCP server leaks every screenshot's parsed result.

**OpenClaw**: No OCR. The accessibility snapshot **is** the text representation — no pixel-to-text needed. Screenshots are passed to the agent's VLM directly (Claude vision, GPT-4V) with ref labels overlaid (`--labels`).

OpenClaw's bet is again the right one for browsers: the DOM/a11y tree is ground truth, OCR is lossy. ComputerUse's vision stack exists because **native apps don't have a DOM** — Omniparser is a workaround for the macOS/Linux stub problem, not a feature.

### 3.6 Sandbox & isolation

**ComputerUse**: None. The MCP server runs with full user privileges. Browser extension WS port has no auth. Workflow execution runs arbitrary TS in-process.

**OpenClaw** ([Dockerfile.sandbox-browser](https://github.com/openclaw/openclaw/blob/main/Dockerfile.sandbox-browser)):
- Chromium + Xvfb + noVNC in `debian:bookworm-slim`, exposes CDP:9222 / VNC:5900 / noVNC:6080
- Browser profiles in dedicated user-data dirs, distinct CDP ports per profile
- Loopback-only control server, gateway auth token auto-generated
- Per-channel DM pairing policy (`dmPolicy="pairing"`)

For the AVB use case (autonomous virtual beings acting on the user's behalf), sandbox isolation is non-negotiable.

---

## 4. What ComputerUse does well (keep these)

Don't throw the baby out:

1. **Selector calculus** — `RightOf(Name("X")) && role:Button && Visible(true)` is more expressive than anything OpenClaw or Playwright offers natively. Keep it, expose refs as a *complement* not a replacement.
2. **Workflow recording → deterministic replay** — OpenClaw has nothing comparable. The rdev hook → YAML → TS pipeline is a real differentiator.
3. **Windows UIAutomation depth** — 12,000 LOC of mature integration. The action overlay, inspect overlay, virtual display are unique.
4. **MCP-native delivery** — `npx -y computeruse-mcp-agent` and you're in Claude Desktop / Cursor. OpenClaw requires running the gateway daemon.
5. **No cursor hijacking** — accessibility-API actions don't move the user's mouse. OpenClaw's Playwright also doesn't, but only in browser.

---

## 5. Gap Summary → Upgrade Targets

Ranked by (impact on agent reliability) × (tractability):

| # | Gap | Evidence | Impact | Tractability |
|---|-----|----------|--------|--------------|
| 1 | Browser bridge fragile vs Playwright/CDP | extension_bridge.rs:32 hardcoded port; 1 file vs 276 | High — most agent tasks are web | High — adopt playwright-core |
| 2 | No ref-based snapshots for agents | server.rs:1338 emits raw YAML w/o refs | High — every misclick = LLM retry | High — one extra column |
| 3 | macOS stubs (76× UnsupportedOperation) | macos/mod.rs L420-1246 | High on mac — children() empty | Medium — AX API is documented |
| 4 | Vision cache unbounded growth | server.rs:~1705 HashMap never evicted | Medium — long-running MCP leaks | High — LRU or TTL |
| 5 | No browser backend abstraction | extension_bridge baked into core lib | Medium — can't swap impl | Medium — trait + 2 impls |
| 6 | Linux stubs (88× UnsupportedOperation) | linux/mod.rs L188-2307 | Medium — Linux servers | Medium — atspi crate exists |
| 7 | No sandbox/isolation | None | Medium for AVB | Low — large surface |
| 8 | No multi-profile / remote browser | extension_bridge.rs single conn | Low-Med | Coupled to #1 |

The Top-5 plan picks #1–#5. Linux (#6) is grouped with macOS (#3) where the fixes overlap. Sandbox (#7) is deferred — depends on #1's choice of backend.

---

## Appendix A: Verified evidence

**Source-confirmed (direct read/grep):**
- `extension_bridge.rs:32` — `const DEFAULT_WS_ADDR: &str = "127.0.0.1:17373";`
- `macos/mod.rs` — 76 grep matches for `UnsupportedOperation`
- `linux/mod.rs` — 88 grep matches for `UnsupportedOperation`
- Platform LOC: Win 12,000 / mac 2,089 / Linux 2,315 (`wc -l`)
- OpenClaw `package.json` — `"playwright-core": "1.58.2"`
- OpenClaw `extensions/` — 91 directories
- OpenClaw browser TS files — 276 (`find -path "*browser*" -name "*.ts" | wc -l`)

**Reported by exploration agent (high confidence, not directly re-verified after /tmp reap):**
- OpenClaw `pw-role-snapshot.ts` aria-ref mechanism
- OpenClaw `Dockerfile.sandbox-browser` chromium+xvfb+noVNC
- ComputerUse 15 MCP tools at server.rs lines 1338–9450
- Omniparser/Gemini default URLs (`app.mediar.ai/api/...`)

## Appendix B: OpenClaw clone

Synced to `/tmp/openclaw-research/openclaw/` (depth-1, 11,308 files). Will be reaped by macOS tmp cleanup; re-clone if needed:
```bash
git clone --depth 1 https://github.com/openclaw/openclaw.git /tmp/openclaw-research/openclaw
```
