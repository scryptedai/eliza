# ComputerUse Upgrade Plan — Top 5 Issues vs OpenClaw

**Companion to:** [OPENCLAW_COMPARISON.md](./OPENCLAW_COMPARISON.md)
**Status:** Draft for human review — **do not implement without sign-off**
**Date:** 2026-04-02

---

## Selection rationale

Issues ranked by `(impact on agent reliability × tractability)`. All five are independently shippable; #1 and #5 share infrastructure but can land in either order.

| # | Issue | Impact | Effort | Files touched |
|---|-------|--------|--------|---------------|
| 1 | Browser bridge → Playwright/CDP backend | ★★★★★ | M-L | new crate + 1 trait |
| 2 | Ref-based snapshots in `get_window_tree` | ★★★★★ | S-M | server.rs, tree_formatter.rs |
| 3 | macOS AX un-stubbing (children/parent/set_value) | ★★★★☆ | M | macos/mod.rs |
| 4 | Cross-modal snapshot coherence | ★★★☆☆ | S | server.rs, utils.rs |
| 5 | `BrowserBackend` trait abstraction | ★★★☆☆ | S | lib.rs, extension_bridge.rs |

Effort: S = ≤2 files, M = 3-6 files, L = new crate.

---

## Issue #1 — Replace browser extension bridge with Playwright/CDP

### Problem

[`extension_bridge.rs`](../crates/computeruse/src/extension_bridge.rs) is a 600-LOC custom WebSocket server on hardcoded `127.0.0.1:17373` that requires a manually-installed Chrome extension. It can `eval()` JS and close tabs. That's it.

OpenClaw's `extensions/browser/` is **34,083 LOC** of `playwright-core@1.58.2` + raw CDP, providing: `page.goto`, `getByRole().click()`, `fill()`, `page.screenshot()`, `page.pdf()`, download/upload interception, cookie/storage R/W, iframe scoping, multi-profile, remote-CDP (Browserless/Browserbase).

For ElizaOS agents, browser is where >70% of "computer use" actually happens (Vercel logs, GCP console, GitHub PRs, SaaS dashboards). The extension bridge can't reliably do any of these.

### Fix

**New crate:** `crates/computeruse-browser-cdp/`

Don't reimplement Playwright in Rust. Use [`chromiumoxide`](https://crates.io/crates/chromiumoxide) (mature Rust CDP client, ~6k stars) or shell to a Node sidecar running playwright-core. **Recommendation: chromiumoxide** — keeps the binary self-contained, no Node runtime dependency at deploy time.

```
crates/computeruse-browser-cdp/
├── Cargo.toml          # chromiumoxide = "0.7", tokio
├── src/
│   ├── lib.rs          # pub struct CdpBrowser impl BrowserBackend
│   ├── session.rs      # Page lifecycle, CDP target attach
│   ├── snapshot.rs     # Accessibility.getFullAXTree → ref-annotated tree
│   ├── actions.rs      # click/type/fill via DOM.querySelector + Input.dispatchMouseEvent
│   └── profile.rs      # Profile { Isolated, AttachToUser{pid}, RemoteCdp{ws_url} }
```

**Surface (exposed via `BrowserBackend` trait, see Issue #5):**
- `navigate(url, wait_until: Load|DomContentLoaded|NetworkIdle)`
- `snapshot() -> AccessibilitySnapshot` (refs included — feeds Issue #2)
- `click_ref(ref_id)`, `type_ref(ref_id, text)`, `fill_ref(ref_id, text)`
- `screenshot(full_page: bool, element_ref: Option<RefId>)`
- `eval(code, await_promise) -> Value` (parity with current bridge)
- `cookies()`, `set_cookie(...)`, `local_storage(origin)`
- `wait_for(condition, timeout)` — visibility, network idle, custom predicate
- `download_to(path)`, `upload(ref_id, paths)`
- `pdf(options)`
- `Profile` enum: `Isolated{user_data_dir}` | `AttachToRunning{cdp_port}` | `Remote{ws_url}`

**MCP tool changes** ([server.rs](../crates/computeruse-mcp-agent/src/server.rs)):
- `navigate_browser` (:6066) → routes to `BrowserBackend::navigate`
- New `browser_snapshot` tool → returns ref-annotated a11y tree
- New `browser_click(ref)`, `browser_type(ref, text)` tools
- Keep `execute_browser_script` as escape hatch → `BrowserBackend::eval`

**Migration:** `extension_bridge` stays as fallback `BrowserBackend` impl (see Issue #5). Agent picks via `BROWSER_BACKEND={cdp|extension}` env or runtime flag.

**Validation:** Integration test against `httpbin.org` and a local fixture page: navigate → snapshot → assert ref count → click ref → assert URL changed.

### Why this is the highest-impact fix

Every other gap in this plan has a workaround (vision fallback, coordinate clicking). Browser has none — if the extension isn't installed, the agent is blind on the web. And the AVB use case (autonomous beings interacting with services) is **almost entirely web**.

---

## Issue #2 — Ref-based snapshots in `get_window_tree`

### Problem

The agent loop today ([server.rs:1338](../crates/computeruse-mcp-agent/src/server.rs)):

1. `get_window_tree(process)` → YAML tree dump
2. Agent parses YAML in-context, spots `Button "Submit"` somewhere
3. Agent **synthesizes a selector string**: `"role:Button name:Submit"`
4. `click_element(process, selector)` → if `ElementNotFound`, retry with different selector

Step 3 is where things break. The agent guesses `name:` vs `text:` vs `id:`. Each wrong guess is a full LLM roundtrip plus a failed click attempt.

OpenClaw's loop:
1. `snapshot` → `[12] button "Submit"`
2. `click 12`

No selector synthesis. The ref is a **promise** — the snapshot side knows exactly which element `12` maps to.

### Fix

**The infrastructure is already there.** [`platforms/mod.rs:38-53`](../crates/computeruse/src/platforms/mod.rs) defines `OverlayDisplayMode::Index` — the visual inspect overlay already numbers elements. The `uia_bounds` cache ([server.rs:815](../crates/computeruse-mcp-agent/src/server.rs)) already stores `index → bounds`. We just need to:

**A) Emit the index in the YAML** ([tree_formatter.rs](../crates/computeruse/src/tree_formatter.rs))

Change `format_tree_as_compact_yaml` to prefix each interactive element with its `uia_bounds` key:

```yaml
# Before
- Button "Submit" {enabled, focusable}
# After
- [42] Button "Submit" {enabled, focusable}
```

**B) Add `ref` parameter to action tools** (server.rs)

```rust
// click_element gets a new arg path:
struct ClickElementArgs {
    process: String,
    selector: Option<String>,   // existing path — keep for power users
    ref_id: Option<u32>,        // NEW — resolve via uia_bounds[ref_id]
    // ... rest unchanged
}
```

When `ref_id` is provided, look up `uia_bounds.get(&ref_id)` → `(x, y, w, h)` → coordinate-click at center. **No selector matching needed.** Same change to `type_into_element`, `scroll_element`, `validate_element`, `activate_element`.

**C) Snapshot generation tag** (server.rs)

Each `get_window_tree` call returns a `snapshot_id: u64` (atomic counter). Action tools optionally accept `snapshot_id` — if provided and it doesn't match the cache's current generation, reject with `StaleSnapshot` instead of clicking the wrong thing. (This also fixes Issue #4 for the UIA path.)

**D) Tool descriptions** (server.rs `#[tool]` macros)

Update descriptions to tell the agent: *"Call `get_window_tree` first. Each interactive element shows a `[N]` ref. Pass that N as `ref_id` to click/type/scroll. Selectors are still available for advanced use."*

### Surface area

- `tree_formatter.rs`: ~30 LOC change in `format_tree_as_compact_yaml`
- `server.rs`: add `ref_id` arg to 5 tools, ~150 LOC; add `snapshot_id` plumbing, ~50 LOC
- `utils.rs`: extend `*Args` structs, ~30 LOC

### Validation

Existing `__test_mcp_workflows_*/calc_add/` fixture: rewrite the workflow to use refs instead of selectors, assert it still passes. Add a test that asserts `click_element(ref_id=999)` returns `RefNotFound`, not a coordinate-click at (0,0).

---

## Issue #3 — Un-stub macOS AX implementation

### Problem

[`macos/mod.rs`](../crates/computeruse/src/platforms/macos/mod.rs) returns `UnsupportedOperation` from **76 call sites**. Critical ones for agent workflows:

| Method | Line | What it should do | Current AX call needed |
|--------|------|-------------------|------------------------|
| `children()` | various | Walk subtree | `AXUIElementCopyAttributeValue(elem, kAXChildrenAttribute, ...)` |
| `parent()` | various | Walk up | `AXUIElementCopyAttributeValue(elem, kAXParentAttribute, ...)` |
| `set_value()` | :450 | Set text field | `AXUIElementSetAttributeValue(elem, kAXValueAttribute, CFString)` |
| `get_text()` | :447 | Read content | Currently returns `Ok(String::new())` — not even an error! |
| `invoke()` | :419 | Press/activate | `AXUIElementPerformAction(elem, kAXPressAction)` |
| `perform_action()` | :467 | Generic action | `AXUIElementPerformAction(elem, action_name)` |
| `is_enabled()` | :458 | Check state | Currently hardcoded `Ok(true)` |
| `is_focused()` | :464 | Check state | Currently hardcoded `Ok(false)` |
| `focus()` | :411 | Set focus | Currently no-op (`Ok(())`) — `AXUIElementSetAttributeValue(elem, kAXFocusedAttribute, true)` |

**The infrastructure is already in place.** [`macos/mod.rs:7`](../crates/computeruse/src/platforms/macos/mod.rs) already imports `AXUIElementCopyAttributeValue` and uses it at lines 202, 211 for `kAXPositionAttribute` and `kAXSizeAttribute`. The `accessibility` crate is already a dependency. The pattern is established.

### Fix

**Phase 1 — Tree traversal (unblocks selectors)**

Without `children()`, no selector ever matches a non-window element. This is the keystone.

```rust
// macos/mod.rs — replace stub at children()
fn children(&self) -> Result<Vec<UIElement>, AutomationError> {
    let ax = self.ax_element()?;
    let kids: CFArray<AXUIElement> = ax
        .attribute(&AXAttribute::new(&CFString::from_static_string("AXChildren")))
        .map_err(|e| AutomationError::PlatformError(format!("AXChildren: {e}")))?;
    Ok(kids.iter()
        .map(|child| UIElement::new(Box::new(MacOSUIElement::from_ax(child.clone()))))
        .collect())
}
```

`parent()` is symmetric (`AXParentAttribute` returns single `AXUIElement`).

The `accessibility` crate already wraps these patterns — see `AXUIElementAttributes` trait already imported at line 5.

**Phase 2 — Element actions**

```rust
fn set_value(&self, value: &str) -> Result<(), AutomationError> {
    let ax = self.ax_element()?;
    ax.set_attribute(
        &AXAttribute::new(&CFString::from_static_string("AXValue")),
        CFString::new(value).as_CFType(),
    ).map_err(|e| AutomationError::PlatformError(format!("set AXValue: {e}")))
}

fn invoke(&self) -> Result<(), AutomationError> {
    let ax = self.ax_element()?;
    ax.perform_action(&CFString::from_static_string("AXPress"))
        .map_err(|e| AutomationError::PlatformError(format!("AXPress: {e}")))
}
```

**Phase 3 — State queries**

Replace hardcoded `Ok(true)` / `Ok(false)` with actual `kAXEnabledAttribute`, `kAXFocusedAttribute` reads.

**Phase 4 — Permission preflight**

Add a startup check: `AXIsProcessTrusted()` (already in `accessibility_sys`). If false, return a clear `AutomationError::PermissionDenied("Grant Accessibility permission: System Settings → Privacy & Security → Accessibility")` instead of crashing at line 243 with `"no native handle"`.

### Surface area

All in `macos/mod.rs`:
- Phase 1: ~80 LOC (children, parent, helper to convert AX→MacOSUIElement)
- Phase 2: ~60 LOC (set_value, get_text, invoke, perform_action)
- Phase 3: ~40 LOC (is_enabled, is_visible, is_focused, focus)
- Phase 4: ~20 LOC (preflight in `MacOSEngine::new`)

**Linux note:** Same shape applies to [`linux/mod.rs`](../crates/computeruse/src/platforms/linux/mod.rs) (88 stubs) using the `atspi` crate's `AccessibleProxy::get_children()` etc. Phase 1 is portable; Phases 2-3 use AT-SPI `Action` and `Value` interfaces. Recommend doing macOS first — more ElizaOS users on macOS, and the AX API is more uniform.

### Validation

Manual: open TextEdit, run `get_window_tree("TextEdit")`, assert tree depth > 2 (currently always ≤ 2). `click_element` on a menu item, assert no `UnsupportedOperation`. `set_value` on a text field, read back, assert match.

Automated: requires a macOS CI runner with Accessibility permission — exists in `.github/workflows/ci.yml` already (verify the permission grant).

---

## Issue #4 — Cross-modal snapshot coherence

### Problem

`DesktopWrapper` holds **five independent caches** ([server.rs:812-816](../crates/computeruse-mcp-agent/src/server.rs)):

```rust
ocr_bounds:       Arc<Mutex<HashMap<u32, (f64,f64,f64,f64)>>>,  // updated by include_ocr
omniparser_items: Arc<Mutex<HashMap<u32, OmniparserItem>>>,     // updated by include_omniparser
vision_items:     Arc<Mutex<HashMap<u32, VisionElement>>>,      // updated by include_gemini_vision
uia_bounds:       Arc<Mutex<HashMap<u32, (f64,f64,f64,f64)>>>,  // updated by every call
dom_bounds:       Arc<Mutex<HashMap<u32, ...>>>,                // updated by include_browser_dom
```

Each is **replaced** (`*cache = new`, lines 1517/1541/1577/1598/1629/1652) — so memory is bounded. But they're updated **independently** by different `include_*` flags. Failure mode:

```
T0:  get_window_tree(include_omniparser=true)  → omniparser_items = snapshot of screen at T0
T1:  user opens a different app, screen changes completely
T2:  get_window_tree(include_ocr=true)         → ocr_bounds = snapshot at T2; omniparser STILL T0
T3:  click_element(vision_type=Omniparser, index=5)  → clicks coordinates from T0 on T2's screen
```

At T3 the agent confidently clicks a stale coordinate. No error, no warning. Worse than `ElementNotFound` — it's a successful click on the wrong pixel.

### Fix

**Unify the snapshot generation under one atomic.**

```rust
// utils.rs — DesktopWrapper
struct SnapshotGeneration {
    id: u64,                                      // monotonic
    timestamp: Instant,
    process: String,                              // which window this snapshot is for
    uia_bounds: HashMap<u32, Bounds>,
    ocr_bounds: Option<HashMap<u32, Bounds>>,     // None if not requested this gen
    omniparser: Option<HashMap<u32, OmniparserItem>>,
    vision: Option<HashMap<u32, VisionElement>>,
    dom_bounds: Option<HashMap<u32, Bounds>>,
}

// DesktopWrapper holds:
current_snapshot: Arc<Mutex<Option<SnapshotGeneration>>>,
snapshot_counter: AtomicU64,
```

`get_window_tree` constructs a fresh `SnapshotGeneration` with whatever `include_*` flags were passed, and **atomically swaps** it in. The returned YAML includes `snapshot_id: <N>` at the top.

Action tools (`click_element` etc.) when using `vision_type=Omniparser`:
1. Lock `current_snapshot`
2. If `snapshot.omniparser.is_none()` → `Err("No omniparser snapshot active — call get_window_tree(include_omniparser=true) first")`
3. If client passed `snapshot_id` and it doesn't match → `Err(StaleSnapshot)` (also covers Issue #2C)
4. Otherwise resolve index and act

**Optionally:** keep last 2-3 generations in a small ring buffer so a slightly-stale `snapshot_id` can still resolve with a warning. Decide based on agent retry patterns observed in practice.

### Surface area

- `utils.rs`: replace 5 `Arc<Mutex<HashMap>>` fields with one `Arc<Mutex<Option<SnapshotGeneration>>>` + `AtomicU64`. ~80 LOC.
- `server.rs`: every cache write site (1517, 1541, 1577, 1598, 1629, 1652) collapses into one `*self.current_snapshot.lock() = Some(gen)`. Net negative LOC. Every cache read site (~1858, 2668, etc.) becomes `self.current_snapshot.lock().as_ref().and_then(|s| s.omniparser.as_ref())...`. ~150 LOC churn.

### Validation

Unit test: construct `DesktopWrapper`, populate snapshot with `omniparser=Some(...)`, call `get_window_tree(include_ocr=true)`, assert `current_snapshot.omniparser.is_none()`. Then assert `click_element(vision_type=Omniparser)` returns the new error, not a click.

---

## Issue #5 — `BrowserBackend` trait abstraction

### Problem

[`lib.rs:16`](../crates/computeruse/src/lib.rs) hardcodes `pub mod extension_bridge;` as a top-level public module. The MCP server's browser tools call `ExtensionBridge::eval_js` directly. There's no way to swap in CDP (Issue #1) without breaking every caller.

This is a small refactor but it's the **enabling change** for Issue #1, and it lets users keep the extension bridge if they prefer it (e.g., already have the extension installed, or are on a locked-down corp environment where CDP is blocked).

### Fix

**A) Define the trait** (new file `crates/computeruse/src/browser_backend.rs`)

```rust
#[async_trait::async_trait]
pub trait BrowserBackend: Send + Sync {
    async fn navigate(&self, url: &str, wait: WaitUntil) -> Result<(), AutomationError>;
    async fn eval(&self, code: &str, await_promise: bool) -> Result<serde_json::Value, AutomationError>;
    async fn close_tab(&self, target: TabTarget) -> Result<CloseTabResult, AutomationError>;
    async fn snapshot(&self) -> Result<BrowserSnapshot, AutomationError>;     // for Issue #2
    async fn screenshot(&self, opts: ScreenshotOpts) -> Result<Vec<u8>, AutomationError>;
    async fn click_ref(&self, ref_id: RefId) -> Result<(), AutomationError>;  // for Issue #2
    async fn type_ref(&self, ref_id: RefId, text: &str) -> Result<(), AutomationError>;
    fn capabilities(&self) -> BrowserCapabilities;  // so callers can degrade gracefully
}

pub struct BrowserCapabilities {
    pub supports_pdf: bool,
    pub supports_downloads: bool,
    pub supports_iframe_scoping: bool,
    pub supports_remote_cdp: bool,
}

pub enum TabTarget { Id(i32), Url(String), Title(String) }
pub enum WaitUntil { Load, DomContentLoaded, NetworkIdle }
```

**B) Implement for `ExtensionBridge`** (in `extension_bridge.rs`)

Most methods already exist; wrap them. `navigate` becomes `eval("location.href = ...")`, `snapshot` calls the existing DOM-capture path, `click_ref` is `eval("document.querySelector('[data-cu-ref=\"N\"]').click()")` once Issue #2 lands. `capabilities()` returns mostly `false`.

**C) `Desktop` holds `Arc<dyn BrowserBackend>`** ([lib.rs](../crates/computeruse/src/lib.rs))

```rust
pub struct Desktop {
    engine: Arc<dyn AccessibilityEngine>,
    browser: Arc<dyn BrowserBackend>,        // NEW — was implicit ExtensionBridge
    // ...
}

impl Desktop {
    pub fn new() -> Self { /* defaults to ExtensionBridge */ }
    pub fn with_browser_backend(backend: Arc<dyn BrowserBackend>) -> Self { ... }
}
```

**D) MCP server picks at startup** ([main.rs](../crates/computeruse-mcp-agent/src/main.rs))

```rust
let browser: Arc<dyn BrowserBackend> = match std::env::var("COMPUTERUSE_BROWSER_BACKEND").as_deref() {
    Ok("cdp") => Arc::new(CdpBrowser::launch_isolated()?),         // from Issue #1
    Ok("cdp-attach") => Arc::new(CdpBrowser::attach(cdp_port)?),
    Ok("remote") => Arc::new(CdpBrowser::remote(ws_url)?),
    _ => Arc::new(ExtensionBridge::start().await?),                // current default
};
```

Document `COMPUTERUSE_BROWSER_BACKEND` in `.env.example` (root repo).

### Surface area

- New file `browser_backend.rs`: ~80 LOC trait + types
- `extension_bridge.rs`: ~50 LOC `impl BrowserBackend for ExtensionBridge`
- `lib.rs`: add field to `Desktop`, ~20 LOC
- `server.rs`: change `ExtensionBridge::` direct calls to `desktop.browser().*`, ~30 LOC sed
- `main.rs`: backend selection, ~20 LOC

### Validation

Existing extension-bridge tests pass unchanged (default backend). Add a mock `BrowserBackend` impl that asserts call sequences for unit-testing the MCP tool layer without a real browser.

---

## Sequencing

```
        Issue #5 (trait)  ─────┐
                               ├──→ Issue #1 (CDP impl)
        Issue #2 (refs)   ─────┘           │
              │                            │
        Issue #4 (coherence) ──────────────┴──→ Full integration
              │
        Issue #3 (macOS) ──── independent, can land any time
```

**Recommended order:** #5 → #2 → #4 → #1 → #3

- #5 first: zero behavior change, unblocks #1
- #2 second: small, immediate agent-UX win, motivates #4
- #4 third: small, fixes a bug class before #1 makes it worse
- #1 fourth: largest, but trait+refs already in place
- #3 anytime: orthogonal, can be a separate workstream

**MVP slice for fastest agent-visible win:** #2 alone. Refs in YAML + `ref_id` in click. ~200 LOC, no new dependencies, ships in one PR.

---

## Out of scope (deliberately deferred)

| Item | Why deferred |
|------|--------------|
| Linux AT-SPI un-stubbing (88 stubs) | Same shape as #3 but lower user impact; do after macOS proves the pattern |
| Sandbox/Docker isolation | Depends on #1's backend choice; large surface; AVB plugin should drive requirements |
| Workflow recording on macOS/Linux | rdev hooks exist; depends on #3 for element-resolution during recording |
| Voice (Swabble equivalent) | OpenClaw's Swabble is mac-only Swift; ElizaOS likely wants this in `plugin-avb` not computeruse |
| Python bindings | PyO3 stub exists; demand-driven |

---

## Risk register

| Risk | Mitigation |
|------|------------|
| chromiumoxide CDP version drift vs Chrome | Pin tested Chrome version; expose `--cdp-version-check` flag |
| macOS AX requires Accessibility permission per-binary | Phase 4 preflight + clear error; document in README |
| Ref renumbering breaks in-flight agent actions | `snapshot_id` validation (#2C/#4) makes this an explicit error not a silent misclick |
| `BrowserBackend` trait churn as #1 reveals new needs | Mark `#[non_exhaustive]` on `BrowserCapabilities`; add methods with default-`Err(Unsupported)` impls |
| #4 changes lock granularity → contention | Single `Mutex<Option<SnapshotGeneration>>` lock held briefly; replacement is `O(1)` swap |

---

## Open questions for human review

1. **#1 backend choice:** chromiumoxide (Rust-native, self-contained) vs Node sidecar running playwright-core (closer to OpenClaw, more battle-tested)? Leaning chromiumoxide.
2. **#2 ref namespace:** Unified ref space across UIA/OCR/Omniparser/DOM (one number line, prefix-tagged like `u42`/`o7`/`d3`) or per-modality? Unified is simpler for agents but requires #4 first.
3. **#3 scope:** Full Phase 1-4 in one PR, or land Phase 1 (children/parent) alone first? Phase 1 alone unblocks selectors and is the highest leverage.
4. **MVP:** Ship #2 standalone first to prove the agent-UX win before committing to #1's larger surface?
