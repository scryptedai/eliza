# Issue #2: Ref-Based Snapshots — Design Document

**Status:** Approved for isolated implementation
**Scope:** `crates/computeruse-mcp-agent` only — no changes to `crates/computeruse` core
**Goal:** Bring computeruse's element-targeting UX to parity with OpenClaw's `[ref=eN]` model while preserving existing capabilities OpenClaw lacks (multi-modal vision, native desktop).

---

## 1. What computeruse Does Today

### 1.1 Snapshot side (`tree_formatter.rs`)

The `format_tree_as_compact_yaml` function ([tree_formatter.rs:56-74](../crates/computeruse-mcp-agent/src/tree_formatter.rs#L56)) walks a `SerializableUIElement` tree in pre-order and emits:

```
#1 [Window] Notepad (bounds: [0,0,1920,1080])
  #2 [MenuBar] (bounds: [0,0,1920,30], focusable)
    #3 [MenuItem] File (bounds: [0,0,40,30])
  #4 [Document] Text Editor (bounds: [0,30,1920,1050], focusable)
```

| Property | Value | Source |
|---|---|---|
| Index assignment | Pre-order counter, 1-based | [tree_formatter.rs:101-102](../crates/computeruse-mcp-agent/src/tree_formatter.rs#L101) |
| Index criteria | **Any element with `bounds.is_some()`** | [tree_formatter.rs:100](../crates/computeruse-mcp-agent/src/tree_formatter.rs#L100) |
| Index format | `#N ` prefix | [tree_formatter.rs:103](../crates/computeruse-mcp-agent/src/tree_formatter.rs#L103) |
| No-bounds format | `- ` prefix | [tree_formatter.rs:116](../crates/computeruse-mcp-agent/src/tree_formatter.rs#L116) |
| Bounds shown to agent | **Yes** — `(bounds: [x,y,w,h])` inline | [tree_formatter.rs:138](../crates/computeruse-mcp-agent/src/tree_formatter.rs#L138) |
| Cache value | `(role, name, bounds, selector)` | [tree_formatter.rs:110-113](../crates/computeruse-mcp-agent/src/tree_formatter.rs#L110) |

The same pattern repeats for each modality with **independent counters**:

| Modality | Cache type (`utils.rs:383-405`) | Selector stored? |
|---|---|---|
| UIA | `HashMap<u32, (String, String, (f64,f64,f64,f64), Option<String>)>` | ✅ `Option<String>` |
| OCR | `HashMap<u32, (String, (f64,f64,f64,f64))>` | ❌ |
| DOM | `HashMap<u32, (String, String, (f64,f64,f64,f64))>` | ❌ |
| Omniparser | `HashMap<u32, OmniparserItem>` | ❌ |
| Vision | `HashMap<u32, VisionElement>` | ❌ |

### 1.2 Click side (`server.rs`)

`click_element` Mode 2 ([server.rs:2605-2819](../crates/computeruse-mcp-agent/src/server.rs#L2605)):

```rust
// Agent must supply BOTH index AND vision_type
let index = args.index.unwrap();           // u32: 5
let vision_type = args.get_vision_type();  // enum: UiTree | Ocr | Omniparser | Gemini | Dom

// Route to one of 5 caches
let (item_label, bounds) = match vision_type {
    VisionType::UiTree => self.uia_bounds.lock()?.get(&index).cloned()?,
    VisionType::Ocr    => self.ocr_bounds.lock()?.get(&index).cloned()?,
    // ... 3 more arms
};

// Click the FROZEN coordinates from snapshot time
let click_x = bounds.0 + bounds.2 / 2.0;
let click_y = bounds.1 + bounds.3 / 2.0;
self.desktop.click_at_coordinates_with_type(click_x, click_y, ...)
```

**The selector is captured but never used for resolution** — [server.rs:2637](../crates/computeruse-mcp-agent/src/server.rs#L2637) stores it only to echo back in the response JSON at [server.rs:2790](../crates/computeruse-mcp-agent/src/server.rs#L2790).

### 1.3 The orphan: `clustered_bounds`

A prefixed-ref system **already exists** but is half-wired:

| Component | Status | Location |
|---|---|---|
| `ElementSource` enum (`Uia`/`Dom`/`Ocr`/`Omniparser`/`Gemini`) | ✅ Built | [computeruse/src/tree_formatter.rs:42-48](../crates/computeruse/src/tree_formatter.rs#L42) |
| `ElementSource::prefix()` → `'u'`/`'d'`/`'o'`/`'p'`/`'g'` | ✅ Built | [computeruse/src/tree_formatter.rs:52-59](../crates/computeruse/src/tree_formatter.rs#L52) |
| `ElementSource::parse_prefixed_index("u42")` → `(Uia, 42)` | ✅ Built | [computeruse/src/tree_formatter.rs:63-79](../crates/computeruse/src/tree_formatter.rs#L63) |
| `clustered_bounds: HashMap<String, (ElementSource, u32, bounds)>` | ✅ Built | [utils.rs:409-420](../crates/computeruse-mcp-agent/src/utils.rs#L409) |
| Cache populated in `ClusteredYaml` mode | ✅ Built | [server.rs:1715](../crates/computeruse-mcp-agent/src/server.rs#L1715) |
| **Click path that consumes prefixed refs** | ❌ **Missing** | — |
| **`parse_prefixed_index` callsite in MCP agent** | ❌ **Zero** | grep verified |

The agent receives `#u5 [Button] Save` in the YAML but cannot say "click u5". It must translate to `{index: 5, vision_type: "ui_tree"}` — defeating the point.

### 1.4 Failure modes (concrete bugs)

| # | Scenario | Result |
|---|---|---|
| F1 | Agent snapshots → user scrolls window → agent calls `click(index: 5)` | Clicks where button **was**, not where it **is** |
| F2 | Agent snapshots with `include_ocr=true` (T1) → snapshots again with `include_tree=true` only (T2) → calls `click(index: 3, vision_type: "ocr")` | Clicks T1's coordinates against T2's screen state. No staleness error. |
| F3 | Agent receives `#u5` from `ClusteredYaml` output → tries `click(index: "u5")` | Type error — `index` is `u32` |
| F4 | Two unrelated windows snapshotted in succession → indices collide | Index 5 from window A clicks coordinates valid only for window B |
| F5 | Agent forgets `vision_type` → defaults to `UiTree` → wrong cache | Clicks UIA element 5 when agent meant OCR element 5 |

---

## 2. What OpenClaw Does

### 2.1 Snapshot side (`pw-role-snapshot.ts`)

`buildRoleSnapshotFromAriaSnapshot` ([pw-role-snapshot.ts:270-327](https://github.com/openclaw/openclaw/blob/main/extensions/browser/src/browser/pw-role-snapshot.ts#L270)) post-processes Playwright's `ariaSnapshot()` text:

```yaml
# Input (Playwright ariaSnapshot)
- heading "Welcome" [level=1]
- button "Submit"
- button "Submit"
- link "Learn more"

# Output (OpenClaw enhanced)
- heading "Welcome" [ref=e1] [level=1]
- button "Submit" [ref=e2]
- button "Submit" [ref=e3] [nth=1]
- link "Learn more" [ref=e4]
```

| Property | Value | Source |
|---|---|---|
| Ref assignment | Counter `e${n}`, 1-based | [pw-role-snapshot.ts:279-282](https://github.com/openclaw/openclaw/blob/main/extensions/browser/src/browser/pw-role-snapshot.ts#L279) |
| Ref criteria | `INTERACTIVE_ROLES.has(role) \|\| (CONTENT_ROLES.has(role) && name)` | [pw-role-snapshot.ts:189](https://github.com/openclaw/openclaw/blob/main/extensions/browser/src/browser/pw-role-snapshot.ts#L189) |
| Ref format | `[ref=eN]` suffix | [pw-role-snapshot.ts:207](https://github.com/openclaw/openclaw/blob/main/extensions/browser/src/browser/pw-role-snapshot.ts#L207) |
| Bounds shown to agent | **No** — never | (absent from output) |
| Cache value (`RoleRef`) | `{ role: string, name?: string, nth?: number }` | [pw-role-snapshot.ts:3-8](https://github.com/openclaw/openclaw/blob/main/extensions/browser/src/browser/pw-role-snapshot.ts#L3) |
| Disambiguation | `[nth=N]` shown **only when role+name collides** | [pw-role-snapshot.ts:110-118](https://github.com/openclaw/openclaw/blob/main/extensions/browser/src/browser/pw-role-snapshot.ts#L110) |
| Compaction | Prunes branches with no `[ref=]` descendants | [pw-role-snapshot.ts:120-153](https://github.com/openclaw/openclaw/blob/main/extensions/browser/src/browser/pw-role-snapshot.ts#L120) |

`INTERACTIVE_ROLES` ([snapshot-roles.ts:8-26](https://github.com/openclaw/openclaw/blob/main/extensions/browser/src/browser/snapshot-roles.ts#L8)): `button`, `checkbox`, `combobox`, `link`, `listbox`, `menuitem`, `menuitemcheckbox`, `menuitemradio`, `option`, `radio`, `searchbox`, `slider`, `spinbutton`, `switch`, `tab`, `textbox`, `treeitem`.

### 2.2 Click side (`pw-session.ts`)

`refLocator(page, "e5")` ([pw-session.ts:519-556](https://github.com/openclaw/openclaw/blob/main/extensions/browser/src/browser/pw-session.ts#L519)):

```typescript
const info = state.roleRefs[normalized];           // {role: "button", name: "Submit", nth: 1}
if (!info) {
  throw new Error(`Unknown ref "${normalized}". Run a new snapshot...`);
}
const locator = info.name
  ? page.getByRole(info.role, { name: info.name, exact: true })
  : page.getByRole(info.role);
return info.nth !== undefined ? locator.nth(info.nth) : locator;
// Caller: await refLocator(page, "e5").click();   ← re-queries live DOM
```

There is also an `"aria"` mode ([pw-session.ts:528-533](https://github.com/openclaw/openclaw/blob/main/extensions/browser/src/browser/pw-session.ts#L528)) that uses Playwright's internal `aria-ref=` selector — refs are stable across calls because Playwright tracks the underlying DOM node.

### 2.3 Why this works

| Failure mode | OpenClaw behavior |
|---|---|
| F1 (scroll) | `getByRole("button", {name: "Submit"})` finds it at new position. Playwright auto-scrolls into view. |
| F2 (cross-modal staleness) | N/A — single modality (DOM only). But: `Unknown ref` error if cache cleared. |
| F3 (string ref) | Refs are strings (`e5`) by design. |
| F4 (window collision) | Refs are per-`Page` (`pageStates: WeakMap<Page, PageState>` at [pw-session.ts:106](https://github.com/openclaw/openclaw/blob/main/extensions/browser/src/browser/pw-session.ts#L106)). |
| F5 (modality confusion) | N/A — single modality. |

### 2.4 What OpenClaw can't do

| Limitation | Reason |
|---|---|
| Native desktop apps | No UIA/AX/AT-SPI integration — browser only |
| OCR / vision-based targeting | No equivalent to Omniparser/Gemini caches |
| Coordinate fallback | Refs that fail to resolve throw; no "last known position" fallback |
| Multi-source clustering | Single source means nothing to cluster |

---

## 3. The Synthesis

### 3.1 Core insight

OpenClaw's advantage is **not** the `[ref=eN]` syntax — computeruse already prints `#N` and already has prefixed `#uN` refs in `ClusteredYaml` mode. The advantage is what the ref **resolves to**:

| | computeruse today | OpenClaw |
|---|---|---|
| Ref → | `(f64, f64, f64, f64)` frozen coordinates | `{role, name, nth}` locator recipe |
| Click → | `enigo.move_mouse(x, y); enigo.click()` | `getByRole(role, {name}).nth(n).click()` |
| Survives scroll? | No | Yes |
| Survives DOM reflow? | No | Yes (if role+name unchanged) |

**But computeruse already stores the recipe** — the UIA cache holds `selector: Option<String>` ([tree_formatter.rs:112](../crates/computeruse-mcp-agent/src/tree_formatter.rs#L112)). It just doesn't use it.

### 3.2 Design: hybrid resolution

Don't pick coordinates **or** locator-recipe. Use both, in priority order:

```
ref → look up cache → has selector? → re-resolve via desktop.locator(sel) → click fresh bounds
                   └→ no selector?  → click cached bounds (OCR/vision can't re-resolve)
                   └→ re-resolve failed? → click cached bounds as fallback
```

This gives:
- **UIA refs**: OpenClaw-grade resilience (selector re-resolves)
- **OCR/Vision refs**: same as today (best effort, coordinates only)
- **No regression**: cached bounds remain the safety net

### 3.3 Design: snapshot generation counter

Add a single `AtomicU64` that increments on every snapshot. Returned to agent in snapshot output. Required (or warned-if-mismatched) on every ref-based click.

```rust
pub snapshot_generation: Arc<AtomicU64>,           // server-side counter
// snapshot returns: { "snapshot_id": 7, "ui_tree": "...", ... }
// click accepts:    { "ref": "u5", "snapshot_id": 7 }
// if 7 != current → "stale ref: snapshot 7 superseded by 9; re-snapshot first"
```

Solves F2 and F4 without changing cache structure.

### 3.4 Design: ref string field

Add `ref: Option<String>` alongside the existing `index: Option<u32>` + `vision_type: Option<VisionType>`. Parse with the **already-existing** `ElementSource::parse_prefixed_index`:

```rust
// New: agent says {"ref": "u5"}
// Old (still works): agent says {"index": 5, "vision_type": "ui_tree"}
```

The `ref` field becomes Mode 2's preferred entry point. `index` + `vision_type` remain for backward compat — both routes converge on the same lookup.

### 3.5 What we deliberately do **not** copy from OpenClaw

| OpenClaw feature | Reason to skip (this PR) |
|---|---|
| `INTERACTIVE_ROLES` filtering | computeruse indexes all bounded elements; agents already rely on this. Filtering is a behavior change → separate PR. |
| `[nth=N]` disambiguation suffix | The selector stored in UIA cache is already a unique chained selector (`role:Window >> role:Button|name:Save`). nth is OpenClaw's workaround for not having that. |
| `compactTree` pruning | Useful but orthogonal — output-size optimization, not targeting correctness. |
| Per-page ref scoping | Single MCP server = single agent session. The generation counter handles cross-window invalidation. |
| Hiding bounds from agent | Agents currently use bounds for spatial reasoning (`RightOf`, etc.). Removing them is a separate decision. |

---

## 4. Implementation Spec

### 4.1 New module: `ref_snapshot.rs`

**File:** `crates/computeruse-mcp-agent/src/ref_snapshot.rs`

Owns the snapshot generation counter and ref resolution logic. Pure functions where possible — testable without `Desktop`.

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use computeruse::ElementSource;

/// Bounds tuple used throughout the MCP agent.
pub type Bounds = (f64, f64, f64, f64);

/// Result of resolving a ref against the current snapshot generation.
#[derive(Debug, Clone, PartialEq)]
pub enum RefResolution {
    /// Ref found, bounds + optional selector for re-resolution.
    Found {
        source: ElementSource,
        label: String,
        bounds: Bounds,
        /// If Some, caller should attempt `desktop.locator(s)` before
        /// falling back to `bounds`. Only populated for UIA refs.
        selector: Option<String>,
    },
    /// Ref string is well-formed but not in any cache.
    NotFound {
        ref_str: String,
        source: ElementSource,
    },
    /// Ref string did not parse (e.g. "x5", "u", "uia5").
    Malformed {
        ref_str: String,
    },
    /// snapshot_id supplied by agent ≠ current generation.
    Stale {
        ref_str: String,
        agent_snapshot: u64,
        current_snapshot: u64,
    },
}

/// Owns the monotonic snapshot generation counter and provides
/// staleness-checked ref resolution against the per-modality caches.
///
/// One instance lives on `DesktopWrapper`. The caches themselves stay
/// where they are (separate `Arc<Mutex<HashMap>>` fields) — this struct
/// holds clones of those Arcs so it can read them without `&DesktopWrapper`.
pub struct RefSnapshotState {
    generation: AtomicU64,

    // UIA: (role, name, bounds, selector)
    uia: Arc<Mutex<HashMap<u32, (String, String, Bounds, Option<String>)>>>,
    // OCR: (text, bounds)
    ocr: Arc<Mutex<HashMap<u32, (String, Bounds)>>>,
    // DOM: (tag, identifier, bounds)
    dom: Arc<Mutex<HashMap<u32, (String, String, Bounds)>>>,
    // Omniparser: full item (label + box_2d)
    omniparser: Arc<Mutex<HashMap<u32, crate::omniparser::OmniparserItem>>>,
    // Vision: full element
    vision: Arc<Mutex<HashMap<u32, crate::vision::VisionElement>>>,
}

impl RefSnapshotState {
    pub fn new(/* the five Arc clones */) -> Self;

    /// Increment and return the new generation. Call once per
    /// `get_window_tree` (or any tool that repopulates a cache).
    pub fn bump(&self) -> u64;

    /// Current generation without bumping.
    pub fn current(&self) -> u64;

    /// Resolve a prefixed ref. `claimed_snapshot` is the snapshot_id
    /// the agent says it's working from — None means "don't check".
    pub fn resolve(&self, ref_str: &str, claimed_snapshot: Option<u64>) -> RefResolution;
}

/// Pure parser exposed for unit tests. Wraps `ElementSource::parse_prefixed_index`
/// but additionally accepts the `#` prefix that appears in YAML output
/// (so agents can copy-paste `#u5` directly).
pub fn parse_ref(s: &str) -> Option<(ElementSource, u32)>;
```

### 4.2 `DesktopWrapper` field addition

**File:** `crates/computeruse-mcp-agent/src/utils.rs`

```rust
// After line 420 (clustered_bounds), before #[cfg(target_os = "windows")]:
#[serde(skip)]
pub ref_state: Arc<crate::ref_snapshot::RefSnapshotState>,
```

Initialized in `server.rs` `new_with_log_capture` (after line 817) by passing clones of the five existing cache `Arc`s.

### 4.3 `ClickElementArgs` field addition

**File:** `crates/computeruse-mcp-agent/src/utils.rs`

```rust
// In struct ClickElementArgs, after vision_type (line 689):

#[schemars(
    description = "Prefixed ref from snapshot output (e.g. 'u5' for UIA element 5, \
                   'o3' for OCR word 3, 'd12' for DOM element 12). Prefixes: \
                   u=ui_tree, o=ocr, d=dom, p=omniparser, g=gemini. \
                   Mutually exclusive with index+vision_type. Preferred over index."
)]
#[serde(rename = "ref")]
pub ref_: Option<String>,

#[schemars(
    description = "The snapshot_id returned by get_window_tree. If provided, \
                   the click is rejected if a newer snapshot has been taken. \
                   Strongly recommended when using ref or index."
)]
pub snapshot_id: Option<u64>,
```

`determine_mode()` updated: `ref_.is_some()` counts as `has_index`. (Both lead to `ClickMode::Index` — the dispatch happens inside the Index arm.)

### 4.4 `click_element` Index arm refactor

**File:** `crates/computeruse-mcp-agent/src/server.rs` (around line 2605)

Pseudocode (actual code keeps existing tracing/highlighting/screenshot machinery):

```rust
ClickMode::Index => {
    // 1. Resolve ref (new path) OR fall back to index+vision_type (old path)
    let resolution = if let Some(ref_str) = args.ref_.as_deref() {
        self.ref_state.resolve(ref_str, args.snapshot_id)
    } else {
        // Legacy path: synthesize prefixed ref from index + vision_type
        let idx = args.index.unwrap();
        let prefix = vision_type_to_prefix(args.get_vision_type());
        self.ref_state.resolve(&format!("{prefix}{idx}"), args.snapshot_id)
    };

    // 2. Handle resolution outcome
    let (label, bounds, maybe_selector) = match resolution {
        RefResolution::Stale { agent_snapshot, current_snapshot, .. } => {
            return Err(McpError::invalid_request(
                format!(
                    "Stale ref: snapshot_id {} is outdated (current: {}). \
                     Call get_window_tree again before clicking.",
                    agent_snapshot, current_snapshot
                ),
                Some(json!({ "agent_snapshot": agent_snapshot, "current": current_snapshot })),
            ));
        }
        RefResolution::Malformed { ref_str } => { /* error */ }
        RefResolution::NotFound { ref_str, source } => { /* error */ }
        RefResolution::Found { label, bounds, selector, .. } => (label, bounds, selector),
    };

    // 3. NEW: Try selector re-resolution for fresh bounds (UIA only)
    let final_bounds = if let Some(sel) = maybe_selector {
        match self.desktop.locator(&sel)
            .first(Some(Duration::from_millis(2000)))
            .await
            .and_then(|el| el.bounds())
        {
            Ok(fresh) => {
                tracing::info!("Re-resolved {} via selector: cached={:?} fresh={:?}", ref_str, bounds, fresh);
                fresh
            }
            Err(e) => {
                tracing::warn!("Selector re-resolve failed ({}), using cached bounds", e);
                bounds  // fallback
            }
        }
    } else {
        bounds
    };

    // 4. Click (existing code, just uses final_bounds)
    let click_x = final_bounds.0 + final_bounds.2 / 2.0;
    let click_y = final_bounds.1 + final_bounds.3 / 2.0;
    self.desktop.click_at_coordinates_with_type(click_x, click_y, ...)
    // ... existing response building, but include "ref", "snapshot_id",
    //     "resolved_via": "selector" | "cached_bounds"
}
```

### 4.5 Snapshot output: emit `snapshot_id`

**File:** `crates/computeruse-mcp-agent/src/server.rs` (in `get_window_tree`, before result is returned)

```rust
// At the top of get_window_tree, after window prep but before any cache writes:
let snapshot_id = self.ref_state.bump();
result_json["snapshot_id"] = json!(snapshot_id);
```

The `#uN` / `#oN` etc. format is already emitted by `ClusteredYaml` mode. For `CompactYaml`, the existing `#N` integer indices remain unchanged — agents using the legacy `index` field continue to work.

---

## 5. Test Plan

All tests in `ref_snapshot.rs` under `#[cfg(test)] mod tests`. Pattern follows existing [tree_formatter.rs:553-649](../crates/computeruse-mcp-agent/src/tree_formatter.rs#L553).

| Test | Asserts |
|---|---|
| `parse_ref_valid_prefixes` | `parse_ref("u5")` → `Some((Uia, 5))`, same for `o`/`d`/`p`/`g` |
| `parse_ref_strips_hash` | `parse_ref("#u5")` → `Some((Uia, 5))` |
| `parse_ref_rejects_unknown_prefix` | `parse_ref("x5")` → `None` |
| `parse_ref_rejects_no_number` | `parse_ref("u")` → `None`, `parse_ref("u-1")` → `None` |
| `parse_ref_rejects_empty` | `parse_ref("")` → `None`, `parse_ref("#")` → `None` |
| `bump_is_monotonic` | `bump()` returns 1, 2, 3 in sequence; `current()` doesn't increment |
| `resolve_uia_found_includes_selector` | UIA cache hit → `Found { selector: Some(...) }` |
| `resolve_ocr_found_no_selector` | OCR cache hit → `Found { selector: None }` |
| `resolve_not_found` | `resolve("u999", None)` with empty cache → `NotFound` |
| `resolve_stale_rejects` | `bump()` to 5 → `resolve("u1", Some(3))` → `Stale { agent: 3, current: 5 }` |
| `resolve_stale_accepts_current` | `bump()` to 5 → `resolve("u1", Some(5))` with populated cache → `Found` |
| `resolve_no_snapshot_id_skips_check` | `resolve("u1", None)` with populated cache → `Found` (lenient mode) |
| `resolve_malformed` | `resolve("hello", None)` → `Malformed` |

No tests require `Desktop` — `RefSnapshotState::new` accepts pre-populated `Arc<Mutex<HashMap>>` fixtures.

---

## 6. Out of Scope (this PR)

| Item | Tracked under |
|---|---|
| `type_into_element` ref support | Issue #2 follow-up — same pattern, separate PR to keep this reviewable |
| `INTERACTIVE_ROLES` filtering | Issue #2 follow-up — behavior change |
| Compaction / tree pruning | Issue #2 follow-up — output optimization |
| Atomic multi-cache swap | Issue #4 |
| `clustered_bounds` cache deprecation | Issue #4 — once `RefSnapshotState` is the source of truth, the orphan can go |

---

## 7. Migration

| Caller | Change required |
|---|---|
| Existing agents using `{index: 5, vision_type: "ui_tree"}` | None — still works |
| Existing agents using `ClusteredYaml` output | Can now actually click `#u5` via `{ref: "u5"}` |
| New agents | Use `{ref: "u5", snapshot_id: <from get_window_tree>}` |

The MCP tool description for `click_element` is updated to recommend `ref` over `index`.
