//! Ref-based snapshot resolution.
//!
//! Owns the monotonic snapshot generation counter and resolves prefixed
//! refs (`u5`, `o3`, `d12`, `p7`, `g2`) against the per-modality bounds
//! caches that already live on `DesktopWrapper`.
//!
//! See `docs/ISSUE2_REF_SNAPSHOTS.md` for the full design rationale and
//! comparison with OpenClaw's `[ref=eN]` model.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use computeruse::ElementSource;

use crate::omniparser::OmniparserItem;
use crate::vision::VisionElement;

/// Bounds tuple used throughout the MCP agent: `(x, y, width, height)`.
pub type Bounds = (f64, f64, f64, f64);

/// Cache type aliases — must match the field types on `DesktopWrapper`.
pub type UiaCache = Arc<Mutex<HashMap<u32, (String, String, Bounds, Option<String>)>>>;
pub type OcrCache = Arc<Mutex<HashMap<u32, (String, Bounds)>>>;
pub type DomCache = Arc<Mutex<HashMap<u32, (String, String, Bounds)>>>;
pub type OmniparserCache = Arc<Mutex<HashMap<u32, OmniparserItem>>>;
pub type VisionCache = Arc<Mutex<HashMap<u32, VisionElement>>>;

/// Result of resolving a ref against the current snapshot generation.
#[derive(Debug, Clone, PartialEq)]
pub enum RefResolution {
    /// Ref found. `selector` is `Some` only for UIA refs; when present,
    /// callers should attempt `desktop.locator(selector)` for fresh
    /// bounds before falling back to `bounds`.
    Found {
        source: ElementSource,
        label: String,
        bounds: Bounds,
        selector: Option<String>,
    },
    /// Ref string parsed but the index is not in the corresponding cache.
    NotFound {
        ref_str: String,
        source: ElementSource,
    },
    /// Ref string did not parse (unknown prefix, no number, etc.).
    Malformed { ref_str: String },
    /// Agent supplied a `snapshot_id` that no longer matches the current
    /// generation — a newer snapshot has invalidated the indices.
    Stale {
        ref_str: String,
        agent_snapshot: u64,
        current_snapshot: u64,
    },
    /// Agent's `snapshot_id` is current, but the cache for this ref's
    /// source was populated at a different generation (or never).
    /// This is the cross-modal staleness gap: e.g. omniparser was last
    /// refreshed at gen=1, agent passes snapshot_id=2 from a later
    /// `get_window_tree(include_ocr=true)` call. `cache_snapshot == 0`
    /// means this modality has never been included.
    CrossModalStale {
        ref_str: String,
        source: ElementSource,
        claimed_snapshot: u64,
        cache_snapshot: u64,
    },
}

/// Per-cache generation tracking. Each `get_window_tree` call only
/// refreshes the caches whose `include_*` flag was set; this records
/// which generation each cache was last written at so `resolve()` can
/// reject refs whose modality wasn't part of the claimed snapshot.
///
/// All-zero on construction → "never populated".
#[derive(Default)]
struct CacheGenerations {
    uia: AtomicU64,
    ocr: AtomicU64,
    dom: AtomicU64,
    omniparser: AtomicU64,
    vision: AtomicU64,
}

impl CacheGenerations {
    fn get(&self, src: ElementSource) -> u64 {
        match src {
            ElementSource::Uia => self.uia.load(Ordering::SeqCst),
            ElementSource::Ocr => self.ocr.load(Ordering::SeqCst),
            ElementSource::Dom => self.dom.load(Ordering::SeqCst),
            ElementSource::Omniparser => self.omniparser.load(Ordering::SeqCst),
            ElementSource::Gemini => self.vision.load(Ordering::SeqCst),
        }
    }

    fn set(&self, src: ElementSource, gen: u64) {
        match src {
            ElementSource::Uia => self.uia.store(gen, Ordering::SeqCst),
            ElementSource::Ocr => self.ocr.store(gen, Ordering::SeqCst),
            ElementSource::Dom => self.dom.store(gen, Ordering::SeqCst),
            ElementSource::Omniparser => self.omniparser.store(gen, Ordering::SeqCst),
            ElementSource::Gemini => self.vision.store(gen, Ordering::SeqCst),
        }
    }
}

/// Owns the snapshot generation counter and provides staleness-checked
/// ref resolution. Holds `Arc` clones of the caches — the caches
/// themselves continue to live on `DesktopWrapper` and are populated by
/// `get_window_tree` exactly as before.
pub struct RefSnapshotState {
    generation: AtomicU64,
    /// Which generation last wrote each cache. Invariant:
    /// `cache_gens.get(src) <= generation` for every source.
    cache_gens: CacheGenerations,
    uia: UiaCache,
    ocr: OcrCache,
    dom: DomCache,
    omniparser: OmniparserCache,
    vision: VisionCache,
}

impl Default for RefSnapshotState {
    /// Empty state with fresh caches. Required by `DesktopWrapper`'s
    /// `#[derive(Deserialize)]` for the `#[serde(skip)]` field — never
    /// used in practice (`DesktopWrapper::default()` panics).
    fn default() -> Self {
        Self::new(
            Arc::new(Mutex::new(HashMap::new())),
            Arc::new(Mutex::new(HashMap::new())),
            Arc::new(Mutex::new(HashMap::new())),
            Arc::new(Mutex::new(HashMap::new())),
            Arc::new(Mutex::new(HashMap::new())),
        )
    }
}

impl RefSnapshotState {
    pub fn new(
        uia: UiaCache,
        ocr: OcrCache,
        dom: DomCache,
        omniparser: OmniparserCache,
        vision: VisionCache,
    ) -> Self {
        Self {
            generation: AtomicU64::new(0),
            cache_gens: CacheGenerations::default(),
            uia,
            ocr,
            dom,
            omniparser,
            vision,
        }
    }

    /// Record that a cache was just populated at `gen`. Call this
    /// immediately after `*cache.lock() = new_data` in `get_window_tree`,
    /// passing the same `snapshot_id` returned by `bump()`.
    pub fn mark_populated(&self, source: ElementSource, gen: u64) {
        self.cache_gens.set(source, gen);
    }

    /// Generation at which `source`'s cache was last populated.
    /// `0` means never. Exposed for diagnostics/error messages.
    pub fn cache_generation(&self, source: ElementSource) -> u64 {
        self.cache_gens.get(source)
    }

    /// Increment the generation counter and return the new value.
    /// Call once per `get_window_tree` (or any tool that repopulates a
    /// cache). Returned to the agent as `snapshot_id`.
    pub fn bump(&self) -> u64 {
        // fetch_add returns the *previous* value; +1 gives the new generation.
        self.generation.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Current generation without incrementing.
    pub fn current(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// Resolve a prefixed ref. `claimed_snapshot` is the `snapshot_id`
    /// the agent says it's working from — `None` skips the staleness
    /// check (lenient mode for backward compat).
    pub fn resolve(&self, ref_str: &str, claimed_snapshot: Option<u64>) -> RefResolution {
        // 1. Staleness check first — if the agent's view is stale,
        //    don't even bother parsing.
        if let Some(claimed) = claimed_snapshot {
            let current = self.current();
            if claimed != current {
                return RefResolution::Stale {
                    ref_str: ref_str.to_string(),
                    agent_snapshot: claimed,
                    current_snapshot: current,
                };
            }
        }

        // 2. Parse the ref.
        let Some((source, index)) = parse_ref(ref_str) else {
            return RefResolution::Malformed {
                ref_str: ref_str.to_string(),
            };
        };

        // 3. Per-cache staleness check (Issue #4 — cross-modal coherence).
        //    The global check above guarantees `claimed == current()`.
        //    Since `cache_gen <= current()` is invariant, the only
        //    mismatch here is `cache_gen < claimed` — i.e. this modality
        //    was not refreshed in the snapshot the agent is citing.
        if let Some(claimed) = claimed_snapshot {
            let cache_gen = self.cache_gens.get(source);
            if cache_gen != claimed {
                return RefResolution::CrossModalStale {
                    ref_str: ref_str.to_string(),
                    source,
                    claimed_snapshot: claimed,
                    cache_snapshot: cache_gen,
                };
            }
        }

        // 4. Route to the right cache. Each arm normalises the cached
        //    value into (label, bounds, selector).
        match source {
            ElementSource::Uia => self.resolve_uia(ref_str, source, index),
            ElementSource::Ocr => self.resolve_ocr(ref_str, source, index),
            ElementSource::Dom => self.resolve_dom(ref_str, source, index),
            ElementSource::Omniparser => self.resolve_omniparser(ref_str, source, index),
            ElementSource::Gemini => self.resolve_vision(ref_str, source, index),
        }
    }

    fn resolve_uia(&self, ref_str: &str, source: ElementSource, index: u32) -> RefResolution {
        let entry = self.uia.lock().ok().and_then(|g| g.get(&index).cloned());
        match entry {
            Some((role, name, bounds, selector)) => RefResolution::Found {
                source,
                label: if name.is_empty() {
                    role
                } else {
                    format!("{role}: {name}")
                },
                bounds,
                selector,
            },
            None => RefResolution::NotFound {
                ref_str: ref_str.to_string(),
                source,
            },
        }
    }

    fn resolve_ocr(&self, ref_str: &str, source: ElementSource, index: u32) -> RefResolution {
        let entry = self.ocr.lock().ok().and_then(|g| g.get(&index).cloned());
        match entry {
            Some((text, bounds)) => RefResolution::Found {
                source,
                label: text,
                bounds,
                selector: None,
            },
            None => RefResolution::NotFound {
                ref_str: ref_str.to_string(),
                source,
            },
        }
    }

    fn resolve_dom(&self, ref_str: &str, source: ElementSource, index: u32) -> RefResolution {
        let entry = self.dom.lock().ok().and_then(|g| g.get(&index).cloned());
        match entry {
            Some((tag, identifier, bounds)) => RefResolution::Found {
                source,
                label: if identifier.is_empty() {
                    tag
                } else {
                    format!("{tag}: {identifier}")
                },
                bounds,
                selector: None,
            },
            None => RefResolution::NotFound {
                ref_str: ref_str.to_string(),
                source,
            },
        }
    }

    fn resolve_omniparser(
        &self,
        ref_str: &str,
        source: ElementSource,
        index: u32,
    ) -> RefResolution {
        let entry = self
            .omniparser
            .lock()
            .ok()
            .and_then(|g| g.get(&index).cloned());
        match entry {
            Some(item) => match item.box_2d {
                // box_2d is [x_min, y_min, x_max, y_max]; convert to (x, y, w, h).
                Some([x1, y1, x2, y2]) => RefResolution::Found {
                    source,
                    label: item.label,
                    bounds: (x1, y1, x2 - x1, y2 - y1),
                    selector: None,
                },
                None => RefResolution::NotFound {
                    ref_str: ref_str.to_string(),
                    source,
                },
            },
            None => RefResolution::NotFound {
                ref_str: ref_str.to_string(),
                source,
            },
        }
    }

    fn resolve_vision(&self, ref_str: &str, source: ElementSource, index: u32) -> RefResolution {
        let entry = self
            .vision
            .lock()
            .ok()
            .and_then(|g| g.get(&index).cloned());
        match entry {
            Some(item) => match item.box_2d {
                Some([x1, y1, x2, y2]) => RefResolution::Found {
                    source,
                    label: item.element_type,
                    bounds: (x1, y1, x2 - x1, y2 - y1),
                    selector: None,
                },
                None => RefResolution::NotFound {
                    ref_str: ref_str.to_string(),
                    source,
                },
            },
            None => RefResolution::NotFound {
                ref_str: ref_str.to_string(),
                source,
            },
        }
    }
}

/// Parse a prefixed ref string into `(source, index)`.
///
/// Wraps `ElementSource::parse_prefixed_index` from the core crate but
/// additionally accepts a leading `#` (so agents can copy-paste `#u5`
/// directly from the YAML output) and trims whitespace.
///
/// | Input    | Output             |
/// |----------|--------------------|
/// | `"u5"`   | `Some((Uia, 5))`   |
/// | `"#u5"`  | `Some((Uia, 5))`   |
/// | `" o3 "` | `Some((Ocr, 3))`   |
/// | `"x5"`   | `None`             |
/// | `"u"`    | `None`             |
/// | `"5"`    | `None`             |
/// | `""`     | `None`             |
pub fn parse_ref(s: &str) -> Option<(ElementSource, u32)> {
    let trimmed = s.trim().strip_prefix('#').unwrap_or_else(|| s.trim());
    ElementSource::parse_prefixed_index(trimmed)
}

/// Map a `VisionType` to its `ElementSource` prefix character.
/// Used by the legacy `index` + `vision_type` click path to synthesise
/// a prefixed ref so both paths converge on `RefSnapshotState::resolve`.
pub fn vision_type_prefix(vt: crate::utils::VisionType) -> char {
    use crate::utils::VisionType;
    match vt {
        VisionType::UiTree => ElementSource::Uia.prefix(),
        VisionType::Ocr => ElementSource::Ocr.prefix(),
        VisionType::Dom => ElementSource::Dom.prefix(),
        VisionType::Omniparser => ElementSource::Omniparser.prefix(),
        VisionType::Gemini => ElementSource::Gemini.prefix(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_caches() -> (UiaCache, OcrCache, DomCache, OmniparserCache, VisionCache) {
        (
            Arc::new(Mutex::new(HashMap::new())),
            Arc::new(Mutex::new(HashMap::new())),
            Arc::new(Mutex::new(HashMap::new())),
            Arc::new(Mutex::new(HashMap::new())),
            Arc::new(Mutex::new(HashMap::new())),
        )
    }

    fn make_state() -> RefSnapshotState {
        let (uia, ocr, dom, omni, vis) = empty_caches();
        RefSnapshotState::new(uia, ocr, dom, omni, vis)
    }

    // ─── parse_ref ────────────────────────────────────────────────────

    #[test]
    fn parse_ref_valid_prefixes() {
        assert_eq!(parse_ref("u5"), Some((ElementSource::Uia, 5)));
        assert_eq!(parse_ref("o3"), Some((ElementSource::Ocr, 3)));
        assert_eq!(parse_ref("d12"), Some((ElementSource::Dom, 12)));
        assert_eq!(parse_ref("p7"), Some((ElementSource::Omniparser, 7)));
        assert_eq!(parse_ref("g2"), Some((ElementSource::Gemini, 2)));
    }

    #[test]
    fn parse_ref_strips_hash() {
        assert_eq!(parse_ref("#u5"), Some((ElementSource::Uia, 5)));
        assert_eq!(parse_ref("#d999"), Some((ElementSource::Dom, 999)));
    }

    #[test]
    fn parse_ref_trims_whitespace() {
        assert_eq!(parse_ref("  u5  "), Some((ElementSource::Uia, 5)));
        assert_eq!(parse_ref(" #o1 "), Some((ElementSource::Ocr, 1)));
    }

    #[test]
    fn parse_ref_rejects_unknown_prefix() {
        assert_eq!(parse_ref("x5"), None);
        assert_eq!(parse_ref("a1"), None);
        assert_eq!(parse_ref("U5"), None); // case-sensitive
    }

    #[test]
    fn parse_ref_rejects_no_number() {
        assert_eq!(parse_ref("u"), None);
        assert_eq!(parse_ref("u-1"), None);
        assert_eq!(parse_ref("uabc"), None);
    }

    #[test]
    fn parse_ref_rejects_empty() {
        assert_eq!(parse_ref(""), None);
        assert_eq!(parse_ref("#"), None);
        assert_eq!(parse_ref("   "), None);
    }

    #[test]
    fn parse_ref_rejects_bare_number() {
        // Bare integers are the legacy `index` field's domain, not refs.
        assert_eq!(parse_ref("5"), None);
        assert_eq!(parse_ref("#5"), None);
    }

    // ─── generation counter ───────────────────────────────────────────

    #[test]
    fn bump_is_monotonic() {
        let state = make_state();
        assert_eq!(state.current(), 0);
        assert_eq!(state.bump(), 1);
        assert_eq!(state.bump(), 2);
        assert_eq!(state.bump(), 3);
        assert_eq!(state.current(), 3);
    }

    #[test]
    fn current_does_not_increment() {
        let state = make_state();
        state.bump();
        assert_eq!(state.current(), 1);
        assert_eq!(state.current(), 1);
        assert_eq!(state.current(), 1);
    }

    // ─── resolve: UIA ─────────────────────────────────────────────────

    #[test]
    fn resolve_uia_found_includes_selector() {
        let (uia, ocr, dom, omni, vis) = empty_caches();
        uia.lock().unwrap().insert(
            5,
            (
                "Button".into(),
                "Save".into(),
                (10.0, 20.0, 100.0, 30.0),
                Some("role:Button|name:Save".into()),
            ),
        );
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);

        let res = state.resolve("u5", None);
        assert_eq!(
            res,
            RefResolution::Found {
                source: ElementSource::Uia,
                label: "Button: Save".into(),
                bounds: (10.0, 20.0, 100.0, 30.0),
                selector: Some("role:Button|name:Save".into()),
            }
        );
    }

    #[test]
    fn resolve_uia_empty_name_uses_role_only() {
        let (uia, ocr, dom, omni, vis) = empty_caches();
        uia.lock().unwrap().insert(
            1,
            ("Pane".into(), "".into(), (0.0, 0.0, 50.0, 50.0), None),
        );
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);

        match state.resolve("u1", None) {
            RefResolution::Found {
                label, selector, ..
            } => {
                assert_eq!(label, "Pane");
                assert_eq!(selector, None);
            }
            other => panic!("expected Found, got {other:?}"),
        }
    }

    // ─── resolve: OCR ─────────────────────────────────────────────────

    #[test]
    fn resolve_ocr_found_no_selector() {
        let (uia, ocr, dom, omni, vis) = empty_caches();
        ocr.lock()
            .unwrap()
            .insert(3, ("hello".into(), (5.0, 6.0, 7.0, 8.0)));
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);

        assert_eq!(
            state.resolve("o3", None),
            RefResolution::Found {
                source: ElementSource::Ocr,
                label: "hello".into(),
                bounds: (5.0, 6.0, 7.0, 8.0),
                selector: None,
            }
        );
    }

    // ─── resolve: DOM ─────────────────────────────────────────────────

    #[test]
    fn resolve_dom_found() {
        let (uia, ocr, dom, omni, vis) = empty_caches();
        dom.lock().unwrap().insert(
            12,
            ("button".into(), "submit-btn".into(), (1.0, 2.0, 3.0, 4.0)),
        );
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);

        assert_eq!(
            state.resolve("d12", None),
            RefResolution::Found {
                source: ElementSource::Dom,
                label: "button: submit-btn".into(),
                bounds: (1.0, 2.0, 3.0, 4.0),
                selector: None,
            }
        );
    }

    // ─── resolve: Omniparser (box_2d → bounds conversion) ─────────────

    #[test]
    fn resolve_omniparser_converts_box2d() {
        let (uia, ocr, dom, omni, vis) = empty_caches();
        omni.lock().unwrap().insert(
            7,
            OmniparserItem {
                label: "icon".into(),
                content: Some("settings".into()),
                box_2d: Some([100.0, 200.0, 150.0, 230.0]), // x1,y1,x2,y2
            },
        );
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);

        // Expect (x, y, w, h) = (100, 200, 50, 30)
        assert_eq!(
            state.resolve("p7", None),
            RefResolution::Found {
                source: ElementSource::Omniparser,
                label: "icon".into(),
                bounds: (100.0, 200.0, 50.0, 30.0),
                selector: None,
            }
        );
    }

    #[test]
    fn resolve_omniparser_no_box2d_is_not_found() {
        let (uia, ocr, dom, omni, vis) = empty_caches();
        omni.lock().unwrap().insert(
            1,
            OmniparserItem {
                label: "text".into(),
                content: None,
                box_2d: None,
            },
        );
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);

        assert_eq!(
            state.resolve("p1", None),
            RefResolution::NotFound {
                ref_str: "p1".into(),
                source: ElementSource::Omniparser,
            }
        );
    }

    // ─── resolve: Vision ──────────────────────────────────────────────

    #[test]
    fn resolve_vision_converts_box2d() {
        let (uia, ocr, dom, omni, vis) = empty_caches();
        vis.lock().unwrap().insert(
            2,
            VisionElement {
                element_type: "button".into(),
                content: Some("Submit".into()),
                description: None,
                box_2d: Some([0.0, 0.0, 80.0, 40.0]),
                interactivity: Some(true),
            },
        );
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);

        assert_eq!(
            state.resolve("g2", None),
            RefResolution::Found {
                source: ElementSource::Gemini,
                label: "button".into(),
                bounds: (0.0, 0.0, 80.0, 40.0),
                selector: None,
            }
        );
    }

    // ─── resolve: not found ───────────────────────────────────────────

    #[test]
    fn resolve_not_found() {
        let state = make_state();
        assert_eq!(
            state.resolve("u999", None),
            RefResolution::NotFound {
                ref_str: "u999".into(),
                source: ElementSource::Uia,
            }
        );
    }

    #[test]
    fn resolve_malformed() {
        let state = make_state();
        assert_eq!(
            state.resolve("hello", None),
            RefResolution::Malformed {
                ref_str: "hello".into(),
            }
        );
        assert_eq!(
            state.resolve("", None),
            RefResolution::Malformed {
                ref_str: "".into(),
            }
        );
    }

    // ─── staleness ────────────────────────────────────────────────────

    #[test]
    fn resolve_stale_rejects_old_snapshot() {
        let state = make_state();
        for _ in 0..5 {
            state.bump();
        }
        // current() == 5, agent claims 3
        assert_eq!(
            state.resolve("u1", Some(3)),
            RefResolution::Stale {
                ref_str: "u1".into(),
                agent_snapshot: 3,
                current_snapshot: 5,
            }
        );
    }

    #[test]
    fn resolve_stale_rejects_future_snapshot() {
        // Defends against agent confusion / replay across server restart.
        let state = make_state();
        state.bump(); // current == 1
        assert_eq!(
            state.resolve("u1", Some(99)),
            RefResolution::Stale {
                ref_str: "u1".into(),
                agent_snapshot: 99,
                current_snapshot: 1,
            }
        );
    }

    #[test]
    fn resolve_stale_accepts_current() {
        let (uia, ocr, dom, omni, vis) = empty_caches();
        uia.lock()
            .unwrap()
            .insert(1, ("Button".into(), "OK".into(), (0.0, 0.0, 1.0, 1.0), None));
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);
        let snap = state.bump();
        state.mark_populated(ElementSource::Uia, snap);

        match state.resolve("u1", Some(snap)) {
            RefResolution::Found { .. } => {}
            other => panic!("expected Found, got {other:?}"),
        }
    }

    #[test]
    fn resolve_no_snapshot_id_skips_check() {
        // Lenient mode: agent didn't pass snapshot_id → no staleness check.
        let (uia, ocr, dom, omni, vis) = empty_caches();
        uia.lock()
            .unwrap()
            .insert(1, ("Button".into(), "OK".into(), (0.0, 0.0, 1.0, 1.0), None));
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);
        // Bump well past 0 — agent's lack of snapshot_id should still resolve.
        for _ in 0..10 {
            state.bump();
        }

        match state.resolve("u1", None) {
            RefResolution::Found { .. } => {}
            other => panic!("expected Found, got {other:?}"),
        }
    }

    #[test]
    fn resolve_stale_short_circuits_before_lookup() {
        // Stale check happens before cache lookup — even a valid ref
        // returns Stale, not Found.
        let (uia, ocr, dom, omni, vis) = empty_caches();
        uia.lock()
            .unwrap()
            .insert(1, ("Button".into(), "OK".into(), (0.0, 0.0, 1.0, 1.0), None));
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);
        state.bump();
        state.bump(); // current == 2

        match state.resolve("u1", Some(1)) {
            RefResolution::Stale { .. } => {}
            other => panic!("expected Stale (not Found), got {other:?}"),
        }
    }

    // ─── cross-modal staleness (Issue #4) ─────────────────────────────

    #[test]
    fn cross_modal_stale_when_cache_not_in_snapshot() {
        // The exact failure mode #4 fixes:
        //   gen=1: get_window_tree(include_omniparser=true) → omniparser populated
        //   gen=2: get_window_tree(include_ocr=true)        → ocr populated
        //   agent: click_element(ref="p5", snapshot_id=2)
        //   → must reject: snapshot 2 didn't refresh omniparser.
        let (uia, ocr, dom, omni, vis) = empty_caches();
        omni.lock().unwrap().insert(
            5,
            OmniparserItem {
                label: "stale-icon".into(),
                content: None,
                box_2d: Some([0.0, 0.0, 10.0, 10.0]),
            },
        );
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);

        let g1 = state.bump();
        state.mark_populated(ElementSource::Omniparser, g1);
        let g2 = state.bump();
        state.mark_populated(ElementSource::Ocr, g2);

        // Global check passes (g2 == current), per-cache check fails.
        assert_eq!(
            state.resolve("p5", Some(g2)),
            RefResolution::CrossModalStale {
                ref_str: "p5".into(),
                source: ElementSource::Omniparser,
                claimed_snapshot: g2,
                cache_snapshot: g1,
            }
        );
    }

    #[test]
    fn cross_modal_never_populated_is_zero() {
        // `cache_snapshot == 0` distinguishes "never included" from
        // "included at an old generation" — server.rs uses this to
        // tailor the error message.
        let state = make_state();
        let g1 = state.bump();
        state.mark_populated(ElementSource::Uia, g1);

        assert_eq!(
            state.resolve("g3", Some(g1)),
            RefResolution::CrossModalStale {
                ref_str: "g3".into(),
                source: ElementSource::Gemini,
                claimed_snapshot: g1,
                cache_snapshot: 0,
            }
        );
    }

    #[test]
    fn cross_modal_passes_when_cache_in_snapshot() {
        // Multiple modalities populated at the same generation —
        // refs into any of them resolve.
        let (uia, ocr, dom, omni, vis) = empty_caches();
        uia.lock()
            .unwrap()
            .insert(1, ("Button".into(), "OK".into(), (0.0, 0.0, 1.0, 1.0), None));
        ocr.lock()
            .unwrap()
            .insert(2, ("hello".into(), (5.0, 5.0, 10.0, 10.0)));
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);

        let g = state.bump();
        state.mark_populated(ElementSource::Uia, g);
        state.mark_populated(ElementSource::Ocr, g);

        assert!(matches!(
            state.resolve("u1", Some(g)),
            RefResolution::Found { .. }
        ));
        assert!(matches!(
            state.resolve("o2", Some(g)),
            RefResolution::Found { .. }
        ));
    }

    #[test]
    fn cross_modal_check_skipped_without_snapshot_id() {
        // Lenient mode: no snapshot_id → no per-cache check either.
        // Backward-compat for legacy {index, vision_type} callers that
        // don't supply snapshot_id.
        let (uia, ocr, dom, omni, vis) = empty_caches();
        omni.lock().unwrap().insert(
            5,
            OmniparserItem {
                label: "icon".into(),
                content: None,
                box_2d: Some([0.0, 0.0, 10.0, 10.0]),
            },
        );
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);

        // Don't mark_populated at all — cache_gen[Omniparser] stays 0.
        // Without snapshot_id the check is skipped, so this still resolves.
        assert!(matches!(
            state.resolve("p5", None),
            RefResolution::Found { .. }
        ));
    }

    #[test]
    fn cross_modal_runs_after_global_stale() {
        // If the global generation is stale, that error takes
        // precedence — no point checking per-cache.
        let state = make_state();
        let g1 = state.bump();
        state.mark_populated(ElementSource::Uia, g1);
        state.bump(); // current == 2

        // Agent passes g1 with a u-ref. Global check fails first.
        match state.resolve("u1", Some(g1)) {
            RefResolution::Stale { .. } => {}
            other => panic!("expected global Stale, got {other:?}"),
        }
    }

    #[test]
    fn cross_modal_runs_before_cache_lookup() {
        // Like resolve_stale_short_circuits_before_lookup but for the
        // per-cache check: even if the index exists in the cache,
        // CrossModalStale fires before Found.
        let (uia, ocr, dom, omni, vis) = empty_caches();
        ocr.lock()
            .unwrap()
            .insert(3, ("text".into(), (0.0, 0.0, 1.0, 1.0)));
        let state = RefSnapshotState::new(uia, ocr, dom, omni, vis);

        let g1 = state.bump();
        state.mark_populated(ElementSource::Ocr, g1);
        let g2 = state.bump();
        state.mark_populated(ElementSource::Uia, g2); // ocr NOT marked at g2

        // o3 exists in cache but its generation is g1, agent claims g2.
        match state.resolve("o3", Some(g2)) {
            RefResolution::CrossModalStale { .. } => {}
            other => panic!("expected CrossModalStale (not Found), got {other:?}"),
        }
    }

    #[test]
    fn mark_populated_overwrites() {
        // Same modality refreshed across generations — latest wins.
        let state = make_state();
        let g1 = state.bump();
        state.mark_populated(ElementSource::Dom, g1);
        let g2 = state.bump();
        state.mark_populated(ElementSource::Dom, g2);

        assert_eq!(state.cache_generation(ElementSource::Dom), g2);
        // Resolving at g2 with a dom ref passes the per-cache check
        // (cache lookup will then return NotFound since cache is empty,
        // but the staleness check itself passed).
        assert!(matches!(
            state.resolve("d1", Some(g2)),
            RefResolution::NotFound { .. }
        ));
    }

    #[test]
    fn cache_generation_initial_zero() {
        let state = make_state();
        for src in [
            ElementSource::Uia,
            ElementSource::Ocr,
            ElementSource::Dom,
            ElementSource::Omniparser,
            ElementSource::Gemini,
        ] {
            assert_eq!(state.cache_generation(src), 0, "src={src:?}");
        }
    }

    // ─── shared Arc semantics ─────────────────────────────────────────

    #[test]
    fn caches_are_shared_not_copied() {
        // Critical: RefSnapshotState must see writes made via the
        // original Arc handles (the ones DesktopWrapper holds).
        let (uia, ocr, dom, omni, vis) = empty_caches();
        let state = RefSnapshotState::new(
            Arc::clone(&uia),
            Arc::clone(&ocr),
            Arc::clone(&dom),
            Arc::clone(&omni),
            Arc::clone(&vis),
        );

        // Write via the original handle AFTER state is constructed.
        uia.lock()
            .unwrap()
            .insert(42, ("Window".into(), "".into(), (0.0, 0.0, 10.0, 10.0), None));

        match state.resolve("u42", None) {
            RefResolution::Found { label, .. } => assert_eq!(label, "Window"),
            other => panic!("expected Found, got {other:?}"),
        }
    }

    // ─── vision_type_prefix ───────────────────────────────────────────

    #[test]
    fn vision_type_prefix_round_trips() {
        use crate::utils::VisionType;
        // Verify the legacy-path bridge: VisionType → prefix → ElementSource
        // must land on the matching cache.
        for (vt, expected_source) in [
            (VisionType::UiTree, ElementSource::Uia),
            (VisionType::Ocr, ElementSource::Ocr),
            (VisionType::Dom, ElementSource::Dom),
            (VisionType::Omniparser, ElementSource::Omniparser),
            (VisionType::Gemini, ElementSource::Gemini),
        ] {
            let prefix = vision_type_prefix(vt);
            let synthesized = format!("{prefix}1");
            let (parsed_source, parsed_idx) = parse_ref(&synthesized).unwrap();
            assert_eq!(parsed_source, expected_source);
            assert_eq!(parsed_idx, 1);
        }
    }
}
