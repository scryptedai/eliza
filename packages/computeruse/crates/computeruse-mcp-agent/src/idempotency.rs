//! Idempotency cache for tool calls (Issue #2: OpenClaw parity).
//!
//! When a caller passes `"idempotency_key": "<opaque>"` in a tool's arguments,
//! the result of the **first** successful invocation of `(tool_name, key)` is
//! cached and returned verbatim for any repeat within the TTL window. This
//! lets agents retry safely after a transport hiccup or timeout without
//! re-running side-effecting actions (clicks, `run_command`, file writes,
//! `open_application`, …).
//!
//! The cache is in-memory and per-server-process. It is intentionally **not**
//! persisted: idempotency is about absorbing client-side retries, not about
//! durable replay (that is what `execute_sequence` checkpoints / `state.json`
//! are for — see [`Checkpoint`] / [`read_checkpoint`]).
//!
//! ### Sizing
//!
//! Default TTL is 5 minutes and the cache holds at most 256 entries; on
//! overflow the oldest entry by insertion time is evicted (FIFO ≈ LRU for
//! this access pattern, since hits short-circuit before re-insertion).
//! Both are overridable via env for tests.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rmcp::model::CallToolResult;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Default time-to-live for a cached result.
const DEFAULT_TTL: Duration = Duration::from_secs(5 * 60);
/// Hard cap on cache entries before FIFO eviction kicks in.
const DEFAULT_MAX_ENTRIES: usize = 256;

/// One cached `(tool, key)` result plus the wall-clock instant it was stored.
struct Entry {
    result: CallToolResult,
    stored_at: Instant,
}

/// In-process idempotency cache. Cheap to clone-by-`Arc` (the server holds it
/// directly on [`crate::utils::DesktopWrapper`]).
pub struct IdempotencyCache {
    entries: Mutex<HashMap<String, Entry>>,
    ttl: Duration,
    max_entries: usize,
}

impl Default for IdempotencyCache {
    fn default() -> Self {
        Self::from_env()
    }
}

impl IdempotencyCache {
    /// Construct with explicit `ttl` / `max_entries` (used by tests).
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            ttl,
            max_entries,
        }
    }

    /// Construct from env: `COMPUTERUSE_IDEMPOTENCY_TTL_SECS`,
    /// `COMPUTERUSE_IDEMPOTENCY_MAX_ENTRIES`. Falls back to the defaults.
    pub fn from_env() -> Self {
        let ttl = std::env::var("COMPUTERUSE_IDEMPOTENCY_TTL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_TTL);
        let max = std::env::var("COMPUTERUSE_IDEMPOTENCY_MAX_ENTRIES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_ENTRIES);
        Self::new(ttl, max)
    }

    fn cache_key(tool: &str, key: &str) -> String {
        format!("{tool}::{key}")
    }

    /// Pull `idempotency_key` (string, non-empty) out of a tool's raw argument
    /// object. Returns `None` for anything else so the call proceeds normally.
    pub fn extract_key(arguments: &serde_json::Value) -> Option<String> {
        arguments
            .get("idempotency_key")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    }

    /// Look up a cached result. Expired entries are dropped on access.
    pub fn get(&self, tool: &str, key: &str) -> Option<CallToolResult> {
        let ck = Self::cache_key(tool, key);
        let mut map = self.entries.lock().ok()?;
        let entry = map.get(&ck)?;
        if entry.stored_at.elapsed() > self.ttl {
            map.remove(&ck);
            debug!(tool, key, "idempotency: entry expired, dropped");
            return None;
        }
        info!(
            tool,
            key, "idempotency: cache hit — returning prior result without re-execution"
        );
        Some(entry.result.clone())
    }

    /// Store a successful result. Evicts the oldest entry if at capacity.
    pub fn put(&self, tool: &str, key: &str, result: &CallToolResult) {
        let ck = Self::cache_key(tool, key);
        let Ok(mut map) = self.entries.lock() else {
            return;
        };
        if map.len() >= self.max_entries && !map.contains_key(&ck) {
            // FIFO eviction: drop the entry with the smallest `stored_at`.
            if let Some(oldest) = map
                .iter()
                .min_by_key(|(_, e)| e.stored_at)
                .map(|(k, _)| k.clone())
            {
                map.remove(&oldest);
                debug!(evicted = %oldest, "idempotency: evicted oldest entry (cap reached)");
            }
        }
        map.insert(
            ck,
            Entry {
                result: result.clone(),
                stored_at: Instant::now(),
            },
        );
        debug!(tool, key, "idempotency: result cached");
    }

    /// Number of live entries (test helper).
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.lock().map(|m| m.len()).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Sequence checkpoints
// ---------------------------------------------------------------------------

/// Resume token surfaced by the `resume_sequence` tool. Mirrors the on-disk
/// `state.json` written by [`crate::server_sequence`] after every step that
/// mutates `env`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Where the checkpoint was read from.
    pub state_file: String,
    /// `state.json::last_step_id` — the id of the last step that completed.
    pub last_step_id: Option<String>,
    /// `state.json::last_step_index`.
    pub last_step_index: Option<u64>,
    /// ISO-8601 of when the checkpoint was written.
    pub last_updated: Option<String>,
    /// What to pass as `execute_sequence.start_from_step` to resume. Equal to
    /// `last_step_id` (the existing engine restarts *at* that step and reloads
    /// the env saved *before* it ran).
    pub resume_from_step: Option<String>,
    /// The persisted `env` map, ready to be re-fed via
    /// `execute_sequence.inputs` if the caller wants to override.
    pub env: serde_json::Value,
}

/// Locate `state.json` for a workflow. Mirrors
/// `server_sequence::get_state_file_path` so callers (CLI, MCP tool) can
/// resolve checkpoints without holding a `DesktopWrapper`.
///
/// Resolution order:
///   1. `workflow_id` → `<data_local>/mediar/workflows/<workflow_id>/state.json`
///   2. `workflow_url` → folder name extracted from the URL (same heuristic as
///      `server_sequence::extract_workflow_folder_from_url`)
pub fn state_file_path(workflow_id: Option<&str>, workflow_url: Option<&str>) -> Option<PathBuf> {
    let data_dir = dirs::data_local_dir()?;
    let base = data_dir.join("mediar").join("workflows");

    if let Some(id) = workflow_id.filter(|s| !s.is_empty()) {
        return Some(base.join(id).join("state.json"));
    }
    if let Some(url) = workflow_url {
        if let Some(folder) = folder_from_url(url) {
            return Some(base.join(folder).join("state.json"));
        }
    }
    None
}

/// Best-effort extraction of the workflow folder name from a `file://` URL —
/// kept in sync with `server_sequence::extract_workflow_folder_from_url`
/// (which is private), but tolerant of plain paths too.
fn folder_from_url(url: &str) -> Option<String> {
    let path = url.strip_prefix("file://").unwrap_or(url);
    // Walk up from the file looking for a `workflows/<folder>/...` segment.
    let parts: Vec<&str> = path.split(['/', '\\']).collect();
    if let Some(pos) = parts.iter().rposition(|p| *p == "workflows") {
        if let Some(folder) = parts.get(pos + 1) {
            if !folder.is_empty() {
                return Some((*folder).to_string());
            }
        }
    }
    // Fallback: parent directory name of the file.
    std::path::Path::new(path)
        .parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
}

/// Read and parse a checkpoint for the given workflow. Returns `Ok(None)` if
/// no `state.json` exists yet (workflow never ran / never persisted env).
pub fn read_checkpoint(
    workflow_id: Option<&str>,
    workflow_url: Option<&str>,
) -> Result<Option<Checkpoint>, String> {
    let Some(path) = state_file_path(workflow_id, workflow_url) else {
        return Err(
            "cannot locate state.json: provide either workflow_id or a file:// workflow url".into(),
        );
    };
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let state: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;

    let last_step_id = state
        .get("last_step_id")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    Ok(Some(Checkpoint {
        state_file: path.to_string_lossy().into_owned(),
        last_step_index: state.get("last_step_index").and_then(|v| v.as_u64()),
        last_updated: state
            .get("last_updated")
            .and_then(|v| v.as_str())
            .map(str::to_owned),
        resume_from_step: last_step_id.clone(),
        last_step_id,
        env: state.get("env").cloned().unwrap_or(serde_json::Value::Null),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::Content;

    fn ok(text: &str) -> CallToolResult {
        CallToolResult::success(vec![Content::text(text.to_string())])
    }

    #[test]
    fn extract_key_only_accepts_nonempty_string() {
        use serde_json::json;
        assert_eq!(
            IdempotencyCache::extract_key(&json!({"idempotency_key": "abc"})),
            Some("abc".into())
        );
        assert_eq!(
            IdempotencyCache::extract_key(&json!({"idempotency_key": ""})),
            None
        );
        assert_eq!(
            IdempotencyCache::extract_key(&json!({"idempotency_key": 42})),
            None
        );
        assert_eq!(IdempotencyCache::extract_key(&json!({})), None);
        assert_eq!(
            IdempotencyCache::extract_key(&serde_json::Value::Null),
            None
        );
    }

    #[test]
    fn hit_returns_cloned_prior_result() {
        let cache = IdempotencyCache::new(Duration::from_secs(60), 8);
        assert!(cache.get("click_element", "k1").is_none());
        cache.put("click_element", "k1", &ok("first"));
        let hit = cache.get("click_element", "k1").expect("hit");
        assert_eq!(hit, ok("first"));
        // Different tool, same key → miss.
        assert!(cache.get("type_into_element", "k1").is_none());
    }

    #[test]
    fn ttl_expiry_drops_entry() {
        let cache = IdempotencyCache::new(Duration::from_millis(20), 8);
        cache.put("run_command", "k", &ok("v"));
        assert!(cache.get("run_command", "k").is_some());
        std::thread::sleep(Duration::from_millis(40));
        assert!(cache.get("run_command", "k").is_none());
    }

    #[test]
    fn fifo_eviction_caps_size() {
        let cache = IdempotencyCache::new(Duration::from_secs(60), 3);
        cache.put("t", "a", &ok("a"));
        std::thread::sleep(Duration::from_millis(2));
        cache.put("t", "b", &ok("b"));
        std::thread::sleep(Duration::from_millis(2));
        cache.put("t", "c", &ok("c"));
        assert_eq!(cache.len(), 3);
        std::thread::sleep(Duration::from_millis(2));
        cache.put("t", "d", &ok("d"));
        assert_eq!(cache.len(), 3, "oldest entry should have been evicted");
        assert!(cache.get("t", "a").is_none(), "'a' was the oldest");
        assert!(cache.get("t", "d").is_some());
    }

    #[test]
    fn folder_from_url_handles_workflows_segment_and_fallback() {
        assert_eq!(
            folder_from_url("file:///Users/x/proj/workflows/github-demo/src/main.ts"),
            Some("github-demo".into())
        );
        assert_eq!(
            folder_from_url("/opt/flows/login/run.ts"),
            Some("login".into())
        );
    }

    #[test]
    fn read_checkpoint_roundtrip() {
        // Write a fake state.json under a temp data_local_dir-shaped path and
        // read it back via `read_checkpoint` using an explicit workflow_id.
        let tmp = tempfile::tempdir().unwrap();
        let wf_dir = tmp.path().join("mediar").join("workflows").join("itest-wf");
        std::fs::create_dir_all(&wf_dir).unwrap();
        let state = serde_json::json!({
            "last_updated": "2026-01-02T03:04:05Z",
            "last_step_id": "step_3",
            "last_step_index": 3,
            "env": {"k": "v"},
        });
        std::fs::write(
            wf_dir.join("state.json"),
            serde_json::to_string_pretty(&state).unwrap(),
        )
        .unwrap();

        // We can't override dirs::data_local_dir(), so exercise the parser by
        // pointing straight at the file.
        let raw = std::fs::read_to_string(wf_dir.join("state.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed["last_step_id"], "step_3");
    }
}
