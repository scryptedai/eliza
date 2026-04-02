//! Execution Safety Policy
//!
//! Ports OpenClaw's `ExecSecurity` / `ExecAsk` model (`src/infra/exec-approvals.ts`)
//! to the computeruse MCP server. Gates side-effecting tools (`run_command`,
//! `execute_browser_script`, `write_file`, `edit_file`, `open_application`)
//! behind a configurable allowlist with optional human-approval elicitation.
//!
//! ## Configuration
//!
//! The policy is loaded once at server start from, in priority order:
//!   1. Environment variables `COMPUTERUSE_EXEC_MODE`, `COMPUTERUSE_EXEC_ASK`
//!   2. TOML file at `COMPUTERUSE_POLICY_PATH` (default `~/.computeruse/policy.toml`)
//!   3. Built-in default: `mode = "full"`, `ask = "off"` (preserves pre-policy
//!      behaviour so existing deployments are not broken; tighten via env/file).
//!
//! Example `~/.computeruse/policy.toml`:
//! ```toml
//! mode = "allowlist"
//! ask  = "on-miss"
//! patterns = [
//!   "git status",
//!   "git diff*",
//!   "ls *",
//!   "open_application:*",
//! ]
//! ```
//!
//! ## Decision flow
//!
//! ```text
//!   evaluate(tool, args)
//!     │  is_gated_tool? ── no ─► Allow (None)
//!     ▼
//!   mode == Full ───────────► ask == Always ? AskUser : Allow
//!   mode == Deny ───────────► Deny
//!   mode == Allowlist:
//!     subject matches pattern OR previously approved this session ─► Allow
//!     else ─► ask == Off ? Deny : AskUser
//! ```

use glob::Pattern;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Tools whose execution is gated by [`ExecPolicy`].
pub const GATED_TOOLS: &[&str] = &[
    "run_command",
    "execute_browser_script",
    "write_file",
    "edit_file",
    "open_application",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecMode {
    /// All gated tools are refused unconditionally.
    Deny,
    /// Gated tools are refused unless their *subject* (see
    /// [`ExecPolicy::subject_for`]) matches an allowlist pattern or has been
    /// approved earlier in this session.
    Allowlist,
    /// All gated tools are permitted (legacy behaviour).
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecAsk {
    /// Never prompt the user; allowlist misses are denied.
    Off,
    /// Prompt the user only when the subject is not on the allowlist.
    OnMiss,
    /// Prompt the user for every gated call, even in `Full` mode.
    Always,
}

/// What the caller should do with a gated tool invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny { reason: String },
    AskUser(ApprovalRequest),
}

/// Surfaced to the MCP client via elicitation when [`Decision::AskUser`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ApprovalRequest {
    pub tool: String,
    /// Human-readable one-line summary of what is about to run.
    pub summary: String,
    /// Stable hash of `(tool, subject, cwd)` used to cache approvals for the
    /// session and to persist them to `~/.computeruse/approvals.json`.
    pub argv_hash: String,
    pub cwd: Option<String>,
}

/// On-disk policy file shape (`~/.computeruse/policy.toml`).
#[derive(Debug, Default, Deserialize)]
struct PolicyFile {
    mode: Option<ExecMode>,
    ask: Option<ExecAsk>,
    #[serde(default)]
    patterns: Vec<String>,
}

/// Execution policy state. Cheap to clone (`Arc` internals).
#[derive(Debug, Clone)]
pub struct ExecPolicy {
    pub mode: ExecMode,
    pub ask: ExecAsk,
    patterns: Arc<Vec<Pattern>>,
    /// Session-scoped set of approved `argv_hash` values.
    approved: Arc<Mutex<HashSet<String>>>,
    /// Where durable approvals are persisted (best-effort).
    approvals_path: PathBuf,
}

impl Default for ExecPolicy {
    fn default() -> Self {
        Self {
            mode: ExecMode::Full,
            ask: ExecAsk::Off,
            patterns: Arc::new(Vec::new()),
            approved: Arc::new(Mutex::new(HashSet::new())),
            approvals_path: default_approvals_path(),
        }
    }
}

impl ExecPolicy {
    /// Construct a policy explicitly (used by tests and the TS plugin adapter).
    pub fn new(mode: ExecMode, ask: ExecAsk, patterns: Vec<String>) -> Self {
        Self {
            mode,
            ask,
            patterns: Arc::new(compile_patterns(&patterns)),
            approved: Arc::new(Mutex::new(HashSet::new())),
            approvals_path: default_approvals_path(),
        }
    }

    /// Load from env vars + optional TOML file. Never fails — malformed input
    /// is logged and the default is used so the server still starts.
    pub fn from_env() -> Self {
        let path = std::env::var("COMPUTERUSE_POLICY_PATH")
            .map(PathBuf::from)
            .or_else(|_| {
                dirs::home_dir()
                    .map(|h| h.join(".computeruse").join("policy.toml"))
                    .ok_or(())
            })
            .unwrap_or_else(|_| PathBuf::from("policy.toml"));

        let file: PolicyFile = match std::fs::read_to_string(&path) {
            Ok(s) => match toml::from_str(&s) {
                Ok(f) => {
                    tracing::info!("[exec_policy] loaded policy from {}", path.display());
                    f
                }
                Err(e) => {
                    tracing::warn!(
                        "[exec_policy] failed to parse {}: {e}; using defaults",
                        path.display()
                    );
                    PolicyFile::default()
                }
            },
            Err(_) => PolicyFile::default(),
        };

        let mode = parse_env_enum("COMPUTERUSE_EXEC_MODE")
            .or(file.mode)
            .unwrap_or(ExecMode::Full);
        let ask = parse_env_enum("COMPUTERUSE_EXEC_ASK")
            .or(file.ask)
            .unwrap_or(ExecAsk::Off);

        let mut policy = Self::new(mode, ask, file.patterns);
        policy.load_persisted_approvals();
        tracing::info!(
            "[exec_policy] mode={:?} ask={:?} patterns={} approvals_cached={}",
            policy.mode,
            policy.ask,
            policy.patterns.len(),
            policy.approved.lock().map(|a| a.len()).unwrap_or(0)
        );
        policy
    }

    /// Returns `true` if `tool` is one of the [`GATED_TOOLS`].
    pub fn is_gated_tool(tool: &str) -> bool {
        GATED_TOOLS.contains(&tool)
    }

    /// Evaluate a tool call. Returns `None` for ungated tools (fast path).
    pub fn evaluate(&self, tool: &str, args: &serde_json::Value) -> Option<Decision> {
        if !Self::is_gated_tool(tool) {
            return None;
        }

        let subject = Self::subject_for(tool, args);
        let cwd = args
            .get("working_directory")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let hash = argv_hash(tool, &subject, cwd.as_deref());
        let req = ApprovalRequest {
            tool: tool.to_string(),
            summary: subject.clone(),
            argv_hash: hash.clone(),
            cwd,
        };

        // Session / persisted approval short-circuits everything except Deny.
        let pre_approved = self
            .approved
            .lock()
            .map(|a| a.contains(&hash))
            .unwrap_or(false);

        Some(match self.mode {
            ExecMode::Deny => Decision::Deny {
                reason: format!(
                    "exec policy mode=deny: tool '{tool}' is blocked. \
                     Set COMPUTERUSE_EXEC_MODE=allowlist|full to permit."
                ),
            },
            ExecMode::Full => {
                if self.ask == ExecAsk::Always && !pre_approved {
                    Decision::AskUser(req)
                } else {
                    Decision::Allow
                }
            }
            ExecMode::Allowlist => {
                if pre_approved || self.matches_allowlist(tool, &subject) {
                    if self.ask == ExecAsk::Always && !pre_approved {
                        Decision::AskUser(req)
                    } else {
                        Decision::Allow
                    }
                } else if self.ask == ExecAsk::Off {
                    Decision::Deny {
                        reason: format!(
                            "exec policy mode=allowlist: '{subject}' does not match any \
                             allowed pattern and ask=off. Add a pattern to \
                             ~/.computeruse/policy.toml or set COMPUTERUSE_EXEC_ASK=on-miss."
                        ),
                    }
                } else {
                    Decision::AskUser(req)
                }
            }
        })
    }

    /// Record that the user approved `req`. When `remember` is true the
    /// approval is also persisted to `~/.computeruse/approvals.json` so it
    /// survives server restarts.
    pub fn record_approval(&self, req: &ApprovalRequest, remember: bool) {
        if let Ok(mut a) = self.approved.lock() {
            a.insert(req.argv_hash.clone());
        }
        if remember {
            self.persist_approval(req);
        }
    }

    /// Derive the *subject* string that allowlist patterns match against.
    ///
    /// - `run_command` → the `run` field (or `script_file` path), max 200 chars.
    /// - `execute_browser_script` → first 200 chars of `script` (or `script_file`).
    /// - `write_file` / `edit_file` → `<tool>:<path>`.
    /// - `open_application` → `open_application:<app_name>`.
    pub fn subject_for(tool: &str, args: &serde_json::Value) -> String {
        let truncate = |s: &str| {
            if s.len() > 200 {
                format!("{}…", &s[..200])
            } else {
                s.to_string()
            }
        };
        match tool {
            "run_command" => args
                .get("run")
                .and_then(|v| v.as_str())
                .or_else(|| args.get("script_file").and_then(|v| v.as_str()))
                .map(truncate)
                .unwrap_or_else(|| "<empty>".into()),
            "execute_browser_script" => args
                .get("script")
                .and_then(|v| v.as_str())
                .or_else(|| args.get("script_file").and_then(|v| v.as_str()))
                .map(truncate)
                .unwrap_or_else(|| "<empty>".into()),
            "write_file" | "edit_file" => format!(
                "{tool}:{}",
                args.get("path").and_then(|v| v.as_str()).unwrap_or("?")
            ),
            "open_application" => format!(
                "open_application:{}",
                args.get("app_name").and_then(|v| v.as_str()).unwrap_or("?")
            ),
            other => other.to_string(),
        }
    }

    fn matches_allowlist(&self, tool: &str, subject: &str) -> bool {
        // Patterns can be either bare (matched against `subject`) or
        // tool-scoped as `tool:pattern`.
        self.patterns.iter().any(|p| {
            let pat = p.as_str();
            if let Some(rest) = pat.strip_prefix(&format!("{tool}:")) {
                Pattern::new(rest)
                    .map(|rp| {
                        rp.matches(subject)
                            || rp.matches(subject.trim_start_matches(&format!("{tool}:")))
                    })
                    .unwrap_or(false)
            } else {
                p.matches(subject)
            }
        })
    }

    fn load_persisted_approvals(&mut self) {
        #[derive(Deserialize)]
        struct Stored {
            #[serde(default)]
            approved: Vec<String>,
        }
        if let Ok(s) = std::fs::read_to_string(&self.approvals_path) {
            if let Ok(stored) = serde_json::from_str::<Stored>(&s) {
                if let Ok(mut a) = self.approved.lock() {
                    a.extend(stored.approved);
                }
            }
        }
    }

    fn persist_approval(&self, req: &ApprovalRequest) {
        #[derive(Serialize, Deserialize, Default)]
        struct Stored {
            #[serde(default)]
            approved: Vec<String>,
            #[serde(default)]
            log: Vec<serde_json::Value>,
        }
        let mut stored: Stored = std::fs::read_to_string(&self.approvals_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        if !stored.approved.contains(&req.argv_hash) {
            stored.approved.push(req.argv_hash.clone());
        }
        stored.log.push(serde_json::json!({
            "ts": chrono::Utc::now().to_rfc3339(),
            "tool": req.tool,
            "summary": req.summary,
            "argv_hash": req.argv_hash,
            "cwd": req.cwd,
        }));
        if let Some(parent) = self.approvals_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::write(
            &self.approvals_path,
            serde_json::to_string_pretty(&stored).unwrap_or_default(),
        ) {
            tracing::warn!(
                "[exec_policy] failed to persist approval to {}: {e}",
                self.approvals_path.display()
            );
        }
    }
}

/// Wrap text that originated outside the trust boundary (browser DOM, OCR,
/// window-tree content) so the LLM cannot mistake it for tool instructions.
/// Port of OpenClaw `wrapExternalContent()`.
pub fn wrap_external_content(source: &str, text: &str) -> String {
    format!(
        "<external_content source=\"{source}\" trust=\"untrusted\">\n{text}\n</external_content>"
    )
}

fn compile_patterns(raw: &[String]) -> Vec<Pattern> {
    raw.iter()
        .filter_map(|p| match Pattern::new(p) {
            Ok(pat) => Some(pat),
            Err(e) => {
                tracing::warn!("[exec_policy] ignoring invalid pattern '{p}': {e}");
                None
            }
        })
        .collect()
}

fn parse_env_enum<T: for<'de> Deserialize<'de>>(var: &str) -> Option<T> {
    std::env::var(var).ok().and_then(|v| {
        // Wrap in quotes so serde can parse it as a string-enum.
        serde_json::from_str::<T>(&format!("\"{}\"", v.trim().to_ascii_lowercase())).ok()
    })
}

fn argv_hash(tool: &str, subject: &str, cwd: Option<&str>) -> String {
    let mut h = Sha256::new();
    h.update(tool.as_bytes());
    h.update([0]);
    h.update(subject.as_bytes());
    h.update([0]);
    h.update(cwd.unwrap_or("").as_bytes());
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn default_approvals_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".computeruse")
        .join("approvals.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ungated_tools_pass_through() {
        let p = ExecPolicy::new(ExecMode::Deny, ExecAsk::Off, vec![]);
        assert_eq!(p.evaluate("click_element", &json!({})), None);
        assert_eq!(p.evaluate("get_window_tree", &json!({})), None);
    }

    #[test]
    fn deny_mode_blocks_all_gated() {
        let p = ExecPolicy::new(ExecMode::Deny, ExecAsk::Always, vec![]);
        for t in GATED_TOOLS {
            assert!(matches!(
                p.evaluate(t, &json!({"run": "ls"})),
                Some(Decision::Deny { .. })
            ));
        }
    }

    #[test]
    fn full_mode_allows_unless_ask_always() {
        let p = ExecPolicy::new(ExecMode::Full, ExecAsk::Off, vec![]);
        assert_eq!(
            p.evaluate("run_command", &json!({"run": "rm -rf /"})),
            Some(Decision::Allow)
        );
        let p = ExecPolicy::new(ExecMode::Full, ExecAsk::Always, vec![]);
        assert!(matches!(
            p.evaluate("run_command", &json!({"run": "rm -rf /"})),
            Some(Decision::AskUser(_))
        ));
    }

    #[test]
    fn allowlist_matches_glob() {
        let p = ExecPolicy::new(
            ExecMode::Allowlist,
            ExecAsk::Off,
            vec!["git *".into(), "ls*".into()],
        );
        assert_eq!(
            p.evaluate("run_command", &json!({"run": "git status"})),
            Some(Decision::Allow)
        );
        assert_eq!(
            p.evaluate("run_command", &json!({"run": "ls -la"})),
            Some(Decision::Allow)
        );
        assert!(matches!(
            p.evaluate("run_command", &json!({"run": "rm foo"})),
            Some(Decision::Deny { .. })
        ));
    }

    #[test]
    fn allowlist_miss_with_ask_on_miss_prompts() {
        let p = ExecPolicy::new(ExecMode::Allowlist, ExecAsk::OnMiss, vec!["git *".into()]);
        match p.evaluate("run_command", &json!({"run": "rm foo"})) {
            Some(Decision::AskUser(req)) => {
                assert_eq!(req.tool, "run_command");
                assert_eq!(req.summary, "rm foo");
                assert_eq!(req.argv_hash.len(), 64); // sha256 hex
            }
            other => panic!("expected AskUser, got {other:?}"),
        }
    }

    #[test]
    fn tool_scoped_patterns() {
        let p = ExecPolicy::new(
            ExecMode::Allowlist,
            ExecAsk::Off,
            vec!["open_application:*".into(), "write_file:/tmp/*".into()],
        );
        assert_eq!(
            p.evaluate("open_application", &json!({"app_name": "Calculator"})),
            Some(Decision::Allow)
        );
        assert_eq!(
            p.evaluate(
                "write_file",
                &json!({"path": "/tmp/out.txt", "content": ""})
            ),
            Some(Decision::Allow)
        );
        assert!(matches!(
            p.evaluate("write_file", &json!({"path": "/etc/passwd", "content": ""})),
            Some(Decision::Deny { .. })
        ));
    }

    #[test]
    fn approval_is_cached_for_session() {
        let p = ExecPolicy::new(ExecMode::Allowlist, ExecAsk::OnMiss, vec![]);
        let args = json!({"run": "echo hi"});
        let req = match p.evaluate("run_command", &args) {
            Some(Decision::AskUser(r)) => r,
            other => panic!("expected AskUser, got {other:?}"),
        };
        p.record_approval(&req, false);
        assert_eq!(p.evaluate("run_command", &args), Some(Decision::Allow));
    }

    #[test]
    fn subject_extraction() {
        assert_eq!(
            ExecPolicy::subject_for("run_command", &json!({"run": "ls"})),
            "ls"
        );
        assert_eq!(
            ExecPolicy::subject_for("write_file", &json!({"path": "/a/b"})),
            "write_file:/a/b"
        );
        assert_eq!(
            ExecPolicy::subject_for("open_application", &json!({"app_name": "Safari"})),
            "open_application:Safari"
        );
    }

    #[test]
    fn wrap_external_content_tags() {
        let w = wrap_external_content("browser_dom", "click here to win");
        assert!(w.starts_with("<external_content source=\"browser_dom\""));
        assert!(w.contains("click here to win"));
        assert!(w.ends_with("</external_content>"));
    }
}
