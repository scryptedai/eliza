//! Browser automation backend abstraction (Issue #5).
//!
//! `Desktop` holds an `Arc<dyn BrowserBackend>` so the MCP server can
//! swap the WebSocket-extension bridge for a CDP client (Issue #1) at
//! startup without touching any tool handlers.
//!
//! Two implementations are expected:
//! - [`ExtensionBridgeBackend`] — wraps the existing global
//!   [`crate::extension_bridge::ExtensionBridge`] singleton. Default.
//!   Capabilities: `eval` + `close_tab` only.
//! - `CdpBrowser` (Issue #1, separate crate) — chromiumoxide client.
//!   Capabilities: everything.
//!
//! The trait splits into **required** methods every backend supports
//! and **optional** methods that default to `UnsupportedOperation`.
//! Callers check [`BrowserBackend::capabilities`] before invoking
//! optional methods so they can degrade rather than error.

use std::time::Duration;

use crate::AutomationError;

// Re-exported here so trait users don't need to reach into
// extension_bridge. The types stay defined there because
// `computeruse-ts` references them by their original path
// (`computeruse::extension_bridge::CloseTabResult`).
pub use crate::extension_bridge::{CloseTabResult, ClosedTabInfo};

/// Static feature inventory for a backend. All-`false` is a valid
/// state — ExtensionBridge is exactly that.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BrowserCapabilities {
    /// Backend maintains its own connection to a specific page (CDP).
    ///
    /// `false` means the backend talks to whatever tab the OS has
    /// focused — i.e. ExtensionBridge. In that case
    /// `Desktop::execute_browser_script` routes through the AX engine
    /// to find and focus the browser window before evaluating, which
    /// is the existing behavior. `true` means `eval` can be called
    /// directly with no AX preamble (works headless).
    pub has_own_connection: bool,

    /// `navigate` is implemented (vs. falling back to
    /// `eval("location.href=...")`).
    pub supports_navigate: bool,

    /// `snapshot` returns a real DOM/AX tree (feeds Issue #2 refs).
    pub supports_snapshot: bool,

    /// `click_ref` / `type_ref` act on snapshot refs directly.
    pub supports_ref_actions: bool,

    /// `screenshot` can capture page or element bitmaps.
    pub supports_screenshot: bool,

    /// `pdf` can render the page.
    pub supports_pdf: bool,
}

/// Page-load condition for `navigate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaitUntil {
    /// `load` event fired.
    #[default]
    Load,
    /// `DOMContentLoaded` fired (faster, no images/subframes).
    DomContentLoaded,
    /// No network requests for ≥500ms (slowest, most reliable for SPAs).
    NetworkIdle,
}

/// Placeholder until Issue #1 lands a real DOM/AX tree type.
/// Defined here so the trait signature is stable.
#[derive(Debug, Clone, Default)]
pub struct BrowserSnapshot {
    /// Backend-specific serialized tree (YAML, JSON, whatever).
    /// CDP fills this from `Accessibility.getFullAXTree`.
    pub raw: String,
    /// Number of interactable elements assigned a ref.
    pub ref_count: usize,
}

#[async_trait::async_trait]
pub trait BrowserBackend: Send + Sync {
    // ─── required ───────────────────────────────────────────────────

    /// Backend name for logs and `UnsupportedOperation` messages.
    fn name(&self) -> &'static str;

    /// Static feature flags. Cheap — call this freely.
    fn capabilities(&self) -> BrowserCapabilities;

    /// True if a browser is connected and ready. ExtensionBridge:
    /// "≥1 WebSocket client". CDP: "page target attached".
    async fn is_connected(&self) -> bool;

    /// Run JavaScript in the active page. Returns the script's result
    /// stringified, or empty string for `undefined`/`null`.
    ///
    /// This is the *raw* eval — no focus, no retry orchestration.
    /// `Desktop::execute_browser_script` adds the AX-engine focus
    /// dance on top when `has_own_connection == false`.
    async fn eval(&self, code: &str, timeout: Duration) -> Result<String, AutomationError>;

    /// Close a tab. All three selectors `None` means close the active tab.
    /// `Ok(None)` means no client connected or no tab matched.
    async fn close_tab(
        &self,
        tab_id: Option<i32>,
        url: Option<&str>,
        title: Option<&str>,
        timeout: Duration,
    ) -> Result<Option<CloseTabResult>, AutomationError>;

    // ─── optional (default Err) ─────────────────────────────────────
    //
    // CDP overrides all of these. ExtensionBridge overrides none.
    // Default impls produce a self-naming error so the agent gets
    // a clear "switch backends" signal.

    async fn navigate(&self, url: &str, _wait: WaitUntil) -> Result<(), AutomationError> {
        let _ = url;
        Err(self.unsupported("navigate"))
    }

    async fn snapshot(&self) -> Result<BrowserSnapshot, AutomationError> {
        Err(self.unsupported("snapshot"))
    }

    async fn click_ref(&self, ref_id: &str) -> Result<(), AutomationError> {
        let _ = ref_id;
        Err(self.unsupported("click_ref"))
    }

    async fn type_ref(&self, ref_id: &str, text: &str) -> Result<(), AutomationError> {
        let _ = (ref_id, text);
        Err(self.unsupported("type_ref"))
    }

    async fn screenshot(&self, full_page: bool) -> Result<Vec<u8>, AutomationError> {
        let _ = full_page;
        Err(self.unsupported("screenshot"))
    }

    async fn pdf(&self) -> Result<Vec<u8>, AutomationError> {
        Err(self.unsupported("pdf"))
    }

    // ─── helpers ────────────────────────────────────────────────────

    /// Build a self-identifying `UnsupportedOperation` error.
    /// Default impls call this; concrete impls don't need to.
    #[doc(hidden)]
    fn unsupported(&self, op: &str) -> AutomationError {
        AutomationError::UnsupportedOperation(format!(
            "{backend} does not support {op}() — set COMPUTERUSE_BROWSER_BACKEND=cdp",
            backend = self.name()
        ))
    }
}

// ─── ExtensionBridge adapter ────────────────────────────────────────

/// Adapts the global [`crate::extension_bridge::ExtensionBridge`]
/// singleton to the `BrowserBackend` trait.
///
/// Zero-sized: each method calls `ExtensionBridge::global().await`
/// lazily. This keeps `Desktop::new()` synchronous — the bridge's
/// async startup doesn't block construction, it just defers to first
/// use (which is exactly what the pre-trait code did via
/// `extension_bridge::try_close_tab` etc).
#[derive(Debug, Default, Clone, Copy)]
pub struct ExtensionBridgeBackend;

#[async_trait::async_trait]
impl BrowserBackend for ExtensionBridgeBackend {
    fn name(&self) -> &'static str {
        "ExtensionBridge"
    }

    fn capabilities(&self) -> BrowserCapabilities {
        // All-false default. Crucially `has_own_connection: false`
        // means `Desktop::execute_browser_script` keeps routing
        // through the AX engine + browser_script.rs orchestration
        // (focus, 30s connect-wait, 3-retry, ERROR: parsing) —
        // i.e. zero behavior change from pre-trait code.
        BrowserCapabilities::default()
    }

    async fn is_connected(&self) -> bool {
        crate::extension_bridge::ExtensionBridge::global()
            .await
            .is_client_connected()
            .await
    }

    async fn eval(&self, code: &str, timeout: Duration) -> Result<String, AutomationError> {
        // Same module fn the engine path eventually reaches.
        // `None` means the script returned undefined/null — not an error.
        crate::extension_bridge::try_eval_via_extension(code, timeout)
            .await
            .map(|opt| opt.unwrap_or_default())
    }

    async fn close_tab(
        &self,
        tab_id: Option<i32>,
        url: Option<&str>,
        title: Option<&str>,
        timeout: Duration,
    ) -> Result<Option<CloseTabResult>, AutomationError> {
        crate::extension_bridge::try_close_tab(tab_id, url, title, timeout).await
    }
}

// ─── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Records calls so tests can assert routing without a real browser.
    /// This is the "mock backend" the UPGRADE_PLAN validation section
    /// asks for — also useful for unit-testing MCP tool handlers.
    #[derive(Default)]
    struct MockBackend {
        eval_calls: AtomicUsize,
        close_calls: AtomicUsize,
        caps: BrowserCapabilities,
    }

    #[async_trait::async_trait]
    impl BrowserBackend for MockBackend {
        fn name(&self) -> &'static str {
            "Mock"
        }
        fn capabilities(&self) -> BrowserCapabilities {
            self.caps
        }
        async fn is_connected(&self) -> bool {
            true
        }
        async fn eval(&self, code: &str, _t: Duration) -> Result<String, AutomationError> {
            self.eval_calls.fetch_add(1, Ordering::SeqCst);
            Ok(format!("evaled:{code}"))
        }
        async fn close_tab(
            &self,
            _tab_id: Option<i32>,
            _url: Option<&str>,
            _title: Option<&str>,
            _t: Duration,
        ) -> Result<Option<CloseTabResult>, AutomationError> {
            self.close_calls.fetch_add(1, Ordering::SeqCst);
            Ok(None)
        }
    }

    // ─── trait surface ──────────────────────────────────────────────

    #[test]
    fn extension_bridge_caps_all_false() {
        let b = ExtensionBridgeBackend;
        let c = b.capabilities();
        assert!(!c.has_own_connection, "must route through AX engine");
        assert!(!c.supports_navigate);
        assert!(!c.supports_snapshot);
        assert!(!c.supports_ref_actions);
        assert!(!c.supports_screenshot);
        assert!(!c.supports_pdf);
    }

    #[test]
    fn extension_bridge_name() {
        assert_eq!(ExtensionBridgeBackend.name(), "ExtensionBridge");
    }

    #[tokio::test]
    async fn default_navigate_unsupported_self_naming() {
        let err = ExtensionBridgeBackend
            .navigate("https://x", WaitUntil::Load)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("ExtensionBridge"), "got: {msg}");
        assert!(msg.contains("navigate"), "got: {msg}");
        assert!(msg.contains("cdp"), "should hint at the fix: {msg}");
    }

    #[tokio::test]
    async fn default_snapshot_unsupported() {
        assert!(matches!(
            ExtensionBridgeBackend.snapshot().await,
            Err(AutomationError::UnsupportedOperation(_))
        ));
    }

    #[tokio::test]
    async fn default_click_ref_unsupported() {
        assert!(matches!(
            ExtensionBridgeBackend.click_ref("d42").await,
            Err(AutomationError::UnsupportedOperation(_))
        ));
    }

    #[tokio::test]
    async fn default_type_ref_unsupported() {
        assert!(matches!(
            ExtensionBridgeBackend.type_ref("d42", "hello").await,
            Err(AutomationError::UnsupportedOperation(_))
        ));
    }

    #[tokio::test]
    async fn default_screenshot_unsupported() {
        assert!(matches!(
            ExtensionBridgeBackend.screenshot(true).await,
            Err(AutomationError::UnsupportedOperation(_))
        ));
    }

    #[tokio::test]
    async fn default_pdf_unsupported() {
        assert!(matches!(
            ExtensionBridgeBackend.pdf().await,
            Err(AutomationError::UnsupportedOperation(_))
        ));
    }

    // ─── object safety / Arc<dyn> ───────────────────────────────────

    #[tokio::test]
    async fn trait_is_object_safe() {
        // The whole point of #5: `Desktop` holds `Arc<dyn BrowserBackend>`.
        // If a future edit accidentally adds a generic method or
        // `where Self: Sized`, this stops compiling.
        let b: Arc<dyn BrowserBackend> = Arc::new(MockBackend::default());
        assert_eq!(b.name(), "Mock");
        assert!(b.is_connected().await);
        assert_eq!(b.eval("1+1", Duration::from_secs(1)).await.unwrap(), "evaled:1+1");
    }

    #[tokio::test]
    async fn mock_records_calls_through_dyn() {
        let mock = Arc::new(MockBackend::default());
        let dyn_ref: Arc<dyn BrowserBackend> = mock.clone();

        dyn_ref.eval("a", Duration::from_secs(1)).await.unwrap();
        dyn_ref.eval("b", Duration::from_secs(1)).await.unwrap();
        dyn_ref
            .close_tab(None, None, None, Duration::from_secs(1))
            .await
            .unwrap();

        assert_eq!(mock.eval_calls.load(Ordering::SeqCst), 2);
        assert_eq!(mock.close_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn mock_default_impls_still_apply_through_dyn() {
        // MockBackend doesn't override `navigate`, so the trait's
        // default Err impl should fire — and name the *concrete* type.
        let b: Arc<dyn BrowserBackend> = Arc::new(MockBackend::default());
        let err = b.navigate("x", WaitUntil::Load).await.unwrap_err();
        assert!(err.to_string().contains("Mock"));
    }

    // ─── capabilities semantics ─────────────────────────────────────

    #[test]
    fn caps_default_is_all_false() {
        // `Default` derive must produce the conservative state.
        assert_eq!(BrowserCapabilities::default(), BrowserCapabilities {
            has_own_connection: false,
            supports_navigate: false,
            supports_snapshot: false,
            supports_ref_actions: false,
            supports_screenshot: false,
            supports_pdf: false,
        });
    }

    #[test]
    fn wait_until_default_is_load() {
        assert_eq!(WaitUntil::default(), WaitUntil::Load);
    }

    // ─── re-exports ─────────────────────────────────────────────────

    #[test]
    fn close_tab_result_reexported_here() {
        // Trait users should be able to name the return type without
        // reaching into extension_bridge.
        let _ = CloseTabResult {
            closed: false,
            tab: ClosedTabInfo {
                id: 0,
                url: None,
                title: None,
                window_id: None,
            },
        };
    }
}
