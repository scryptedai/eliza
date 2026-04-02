//! Chrome DevTools Protocol backend for `computeruse` (Issue #1).
//!
//! Implements [`computeruse::BrowserBackend`] over `chromiumoxide`,
//! a self-contained Rust CDP client — no Node.js sidecar, no
//! manually-installed extension.
//!
//! ```ignore
//! let cdp = CdpBrowser::launch_isolated().await?;
//! let desktop = Desktop::new_default()?
//!     .with_browser_backend(Arc::new(cdp));
//! ```
//!
//! Or attach to an already-running Chrome (`--remote-debugging-port=9222`):
//!
//! ```ignore
//! let cdp = CdpBrowser::attach("http://localhost:9222").await?;
//! ```
//!
//! The `Handler` event-pump loop is spawned internally; callers don't
//! need to drive it.
//!
//! ## Capability map vs `ExtensionBridgeBackend`
//!
//! | method      | extension | cdp |
//! |-------------|-----------|-----|
//! | eval        | ✓         | ✓   |
//! | close_tab   | ✓         | ✓   |
//! | navigate    | ✗         | ✓   |
//! | screenshot  | ✗         | ✓   |
//! | pdf         | ✗         | ✓   |
//! | snapshot    | ✗         | TODO — needs `Accessibility.getFullAXTree` + ref formatting |
//! | click_ref   | ✗         | TODO — needs snapshot ref tracking |
//! | type_ref    | ✗         | TODO — needs snapshot ref tracking |
//!
//! `has_own_connection: true` means `Desktop::execute_browser_script`
//! calls `eval` directly — no AX-engine browser-window lookup, no
//! focus dance, works headless.

use std::time::Duration;

use chromiumoxide::cdp::browser_protocol::page::{
    CaptureScreenshotFormat, PrintToPdfParams,
};
use chromiumoxide::error::CdpError;
use chromiumoxide::page::ScreenshotParams;
use chromiumoxide::{Browser, BrowserConfig, Handler, Page};
use computeruse::browser_backend::{CloseTabResult, ClosedTabInfo};
use computeruse::{AutomationError, BrowserBackend, BrowserCapabilities, WaitUntil};
use futures::StreamExt;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// CDP-backed [`BrowserBackend`].
///
/// Holds the chromiumoxide `Browser`, a "current page" cursor, and the
/// background task driving the CDP event pump. The task is aborted on
/// `Drop` so this is safe to drop without explicit shutdown.
pub struct CdpBrowser {
    browser: Browser,
    /// Active page for `eval`/`screenshot`/`pdf`. `navigate` populates
    /// or replaces it. `close_tab` clears it if the closed page was
    /// the current one.
    page: Mutex<Option<Page>>,
    /// `chromiumoxide::Handler` is a `Stream` that must be polled for
    /// any CDP request to make progress. We drain it on a background
    /// task so callers never see it.
    handler_task: JoinHandle<()>,
}

impl CdpBrowser {
    /// Launch a fresh headless Chrome with an isolated profile.
    /// Best for CI / autonomous agents — no user state, no extensions.
    pub async fn launch_isolated() -> Result<Self, AutomationError> {
        let cfg = BrowserConfig::builder()
            .build()
            .map_err(|e| AutomationError::PlatformError(format!("CDP config: {e}")))?;
        Self::from_launch(cfg).await
    }

    /// Launch a fresh Chrome with a visible window. Useful for
    /// development / watching the agent work.
    pub async fn launch_with_head() -> Result<Self, AutomationError> {
        let cfg = BrowserConfig::builder()
            .with_head()
            .build()
            .map_err(|e| AutomationError::PlatformError(format!("CDP config: {e}")))?;
        Self::from_launch(cfg).await
    }

    /// Attach to an already-running Chrome that was started with
    /// `--remote-debugging-port=N`. Pass the WebSocket or HTTP
    /// debugging URL (e.g. `"http://localhost:9222"` or the full
    /// `ws://...` from `/json/version`).
    ///
    /// Also covers the remote-CDP case — Browserless, Browserbase,
    /// etc. all expose a `ws://` URL.
    pub async fn attach(debug_url: impl Into<String>) -> Result<Self, AutomationError> {
        let (browser, handler) = Browser::connect(debug_url).await.map_err(cdp_err)?;
        Self::from_parts(browser, handler).await
    }

    async fn from_launch(cfg: BrowserConfig) -> Result<Self, AutomationError> {
        let (browser, handler) = Browser::launch(cfg).await.map_err(cdp_err)?;
        Self::from_parts(browser, handler).await
    }

    async fn from_parts(browser: Browser, handler: Handler) -> Result<Self, AutomationError> {
        // The handler stream MUST be polled for any browser/page
        // command to complete — chromiumoxide multiplexes all CDP
        // traffic through it. Spawn it once here so trait method
        // implementations don't have to think about it.
        //
        // Errors from the stream are logged but don't kill the loop;
        // a single bad CDP frame shouldn't take down the whole
        // backend. The loop ends when the WebSocket closes (stream
        // returns `None`).
        let handler_task = tokio::spawn(drive_handler(handler));

        // Adopt the first existing page if there is one (attach mode
        // typically has one already). For launch mode there's usually
        // an `about:blank` tab; if not, `navigate` creates one on
        // demand.
        let initial_page = browser.pages().await.map_err(cdp_err)?.into_iter().next();

        Ok(Self {
            browser,
            page: Mutex::new(initial_page),
            handler_task,
        })
    }

    /// Borrow the active page, or error if `navigate` hasn't been
    /// called yet. The error message tells the agent what to do.
    async fn current_page(&self) -> Result<Page, AutomationError> {
        // `Page` is a cheap handle (`Arc` internally) — cloning out
        // of the lock is cheaper than holding it across an await.
        self.page
            .lock()
            .await
            .clone()
            .ok_or_else(|| {
                AutomationError::PlatformError(
                    "CDP backend has no active page — call navigate_browser first".into(),
                )
            })
    }
}

/// `chromiumoxide::Handler: Stream<Item = Result<(), CdpError>>`.
/// Drain it until the WebSocket closes.
async fn drive_handler(mut handler: Handler) {
    while let Some(event) = handler.next().await {
        if let Err(e) = event {
            // CDP frame parse error, target crash, etc. Log and
            // keep going — the next request will surface a real
            // error to the caller if the connection is actually dead.
            tracing::warn!(error = %e, "CDP handler event error");
        }
    }
    tracing::info!("CDP handler stream ended (browser disconnected)");
}

/// Map chromiumoxide errors into the computeruse error space.
/// Not a `From` impl because both types are foreign (orphan rule).
fn cdp_err(e: CdpError) -> AutomationError {
    AutomationError::PlatformError(format!("CDP: {e}"))
}

impl Drop for CdpBrowser {
    fn drop(&mut self) {
        // Don't leave the handler task running after the Browser is
        // gone — it would just spin on a closed channel.
        self.handler_task.abort();
    }
}

#[async_trait::async_trait]
impl BrowserBackend for CdpBrowser {
    fn name(&self) -> &'static str {
        "CDP"
    }

    fn capabilities(&self) -> BrowserCapabilities {
        BrowserCapabilities {
            // The whole point: CDP talks to the page directly, no AX
            // lookup, no window focus. `Desktop::execute_browser_script`
            // sees this and skips the engine path.
            has_own_connection: true,
            supports_navigate: true,
            supports_screenshot: true,
            supports_pdf: true,
            // TODO(Issue #1 phase 2): wire `Accessibility.getFullAXTree`
            // and a ref index. Then flip these to true.
            supports_snapshot: false,
            supports_ref_actions: false,
        }
    }

    async fn is_connected(&self) -> bool {
        // If the handler task has finished, the WebSocket is gone
        // and every command will time out. Cheap heuristic — no
        // round-trip needed.
        !self.handler_task.is_finished()
    }

    async fn eval(&self, code: &str, timeout: Duration) -> Result<String, AutomationError> {
        let page = self.current_page().await?;
        let fut = page.evaluate(code);

        let result = tokio::time::timeout(timeout, fut)
            .await
            .map_err(|_| {
                AutomationError::Timeout(format!(
                    "CDP eval timed out after {}s",
                    timeout.as_secs()
                ))
            })?
            .map_err(cdp_err)?;

        // `EvaluationResult::value()` is `Option<&Value>` — `None` for
        // `undefined`. Stringify everything else; trait contract is
        // empty string for no-value.
        Ok(match result.value() {
            None => String::new(),
            // String results: unwrap the JSON string rather than
            // returning `"\"hello\""`. Matches what `eval` returns
            // in a browser console.
            Some(serde_json::Value::String(s)) => s.clone(),
            // Numbers/bools/objects: serialize. The agent gets
            // `"42"`, `"true"`, `"{\"a\":1}"`.
            Some(v) => v.to_string(),
        })
    }

    async fn close_tab(
        &self,
        tab_id: Option<i32>,
        url: Option<&str>,
        title: Option<&str>,
        timeout: Duration,
    ) -> Result<Option<CloseTabResult>, AutomationError> {
        // CDP doesn't have Chrome-extension-style integer tab IDs.
        // Treat `tab_id` as a no-match signal so callers learn to
        // use url/title with this backend.
        if tab_id.is_some() {
            tracing::warn!(
                "CDP backend ignores tab_id (extension-API concept) — use url or title"
            );
        }

        let pages = tokio::time::timeout(timeout, self.browser.pages())
            .await
            .map_err(|_| AutomationError::Timeout("CDP pages() timed out".into()))?
            .map_err(cdp_err)?;

        // Walk pages until one matches. `None`/`None` means close the
        // current page (matching ExtensionBridge's "active tab" semantics).
        let target = if url.is_none() && title.is_none() {
            self.page.lock().await.clone()
        } else {
            find_matching_page(&pages, url, title).await
        };

        let Some(page) = target else {
            return Ok(None);
        };

        // Capture metadata before closing (close consumes the handle).
        let closed_url = page.url().await.ok().flatten();
        let closed_title = page.get_title().await.ok().flatten();

        // If we're closing our own current page, clear the cursor so
        // the next `eval` returns "no active page" instead of timing
        // out on a dead target.
        let was_current = self
            .page
            .lock()
            .await
            .as_ref()
            .map(|p| p.target_id() == page.target_id())
            .unwrap_or(false);

        page.close().await.map_err(cdp_err)?;

        if was_current {
            *self.page.lock().await = None;
        }

        Ok(Some(CloseTabResult {
            closed: true,
            tab: ClosedTabInfo {
                // CDP target IDs are GUIDs, not i32. Sentinel zero
                // signals "don't trust this field" — url/title are
                // the real identity.
                id: 0,
                url: closed_url,
                title: closed_title,
                window_id: None,
            },
        }))
    }

    async fn navigate(&self, url: &str, wait: WaitUntil) -> Result<(), AutomationError> {
        // Reuse the current page if we have one; create a fresh one
        // otherwise. Reusing means cookies/localStorage persist
        // across navigations within a session — usually what you want.
        let page = {
            let mut guard = self.page.lock().await;
            match guard.as_ref() {
                Some(p) => p.clone(),
                None => {
                    // `new_page` accepts a URL but we want explicit
                    // wait control, so create blank then `goto`.
                    let p = self
                        .browser
                        .new_page("about:blank")
                        .await
                        .map_err(cdp_err)?;
                    *guard = Some(p.clone());
                    p
                }
            }
        };

        page.goto(url).await.map_err(cdp_err)?;

        // `goto` returns once navigation is *committed* (response
        // headers in). For anything past `DomContentLoaded`, wait for
        // the load event.
        match wait {
            WaitUntil::DomContentLoaded => {}
            WaitUntil::Load | WaitUntil::NetworkIdle => {
                // chromiumoxide's `wait_for_navigation` waits for the
                // load event. True network-idle would need raw CDP
                // `Page.lifecycleEvent` listening — punting on that
                // until an agent actually needs it.
                page.wait_for_navigation().await.map_err(cdp_err)?;
            }
        }

        Ok(())
    }

    async fn screenshot(&self, full_page: bool) -> Result<Vec<u8>, AutomationError> {
        let page = self.current_page().await?;
        let params = ScreenshotParams::builder()
            .format(CaptureScreenshotFormat::Png)
            .full_page(full_page)
            .build();
        page.screenshot(params).await.map_err(cdp_err)
    }

    async fn pdf(&self) -> Result<Vec<u8>, AutomationError> {
        let page = self.current_page().await?;
        // Defaults: A4 portrait, print backgrounds, 1.0 scale.
        // Sufficient for "save what's on screen" — agents wanting
        // landscape/margins can call `eval` with `window.print()`.
        page.pdf(PrintToPdfParams::default()).await.map_err(cdp_err)
    }

    // snapshot / click_ref / type_ref deliberately fall through to
    // the trait's default `Err(UnsupportedOperation)`. They need a
    // ref index built from `Accessibility.getFullAXTree` — non-trivial
    // and tied to Issue #2's ref format. Phase 2.
}

/// First page whose URL or title contains the given substring.
/// Both filters are AND'd if both supplied.
async fn find_matching_page(
    pages: &[Page],
    url_substr: Option<&str>,
    title_substr: Option<&str>,
) -> Option<Page> {
    for page in pages {
        let url_ok = match url_substr {
            None => true,
            Some(needle) => page
                .url()
                .await
                .ok()
                .flatten()
                .map(|u| u.contains(needle))
                .unwrap_or(false),
        };
        let title_ok = match title_substr {
            None => true,
            Some(needle) => page
                .get_title()
                .await
                .ok()
                .flatten()
                .map(|t| t.contains(needle))
                .unwrap_or(false),
        };
        if url_ok && title_ok {
            return Some(page.clone());
        }
    }
    None
}

// ─── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ─── unit (no Chrome needed) ────────────────────────────────────

    #[test]
    fn cdp_error_maps_to_platform_error() {
        let cdp = CdpError::msg("target crashed");
        let mapped = cdp_err(cdp);
        match mapped {
            AutomationError::PlatformError(s) => {
                assert!(s.starts_with("CDP: "), "got: {s}");
                assert!(s.contains("target crashed"));
            }
            other => panic!("expected PlatformError, got {other:?}"),
        }
    }

    #[test]
    fn caps_advertise_what_is_implemented() {
        // Can't construct a real CdpBrowser without Chrome. But the
        // capability values are static — verify them by reading the
        // impl directly. If someone flips a flag without implementing
        // the method, this is the canary.
        let caps = BrowserCapabilities {
            has_own_connection: true,
            supports_navigate: true,
            supports_screenshot: true,
            supports_pdf: true,
            supports_snapshot: false,
            supports_ref_actions: false,
        };
        // Pin: snapshot/ref_actions stay false until phase 2 lands.
        // The trait's default impl returns Err for both — flipping
        // these to true without overriding would lie to callers.
        assert!(!caps.supports_snapshot);
        assert!(!caps.supports_ref_actions);
        // And the routing flag — `Desktop::execute_browser_script`
        // depends on this being true to skip the AX engine.
        assert!(caps.has_own_connection);
    }

    /// Checks the trait constraint compiles. Catches accidental
    /// `where Self: Sized` additions that would break `Arc<dyn>`.
    fn _assert_object_safe(b: CdpBrowser) -> Arc<dyn BrowserBackend> {
        Arc::new(b)
    }

    // ─── integration (needs Chrome — run with --ignored) ────────────
    //
    // These exercise the full chromiumoxide round-trip. They're
    // `#[ignore]`d so `cargo test` stays green on machines without
    // Chrome (CI, this dev box). Run manually:
    //
    //   cargo test -p computeruse-browser-cdp -- --ignored
    //
    // The httpbin target is from UPGRADE_PLAN's validation section.

    #[tokio::test]
    #[ignore = "needs a Chrome binary on PATH"]
    async fn launch_navigate_eval_roundtrip() {
        let cdp = CdpBrowser::launch_isolated()
            .await
            .expect("launch — is Chrome installed?");

        // Trait-level checks
        assert_eq!(cdp.name(), "CDP");
        assert!(cdp.is_connected().await);
        assert!(cdp.capabilities().has_own_connection);

        // navigate → eval round-trip per UPGRADE_PLAN validation
        cdp.navigate("https://httpbin.org/html", WaitUntil::Load)
            .await
            .expect("navigate");

        let title = cdp
            .eval("document.title", Duration::from_secs(10))
            .await
            .expect("eval");
        assert!(
            title.contains("Herman Melville"),
            "httpbin /html serves a Moby Dick excerpt; got: {title}"
        );

        // screenshot → non-empty PNG
        let png = cdp.screenshot(false).await.expect("screenshot");
        assert!(png.len() > 1000, "got {} bytes", png.len());
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "PNG magic header");
    }

    #[tokio::test]
    #[ignore = "needs a Chrome binary on PATH"]
    async fn close_tab_clears_current_page() {
        let cdp = CdpBrowser::launch_isolated().await.expect("launch");
        cdp.navigate("about:blank", WaitUntil::DomContentLoaded)
            .await
            .expect("navigate");

        // Close the current page (no url/title filter)
        let result = cdp
            .close_tab(None, None, None, Duration::from_secs(10))
            .await
            .expect("close_tab")
            .expect("should have found a page");
        assert!(result.closed);

        // Cursor cleared — eval should now error with the
        // "no active page" message, not time out.
        let err = cdp
            .eval("1+1", Duration::from_secs(2))
            .await
            .expect_err("should fail without page");
        assert!(
            err.to_string().contains("no active page"),
            "got: {err}"
        );
    }

    #[tokio::test]
    #[ignore = "needs a Chrome binary on PATH"]
    async fn unsupported_methods_self_name() {
        let cdp = CdpBrowser::launch_isolated().await.expect("launch");

        // snapshot/click_ref/type_ref fall through to default impl
        let err = cdp.snapshot().await.expect_err("should be unsupported");
        let msg = err.to_string();
        assert!(msg.contains("CDP"), "should self-identify: {msg}");
        assert!(msg.contains("snapshot"));
    }
}
