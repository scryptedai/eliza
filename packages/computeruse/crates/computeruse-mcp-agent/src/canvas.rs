//! Canvas-lite: a tiny localhost HTTP surface the agent can render onto.
//!
//! OpenClaw ships a "Canvas" — a browser window the agent owns and can paint
//! arbitrary HTML / images / URLs into, so the user sees what the agent is
//! doing (progress, results, generated artefacts) without the agent having
//! to commandeer an existing application window.
//!
//! This is the *lite* counterpart: an in-process axum server bound to
//! `127.0.0.1` that serves a single host page. The host page opens an SSE
//! stream to `/events`; whenever an MCP tool calls
//! [`CanvasServer::present_html`] / `present_url` / `present_image`, the
//! server bumps a version counter and the host page hot-reloads its content
//! frame from `/content`.
//!
//! Design choices:
//! * **In-memory, not on-disk.** Content is held in a `RwLock` rather than
//!   written under `~/.computeruse/canvas/` — simpler, no cleanup, and the
//!   integration test can assert the bytes round-trip exactly.
//! * **SSE, not WebSocket.** axum's `ws` feature is not enabled in this
//!   crate; SSE is in axum core and is sufficient for one-way reload pushes.
//! * **Lazy global.** The server only binds a port on the first
//!   `canvas_present` call, so hosts that never use the canvas pay nothing.
//! * **Port fallback.** Tries `17374` first (memorable, matches OpenClaw),
//!   falls back to an ephemeral port if busy so two agents on one host
//!   don't collide.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;
use tokio::sync::{broadcast, OnceCell, RwLock};

/// Preferred port — chosen to match OpenClaw's canvas so docs/tutorials
/// that reference `http://localhost:17374` work unchanged.
const PREFERRED_PORT: u16 = 17374;

/// What the canvas is currently showing.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanvasContent {
    /// Nothing presented yet, or `canvas_hide` was called.
    #[default]
    Empty,
    /// Inline HTML rendered inside the host page's content frame.
    Html { html: String },
    /// External URL loaded in an `<iframe>`.
    Url { url: String },
    /// Image bytes (PNG/JPEG) served at `/content` with the right MIME.
    Image {
        mime: String,
        /// base64 so the struct stays `Serialize` for `canvas_snapshot`.
        data_base64: String,
    },
}

/// Shared state behind the axum router. Cheap to clone (all `Arc`).
#[derive(Clone)]
struct CanvasState {
    content: Arc<RwLock<CanvasContent>>,
    version: Arc<AtomicU64>,
    reload_tx: broadcast::Sender<u64>,
}

/// Handle to the running canvas server.
pub struct CanvasServer {
    addr: SocketAddr,
    state: CanvasState,
}

/// Process-wide singleton — a single agent process gets one canvas.
static CANVAS: OnceCell<CanvasServer> = OnceCell::const_new();

impl CanvasServer {
    /// Get the running canvas, starting it on first call.
    pub async fn ensure_running() -> anyhow::Result<&'static CanvasServer> {
        CANVAS
            .get_or_try_init(|| async { CanvasServer::start().await })
            .await
    }

    /// Returns the canvas if it has already been started, without starting it.
    pub fn get() -> Option<&'static CanvasServer> {
        CANVAS.get()
    }

    async fn start() -> anyhow::Result<CanvasServer> {
        let (reload_tx, _) = broadcast::channel(16);
        let state = CanvasState {
            content: Arc::new(RwLock::new(CanvasContent::Empty)),
            version: Arc::new(AtomicU64::new(0)),
            reload_tx,
        };

        let router = Router::new()
            .route("/", get(host_page))
            .route("/content", get(content))
            .route("/version", get(version))
            .route("/events", get(events))
            .with_state(state.clone());

        // Try the well-known port first; if it's taken (second agent, or
        // something else parked there) fall back to an OS-assigned port.
        let listener = match tokio::net::TcpListener::bind(("127.0.0.1", PREFERRED_PORT)).await {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(
                    "[canvas] port {PREFERRED_PORT} unavailable ({e}); using ephemeral port"
                );
                tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?
            }
        };
        let addr = listener.local_addr()?;
        tracing::info!("[canvas] serving at http://{addr}/");

        // Detached: lives for the process. We deliberately do not keep a
        // shutdown handle — the canvas is idempotent to restart and the
        // process exiting is the only meaningful shutdown.
        tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, router).await {
                tracing::error!("[canvas] server exited: {e}");
            }
        });

        Ok(CanvasServer { addr, state })
    }

    /// `http://127.0.0.1:<port>/` — the URL to open in a browser.
    pub fn url(&self) -> String {
        format!("http://{}/", self.addr)
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Replace the canvas content and notify connected host pages.
    pub async fn present(&self, content: CanvasContent) -> u64 {
        *self.state.content.write().await = content;
        let v = self.state.version.fetch_add(1, Ordering::SeqCst) + 1;
        // Ignore send error: no host page connected yet is fine.
        let _ = self.state.reload_tx.send(v);
        v
    }

    pub async fn present_html(&self, html: impl Into<String>) -> u64 {
        self.present(CanvasContent::Html { html: html.into() })
            .await
    }

    pub async fn present_url(&self, url: impl Into<String>) -> u64 {
        self.present(CanvasContent::Url { url: url.into() }).await
    }

    pub async fn present_image(&self, bytes: &[u8], mime: impl Into<String>) -> u64 {
        use base64::Engine;
        self.present(CanvasContent::Image {
            mime: mime.into(),
            data_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        })
        .await
    }

    /// Clear the canvas (host page shows the idle placeholder).
    pub async fn hide(&self) -> u64 {
        self.present(CanvasContent::Empty).await
    }

    /// Snapshot of what's currently rendered — returned by `canvas_snapshot`.
    pub async fn snapshot(&self) -> (u64, CanvasContent) {
        (
            self.state.version.load(Ordering::SeqCst),
            self.state.content.read().await.clone(),
        )
    }
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

/// Host page: minimal shell that subscribes to `/events` and re-fetches
/// `/content` on every `reload` message. Embedded so there's no asset
/// directory to ship.
const HOST_HTML: &str = r#"<!doctype html>
<html><head><meta charset="utf-8">
<title>computeruse canvas</title>
<style>
  html,body{margin:0;height:100%;background:#111;color:#ddd;
    font:14px/1.4 ui-monospace,Menlo,Consolas,monospace}
  #bar{position:fixed;top:0;left:0;right:0;padding:6px 10px;
    background:#000a;font-size:12px;z-index:2}
  #stage{position:fixed;inset:28px 0 0 0;overflow:auto}
  #stage iframe,#stage img{width:100%;height:100%;border:0;display:block}
  #stage .html{padding:16px;background:#fff;color:#000;min-height:100%;
    box-sizing:border-box}
  #idle{display:flex;align-items:center;justify-content:center;
    height:100%;opacity:.5}
</style></head><body>
<div id="bar">computeruse canvas — <span id="status">connecting…</span></div>
<div id="stage"><div id="idle">waiting for canvas_present…</div></div>
<script>
const stage=document.getElementById('stage');
const status=document.getElementById('status');
async function load(){
  const r=await fetch('/content',{cache:'no-store'});
  const ct=r.headers.get('content-type')||'';
  stage.innerHTML='';
  if(r.status===204){
    stage.innerHTML='<div id="idle">canvas hidden</div>';
  }else if(ct.startsWith('image/')){
    const b=await r.blob();
    const img=document.createElement('img');
    img.src=URL.createObjectURL(b);stage.appendChild(img);
  }else if(ct.includes('json')){
    const j=await r.json();
    if(j.kind==='url'){
      const f=document.createElement('iframe');f.src=j.url;stage.appendChild(f);
    }
  }else{
    const d=document.createElement('div');d.className='html';
    d.innerHTML=await r.text();stage.appendChild(d);
  }
}
const es=new EventSource('/events');
es.onopen =()=>{status.textContent='live';load();};
es.onerror=()=>{status.textContent='reconnecting…';};
es.onmessage=()=>load();
</script></body></html>"#;

async fn host_page() -> Html<&'static str> {
    Html(HOST_HTML)
}

/// Returns the current content in whatever shape the host page expects:
/// * `Empty` → `204 No Content`
/// * `Html`  → `text/html` body
/// * `Url`   → JSON `{kind:"url",url:…}` (host page builds the iframe)
/// * `Image` → raw image bytes with the stored MIME
async fn content(State(state): State<CanvasState>) -> Response {
    match state.content.read().await.clone() {
        CanvasContent::Empty => axum::http::StatusCode::NO_CONTENT.into_response(),
        CanvasContent::Html { html } => Html(html).into_response(),
        CanvasContent::Url { url } => {
            Json(serde_json::json!({"kind":"url","url":url})).into_response()
        }
        CanvasContent::Image { mime, data_base64 } => {
            use base64::Engine;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data_base64)
                .unwrap_or_default();
            ([(axum::http::header::CONTENT_TYPE, mime)], bytes).into_response()
        }
    }
}

async fn version(State(state): State<CanvasState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "version": state.version.load(Ordering::SeqCst) }))
}

/// SSE stream that emits one event per `present()` call. The host page
/// doesn't care about the payload, only that *something* arrived.
async fn events(
    State(state): State<CanvasState>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.reload_tx.subscribe();
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        match rx.recv().await {
            Ok(v) => Some((Ok(Event::default().data(format!("reload:{v}"))), rx)),
            // Lagged: we missed some reloads — emit one catch-up event and
            // keep going. Closed: sender dropped, end the stream.
            Err(broadcast::error::RecvError::Lagged(_)) => {
                Some((Ok(Event::default().data("reload:lagged")), rx))
            }
            Err(broadcast::error::RecvError::Closed) => None,
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end: start the server, present HTML, fetch it back over HTTP.
    /// Uses the global singleton so this also exercises `ensure_running`'s
    /// idempotence (second call returns the same instance).
    #[tokio::test]
    async fn present_html_round_trips_over_http() {
        let canvas = CanvasServer::ensure_running().await.expect("start");
        let again = CanvasServer::ensure_running().await.expect("start");
        assert_eq!(canvas.addr(), again.addr(), "singleton");

        let v0 = canvas.present_html("<h1>hello canvas</h1>").await;
        assert!(v0 >= 1);

        let base = format!("http://{}", canvas.addr());
        let body = reqwest::get(format!("{base}/content"))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert_eq!(body, "<h1>hello canvas</h1>");

        // /version reflects the bump
        let ver: serde_json::Value = reqwest::get(format!("{base}/version"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(ver["version"].as_u64(), Some(v0));

        // hide → 204
        canvas.hide().await;
        let resp = reqwest::get(format!("{base}/content")).await.unwrap();
        assert_eq!(resp.status().as_u16(), 204);

        // snapshot reflects Empty
        let (_, snap) = canvas.snapshot().await;
        assert!(matches!(snap, CanvasContent::Empty));

        // image: bytes and MIME round-trip. Kept in the same test because
        // the canvas is a process-global singleton and parallel tests would
        // race on the shared content.
        let png = [0x89u8, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        canvas.present_image(&png, "image/png").await;
        let resp = reqwest::get(format!("{base}/content")).await.unwrap();
        assert_eq!(
            resp.headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok()),
            Some("image/png")
        );
        assert_eq!(resp.bytes().await.unwrap().as_ref(), &png[..]);
    }
}
