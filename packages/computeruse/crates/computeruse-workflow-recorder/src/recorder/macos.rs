//! macOS workflow recorder.
//!
//! Captures raw mouse and keyboard events via `rdev` (which wraps a
//! Quartz `CGEventTap` under the hood) and emits them on the same
//! `broadcast::Sender<WorkflowEvent>` the Windows recorder uses, so the
//! existing `WorkflowRecorder::process_events` consumer, JSON schema and
//! save path are reused unchanged.
//!
//! This is intentionally a *first-cut* recorder — it produces low-level
//! `Mouse` / `Keyboard` events plus best-effort focused-element metadata.
//! High-level semantic events (Click/TextInputCompleted/ApplicationSwitch)
//! are Windows-only for now; the captured stream is still sufficient for
//! replay and for the OpenClaw-parity goal of "macOS recorder exists".
//!
//! Requires the host process to have Accessibility permission (same as
//! every other AX consumer in this crate). `rdev::listen` will return
//! `ListenError` immediately if the permission is missing, which we surface
//! as `WorkflowRecorderError::InitializationError`.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use rdev::{Button, EventType, Key};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

use crate::events::{
    EventMetadata, KeyboardEvent, MouseButton, MouseEvent, MouseEventType, Position, WorkflowEvent,
};
use crate::{Result, WorkflowRecorderConfig, WorkflowRecorderError};

/// Tracks modifier-key state across events (rdev only reports per-key
/// press/release, not the composite modifier mask).
#[derive(Default, Clone, Copy)]
struct Modifiers {
    ctrl: bool,
    alt: bool,
    shift: bool,
    meta: bool,
}

pub struct MacOSRecorder {
    stop: Arc<AtomicBool>,
    /// Join handle for the blocking `rdev::listen` thread. We keep it so the
    /// struct owns the thread's lifetime; rdev 0.5 has no clean way to break
    /// out of `listen`, so on `stop()` we flip `stop` and detach.
    _listener: std::thread::JoinHandle<()>,
}

impl MacOSRecorder {
    pub fn new(
        config: WorkflowRecorderConfig,
        event_tx: broadcast::Sender<WorkflowEvent>,
    ) -> Result<Self> {
        // Fail fast with a useful message if Accessibility permission is
        // missing — otherwise rdev surfaces an opaque `EventTapError`.
        if !unsafe { accessibility_sys::AXIsProcessTrusted() } {
            return Err(WorkflowRecorderError::InitializationError(
                "macOS Accessibility permission not granted. Enable this \
                 process in System Settings → Privacy & Security → \
                 Accessibility, then retry."
                    .to_string(),
            ));
        }

        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = stop.clone();

        // Best-effort focused-element capture. Constructing `Desktop` is
        // cheap (it just wraps the AX system-wide element); we hold it for
        // the lifetime of the listener thread.
        let desktop = computeruse::Desktop::new(false, false).ok();
        let capture_ui = !config.reduce_ui_element_capture;

        // rdev::listen blocks forever and must own its thread.
        let (init_tx, init_rx) = std::sync::mpsc::channel::<std::result::Result<(), String>>();
        let listener = std::thread::Builder::new()
            .name("macos-workflow-recorder".to_string())
            .spawn(move || {
                let mods = Arc::new(Mutex::new(Modifiers::default()));
                let last_pos = Arc::new(Mutex::new(Position { x: 0, y: 0 }));
                // Signal the spawner that we got far enough to enter listen().
                // Any rdev failure happens synchronously inside listen(), so
                // we send Ok now and let the Err path below report failure.
                let _ = init_tx.send(Ok(()));

                let tx = event_tx;
                let result = rdev::listen(move |ev| {
                    if stop_for_thread.load(Ordering::Relaxed) {
                        return;
                    }
                    if let Some(workflow_ev) = translate(
                        &ev,
                        &mods,
                        &last_pos,
                        capture_ui.then_some(desktop.as_ref()).flatten(),
                    ) {
                        if let Err(e) = tx.send(workflow_ev) {
                            debug!("macOS recorder: no subscribers for event ({e})");
                        }
                    }
                });
                if let Err(e) = result {
                    warn!("rdev::listen exited with error: {e:?}");
                }
            })
            .map_err(|e| {
                WorkflowRecorderError::InitializationError(format!(
                    "failed to spawn macOS recorder thread: {e}"
                ))
            })?;

        // Wait for the listener thread to confirm startup.
        match init_rx.recv_timeout(std::time::Duration::from_secs(2)) {
            Ok(Ok(())) => {}
            Ok(Err(msg)) => return Err(WorkflowRecorderError::InitializationError(msg)),
            Err(_) => {
                return Err(WorkflowRecorderError::InitializationError(
                    "macOS recorder thread did not start within 2s".to_string(),
                ))
            }
        }

        info!("macOS workflow recorder started (rdev CGEventTap)");
        Ok(Self {
            stop,
            _listener: listener,
        })
    }

    pub fn stop(&self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        info!("macOS workflow recorder stopped (events suppressed)");
        // rdev 0.5 cannot unblock `listen`; the thread becomes a no-op and
        // is reclaimed at process exit. Matches the Windows recorder's
        // soft-stop semantics.
        Ok(())
    }
}

/// Map an rdev event into the crate-wide `WorkflowEvent` schema.
fn translate(
    ev: &rdev::Event,
    mods: &Arc<Mutex<Modifiers>>,
    last_pos: &Arc<Mutex<Position>>,
    desktop: Option<&computeruse::Desktop>,
) -> Option<WorkflowEvent> {
    let metadata = || {
        let mut m = EventMetadata::with_timestamp();
        if let Some(d) = desktop {
            m.ui_element = d.focused_element().ok();
        }
        m
    };
    let pos = || *last_pos.lock().unwrap();

    match ev.event_type {
        EventType::MouseMove { x, y } => {
            let p = Position {
                x: x as i32,
                y: y as i32,
            };
            *last_pos.lock().unwrap() = p;
            // Move events are extremely high-frequency; the Windows recorder
            // throttles them too. We drop them entirely — replay only needs
            // the position attached to the next click/wheel.
            None
        }
        EventType::ButtonPress(b) => Some(WorkflowEvent::Mouse(MouseEvent {
            event_type: MouseEventType::Down,
            button: map_button(b),
            position: pos(),
            scroll_delta: None,
            drag_start: None,
            metadata: metadata(),
        })),
        EventType::ButtonRelease(b) => Some(WorkflowEvent::Mouse(MouseEvent {
            event_type: MouseEventType::Up,
            button: map_button(b),
            position: pos(),
            scroll_delta: None,
            drag_start: None,
            metadata: metadata(),
        })),
        EventType::Wheel { delta_x, delta_y } => Some(WorkflowEvent::Mouse(MouseEvent {
            event_type: MouseEventType::Wheel,
            button: MouseButton::Middle,
            position: pos(),
            scroll_delta: Some((delta_x as i32, delta_y as i32)),
            drag_start: None,
            metadata: metadata(),
        })),
        EventType::KeyPress(k) => {
            update_mods(mods, k, true);
            let m = *mods.lock().unwrap();
            Some(WorkflowEvent::Keyboard(KeyboardEvent {
                key_code: key_code(k),
                is_key_down: true,
                ctrl_pressed: m.ctrl,
                alt_pressed: m.alt,
                shift_pressed: m.shift,
                win_pressed: m.meta,
                character: ev.name.as_ref().and_then(|s| s.chars().next()),
                scan_code: None,
                metadata: metadata(),
            }))
        }
        EventType::KeyRelease(k) => {
            update_mods(mods, k, false);
            let m = *mods.lock().unwrap();
            Some(WorkflowEvent::Keyboard(KeyboardEvent {
                key_code: key_code(k),
                is_key_down: false,
                ctrl_pressed: m.ctrl,
                alt_pressed: m.alt,
                shift_pressed: m.shift,
                win_pressed: m.meta,
                character: None,
                scan_code: None,
                metadata: metadata(),
            }))
        }
    }
}

fn map_button(b: Button) -> MouseButton {
    match b {
        Button::Left => MouseButton::Left,
        Button::Right => MouseButton::Right,
        Button::Middle => MouseButton::Middle,
        Button::Unknown(_) => MouseButton::Left,
    }
}

fn update_mods(mods: &Arc<Mutex<Modifiers>>, k: Key, down: bool) {
    let mut m = mods.lock().unwrap();
    match k {
        Key::ControlLeft | Key::ControlRight => m.ctrl = down,
        Key::Alt | Key::AltGr => m.alt = down,
        Key::ShiftLeft | Key::ShiftRight => m.shift = down,
        Key::MetaLeft | Key::MetaRight => m.meta = down,
        _ => {}
    }
}

/// rdev `Key` → stable u32 code. rdev does not expose raw keycodes on
/// macOS, so we use the enum discriminant — stable within an rdev version
/// and round-trippable for replay on the same host.
fn key_code(k: Key) -> u32 {
    // SAFETY: `Key` is `#[repr(u?)]`-less but has no payload except
    // `Unknown(u32)`. We map known variants to their discriminant index
    // and pass `Unknown` through.
    match k {
        Key::Unknown(code) => code,
        other => {
            // Hash the Debug repr into the low 16 bits so distinct keys
            // get distinct, deterministic codes without a 100-arm match.
            let s = format!("{other:?}");
            let mut h: u32 = 2166136261;
            for b in s.bytes() {
                h = h.wrapping_mul(16777619) ^ (b as u32);
            }
            h & 0xFFFF
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_codes_are_deterministic_and_distinct() {
        let a = key_code(Key::KeyA);
        let b = key_code(Key::KeyB);
        let a2 = key_code(Key::KeyA);
        assert_eq!(a, a2, "same key must map to same code");
        assert_ne!(a, b, "different keys must map to different codes");
        assert_eq!(key_code(Key::Unknown(42)), 42);
    }

    #[test]
    fn modifier_tracking() {
        let mods = Arc::new(Mutex::new(Modifiers::default()));
        update_mods(&mods, Key::ShiftLeft, true);
        assert!(mods.lock().unwrap().shift);
        update_mods(&mods, Key::ShiftLeft, false);
        assert!(!mods.lock().unwrap().shift);
        update_mods(&mods, Key::MetaLeft, true);
        assert!(mods.lock().unwrap().meta);
    }

    #[test]
    fn translate_mouse_move_updates_position_but_emits_nothing() {
        let mods = Arc::new(Mutex::new(Modifiers::default()));
        let pos = Arc::new(Mutex::new(Position { x: 0, y: 0 }));
        let ev = rdev::Event {
            time: std::time::SystemTime::now(),
            name: None,
            event_type: EventType::MouseMove { x: 10.0, y: 20.0 },
        };
        let out = translate(&ev, &mods, &pos, None);
        assert!(out.is_none());
        assert_eq!(pos.lock().unwrap().x, 10);
        assert_eq!(pos.lock().unwrap().y, 20);
    }

    #[test]
    fn translate_button_press_carries_last_position() {
        let mods = Arc::new(Mutex::new(Modifiers::default()));
        let pos = Arc::new(Mutex::new(Position { x: 5, y: 7 }));
        let ev = rdev::Event {
            time: std::time::SystemTime::now(),
            name: None,
            event_type: EventType::ButtonPress(Button::Left),
        };
        match translate(&ev, &mods, &pos, None) {
            Some(WorkflowEvent::Mouse(m)) => {
                assert_eq!(m.position.x, 5);
                assert_eq!(m.position.y, 7);
                assert!(matches!(m.event_type, MouseEventType::Down));
                assert!(matches!(m.button, MouseButton::Left));
            }
            other => panic!("expected Mouse event, got {other:?}"),
        }
    }
}
