//! Cross-platform health check system for monitoring automation API availability
//!
//! This module provides a unified health checking interface that works across
//! all platforms, with platform-specific implementations for checking the
//! underlying automation APIs (UIAutomation on Windows, AX on macOS, etc.)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Overall system health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// Everything is working correctly
    Healthy,
    /// Some functionality is degraded but system is operational
    Degraded,
    /// System is not operational
    Unhealthy,
}

impl HealthStatus {
    /// Convert to HTTP status code for health endpoints
    pub fn to_http_status(&self) -> u16 {
        match self {
            HealthStatus::Healthy => 200,   // OK
            HealthStatus::Degraded => 206,  // Partial Content
            HealthStatus::Unhealthy => 503, // Service Unavailable
        }
    }
}

/// Common health check result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Overall health status
    pub status: HealthStatus,

    /// Whether the automation API is available
    pub api_available: bool,

    /// Whether we can access the desktop/screen
    pub desktop_accessible: bool,

    /// Whether we can enumerate UI elements
    pub can_enumerate_elements: bool,

    /// Time taken to perform the health check in milliseconds
    pub check_duration_ms: u64,

    /// Platform name (e.g., "windows", "macos", "linux")
    pub platform: String,

    /// Error message if any check failed
    pub error_message: Option<String>,

    /// Additional platform-specific diagnostics
    pub diagnostics: HashMap<String, serde_json::Value>,
}

impl Default for HealthCheckResult {
    fn default() -> Self {
        Self {
            status: HealthStatus::Unhealthy,
            api_available: false,
            desktop_accessible: false,
            can_enumerate_elements: false,
            check_duration_ms: 0,
            platform: std::env::consts::OS.to_string(),
            error_message: Some("Health check not performed".to_string()),
            diagnostics: HashMap::new(),
        }
    }
}

impl HealthCheckResult {
    /// Create a new healthy result
    pub fn healthy(platform: impl Into<String>) -> Self {
        Self {
            status: HealthStatus::Healthy,
            api_available: true,
            desktop_accessible: true,
            can_enumerate_elements: true,
            check_duration_ms: 0,
            platform: platform.into(),
            error_message: None,
            diagnostics: HashMap::new(),
        }
    }

    /// Create a new unhealthy result with error
    pub fn unhealthy(platform: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            status: HealthStatus::Unhealthy,
            api_available: false,
            desktop_accessible: false,
            can_enumerate_elements: false,
            check_duration_ms: 0,
            platform: platform.into(),
            error_message: Some(error.into()),
            diagnostics: HashMap::new(),
        }
    }

    /// Update the overall status based on component health
    pub fn update_status(&mut self) {
        self.status =
            if self.api_available && self.desktop_accessible && self.can_enumerate_elements {
                HealthStatus::Healthy
            } else if self.api_available {
                HealthStatus::Degraded
            } else {
                HealthStatus::Unhealthy
            };
    }

    /// Add a diagnostic value
    pub fn add_diagnostic(&mut self, key: impl Into<String>, value: impl Serialize) {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.diagnostics.insert(key.into(), json_value);
        }
    }
}

/// Trait for platform-specific health check implementations
#[async_trait]
pub trait PlatformHealthCheck: Send + Sync {
    /// Perform a health check of the platform's automation API
    async fn check_health(&self) -> HealthCheckResult;

    /// Quick health check (just basic availability)
    async fn quick_check(&self) -> bool {
        self.check_health().await.api_available
    }
}

/// Get the platform-specific health checker
pub async fn get_platform_health_checker() -> Box<dyn PlatformHealthCheck> {
    #[cfg(target_os = "windows")]
    {
        Box::new(super::platforms::windows::health::WindowsHealthChecker::new())
    }

    #[cfg(target_os = "macos")]
    {
        Box::new(MacOSHealthChecker)
    }

    #[cfg(target_os = "linux")]
    {
        Box::new(LinuxHealthChecker)
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        Box::new(UnsupportedPlatformHealthChecker)
    }
}

/// Convenience function to perform a health check on the current platform
pub async fn check_automation_health() -> HealthCheckResult {
    let checker = get_platform_health_checker().await;
    checker.check_health().await
}

/// macOS health checker
#[cfg(target_os = "macos")]
struct MacOSHealthChecker;

#[cfg(target_os = "macos")]
#[async_trait]
impl PlatformHealthCheck for MacOSHealthChecker {
    async fn check_health(&self) -> HealthCheckResult {
        use std::time::Instant;
        let started = Instant::now();

        let mut result = HealthCheckResult {
            platform: "macos".to_string(),
            error_message: None,
            ..Default::default()
        };

        // 1. Accessibility permission — the gate for the entire AX API.
        //    AXIsProcessTrusted() returns false until the user grants the
        //    hosting process permission in System Settings → Privacy &
        //    Security → Accessibility.
        let trusted = unsafe { accessibility_sys::AXIsProcessTrusted() };
        result.api_available = trusted;
        result.add_diagnostic("accessibility_permission", trusted);

        // 2. Desktop / system-wide element reachable
        let system_wide = accessibility::AXUIElement::system_wide();
        let focused_app = system_wide.attribute(&accessibility::AXAttribute::new(
            &core_foundation::string::CFString::new("AXFocusedApplication"),
        ));
        result.desktop_accessible = focused_app.is_ok();
        match &focused_app {
            Ok(_) => {
                result.add_diagnostic("system_wide_element", "ok");
            }
            Err(e) => {
                result.add_diagnostic("system_wide_element", format!("{e:?}"));
            }
        }

        // 3. Can we actually enumerate UI elements? Try listing running apps
        //    via the cross-platform Desktop façade so the result reflects
        //    what callers will experience.
        match crate::Desktop::new(false, false) {
            Ok(desktop) => match desktop.applications() {
                Ok(apps) => {
                    result.can_enumerate_elements = !apps.is_empty();
                    result.add_diagnostic("application_count", apps.len());
                }
                Err(e) => {
                    result.can_enumerate_elements = false;
                    result.add_diagnostic("enumerate_error", e.to_string());
                }
            },
            Err(e) => {
                result.can_enumerate_elements = false;
                result.add_diagnostic("desktop_init_error", e.to_string());
            }
        }

        if !trusted {
            result.error_message = Some(
                "Accessibility permission not granted. Open System Settings → \
                 Privacy & Security → Accessibility and enable this process \
                 (or the terminal/IDE hosting it), then restart."
                    .to_string(),
            );
            result.add_diagnostic(
                "remediation",
                "System Settings → Privacy & Security → Accessibility → enable this app",
            );
        }

        result.check_duration_ms = started.elapsed().as_millis() as u64;
        result.update_status();
        result
    }
}

/// Linux health checker
#[cfg(target_os = "linux")]
struct LinuxHealthChecker;

#[cfg(target_os = "linux")]
#[async_trait]
impl PlatformHealthCheck for LinuxHealthChecker {
    async fn check_health(&self) -> HealthCheckResult {
        use std::time::Instant;
        let started = Instant::now();

        let mut result = HealthCheckResult {
            platform: "linux".to_string(),
            error_message: None,
            ..Default::default()
        };

        // Display server presence (X11 or Wayland)
        let display = std::env::var("DISPLAY").ok();
        let wayland = std::env::var("WAYLAND_DISPLAY").ok();
        result.add_diagnostic("display", display.clone().unwrap_or_default());
        result.add_diagnostic("wayland_display", wayland.clone().unwrap_or_default());
        let has_display = display.is_some() || wayland.is_some();
        result.desktop_accessible = has_display;

        // AT-SPI accessibility bus — this is the Linux equivalent of the
        // macOS Accessibility permission. We probe via atspi (already a
        // platform dep) which speaks D-Bus to org.a11y.Bus.
        match atspi::AccessibilityConnection::new().await {
            Ok(conn) => {
                result.api_available = true;
                result.add_diagnostic("atspi_bus", "connected");
                // Try a single round-trip to confirm the registry responds.
                match conn
                    .connection()
                    .call_method(
                        Some("org.a11y.atspi.Registry"),
                        "/org/a11y/atspi/accessible/root",
                        Some("org.a11y.atspi.Accessible"),
                        "GetChildCount",
                        &(),
                    )
                    .await
                {
                    Ok(reply) => {
                        let count: i32 = reply.body().deserialize().unwrap_or(0);
                        result.can_enumerate_elements = true;
                        result.add_diagnostic("root_child_count", count);
                    }
                    Err(e) => {
                        result.can_enumerate_elements = false;
                        result.add_diagnostic("registry_error", e.to_string());
                    }
                }
            }
            Err(e) => {
                result.api_available = false;
                result.add_diagnostic("atspi_bus", format!("unavailable: {e}"));
                result.error_message = Some(format!(
                    "AT-SPI accessibility bus unreachable ({e}). Ensure the \
                     `at-spi2-core` package is installed and a desktop session \
                     is running. On GNOME: gsettings set \
                     org.gnome.desktop.interface toolkit-accessibility true"
                ));
            }
        }

        // Window-management helpers — not hard requirements, but their
        // absence degrades activate_window / move_window on X11.
        for tool in ["wmctrl", "xdotool"] {
            let found = which_in_path(tool);
            result.add_diagnostic(format!("has_{tool}"), found);
        }

        if !has_display && result.error_message.is_none() {
            result.error_message = Some(
                "No DISPLAY or WAYLAND_DISPLAY set — running headless. \
                 UI automation requires a graphical session (or Xvfb)."
                    .to_string(),
            );
        }

        result.check_duration_ms = started.elapsed().as_millis() as u64;
        result.update_status();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn health_report_is_well_formed() {
        let r = check_automation_health().await;
        assert_eq!(r.platform, std::env::consts::OS);
        // status must be consistent with the component flags
        let expected = if r.api_available && r.desktop_accessible && r.can_enumerate_elements {
            HealthStatus::Healthy
        } else if r.api_available {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };
        assert_eq!(r.status, expected);
        #[cfg(target_os = "macos")]
        assert!(
            r.diagnostics.contains_key("accessibility_permission"),
            "macOS report must include accessibility_permission diagnostic"
        );
        #[cfg(target_os = "linux")]
        assert!(
            r.diagnostics.contains_key("atspi_bus"),
            "linux report must include atspi_bus diagnostic"
        );
    }
}

#[cfg(target_os = "linux")]
fn which_in_path(bin: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| {
                let full = dir.join(bin);
                full.is_file()
            })
        })
        .unwrap_or(false)
}

/// Health checker for unsupported platforms
#[allow(dead_code)]
struct UnsupportedPlatformHealthChecker;

#[async_trait]
impl PlatformHealthCheck for UnsupportedPlatformHealthChecker {
    async fn check_health(&self) -> HealthCheckResult {
        HealthCheckResult {
            status: HealthStatus::Healthy,
            api_available: true,
            desktop_accessible: true,
            can_enumerate_elements: true,
            check_duration_ms: 0,
            platform: std::env::consts::OS.to_string(),
            error_message: None,
            diagnostics: {
                let mut diag = HashMap::new();
                diag.insert(
                    "note".to_string(),
                    serde_json::Value::String(
                        "Platform-specific health checks not implemented".to_string(),
                    ),
                );
                diag
            },
        }
    }
}
