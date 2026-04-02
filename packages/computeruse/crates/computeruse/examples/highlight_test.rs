use computeruse::{AutomationError, Desktop, FontStyle, TextPosition};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), AutomationError> {
    println!("🎯 Testing Highlight Functionality");
    println!("{}", "=".repeat(50));

    // Create desktop instance
    let desktop = Desktop::new(false, false)?;

    println!("\n1. Finding Calculator application...");

    // Get all applications and find Calculator
    let apps = desktop.applications()?;
    let calculator = apps
        .iter()
        .find(|app| {
            app.name()
                .unwrap_or_default()
                .to_lowercase()
                .contains("calculator")
        })
        .ok_or_else(|| {
            AutomationError::PlatformError(
                "Calculator not found. Please open Calculator first.".to_string(),
            )
        })?;

    println!(
        "✅ Found Calculator: {}",
        calculator.name().unwrap_or("Unknown".to_string())
    );

    println!("\n2. Finding Calculator buttons to highlight...");

    // First, let's find any button in Calculator
    let button = match calculator.locator("role:Button") {
        Ok(locator) => match locator.first(None).await {
            Ok(button) => {
                println!(
                    "✅ Found a button: {}",
                    button.name().unwrap_or("Unknown".to_string())
                );
                button
            }
            Err(e) => {
                println!("❌ No buttons found: {e}");
                // Try to find any clickable element
                match calculator.locator("role:*") {
                    Ok(locator) => match locator.first(None).await {
                        Ok(element) => {
                            println!(
                                "✅ Found a clickable element: {} ({})",
                                element.name().unwrap_or("Unknown".to_string()),
                                element.role()
                            );
                            element
                        }
                        Err(e) => {
                            println!("❌ No clickable elements found: {e}");
                            return Ok(());
                        }
                    },
                    Err(e) => {
                        println!("❌ Failed to create locator: {e}");
                        return Ok(());
                    }
                }
            }
        },
        Err(e) => {
            println!("❌ Failed to create locator for buttons: {e}");
            return Ok(());
        }
    };

    println!("\n3. Testing highlight with different font colors...");

    // Test 1: Black text (most readable)
    let black_font_style = FontStyle {
        size: 16,        // Larger font size
        bold: true,      // Bold text
        color: 0x000000, // Black text (BGR format)
    };

    println!("   🔸 Test 1: Top position with black text");
    match button.highlight(
        Some(0x0000FF),                    // Red border (BGR format)
        Some(Duration::from_millis(2000)), // Duration: 2 seconds
        Some("BLACK TEXT"),                // Text overlay
        Some(TextPosition::Top),           // Position text on top (now left-aligned!)
        Some(black_font_style),            // Black font style
    ) {
        Ok(handle) => {
            println!("   ✅ Black text highlight started - should be readable!");
            sleep(Duration::from_millis(2500)).await;
            drop(handle);
            println!("   📝 Top highlight completed");
        }
        Err(e) => {
            println!("   ❌ Failed to highlight button: {e}");
        }
    }

    // Small pause between tests
    sleep(Duration::from_millis(500)).await;

    // Test 2: Text matching border color (red)
    let red_font_style = FontStyle {
        size: 16,        // Larger font size
        bold: true,      // Bold text
        color: 0x0000FF, // Red text matching border (BGR format)
    };

    println!("   🔸 Test 2: Text color matching border color (red)");
    match button.highlight(
        Some(0x0000FF),                    // Red border (BGR format)
        Some(Duration::from_millis(2000)), // Duration: 2 seconds
        Some("RED TEXT"),                  // Text overlay
        Some(TextPosition::Inside),        // Position text inside element
        Some(red_font_style),              // Red font style matching border
    ) {
        Ok(handle) => {
            println!("   ✅ Red text highlight started - matching border color!");
            sleep(Duration::from_millis(2500)).await;
            drop(handle);
            println!("   📝 Inside highlight completed");
        }
        Err(e) => {
            println!("   ❌ Failed to highlight button: {e}");
        }
    }

    println!("\n🎉 Highlight test completed!");
    println!("\nFeatures tested:");
    println!("  ✅ Element highlighting with custom colors");
    println!("  ✅ TOP-LEFT aligned text positioning");
    println!("  ✅ Text inside element positioning");
    println!("  ✅ Black text for readability");
    println!("  ✅ Text color matching border color");
    println!("  ✅ Custom font styling (size, bold)");

    Ok(())
}
