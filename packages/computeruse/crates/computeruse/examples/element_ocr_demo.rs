/// Example demonstrating element OCR functionality
///
/// This example shows how to:
/// 1. Open an application (Notepad)
/// 2. Find UI elements
/// 3. Capture screenshots of specific elements and perform OCR
/// 4. Compare different text extraction methods
use computeruse::{AutomationError, Desktop};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), AutomationError> {
    println!("🔍 Element OCR Demo");
    println!("==================");

    // Initialize desktop automation
    let desktop = Desktop::new(false, false)?;

    // Open Notepad for a reliable text-based application
    println!("\n📱 Opening Notepad...");
    let notepad = desktop.open_application("notepad")?;
    std::thread::sleep(Duration::from_millis(2000));

    // Type some text to have content for OCR
    println!("⌨️  Typing sample text...");
    let locator = desktop.locator("role:Document");
    let text_area = locator.first(None).await?;
    text_area.type_text(
        "Hello World!\nThis is a test for OCR functionality.\nLine 3: Special characters @#$%",
        false,
    )?;

    // Wait for text to be rendered
    std::thread::sleep(Duration::from_millis(1000));

    println!("\n🔍 Testing OCR on different elements:");

    // Test 1: OCR on the text document area
    println!("\n1. OCR on document text area:");
    match text_area.ocr().await {
        Ok(ocr_text) => {
            println!("   ✅ OCR Success!");
            println!("   📄 Extracted text: \"{}\"", ocr_text.trim());
            println!("   📏 Length: {} characters", ocr_text.len());
        }
        Err(e) => {
            println!("   ❌ OCR failed: {e}");
        }
    }

    // Test 2: Compare with native text() method
    println!("\n2. Comparison with native text() method:");
    match text_area.text(3) {
        Ok(native_text) => {
            println!("   ✅ Native text: \"{}\"", native_text.trim());
            println!("   📏 Length: {} characters", native_text.len());
        }
        Err(e) => {
            println!("   ❌ Native text failed: {e}");
        }
    }

    // Test 3: OCR on the window title bar (should contain "Notepad")
    println!("\n3. OCR on window title bar:");
    let title_locator = desktop.locator("role:TitleBar");
    match title_locator.first(None).await {
        Ok(title_bar) => match title_bar.ocr().await {
            Ok(title_text) => {
                println!("   ✅ Title OCR Success!");
                println!("   📄 Title text: \"{}\"", title_text.trim());

                if title_text.to_lowercase().contains("notepad") {
                    println!("   🎯 Correctly detected 'Notepad' in title!");
                }
            }
            Err(e) => {
                println!("   ❌ Title OCR failed: {e}");
            }
        },
        Err(e) => {
            println!("   ❌ Could not find title bar: {e}");
        }
    }

    // Test 4: OCR on menu items (File, Edit, etc.)
    println!("\n4. OCR on menu items:");
    let menu_locator = desktop.locator("role:MenuBar");
    match menu_locator.first(None).await {
        Ok(menu_bar) => {
            println!("   📋 Found menu bar, testing OCR...");
            match menu_bar.ocr().await {
                Ok(menu_text) => {
                    println!("   ✅ Menu OCR Success!");
                    println!("   📄 Menu text: \"{}\"", menu_text.trim());

                    // Check for common menu items
                    let menu_lower = menu_text.to_lowercase();
                    let found_items: Vec<&str> = ["file", "edit", "format", "view", "help"]
                        .iter()
                        .filter(|&&item| menu_lower.contains(item))
                        .copied()
                        .collect();

                    if !found_items.is_empty() {
                        println!("   🎯 Detected menu items: {found_items:?}");
                    }
                }
                Err(e) => {
                    println!("   ❌ Menu OCR failed: {e}");
                }
            }
        }
        Err(e) => {
            println!("   ❌ Could not find menu bar: {e}");
        }
    }

    // Test 5: Performance comparison
    println!("\n5. Performance comparison:");

    // Time the OCR method
    let start = std::time::Instant::now();
    let _ocr_result = text_area.ocr().await;
    let ocr_duration = start.elapsed();

    // Time just the capture method
    let start = std::time::Instant::now();
    let _capture_result = text_area.capture();
    let capture_duration = start.elapsed();

    println!("   ⏱️  Capture only: {capture_duration:?}");
    println!("   ⏱️  OCR (capture + recognition): {ocr_duration:?}");
    println!("   📊 OCR overhead: {:?}", ocr_duration - capture_duration);

    // Clean up
    println!("\n🧹 Cleaning up...");
    notepad.close()?;

    println!("\n✅ Element OCR demo completed!");
    println!("\n💡 Key takeaways:");
    println!("   • Use element.ocr() for extracting text from visual elements");
    println!("   • OCR works best on clear, well-rendered text");
    println!("   • Compare with native text() methods when available");
    println!("   • OCR is particularly useful for elements that don't expose text via accessibility APIs");

    Ok(())
}
