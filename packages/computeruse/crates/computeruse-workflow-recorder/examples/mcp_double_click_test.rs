use computeruse_workflow_recorder::{
    MouseEventType, WorkflowEvent, WorkflowRecorder, WorkflowRecorderConfig,
};
use std::time::Duration;
use tokio::time::timeout;
use tokio_stream::StreamExt;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with more detailed output
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    println!("🚀 Starting MCP Double Click Test");
    info!("This test will use the workflow recorder to capture double clicks");

    println!("📋 Creating recorder configuration...");
    // Create recorder configuration
    let config = WorkflowRecorderConfig {
        capture_ui_elements: true,
        record_mouse: true,
        record_hotkeys: false,
        record_clipboard: false,
        record_application_switches: false,
        record_browser_tab_navigation: false,
        record_text_input_completion: false,
        mouse_move_throttle_ms: 500,
        ..Default::default()
    };
    println!("✅ Configuration created");

    // Start the recorder
    println!("📹 Creating workflow recorder...");
    let mut recorder = WorkflowRecorder::new("mcp_double_click_test".to_string(), config);
    println!("✅ Recorder created");

    println!("🔄 Starting recorder...");
    match recorder.start().await {
        Ok(_) => println!("✅ Recorder started successfully!"),
        Err(e) => {
            println!("❌ Failed to start recorder: {e}");
            return Err(e.into());
        }
    }

    // Get event stream
    println!("📡 Getting event stream...");
    let mut event_stream = recorder.event_stream();
    println!("✅ Event stream obtained");

    // Give recorder time to initialize
    println!("⏱️ Initializing recorder (1 second)...");
    tokio::time::sleep(Duration::from_millis(1000)).await;
    println!("✅ Recorder initialization complete");

    println!("🖱️ Now performing test double clicks...");

    // Start event collection
    println!("🔄 Starting event collector...");
    let event_collector = tokio::spawn(async move {
        println!("📊 Event collector started");
        let mut events = Vec::new();
        let mut double_click_count = 0;
        let mut single_click_count = 0;

        let start_time = std::time::Instant::now();
        while start_time.elapsed() < Duration::from_secs(15) {
            match timeout(Duration::from_millis(200), event_stream.next()).await {
                Ok(Some(event)) => {
                    println!("📨 Event received: {event:?}");
                    events.push(event.clone());

                    if let WorkflowEvent::Mouse(mouse_event) = &event {
                        match mouse_event.event_type {
                            MouseEventType::DoubleClick => {
                                double_click_count += 1;
                                println!(
                                    "🖱️🖱️ DOUBLE CLICK #{} detected at ({}, {})",
                                    double_click_count,
                                    mouse_event.position.x,
                                    mouse_event.position.y
                                );

                                if let Some(ui_element) = &mouse_event.metadata.ui_element {
                                    println!(
                                        "   Element: '{}' ({})",
                                        ui_element.name_or_empty(),
                                        ui_element.role()
                                    );
                                }
                            }
                            MouseEventType::Down | MouseEventType::Up => {
                                single_click_count += 1;
                                println!("📍 Single click event: {:?}", mouse_event.event_type);
                            }
                            _ => {}
                        }
                    }
                }
                Ok(None) => {
                    println!("📡 Event stream ended");
                    break;
                }
                Err(_) => {
                    // Timeout, continue
                    continue;
                }
            }
        }

        println!("📊 Event collector finished");
        (events, double_click_count, single_click_count)
    });

    // Give collector time to start
    println!("⏱️ Waiting for collector to start...");
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Perform the test
    println!("🎯 Test Instructions:");
    println!("   1. Please manually double-click anywhere on the screen");
    println!("   2. Try double-clicking on different UI elements");
    println!("   3. The test will run for 10 seconds to capture your double clicks");
    println!("   4. Watch the console for detected double click events");

    println!("⏰ Starting 10-second test period...");
    // Wait for the user to perform double clicks
    tokio::time::sleep(Duration::from_secs(10)).await;

    println!("⏰ Test time completed, stopping recorder...");

    // Stop the recorder
    match recorder.stop().await {
        Ok(_) => println!("✅ Recorder stopped successfully"),
        Err(e) => {
            println!("⚠️ Error stopping recorder: {e}");
        }
    }

    // Get the results
    println!("📊 Getting test results...");
    let (events, double_click_count, single_click_count) = event_collector.await?;

    // Print comprehensive results
    println!("📊 TEST RESULTS:");
    println!("   Total events captured: {}", events.len());
    println!("   Double clicks detected: {double_click_count}");
    println!("   Single click events: {single_click_count}");

    // Print detailed event log
    if !events.is_empty() {
        println!("📝 Event Log:");
        for (i, event) in events.iter().enumerate() {
            if let WorkflowEvent::Mouse(mouse_event) = event {
                println!(
                    "   Event {}: {:?} at ({}, {})",
                    i, mouse_event.event_type, mouse_event.position.x, mouse_event.position.y
                );
            }
        }
    } else {
        println!("⚠️ No events were captured!");
    }

    // Validate results
    if double_click_count > 0 {
        println!("✅ SUCCESS: Double click detection is working correctly!");
        println!("   {double_click_count} double click(s) were successfully detected and recorded");
    } else {
        println!("⚠️  NO DOUBLE CLICKS DETECTED");
        println!("   This could mean:");
        println!("   - No double clicks were performed during the test");
        println!("   - The double click detection is not working");
        println!("   - Double clicks were too slow/far apart");
    }

    if single_click_count >= double_click_count * 2 {
        println!("✅ Single click events are also being captured correctly");
    }

    println!("🏁 MCP Double Click Test completed!");

    Ok(())
}
