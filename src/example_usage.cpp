#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <csignal>

#include "ximea_manager.h"

// Simple atomic to handle Ctrl+C
std::atomic<bool> keep_running(true);
void signal_handler(int) { keep_running = false; }

int main() {
    // 1. Setup signal handling for clean exit
    std::signal(SIGINT, signal_handler);

    // 2. Instantiate the Manager
    // This class coordinates the camera (1000Hz), recorder (1000Hz), and previewer (60Hz)
    XimeaManager manager;

    // 3. Initialize Settings
    // Params: save_fps, display_fps, codec ("h264", "h265", "av1")
    int save_fps = 1000;
    int display_fps = 60;
    std::string codec = "h265"; // Using HEVC for better high-speed compression

    std::cout << "--- Initializing XimeaManager ---" << std::endl;
    if (!manager.Initialize(save_fps, display_fps, codec)) {
        std::cerr << "Failed to initialize. Ensure XIMEA camera is connected and Above 4G Decoding is enabled." << std::endl;
        return 1;
    }

    // 4. Start Acquisition
    // This launches background threads for acquisition and encoding.
    // It returns immediately (asynchronous).
    // An optional error callback can be passed to handle runtime failures.
    auto error_callback = [](const std::string& msg) {
        std::cerr << "\n[CALLBACK ERROR] " << msg << std::endl;
        keep_running = false;
    };

    std::cout << "--- Starting High-Speed Capture ---" << std::endl;
    if (!manager.Start(".", "cpp_example_output", error_callback)) {
        std::cerr << "Failed to start capture." << std::endl;
        return 1;
    }

    std::cout << "Capture running for 10 seconds. Press Ctrl+C to stop." << std::endl;

    // 5. Telemetry Loop
    // Periodically poll the manager for real-time performance statistics.
    auto start_time = std::chrono::steady_clock::now();
    while (keep_running && manager.IsRunning()) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= 10) {
            break;
        }

        XimeaTelemetry tel = manager.GetTelemetry();

        // Print a simple live status line
        std::cout << "\r[CAM] Captured: " << tel.cam_captured 
                  << " Dropped: " << tel.cam_dropped 
                  << " | [REC] Encoded: " << tel.rec_encoded 
                  << " Queue: " << tel.rec_queue 
                  << " | [DISP] " << tel.prev_displayed << std::flush;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 6. Graceful Stop
    // This stops threads, flushes GStreamer pipelines, and closes the camera.
    std::cout << "\n--- Stopping ---" << std::endl;
    manager.Stop();
    std::cout << "Capture finished. Output saved to cpp_example_output.mp4" << std::endl;

    return 0;
}
