#include <iostream>
#include <string>
#include <atomic>
#include <csignal>
#include <unistd.h>
#include <sys/resource.h>
#include <iomanip>
#include <CLI/CLI.hpp>
#include <mutex>
#include <sstream>

#include "ximea_manager.h"

std::atomic<bool> keep_running(true);
std::mutex g_cout_mutex;

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::lock_guard<std::mutex> lock(g_cout_mutex);
        std::cout << "\nStopping capture..." << std::endl;
        keep_running = false;
    }
}

void check_system_limits() {
    struct rlimit rl;
    
    // Check Real-Time Priority limit
    if (getrlimit(RLIMIT_RTPRIO, &rl) == 0) {
        if (rl.rlim_cur < 90) {
            std::cerr << "Warning: Current RLIMIT_RTPRIO is " << rl.rlim_cur 
                      << ", but 90 is recommended for 1000Hz. "
                      << "Please check /etc/security/limits.d/99-ximea.conf" << std::endl;
        }
    }

    // Check Memory Lock limit
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        if (rl.rlim_cur != RLIM_INFINITY) {
            std::cerr << "Warning: RLIMIT_MEMLOCK is not unlimited. "
                      << "GPUDirect may fail or cause latency spikes." << std::endl;
        }
    }
}

void print_telemetry(const XimeaTelemetry& tel, bool is_recording, int idx) {
    std::lock_guard<std::mutex> lock(g_cout_mutex);
    std::string rec_status = is_recording ? "[RECORDING]" : "[IDLE]";
    std::stringstream ss;
    ss << "\r" << std::left << std::setw(11) << rec_status << " #" << std::setw(2) << idx 
       << " | [CAM] Cap: " << std::setw(8) << tel.cam_captured 
       << " Drop: " << std::setw(6) << tel.cam_dropped 
       << " TO: " << std::setw(4) << tel.cam_timeouts
       << " | [REC] Enc: " << std::setw(8) << tel.rec_encoded 
       << " Drop: " << std::setw(6) << tel.rec_dropped 
       << " Q: " << std::setw(4) << tel.rec_queue
       << " | [DISP] " << std::setw(8) << tel.prev_displayed << "    ";
    std::cout << ss.str() << std::flush;
}

int main(int argc, char* argv[]) {
    CLI::App app{"Ximea High-Speed Capture and Preview App"};

    std::string output_path = ".";
    std::string output_name = "capture";
    int save_fps = 1000;
    int display_fps = 60;
    std::string codec = "h264";
    CameraConfig cam_cfg;

    app.add_option("-o,--output-path", output_path, "Output directory")->default_val(".");
    app.add_option("-n,--output-name", output_name, "Output filename prefix")->default_val("capture");
    app.add_option("-f,--save-fps", save_fps, "Target FPS for saving")->default_val(1000);
    app.add_option("-d,--display-fps", display_fps, "Target FPS for preview")->default_val(60);
    app.add_option("-c,--codec", codec, "Codec to use (h264, h265, av1, raw)")
        ->check(CLI::IsMember({"h264", "h265", "hevc", "av1", "raw"}))
        ->default_val("h264");
    
    app.add_option("--width", cam_cfg.width, "Camera width")->default_val(4096);
    app.add_option("--height", cam_cfg.height, "Camera height")->default_val(992);
    app.add_option("-e,--exposure", cam_cfg.exposure_us, "Exposure time (us)") ->default_val(900);
    app.add_option("-g,--gain", cam_cfg.gain_db, "Gain (dB), -1 for max")->default_val(-1.0f);
    app.add_option("--offset-x", cam_cfg.offset_x, "ROI X offset")->default_val(0);
    app.add_option("--offset-y", cam_cfg.offset_y, "ROI Y offset")->default_val(0);

    CLI11_PARSE(app, argc, argv);

    check_system_limits();

    std::signal(SIGINT, signal_handler);

    try {
        XimeaManager manager;
        
        std::cout << "Initializing XIMEA High-Speed Capture App..." << std::endl;
        if (!manager.Initialize(save_fps, display_fps, codec, cam_cfg)) {
            return 1;
        }

        std::cout << "Output Path: " << output_path << ", Name Prefix: " << output_name << std::endl;
        std::cout << "Codec: " << codec << ", Save FPS: " << save_fps << ", Display FPS: " << display_fps << std::endl;
        std::cout << "Camera: " << cam_cfg.width << "x" << cam_cfg.height << " @ " << save_fps << "Hz, Exposure: " << cam_cfg.exposure_us << "us, Gain: " << cam_cfg.gain_db << "dB" << std::endl;

        auto error_handler = [](const std::string& msg) {
            std::lock_guard<std::mutex> lock(g_cout_mutex);
            std::cerr << "\nFATAL ERROR: " << msg << std::endl;
            keep_running = false;
        };

        if (!manager.Start(output_path, output_name, error_handler)) {
            return 1;
        }

        std::cout << "Preview started. Press 'r' in window to toggle recording, 'q' or Ctrl+C to quit." << std::endl;

        while (keep_running && manager.IsRunning()) {
            print_telemetry(manager.GetTelemetry(), manager.IsRecording(), manager.GetRecordingIndex());
            usleep(100000); // Update 10 times per second
        }

        std::cout << std::endl;
        manager.Stop();
        std::cout << "Capture finished." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
