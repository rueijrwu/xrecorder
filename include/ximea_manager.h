#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <functional>

#include "ximea_capture.h"
#include "gst_recorder.h"
#include "gst_previewer.h"

struct XimeaTelemetry {
    uint64_t cam_captured = 0;
    uint64_t cam_dropped = 0;
    uint64_t cam_timeouts = 0;
    uint64_t rec_encoded = 0;
    uint64_t rec_dropped = 0;
    uint32_t rec_queue = 0;
    uint64_t prev_displayed = 0;
};

struct CameraConfig {
    int width = 4096;
    int height = 996;
    int exposure_us = 900;
    float gain_db = -1.0f; // -1 for max
    int offset_x = 0;
    int offset_y = 0;
};

class XimeaManager {
public:
    using ErrorCallback = std::function<void(const std::string& error_msg)>;

    XimeaManager();
    ~XimeaManager();

    bool Initialize(int save_fps = 1000, int display_fps = 60, const std::string& codec = "h264", 
                    const CameraConfig& cam_cfg = CameraConfig());
    bool Start(const std::string& output_path, const std::string& output_name, ErrorCallback error_cb = nullptr);
    void Stop();
    bool IsRunning() const;
    bool IsRecording() const { return is_recording_; }
    int  GetRecordingIndex() const { return recording_index_; }

    XimeaTelemetry GetTelemetry() const;

private:
    std::unique_ptr<XimeaCapture> camera_;
    std::unique_ptr<GstRecorder> recorder_;
    std::unique_ptr<GstPreviewer> previewer_;

    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_recording_{false};      // true when active recording to disk
    std::atomic<int>  recording_index_{0};       // suffix for filename_idx.mp4
    std::atomic<bool> stop_done_{false};         // ensures Stop() executes exactly once
    int save_fps_ = 1000;
    int display_fps_ = 60;
    std::string codec_ = "h264";
    std::string output_path_;
    std::string base_output_name_;
    ErrorCallback error_callback_;
};
