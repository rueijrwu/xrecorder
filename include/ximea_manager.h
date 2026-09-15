#pragma once

#include <string>
#include <memory>
#include <vector>
#include <atomic>
#include <functional>

#include "ximea_capture.h"
#include "gst_recorder.h"
#include "gst_previewer.h"
#include "multi_nv_recorder.h"

struct XimeaTelemetry {
    uint64_t cam_captured = 0;
    uint64_t cam_dropped = 0;
    uint64_t cam_timeouts = 0;
    uint64_t rec_encoded = 0;
    uint64_t rec_dropped = 0;
    uint32_t rec_queue = 0;
    uint64_t prev_displayed = 0;

    // Populated only when the multi-lane NVENC recorder (codec == "h264")
    // is active; see RECORD.md "Telemetry".
    uint64_t rec_cross_gpu_frames = 0;
    uint64_t rec_pending_gops = 0;
    uint64_t rec_gop_gaps = 0;
    uint64_t rec_frame_gaps = 0;
    uint64_t rec_bytes_written = 0;
};

struct CameraConfig {
    int width = 4096;
    int height = 996;
    int exposure_us = 900;
    float gain_db = -1.0f; // -1 for max
    int offset_x = 0;
    int offset_y = 0;
    int gpu_id = 0;  // GPU the XIMEA GPUDirect frames land on
};

struct LaneConfig {
    int gpu_id = 0;
    double weight = 1.0;  // relative sustained encode capacity for GOP scheduling
};

struct RecorderConfig {
    int gop_size = 30;
    int pool_size_per_lane = 48;
    uint32_t max_queue_depth = 16;
    // Empty => XimeaManager auto-detects GPUs and builds a default lane set
    // matching RECORD.md (two lanes on the capture GPU, one on the second
    // GPU if present).
    std::vector<LaneConfig> lanes;
};

class XimeaManager {
public:
    using ErrorCallback = std::function<void(const std::string& error_msg)>;

    XimeaManager();
    ~XimeaManager();

    bool Initialize(int save_fps = 1000, int display_fps = 60, const std::string& codec = "h264",
                    const CameraConfig& cam_cfg = CameraConfig(),
                    const RecorderConfig& rec_cfg = RecorderConfig());
    bool Start(const std::string& output_path, const std::string& output_name, ErrorCallback error_cb = nullptr);
    void Stop();
    bool IsRunning() const;
    bool IsRecording() const { return is_recording_; }
    int  GetRecordingIndex() const { return recording_index_; }

    XimeaTelemetry GetTelemetry() const;

private:
    std::unique_ptr<XimeaCapture> camera_;
    std::unique_ptr<GstRecorder> recorder_;         // used for codec != h264 (raw/h265/av1)
    std::unique_ptr<MultiNvRecorder> multi_recorder_;  // used for codec == h264
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
    RecorderConfig rec_cfg_;
};
