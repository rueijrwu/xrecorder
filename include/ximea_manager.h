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
    uint64_t rec_cross_gpu_frames = 0;
    uint64_t rec_pending_gops = 0;
    uint64_t rec_gop_gaps = 0;
    uint64_t rec_frame_gaps = 0;
    uint64_t rec_bytes_written = 0;
};

struct CameraConfig {
    int width = 2048;
    int height = 1024;
    int exposure_us = 900;
    float gain_db = -1.0f;
    int offset_x = 0;
    int offset_y = 1040;
    int gpu_id = 0;
};

struct LaneConfig {
    int gpu_id = 0;
    double weight = 1.0;
};

struct RecorderConfig {
    // Group-of-frames size. Capture frames are accumulated into this logical
    // group and the completed/active group is routed as one scheduling unit to
    // one encoder lane. The same value is used for the independently decodable
    // H.264 GOP produced by that lane.
    int gop_size = 30;
    int pool_size_per_lane = 48;
    uint32_t max_queue_depth = 16;
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
    std::unique_ptr<GstRecorder> recorder_;
    std::unique_ptr<MultiNvRecorder> multi_recorder_;
    std::unique_ptr<GstPreviewer> previewer_;

    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_recording_{false};
    std::atomic<int>  recording_index_{0};
    std::atomic<bool> stop_done_{false};
    int save_fps_ = 1000;
    int display_fps_ = 60;
    std::string codec_ = "h264";
    std::string output_path_;
    std::string base_output_name_;
    ErrorCallback error_callback_;
    RecorderConfig rec_cfg_;
};
