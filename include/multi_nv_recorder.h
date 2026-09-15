#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "encoder_lane.h"
#include "gop_scheduler.h"
#include "ordered_bitstream_serializer.h"

// Top-level multi-lane NVENC recorder.
//
// gop_size defines a LOGICAL FrameGroup interval and the matching H.264 GOP.
// It is not a physical prebuffer: frames stream into the selected lane as soon
// as each ownership copy completes. The first actual frame of a FrameGroup
// selects the lane; all later frames in that source-index interval use the same
// route. This preserves low latency while keeping one encoder reference history
// per independently decodable GOP.
//
// Execution model:
//   capture/RDMA + ownership copy
//   + three concurrently progressing lane process/NVENC pipelines.
class MultiNvRecorder {
public:
    using ErrorCallback = std::function<void(const std::string& error_msg)>;

    struct LaneSpec {
        int gpu_id;
        uint32_t bitrate_kbps = 100000;
        double weight = 1.0;
    };

    struct Config {
        int capture_gpu_id = 0;
        int width = 2048;
        int height = 1024;
        int fps = 1000;
        int gop_size = 30;  // logical FrameGroup size + encoder GOP size
        int pool_size_per_lane = 48;
        uint32_t max_queue_depth = 16;
        uint32_t stall_timeout_ms = 2000;
        std::vector<LaneSpec> lanes;
    };

    struct Stats {
        uint64_t frames_submitted = 0;
        uint64_t frames_dropped = 0;
        uint64_t cross_gpu_frames = 0;
        std::vector<EncoderLane::Stats> lane_stats;
        OrderedBitstreamSerializer::Stats serializer_stats;
    };

    MultiNvRecorder(const Config& cfg, const std::string& output_path,
                    const std::string& output_name);
    ~MultiNvRecorder();

    MultiNvRecorder(const MultiNvRecorder&) = delete;
    MultiNvRecorder& operator=(const MultiNvRecorder&) = delete;

    bool Start(ErrorCallback error_cb = nullptr);
    void Stop();
    void SetOutputName(const std::string& name) { output_name_ = name; }

    // One frame enters its logical FrameGroup and is processed immediately.
    // Capture waits only for ownership copy, never for the rest of the group,
    // conversion, or NVENC completion.
    void PushFrame(void* xi_gpu_ptr, uint64_t timestamp_us, uint32_t acq_nframe);

    Stats GetStats() const;

private:
    struct LaneTransfer {
        int gpu_id = -1;
        bool peer_ok = false;
        std::vector<void*> gray8_staging;
        std::vector<cudaEvent_t> copy_done_events;
        cudaStream_t copy_stream = nullptr;
        std::vector<void*> pinned_staging;
        std::vector<cudaEvent_t> d2h_done_events;
        cudaStream_t capture_stream = nullptr;
    };

    struct FrameGroupRoute {
        int lane_idx = -1;
        bool needs_idr = true;
    };

    void SetupLaneTransfer(int lane_id, int lane_gpu_id);
    void BuildSerializer();
    void RegisterMissingFrames(uint64_t first_source_index, uint64_t count);

    Config cfg_;
    std::string output_path_;
    std::string output_name_;

    std::vector<std::unique_ptr<EncoderLane>> lanes_;
    std::unique_ptr<GopScheduler> scheduler_;
    std::unique_ptr<OrderedBitstreamSerializer> serializer_;

    std::map<int, LaneTransfer> lane_transfers_;
    size_t gray8_size_ = 0;

    ErrorCallback error_callback_;
    std::atomic<bool> started_{false};
    mutable std::mutex stop_mutex_;

    bool have_acq_sequence_ = false;
    uint32_t last_acq_nframe_ = 0;
    uint64_t last_source_index_ = 0;
    uint64_t first_ts_us_ = 0;

    // group_index -> fixed route for the entire logical FrameGroup.
    std::map<uint64_t, FrameGroupRoute> frame_group_routes_;

    std::atomic<uint64_t> frames_submitted_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::atomic<uint64_t> cross_gpu_frames_{0};
};
