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
// Intended two-GPU deployment:
//
//   XIMEA @ 1000 fps (GPUDirect -> RTX PRO capture GPU)
//         |
//         +--> EncoderLane 0 (RTX PRO, local) --------+
//         +--P2P--> EncoderLane 1 (RTX 5070 Ti) ------+--> OrderedBitstreamSerializer --> one MKV
//         +--P2P--> EncoderLane 2 (RTX 5070 Ti) ------+
//
// Each encoder lane owns its own CUDA conversion stream. Each remote lane
// also owns its own capture-GPU transfer stream for the pinned-host fallback,
// so no two pipelines are serialized through a shared CUDA stream.
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
        int width = 4096;
        int height = 992;
        int fps = 1000;
        int gop_size = 30;
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

    MultiNvRecorder(const Config& cfg, const std::string& output_path, const std::string& output_name);
    ~MultiNvRecorder();

    MultiNvRecorder(const MultiNvRecorder&) = delete;
    MultiNvRecorder& operator=(const MultiNvRecorder&) = delete;

    bool Start(ErrorCallback error_cb = nullptr);
    void Stop();
    void SetOutputName(const std::string& name) { output_name_ = name; }

    // Non-blocking; called from the XIMEA capture callback. acq_nframe is the
    // authoritative source sequence. source_index is derived from successive
    // acq_nframe deltas, so camera-side gaps remain visible in the recording
    // timeline instead of being hidden by a dense application counter.
    void PushFrame(void* xi_gpu_ptr, uint64_t timestamp_us, uint32_t acq_nframe);

    Stats GetStats() const;

private:
    struct RemoteTransfer {
        int gpu_id = -1;
        bool peer_ok = false;
        std::vector<void*> gray8_staging;       // on this lane's GPU
        std::vector<void*> pinned_staging;      // host, only used if !peer_ok
        std::vector<cudaEvent_t> xfer_events;   // capture-GPU events, only if !peer_ok
        cudaStream_t capture_stream = nullptr;  // per-lane capture-GPU fallback stream
    };

    struct GopRoute {
        int lane_idx = -1;
        bool needs_idr = true;
    };

    void SetupRemoteTransfer(int lane_id, int lane_gpu_id);
    void BuildSerializer();
    void RegisterMissingFrames(uint64_t first_source_index, uint64_t count);

    Config cfg_;
    std::string output_path_;
    std::string output_name_;

    std::vector<std::unique_ptr<EncoderLane>> lanes_;
    std::unique_ptr<GopScheduler> scheduler_;
    std::unique_ptr<OrderedBitstreamSerializer> serializer_;

    // Keyed by lane_id, deliberately NOT by gpu_id. Two lanes on the same
    // RTX 5070 Ti must have independent GRAY8 staging and transfer streams.
    std::map<int, RemoteTransfer> remote_transfers_;
    size_t gray8_size_ = 0;

    ErrorCallback error_callback_;
    std::atomic<bool> started_{false};
    mutable std::mutex stop_mutex_;

    // Capture-thread-only sequence state. last_source_index_ advances by the
    // unsigned acq_nframe delta, preserving XiAPI gaps and natural uint32 wrap.
    bool have_acq_sequence_ = false;
    uint32_t first_acq_nframe_ = 0;
    uint32_t last_acq_nframe_ = 0;
    uint64_t last_source_index_ = 0;
    uint64_t first_ts_us_ = 0;

    std::map<uint64_t, GopRoute> gop_routes_;  // capture-thread only

    std::atomic<uint64_t> frames_submitted_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::atomic<uint64_t> cross_gpu_frames_{0};
};
