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

// Top-level multi-lane NVENC recorder (RECORD.md "Recommended Architecture").
//
//   XIMEA @ 1000 fps (GPUDirect -> capture GPU)
//         |
//         +--> EncoderLane 0 (e.g. 5070 Ti) --+
//         +--> EncoderLane 1 (e.g. 5070 Ti) --+--> OrderedBitstreamSerializer --> one MKV
//         +--> EncoderLane 2 (e.g. PRO 2000) -+
//
// GOPs (not individual frames) are scheduled to lanes. Frames destined for
// a lane on a different GPU than the capture GPU get a GRAY8 P2P/staged
// cross-GPU copy before conversion; frames destined for a lane on the
// capture GPU are converted directly from the XIMEA GPUDirect pointer with
// no intermediate copy.
class MultiNvRecorder {
public:
    using ErrorCallback = std::function<void(const std::string& error_msg)>;

    struct LaneSpec {
        int gpu_id;
        uint32_t bitrate_kbps = 100000;
        double weight = 1.0;   // relative sustained capacity for scheduling
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

    // Non-blocking; called from the XIMEA capture callback. Must never wait
    // on GPU or encoder completion (RECORD.md "Capture thread requirements").
    void PushFrame(void* xi_gpu_ptr, uint64_t timestamp_us, uint32_t acq_nframe);

    Stats GetStats() const;

private:
    struct RemoteTransfer {
        bool peer_ok = false;
        std::vector<void*> gray8_staging;      // on lane's GPU
        std::vector<void*> pinned_staging;      // host, only used if !peer_ok
        std::vector<cudaEvent_t> xfer_events;   // only used if !peer_ok
    };

    void SetupRemoteTransfer(int lane_gpu_id);

    Config cfg_;
    std::string output_path_;
    std::string output_name_;

    std::vector<std::unique_ptr<EncoderLane>> lanes_;
    std::unique_ptr<GopScheduler> scheduler_;
    std::unique_ptr<OrderedBitstreamSerializer> serializer_;

    std::map<int, RemoteTransfer> remote_transfers_;  // keyed by lane gpu_id
    cudaStream_t capture_xfer_stream_ = nullptr;
    size_t gray8_size_ = 0;

    ErrorCallback error_callback_;
    std::atomic<bool> started_{false};
    mutable std::mutex stop_mutex_;

    uint64_t source_index_ = 0;
    uint64_t first_ts_us_ = 0;
    std::map<uint64_t, int> gop_lane_map_;  // gop_index -> lane index, capture-thread only

    std::atomic<uint64_t> frames_submitted_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::atomic<uint64_t> cross_gpu_frames_{0};
};
