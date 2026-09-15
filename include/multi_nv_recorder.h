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
// Execution model:
//   1) XIMEA RDMA lands one frame in RTX PRO VRAM.
//   2) Capture path copies that frame into lane-owned GRAY8 memory.
//   3) Capture synchronizes ONLY that ownership copy.
//   4) The selected lane's independent process stream converts GRAY8->NV12
//      and feeds its independent NVENC pipeline while capture continues.
//
// Therefore the intended steady state is four concurrently progressing paths:
//   capture/RDMA+copy + PRO pipeline + 5070 lane A + 5070 lane B.
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

    MultiNvRecorder(const Config& cfg, const std::string& output_path,
                    const std::string& output_name);
    ~MultiNvRecorder();

    MultiNvRecorder(const MultiNvRecorder&) = delete;
    MultiNvRecorder& operator=(const MultiNvRecorder&) = delete;

    bool Start(ErrorCallback error_cb = nullptr);
    void Stop();
    void SetOutputName(const std::string& name) { output_name_ = name; }

    // Called from the XIMEA capture callback. acq_nframe is authoritative.
    // This function waits only until the XiAPI RDMA source frame has been
    // copied into the selected lane's application-owned GRAY8 slot. It never
    // waits for conversion or NVENC completion.
    void PushFrame(void* xi_gpu_ptr, uint64_t timestamp_us, uint32_t acq_nframe);

    Stats GetStats() const;

private:
    struct LaneTransfer {
        int gpu_id = -1;
        bool peer_ok = false;

        // Every lane, including the local PRO lane, owns a GRAY8 ring. The
        // XiAPI pointer is never consumed directly by a process pipeline.
        std::vector<void*> gray8_staging;

        // Per-slot event recorded when ownership copy into gray8_staging is
        // complete. Capture synchronizes only this event.
        std::vector<cudaEvent_t> copy_done_events;

        // Copy stream lives on the lane/destination GPU. Local PRO uses D2D;
        // remote 5070 lanes use P2P on this stream.
        cudaStream_t copy_stream = nullptr;

        // Only used if direct CUDA P2P is unavailable. D2H is issued on a
        // dedicated capture-GPU stream, then this lane's copy_stream performs
        // H2D. These resources are per lane so the two 5070 pipelines never
        // serialize through one fallback stream.
        std::vector<void*> pinned_staging;
        std::vector<cudaEvent_t> d2h_done_events;
        cudaStream_t capture_stream = nullptr;
    };

    struct GopRoute {
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

    // Keyed by lane_id. Each lane has independent GRAY8 ownership and copy
    // streams even when two lanes share the same RTX 5070 Ti.
    std::map<int, LaneTransfer> lane_transfers_;
    size_t gray8_size_ = 0;

    ErrorCallback error_callback_;
    std::atomic<bool> started_{false};
    mutable std::mutex stop_mutex_;

    // Capture-thread-only sequence state. last_source_index_ advances by the
    // unsigned acq_nframe delta, preserving XiAPI gaps and uint32 wrap.
    bool have_acq_sequence_ = false;
    uint32_t last_acq_nframe_ = 0;
    uint64_t last_source_index_ = 0;
    uint64_t first_ts_us_ = 0;

    std::map<uint64_t, GopRoute> gop_routes_;

    std::atomic<uint64_t> frames_submitted_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::atomic<uint64_t> cross_gpu_frames_{0};
};
