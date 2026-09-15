#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/cuda/gstcuda.h>
#include <gst/gst.h>

#include "frame_types.h"
#include "latency_stats.h"

// One independent NVENC encode session bound to a single CUDA/GPU id.
// Pipeline: appsrc(CUDAMemory NV12) ! nvh264enc ! h264parse ! appsink
//
// Frames are submitted already converted to NV12 in an application-owned
// GPU buffer belonging to this lane's own bounded pool (see RECORD.md
// "Remove unnecessary local GRAY8 copy" / "Avoid the current 1000-frame
// recorder pool design"). The lane never muxes or writes to disk; encoded
// access units are handed to a caller-supplied callback in submission
// order for the OrderedBitstreamSerializer to consume.
//
// The caller (MultiNvRecorder) is responsible for launching the GRAY8->NV12
// conversion kernel that fills a reserved slot, on the stream returned by
// ConvertStream() (which runs in this lane's GPU context) — this keeps the
// capture-adjacent path non-blocking: SubmitSlot only records an event and
// enqueues, it never waits on the GPU.
class EncoderLane {
public:
    using ErrorCallback = std::function<void(const std::string& error_msg)>;
    using ChunkCallback = std::function<void(EncodedChunk&& chunk)>;

    struct Config {
        int lane_id = 0;
        int gpu_id = 0;
        int width = 4096;
        int height = 992;
        int fps = 1000;
        int gop_size = 30;
        int pool_size = 48;       // bounded NV12 slots (see RECORD.md pool sizing)
        uint32_t bitrate_kbps = 100000;
    };

    struct Stats {
        uint64_t submitted = 0;
        uint64_t completed = 0;
        uint64_t dropped = 0;
        uint32_t queue_depth = 0;
        LatencyStats::Snapshot encode_latency;
    };

    explicit EncoderLane(const Config& cfg);
    ~EncoderLane();

    EncoderLane(const EncoderLane&) = delete;
    EncoderLane& operator=(const EncoderLane&) = delete;

    bool Start(ChunkCallback chunk_cb, ErrorCallback error_cb = nullptr);
    void Stop();

    // Returns a pool slot's GPU (NV12) pointer to convert INTO for the given
    // frame, or nullptr if the lane's bounded pool is saturated (caller must
    // treat this as a drop — the capture path must never block on backlog).
    void* ReserveSlot(int* out_slot_index);
    cudaStream_t ConvertStream() const { return convert_stream_; }

    // Submits a previously reserved+filled slot. `force_idr` must be true
    // for the first frame of every GOP this lane encodes so independently
    // scheduled GOPs remain independently decodable.
    void SubmitSlot(int slot_index, uint64_t gop_index, uint64_t source_index,
                     uint64_t pts_ns, bool force_idr);

    int gpu_id() const { return cfg_.gpu_id; }
    int lane_id() const { return cfg_.lane_id; }
    uint32_t QueueDepth() const { return queue_depth_.load(std::memory_order_relaxed); }
    Stats GetStats() const;

private:
    struct PendingSubmit {
        int slot_index;
        uint64_t gop_index;
        uint64_t source_index;
        uint64_t pts_ns;
        bool force_idr;
        uint64_t enqueue_time_us;
    };

    void PushLoop();
    void PullLoop();
    void RequestKeyframe();
    void ReleaseSlot(int index);
    static gboolean on_bus_message(GstBus* bus, GstMessage* msg, gpointer user_data);

    Config cfg_;
    ChunkCallback chunk_callback_;
    ErrorCallback error_callback_;

    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GstElement* appsink_ = nullptr;
    GstCudaContext* cuda_ctx_ = nullptr;
    GstCudaAllocator* cuda_allocator_ = nullptr;
    GstVideoInfo video_info_;

    GMainLoop* loop_ = nullptr;
    std::thread loop_thread_;
    std::thread push_thread_;
    std::thread pull_thread_;
    std::atomic<bool> keep_running_{false};
    std::atomic<bool> stopping_{false};
    std::atomic<bool> started_{false};
    std::atomic<bool> pipeline_failed_{false};
    mutable std::mutex stop_mutex_;

    // Bounded application-owned NV12 pool for this lane (RECORD.md warns
    // against the old 1000-slot GRAY8+NV12 pool design).
    std::vector<void*> nv12_pool_;
    std::vector<std::atomic<bool>> slot_in_use_;
    std::vector<cudaEvent_t> pool_events_;
    std::atomic<int> next_slot_{0};
    cudaStream_t convert_stream_ = nullptr;

    std::queue<PendingSubmit> submit_queue_;
    std::mutex submit_mutex_;
    std::condition_variable submit_cv_;

    // FIFO of (gop_index, source_index, pts) for frames pushed to appsrc, in
    // submission order, so appsink output (which GStreamer preserves in
    // order for a single appsrc/appsink pipeline) can be re-tagged.
    std::mutex pending_mutex_;
    std::deque<std::tuple<uint64_t, uint64_t, uint64_t>> pending_frames_;

    std::atomic<uint64_t> submitted_{0};
    std::atomic<uint64_t> completed_{0};
    std::atomic<uint64_t> dropped_{0};
    std::atomic<uint32_t> queue_depth_{0};
    LatencyStats encode_latency_;
};
