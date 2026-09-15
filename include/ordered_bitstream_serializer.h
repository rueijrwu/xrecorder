#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include "frame_types.h"

// Consumes EncodedChunk objects arriving out of order from up to N
// EncoderLanes and re-emits them strictly in source order, one whole GOP at
// a time, into a single final mux/filesink pipeline (RECORD.md "Encoder
// pipeline design" / "Scheduler").
//
// A GOP is "ready" once it has received `expected` chunks (registered by
// the caller when the GOP is assigned to a lane). If a GOP stalls for
// longer than `stall_timeout_ms` (e.g. a lane died) it is force-flushed
// with whatever was received, and the gap is counted in telemetry rather
// than blocking the whole recording forever.
class OrderedBitstreamSerializer {
public:
    using ErrorCallback = std::function<void(const std::string& error_msg)>;

    struct Config {
        int width = 4096;
        int height = 992;
        int fps = 1000;
        std::string output_path;
        std::string output_name;
        uint32_t stall_timeout_ms = 2000;
    };

    struct Stats {
        uint64_t bytes_written = 0;
        uint64_t frames_written = 0;
        uint64_t pending_gops = 0;
        uint64_t frame_gaps = 0;
        uint64_t gop_gaps = 0;
    };

    explicit OrderedBitstreamSerializer(const Config& cfg);
    ~OrderedBitstreamSerializer();

    bool Start(ErrorCallback error_cb = nullptr);
    void Stop();

    // Must be called once, before any frame of `gop_index` is submitted to
    // its lane, with the nominal frame count for that GOP. May be called
    // again later with a smaller count to finalize a short trailing GOP.
    void SetGopExpectedCount(uint64_t gop_index, uint32_t count);

    // Called when a frame belonging to gop_index was dropped instead of
    // handed to a lane, so the completion count stays reachable.
    void NotifyFrameDropped(uint64_t gop_index);

    // Thread-safe; called concurrently from every EncoderLane's pull thread.
    void PushChunk(EncodedChunk&& chunk);

    Stats GetStats() const;

private:
    struct GopBuffer {
        std::vector<EncodedChunk> chunks;
        uint32_t expected = 0;
        uint64_t first_seen_us = 0;
    };

    void EmitLoop();
    void EmitChunk(const EncodedChunk& chunk);
    static gboolean on_bus_message(GstBus* bus, GstMessage* msg, gpointer user_data);

    Config cfg_;
    ErrorCallback error_callback_;

    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GMainLoop* loop_ = nullptr;
    std::thread loop_thread_;
    std::thread emit_thread_;
    std::atomic<bool> keep_running_{false};
    std::atomic<bool> stopping_{false};
    std::atomic<bool> pipeline_failed_{false};
    std::atomic<bool> started_{false};
    mutable std::mutex stop_mutex_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::map<uint64_t, GopBuffer> pending_;
    uint64_t next_expected_gop_ = 0;

    std::atomic<uint64_t> bytes_written_{0};
    std::atomic<uint64_t> frames_written_{0};
    std::atomic<uint64_t> frame_gaps_{0};
    std::atomic<uint64_t> gop_gaps_{0};
};
