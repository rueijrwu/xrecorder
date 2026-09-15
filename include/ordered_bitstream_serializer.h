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
// a time, into a single final mux/filesink pipeline.
//
// A GOP has a nominal frame count and an explicit drop count. This matters
// because acq_nframe can expose camera-side gaps before the GOP's first actual
// frame arrives. Completion is therefore based on:
//
//   expected_chunks = nominal_expected - dropped
//
// rather than mutating a single expected counter whose initialization order
// can race with drop notification.
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

    // Register the nominal source-frame count for a GOP. Repeated calls are
    // safe and preserve any drop notifications that arrived earlier.
    void SetGopExpectedCount(uint64_t gop_index, uint32_t count);

    // Register one or more source frames that will never produce an encoded
    // chunk (camera-side acq_nframe gap or application-side drop).
    void NotifyFramesDropped(uint64_t gop_index, uint32_t count = 1);
    void NotifyFrameDropped(uint64_t gop_index) { NotifyFramesDropped(gop_index, 1); }

    // Thread-safe; called concurrently from every EncoderLane's pull thread.
    void PushChunk(EncodedChunk&& chunk);

    Stats GetStats() const;

private:
    struct GopBuffer {
        std::vector<EncodedChunk> chunks;
        uint32_t nominal_expected = 0;
        uint32_t dropped = 0;
        bool expected_set = false;
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
