#pragma once

#include <string>
#include <mutex>
#include <atomic>
#include <thread>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <functional>
#include <cuda_runtime.h>

class GstPreviewer {
public:
    using ErrorCallback = std::function<void(const std::string& error_msg)>;
    using KeyCallback = std::function<void(const std::string& key)>;

    GstPreviewer(int width, int height, int display_fps);
    ~GstPreviewer();

    bool Start(ErrorCallback error_cb = nullptr, KeyCallback key_cb = nullptr);
    void Stop();
    void UpdateFrame(void* gpu_buffer, uint64_t timestamp_us);

    struct Stats {
        uint64_t frames_displayed = 0;
    };
    Stats GetStats() const;

private:
    void PreviewLoop();
    static gboolean on_bus_message(GstBus* bus, GstMessage* msg, gpointer user_data);

    int width_;
    int height_;
    int display_fps_;

    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GMainLoop* loop_ = nullptr;
    std::thread loop_thread_;
    std::thread preview_thread_;

    std::atomic<bool> keep_running_{false};
    std::atomic<bool> frame_available_{false};
    mutable std::mutex data_mutex_;
    std::mutex gpu_copy_mutex_;
    uint64_t latest_timestamp_us_ = 0;
    uint64_t last_preview_copy_us_ = 0;

    // Owned GPU GRAY8 buffer — image.bp is deep-copied here in UpdateFrame so
    // the XIMEA ring buffer slot is released before frame_callback_ returns.
    void* preview_gray8_gpu_ = nullptr;

    void* pinned_buffer_ = nullptr;
    ErrorCallback error_callback_;
    KeyCallback key_callback_;
    std::atomic<uint64_t> frames_displayed_{0};
};
