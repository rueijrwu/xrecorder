#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <cuda_runtime.h>
#include <gst/app/gstappsrc.h>
#include <gst/cuda/gstcuda.h>
#include <gst/gst.h>

class GstRecorder {
public:
    using ErrorCallback = std::function<void(const std::string& error_msg)>;

    struct Stats {
        uint64_t frames_encoded = 0;
        uint64_t frames_dropped = 0;
        uint32_t queue_current = 0;
    };

    struct FrameData {
        void* gpu_buffer;
        uint64_t timestamp_us;
        cudaEvent_t ready_event;
        int pool_index;
    };

    GstRecorder(const std::string& path, const std::string& name, int width, int height, int fps, const std::string& codec = "h264");
    ~GstRecorder();

    bool Start(ErrorCallback error_cb = nullptr);
    void Stop();
    void PushFrame(void* gpu_buffer, uint64_t timestamp_us);
    void SetOutputName(const std::string& name) { output_name_ = name; }
    Stats GetStats() const;

private:
    void RecorderLoop();
    void ReleaseSlot(int index);
    static gboolean on_bus_message(GstBus* bus, GstMessage* msg, gpointer user_data);

    std::string output_path_;
    std::string output_name_;
    int width_;
    int height_;
    int fps_;
    std::string codec_;

    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GstCudaContext* cuda_ctx_ = nullptr;
    GstCudaAllocator* cuda_allocator_ = nullptr;
    GstVideoInfo video_info_;

    GMainLoop* loop_ = nullptr;
    std::thread loop_thread_;
    std::thread recorder_thread_;
    std::atomic<bool> keep_running_{false};
    std::atomic<bool> stopping_{false};
    std::atomic<bool> started_{false};
    mutable std::mutex stop_mutex_;

    std::queue<FrameData> frame_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    ErrorCallback error_callback_;
    std::atomic<uint64_t> frames_encoded_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::atomic<bool> pipeline_failed_{false};
    uint64_t first_ts_ = 0;

    static constexpr int POOL_SIZE = 1000;
    void* gray8_pool_[POOL_SIZE] = {nullptr};
    void* gpu_pool_[POOL_SIZE] = {nullptr};
    void* host_pool_[POOL_SIZE] = {nullptr};
    std::atomic<bool> slot_in_use_[POOL_SIZE] = {};
    int pool_index_ = 0;

    cudaStream_t record_stream_ = nullptr;
    cudaEvent_t pool_events_[POOL_SIZE] = {};
};
