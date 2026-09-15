#pragma once

#include <string>
#include <atomic>
#include <thread>
#include <m3api/xiApi.h>
#include <cuda_runtime.h>
#include <functional>

class XimeaCapture {
public:
    using FrameCallback = std::function<void(void* gpu_buffer, uint64_t timestamp_us)>;
    using ErrorCallback = std::function<void(const std::string& error_msg)>;

    XimeaCapture(int target_fps = 1000);
    ~XimeaCapture();

    void SetResolution(int w, int h) { width_ = w; height_ = h; }
    void SetExposure(int us) { exposure_us_ = us; }
    void SetGain(float db) { gain_db_ = db; }
    void SetOffsets(int x, int y) { offset_x_ = x; offset_y_ = y; }

    bool Open();
    void Close();
    void Cleanup();

    bool StartAcquisition(FrameCallback callback, ErrorCallback error_cb = nullptr);
    bool StopAcquisition();

    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }
    int GetFPS() const { return fps_; }

    struct Stats {
        uint64_t frames_captured = 0;
        uint64_t frames_dropped = 0;
        uint64_t timeouts = 0;
    };
    Stats GetStats() const;

private:
    void CaptureLoop();

    void* camera_handle_ = nullptr;
    int width_ = 4096;
    int height_ = 996;
    int fps_ = 1000;
    int exposure_us_ = 900;
    float gain_db_ = -1.0f; // -1 means max gain
    int offset_x_ = 0;
    int offset_y_ = 0;

    std::atomic<bool> keep_running_{false};
    std::thread capture_thread_;
    FrameCallback frame_callback_;
    ErrorCallback error_callback_;

    std::atomic<uint64_t> frames_captured_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::atomic<uint64_t> timeouts_{0};
};
