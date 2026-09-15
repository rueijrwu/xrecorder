#include "ximea_manager.h"
#include <iostream>
#include <cuda.h>
#include <gst/gst.h>
#include <filesystem>

extern "C" gboolean gst_cuda_load_library(void);

XimeaManager::XimeaManager() {}

XimeaManager::~XimeaManager() {
    Stop();
}

bool XimeaManager::Initialize(int save_fps, int display_fps, const std::string& codec, const CameraConfig& cam_cfg) {
    // Force CUDA driver initialization early.
    if (cuInit(0) != CUDA_SUCCESS) {
        std::cerr << "XimeaManager: Failed to initialize CUDA driver." << std::endl;
        return false;
    }

    // Initialize GStreamer and its CUDA loader early to prevent assertion failures in nvh264enc
    gst_init(NULL, NULL);
    if (!gst_cuda_load_library()) {
        std::cerr << "XimeaManager: Warning: gst_cuda_load_library() failed. GStreamer CUDA elements may crash." << std::endl;
    }

    save_fps_ = save_fps;
    display_fps_ = display_fps;
    codec_ = codec;

    camera_ = std::make_unique<XimeaCapture>(save_fps_);
    camera_->SetResolution(cam_cfg.width, cam_cfg.height);
    camera_->SetExposure(cam_cfg.exposure_us);
    camera_->SetGain(cam_cfg.gain_db);
    camera_->SetOffsets(cam_cfg.offset_x, cam_cfg.offset_y);

    if (!camera_->Open()) {
        std::cerr << "XimeaManager: Failed to open camera." << std::endl;
        return false;
    }

    return true;
}

bool XimeaManager::Start(const std::string& output_path, const std::string& output_name, ErrorCallback error_cb) {
    if (is_running_) return true;

    stop_done_ = false;
    error_callback_ = error_cb;
    output_path_ = output_path;
    base_output_name_ = output_name;
    is_recording_ = false;
    recording_index_ = 0;

    // Ensure output directory exists
    try {
        if (!output_path_.empty()) {
            std::filesystem::create_directories(output_path_);
        }
    } catch (const std::exception& e) {
        std::cerr << "XimeaManager: Failed to create output directory: " << e.what() << ": " << output_path_ << std::endl;
        return false;
    }

    int width = camera_->GetWidth();
    int height = camera_->GetHeight();

    recorder_ = std::make_unique<GstRecorder>(output_path_, base_output_name_, width, height, save_fps_, codec_);
    previewer_ = std::make_unique<GstPreviewer>(width, height, display_fps_);

    auto on_error = [this](const std::string& msg) {
        if (error_callback_) error_callback_(msg);
        is_running_ = false;
    };

    auto on_key = [this](const std::string& key) {
        if (key == "r") {
            if (!is_recording_) {
                recording_index_++;
                std::string indexed_name = base_output_name_ + "_" + std::to_string(recording_index_);
                recorder_->SetOutputName(indexed_name);
                if (recorder_->Start(error_callback_)) { // Reuse manager's error callback
                    is_recording_ = true;
                } else {
                    std::cerr << "Failed to start recording." << std::endl;
                }
            } else {
                recorder_->Stop();
                is_recording_ = false;
            }
        } else if (key == "q") {
            std::cout << "\nQuit requested via preview window." << std::endl;
            is_running_ = false;
        }
    };

    if (!previewer_->Start(on_error, on_key)) {
        std::cerr << "XimeaManager: Failed to start previewer." << std::endl;
        return false;
    }

    bool started = camera_->StartAcquisition([this](void* gpu_buffer, uint64_t timestamp_us) {
        if (is_recording_ && recorder_) recorder_->PushFrame(gpu_buffer, timestamp_us);
        if (previewer_) previewer_->UpdateFrame(gpu_buffer, timestamp_us);
    }, on_error);

    if (started) {
        is_running_ = true;
    }

    return started;
}

void XimeaManager::Stop() {
    if (stop_done_.exchange(true)) return;  // execute exactly once from any thread
    is_running_ = false;
    is_recording_ = false;

    // Close camera BEFORE recorder: Close() stops acquisition then calls xiCloseDevice,
    // which cudaFrees the RDMA buffers — must happen while GstRecorder's CUDA context is alive.
    if (camera_) camera_->Close();
    if (recorder_) recorder_->Stop();
    if (previewer_) previewer_->Stop();
}

bool XimeaManager::IsRunning() const {
    return is_running_;
}

XimeaTelemetry XimeaManager::GetTelemetry() const {
    XimeaTelemetry tel;
    if (camera_) {
        auto s = camera_->GetStats();
        tel.cam_captured = s.frames_captured;
        tel.cam_dropped = s.frames_dropped;
        tel.cam_timeouts = s.timeouts;
    }
    if (recorder_) {
        auto s = recorder_->GetStats();
        tel.rec_encoded = s.frames_encoded;
        tel.rec_dropped = s.frames_dropped;
        tel.rec_queue = s.queue_current;
    }
    if (previewer_) {
        auto s = previewer_->GetStats();
        tel.prev_displayed = s.frames_displayed;
    }
    return tel;
}
