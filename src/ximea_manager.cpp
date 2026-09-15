#include "ximea_manager.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gst/gst.h>
#include <filesystem>

extern "C" gboolean gst_cuda_load_library(void);

XimeaManager::XimeaManager() {}

XimeaManager::~XimeaManager() {
    Stop();
}

bool XimeaManager::Initialize(int save_fps, int display_fps, const std::string& codec,
                               const CameraConfig& cam_cfg, const RecorderConfig& rec_cfg) {
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
    rec_cfg_ = rec_cfg;

    if (rec_cfg_.lanes.empty()) {
        // Default lane placement matches RECORD.md: two lanes on the
        // capture GPU (e.g. RTX 5070 Ti, 2 NVENC engines), one lane on the
        // second GPU (e.g. RTX PRO 2000) if present.
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        int capture_gpu = cam_cfg.gpu_id;
        if (device_count >= 2) {
            int other_gpu = (capture_gpu == 0) ? 1 : 0;
            rec_cfg_.lanes = {{capture_gpu, 1.0}, {capture_gpu, 1.0}, {other_gpu, 1.0}};
        } else {
            rec_cfg_.lanes = {{capture_gpu, 1.0}, {capture_gpu, 1.0}};
        }
    }

    camera_ = std::make_unique<XimeaCapture>(save_fps_);
    camera_->SetResolution(cam_cfg.width, cam_cfg.height);
    camera_->SetExposure(cam_cfg.exposure_us);
    camera_->SetGain(cam_cfg.gain_db);
    camera_->SetOffsets(cam_cfg.offset_x, cam_cfg.offset_y);
    camera_->SetGpuId(cam_cfg.gpu_id);

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

    if (codec_ == "h264") {
        MultiNvRecorder::Config mcfg;
        mcfg.capture_gpu_id = camera_->GetGpuId();
        mcfg.width = width;
        mcfg.height = height;
        mcfg.fps = save_fps_;
        mcfg.gop_size = rec_cfg_.gop_size;
        mcfg.pool_size_per_lane = rec_cfg_.pool_size_per_lane;
        mcfg.max_queue_depth = rec_cfg_.max_queue_depth;
        for (const auto& l : rec_cfg_.lanes) {
            mcfg.lanes.push_back({l.gpu_id, 100000, l.weight});
        }
        multi_recorder_ = std::make_unique<MultiNvRecorder>(mcfg, output_path_, base_output_name_);
    } else {
        recorder_ = std::make_unique<GstRecorder>(output_path_, base_output_name_, width, height, save_fps_, codec_);
    }
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
                bool ok = false;
                if (codec_ == "h264" && multi_recorder_) {
                    multi_recorder_->SetOutputName(indexed_name);
                    ok = multi_recorder_->Start(error_callback_);
                } else if (recorder_) {
                    recorder_->SetOutputName(indexed_name);
                    ok = recorder_->Start(error_callback_);
                }
                if (ok) {
                    is_recording_ = true;
                } else {
                    std::cerr << "Failed to start recording." << std::endl;
                }
            } else {
                if (codec_ == "h264" && multi_recorder_) {
                    multi_recorder_->Stop();
                } else if (recorder_) {
                    recorder_->Stop();
                }
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

    bool started = camera_->StartAcquisition(
        [this](void* gpu_buffer, uint64_t timestamp_us, uint32_t acq_nframe) {
            if (is_recording_) {
                if (codec_ == "h264" && multi_recorder_) {
                    multi_recorder_->PushFrame(gpu_buffer, timestamp_us, acq_nframe);
                } else if (recorder_) {
                    recorder_->PushFrame(gpu_buffer, timestamp_us);
                }
            }
            if (previewer_) previewer_->UpdateFrame(gpu_buffer, timestamp_us);
        },
        on_error);

    if (started) {
        is_running_ = true;
    }

    return started;
}

void XimeaManager::Stop() {
    if (stop_done_.exchange(true)) return;  // execute exactly once from any thread
    is_running_ = false;
    is_recording_ = false;

    // Close camera BEFORE recorders: Close() stops acquisition then calls
    // xiCloseDevice, which cudaFrees the RDMA buffers — must happen while
    // the recorders' CUDA contexts are still alive.
    if (camera_) camera_->Close();
    if (multi_recorder_) multi_recorder_->Stop();
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
    if (codec_ == "h264" && multi_recorder_) {
        auto s = multi_recorder_->GetStats();
        tel.rec_encoded = s.serializer_stats.frames_written;
        tel.rec_dropped = s.frames_dropped;
        uint32_t queue_sum = 0;
        for (const auto& lane_stats : s.lane_stats) queue_sum += lane_stats.queue_depth;
        tel.rec_queue = queue_sum;
        tel.rec_cross_gpu_frames = s.cross_gpu_frames;
        tel.rec_pending_gops = s.serializer_stats.pending_gops;
        tel.rec_gop_gaps = s.serializer_stats.gop_gaps;
        tel.rec_frame_gaps = s.serializer_stats.frame_gaps;
        tel.rec_bytes_written = s.serializer_stats.bytes_written;
    } else if (recorder_) {
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
