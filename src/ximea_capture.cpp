#include "ximea_capture.h"
#include <iostream>
#include <cstring>
#include <pthread.h>
#include <sched.h>

XimeaCapture::XimeaCapture(int target_fps) 
    : fps_(target_fps)
{
}

XimeaCapture::~XimeaCapture() {
    Close();
}

bool XimeaCapture::Open() {
    XI_RETURN stat = xiOpenDevice(0, &camera_handle_);
    if (stat != XI_OK) {
        std::cerr << "xiOpenDevice failed: " << stat << std::endl;
        return false;
    }

    // Set basic parameters
    xiSetParamInt(camera_handle_, XI_PRM_WIDTH, width_);
    xiSetParamInt(camera_handle_, XI_PRM_HEIGHT, height_);
    xiSetParamInt(camera_handle_, XI_PRM_OFFSET_X, offset_x_);
    xiSetParamInt(camera_handle_, XI_PRM_OFFSET_Y, offset_y_);

    // Set Framerate first, then Exposure. This ensures the frame period is set
    // before we apply the exposure time, preventing the driver from overriding
    // our exposure setting based on a previous (possibly higher) framerate.
    xiSetParamInt(camera_handle_, XI_PRM_ACQ_TIMING_MODE, XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT);
    xiSetParamFloat(camera_handle_, XI_PRM_FRAMERATE, (float)fps_);
    xiSetParamInt(camera_handle_, XI_PRM_EXPOSURE, exposure_us_);

    // Disable supported auto features to ensure stable performance at 1000fps.
    // Do not touch AUTO_WB: this camera/transport stack reports it unsupported.
    xiSetParamInt(camera_handle_, XI_PRM_AEAG, XI_OFF);
    xiSetParamInt(camera_handle_, XI_PRM_FFC, XI_OFF);

    // Set analog gain
    xiSetParamInt(camera_handle_, XI_PRM_GAIN_SELECTOR, XI_GAIN_SELECTOR_ANALOG_ALL);
    float max_gain = 0.0f;
    xiGetParamFloat(camera_handle_, XI_PRM_GAIN XI_PRM_INFO_MAX, &max_gain);
    
    float target_gain = (gain_db_ >= 0.0f) ? std::min(gain_db_, max_gain) : max_gain;
    xiSetParamFloat(camera_handle_, XI_PRM_GAIN, target_gain);
    std::cout << "Camera analog gain set to: " << target_gain << " dB (max: " << max_gain << ")" << std::endl;

    // Configure GPUDirect RDMA based on xiCUDASample
    cudaSetDevice(0);
    cudaSetDeviceFlags(cudaDeviceMapHost); // Recommended in sample

    xiSetParamInt(camera_handle_, XI_PRM_TRANSPORT_DATA_TARGET, XI_TRANSPORT_DATA_TARGET_GPU_RAM);
    xiSetParamInt(camera_handle_, XI_PRM_IMAGE_DATA_FORMAT, XI_FRM_TRANSPORT_DATA);
    xiSetParamInt(camera_handle_, XI_PRM_OUTPUT_DATA_BIT_DEPTH, 8);

    // Buffer management tuned for high-rate acquisition
    xiSetParamInt(camera_handle_, XI_PRM_BUFFER_POLICY, XI_BP_UNSAFE);
    xiSetParamInt(camera_handle_, XI_PRM_BUFFERS_QUEUE_SIZE, 129);

    // Get actual payload size to set acquisition buffer size
    int payload_size = 0;
    xiGetParamInt(camera_handle_, XI_PRM_IMAGE_PAYLOAD_SIZE, &payload_size);
    if (payload_size <= 0) payload_size = width_ * height_;

    // Set Acquisition Buffer Size (RDMA is limited by BAR size, often 256MB)
    // With Above 4G Decoding/ReBAR enabled, we set this to 150 frames (approx 600MB)
    // to provide a large safety margin against OS scheduling jitter at 1000Hz.
    xiSetParamInt(camera_handle_, XI_PRM_ACQ_BUFFER_SIZE, payload_size * 150);

    // Get actual dimensions
    xiGetParamInt(camera_handle_, XI_PRM_WIDTH, &width_);
    xiGetParamInt(camera_handle_, XI_PRM_HEIGHT, &height_);

    return true;
}

void XimeaCapture::Close() {
    StopAcquisition();
    Cleanup();
}

void XimeaCapture::Cleanup() {
    if (camera_handle_) {
        xiCloseDevice(camera_handle_);
        camera_handle_ = nullptr;
    }
}

bool XimeaCapture::StartAcquisition(FrameCallback callback, ErrorCallback error_cb) {
    frame_callback_ = callback;
    error_callback_ = error_cb;
    
    frames_captured_ = 0;
    frames_dropped_ = 0;
    timeouts_ = 0;

    XI_RETURN stat = xiStartAcquisition(camera_handle_);
    if (stat != XI_OK) {
        std::cerr << "xiStartAcquisition failed: " << stat << std::endl;
        return false;
    }
    keep_running_ = true;
    capture_thread_ = std::thread(&XimeaCapture::CaptureLoop, this);
    return true;
}

bool XimeaCapture::StopAcquisition() {
    keep_running_ = false;
    if (capture_thread_.joinable()) capture_thread_.join();
    if (camera_handle_) {
        xiStopAcquisition(camera_handle_);
    }
    return true;
}

XimeaCapture::Stats XimeaCapture::GetStats() const {
    return {frames_captured_, frames_dropped_, timeouts_};
}

void XimeaCapture::CaptureLoop() {
    // Set this thread to real-time priority (SCHED_FIFO)
    // Requires permissions set in /etc/security/limits.d/99-ximea.conf
    struct sched_param param;
    param.sched_priority = 90;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        std::cerr << "Warning: Failed to set thread priority to SCHED_FIFO." << std::endl;
    }

    uint64_t last_expected_ts = 0;
    uint64_t frame_interval_us = 1000000 / fps_;

    while (keep_running_) {
        XI_IMG image = {};
        image.size = sizeof(image);
        // Block up to 100ms for an image
        XI_RETURN res = xiGetImage(camera_handle_, 100, &image); 
        
        if (res == XI_OK) {
            uint64_t ts = (uint64_t)image.tsSec * 1000000ULL + (uint64_t)image.tsUSec;
            
            // Basic drop detection based on timestamps
            if (last_expected_ts != 0) {
                if (ts > last_expected_ts + frame_interval_us * 1.5) {
                    uint64_t dropped = (ts - last_expected_ts) / frame_interval_us - 1;
                    frames_dropped_ += dropped;
                }
            }
            last_expected_ts = ts;
            frames_captured_++;

            if (frame_callback_) {
                // In RDMA mode, image.bp is a pointer to GPU memory
                frame_callback_(image.bp, ts);
            }
        } else if (res == XI_TIMEOUT) {
            timeouts_++;
        } else {
            std::cerr << "xiGetImage failed: " << res << std::endl;
            if (error_callback_) {
                error_callback_("xiGetImage failed with error " + std::to_string(res));
            }
            break; 
        }
    }
}
