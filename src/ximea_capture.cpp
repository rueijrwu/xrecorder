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
    // Match XIMEA's GPUDirect sample ordering as closely as possible:
    // select/configure the CUDA device before xiOpenDevice().
    cudaError_t cuda_stat = cudaSetDevice(gpu_id_);
    if (cuda_stat != cudaSuccess) {
        std::cerr << "cudaSetDevice(" << gpu_id_ << ") before xiOpenDevice failed: "
                  << cudaGetErrorString(cuda_stat) << std::endl;
        return false;
    }

    cuda_stat = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (cuda_stat != cudaSuccess && cuda_stat != cudaErrorSetOnActiveProcess) {
        std::cerr << "cudaSetDeviceFlags(cudaDeviceMapHost) failed: "
                  << cudaGetErrorString(cuda_stat) << std::endl;
        return false;
    }
    if (cuda_stat == cudaErrorSetOnActiveProcess) {
        // CUDA runtime may already be initialized by manager startup. This is
        // non-fatal for RDMA, but log it because XIMEA's sample sets the flag
        // before any context is created.
        cudaGetLastError();
        std::cerr << "Warning: cudaDeviceMapHost could not be set because CUDA is already active; "
                     "continuing with the existing context."
                  << std::endl;
    }

    XI_RETURN stat = xiOpenDevice(0, &camera_handle_);
    if (stat != XI_OK) {
        std::cerr << "xiOpenDevice failed: " << stat << std::endl;
        return false;
    }

    xiSetParamInt(camera_handle_, XI_PRM_WIDTH, width_);
    xiSetParamInt(camera_handle_, XI_PRM_HEIGHT, height_);
    xiSetParamInt(camera_handle_, XI_PRM_OFFSET_X, offset_x_);
    xiSetParamInt(camera_handle_, XI_PRM_OFFSET_Y, offset_y_);

    xiSetParamInt(camera_handle_, XI_PRM_ACQ_TIMING_MODE, XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT);
    xiSetParamFloat(camera_handle_, XI_PRM_FRAMERATE, (float)fps_);
    xiSetParamInt(camera_handle_, XI_PRM_EXPOSURE, exposure_us_);

    xiSetParamInt(camera_handle_, XI_PRM_AEAG, XI_OFF);
    xiSetParamInt(camera_handle_, XI_PRM_FFC, XI_OFF);

    xiSetParamInt(camera_handle_, XI_PRM_GAIN_SELECTOR, XI_GAIN_SELECTOR_ANALOG_ALL);
    float max_gain = 0.0f;
    xiGetParamFloat(camera_handle_, XI_PRM_GAIN XI_PRM_INFO_MAX, &max_gain);

    float target_gain = (gain_db_ >= 0.0f) ? std::min(gain_db_, max_gain) : max_gain;
    xiSetParamFloat(camera_handle_, XI_PRM_GAIN, target_gain);
    std::cout << "Camera analog gain set to: " << target_gain << " dB (max: " << max_gain << ")" << std::endl;

    xiSetParamInt(camera_handle_, XI_PRM_IMAGE_DATA_FORMAT, XI_FRM_TRANSPORT_DATA);
    xiSetParamInt(camera_handle_, XI_PRM_OUTPUT_DATA_BIT_DEPTH, 8);
    xiSetParamInt(camera_handle_, XI_PRM_TRANSPORT_DATA_TARGET,
                  XI_TRANSPORT_DATA_TARGET_GPU_RAM);
    xiSetParamInt(camera_handle_, XI_PRM_BUFFER_POLICY, XI_BP_UNSAFE);

    int payload_size = 0;
    xiGetParamInt(camera_handle_, XI_PRM_IMAGE_PAYLOAD_SIZE, &payload_size);
    if (payload_size <= 0) payload_size = width_ * height_;

    // XIMEA's GPUDirect sample uses exactly four payload buffers for RDMA.
    // Keep the diagnostic path identical and let XiAPI manage its queue size.
    constexpr int kAcqBufferFrames = 4;
    const int requested_acq_buffer_size = payload_size * kAcqBufferFrames;
    stat = xiSetParamInt(camera_handle_, XI_PRM_ACQ_BUFFER_SIZE,
                         requested_acq_buffer_size);
    if (stat != XI_OK) {
        std::cerr << "xiSetParam(ACQ_BUFFER_SIZE) failed: " << stat << std::endl;
        return false;
    }

    int actual_acq_buffer_size = 0;
    int actual_queue_size = 0;
    xiGetParamInt(camera_handle_, XI_PRM_ACQ_BUFFER_SIZE, &actual_acq_buffer_size);
    xiGetParamInt(camera_handle_, XI_PRM_BUFFERS_QUEUE_SIZE, &actual_queue_size);
    std::cout << "XimeaCapture: ACQ_BUFFER_SIZE requested=" << requested_acq_buffer_size
              << " actual=" << actual_acq_buffer_size
              << ", BUFFERS_QUEUE_SIZE left at XiAPI default actual=" << actual_queue_size
              << std::endl;

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

    // Rebind the selected GPUDirect GPU immediately before acquisition in case
    // recorder construction touched the other CUDA device.
    cudaError_t cuda_stat = cudaSetDevice(gpu_id_);
    if (cuda_stat != cudaSuccess) {
        std::cerr << "cudaSetDevice(" << gpu_id_ << ") before xiStartAcquisition failed: "
                  << cudaGetErrorString(cuda_stat) << std::endl;
        return false;
    }
    std::cout << "XimeaCapture: binding CUDA device " << gpu_id_
              << " immediately before xiStartAcquisition" << std::endl;

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
    struct sched_param param;
    param.sched_priority = 90;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        std::cerr << "Warning: Failed to set thread priority to SCHED_FIFO." << std::endl;
    }

    have_last_acq_nframe_ = false;

    while (keep_running_) {
        XI_IMG image = {};
        image.size = sizeof(image);
        XI_RETURN res = xiGetImage(camera_handle_, 100, &image);

        if (res == XI_OK) {
            uint64_t ts = (uint64_t)image.tsSec * 1000000ULL + (uint64_t)image.tsUSec;

            if (have_last_acq_nframe_ && image.acq_nframe > last_acq_nframe_ + 1) {
                frames_dropped_ += (image.acq_nframe - last_acq_nframe_ - 1);
            }
            last_acq_nframe_ = image.acq_nframe;
            have_last_acq_nframe_ = true;
            frames_captured_++;

            if (frame_callback_) {
                frame_callback_(image.bp, ts, image.acq_nframe);
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
