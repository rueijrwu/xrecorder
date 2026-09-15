#include "gst_previewer.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>
#include <gst/video/navigation.h>

GstPreviewer::GstPreviewer(int width, int height, int display_fps)
    : width_(width), height_(height), display_fps_(display_fps)
{
}

GstPreviewer::~GstPreviewer() {
    Stop();
}

bool GstPreviewer::Start(ErrorCallback error_cb, KeyCallback key_cb) {
    gst_init(NULL, NULL);
    error_callback_ = error_cb;
    key_callback_ = key_cb;
    frames_displayed_ = 0;
    frame_available_ = false;
    latest_timestamp_us_ = 0;
    last_preview_copy_us_ = 0;

    size_t size = static_cast<size_t>(width_) * height_;

    // Owned GPU GRAY8 buffer — image.bp is deep-copied here so the XIMEA ring
    // buffer slot is released as soon as frame_callback_ returns.
    cudaError_t err = cudaMalloc(&preview_gray8_gpu_, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate preview GPU buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Pinned host memory for fast D2H transfer
    err = cudaHostAlloc(&pinned_buffer_, size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory: " << cudaGetErrorString(err) << std::endl;
        cudaFree(preview_gray8_gpu_);
        preview_gray8_gpu_ = nullptr;
        return false;
    }

    // Calculate preview dimensions that maintain the original aspect ratio
    // Target a reasonable width (e.g., 1280) and calculate proportional height
    int preview_w = 1280;
    int preview_h = (height_ * preview_w) / width_;
    preview_h = (preview_h / 2) * 2; // Ensure even height

    std::stringstream ss;
    ss << "appsrc name=mysrc caps=\"video/x-raw, format=GRAY8, width=" << width_ << ", height=" << height_ 
         << ", framerate=" << display_fps_ << "/1\" ! queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 "
         << "! videoconvert ! videoscale ! video/x-raw, width="
         << preview_w << ", height=" << preview_h << " ! autovideosink sync=false";

    pipeline_ = gst_parse_launch(ss.str().c_str(), NULL);
    if (!pipeline_) {
        std::cerr << "Failed to create GStreamer pipeline for previewer." << std::endl;
        cudaFreeHost(pinned_buffer_);
        return false;
    }

    appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "mysrc");
    g_object_set(G_OBJECT(appsrc_),
                 "is-live", TRUE,
                 "format", GST_FORMAT_TIME,
                 "do-timestamp", FALSE,
                 "block", FALSE,
                 NULL);
    
    // Setup error monitoring on the bus
    GstBus* bus = gst_element_get_bus(pipeline_);
    gst_bus_add_watch(bus, on_bus_message, this);
    gst_object_unref(bus);

    // Set up loop
    loop_ = g_main_loop_new(NULL, FALSE);
    loop_thread_ = std::thread([this]() {
        g_main_loop_run(loop_);
    });

    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    
    keep_running_ = true;
    preview_thread_ = std::thread(&GstPreviewer::PreviewLoop, this);
    
    return true;
}

void GstPreviewer::Stop() {
    keep_running_ = false;
    if (preview_thread_.joinable()) preview_thread_.join();

    if (pipeline_)
        gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));

    // Stop GLib loop before unreffing — same race condition as GstRecorder.
    if (loop_) {
        g_main_loop_quit(loop_);
        if (loop_thread_.joinable()) loop_thread_.join();
        g_main_loop_unref(loop_);
        loop_ = nullptr;
    }

    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(appsrc_);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }

    if (preview_gray8_gpu_) {
        cudaFree(preview_gray8_gpu_);
        preview_gray8_gpu_ = nullptr;
    }
    if (pinned_buffer_) {
        cudaFreeHost(pinned_buffer_);
        pinned_buffer_ = nullptr;
    }
}

void GstPreviewer::UpdateFrame(void* gpu_buffer, uint64_t timestamp_us) {
    const uint64_t copy_interval_us = 1000000ULL / static_cast<uint64_t>(std::max(1, display_fps_));
    const uint64_t now_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    bool should_copy = false;

    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        // Throttle preview deep-copy to display cadence (e.g., 60 Hz) using
        // local monotonic time, so preview does not freeze if camera timestamps
        // repeat/stall.
        if (!frame_available_ || now_us >= (last_preview_copy_us_ + copy_interval_us)) {
            last_preview_copy_us_ = now_us;
            should_copy = true;
        }
    }

    if (!should_copy || !preview_gray8_gpu_) return;

    // Synchronous D2D avoids building an async stream backlog at high capture FPS.
    // This keeps preview latency bounded to the most recent snapshot.
    {
        std::lock_guard<std::mutex> gpu_lock(gpu_copy_mutex_);
        cudaMemcpy(preview_gray8_gpu_, gpu_buffer,
                   static_cast<size_t>(width_) * height_,
                   cudaMemcpyDeviceToDevice);
    }

    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_timestamp_us_ = timestamp_us;
        frame_available_ = true;
    }
}

GstPreviewer::Stats GstPreviewer::GetStats() const {
    return {frames_displayed_};
}

gboolean GstPreviewer::on_bus_message(GstBus* bus, GstMessage* msg, gpointer user_data) {
    GstPreviewer* self = static_cast<GstPreviewer*>(user_data);
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError* err = NULL;
            gchar* debug = NULL;
            gst_message_parse_error(msg, &err, &debug);
            std::string err_msg = "GStreamer Previewer Error: " + std::string(err->message);
            std::cerr << err_msg << std::endl;
            if (self->error_callback_) self->error_callback_(err_msg);
            g_error_free(err);
            g_free(debug);
            break;
        }
        case GST_MESSAGE_ELEMENT: {
            if (gst_navigation_message_get_type(msg) == GST_NAVIGATION_MESSAGE_EVENT) {
                GstEvent* event = NULL;
                if (gst_navigation_message_parse_event(msg, &event)) {
                    GstNavigationEventType type = gst_navigation_event_get_type(event);
                    if (type == GST_NAVIGATION_EVENT_KEY_PRESS) {
                        const gchar* key;
                        if (gst_navigation_event_parse_key_event(event, &key)) {
                            if (self->key_callback_) self->key_callback_(key);
                        }
                    }
                    gst_event_unref(event);
                }
            }
            break;
        }
        default:
            break;
    }
    return TRUE;
}

void GstPreviewer::PreviewLoop() {
    int interval_ms = 1000 / display_fps_;
    size_t size = static_cast<size_t>(width_) * height_;
    const GstClockTime frame_duration = gst_util_uint64_scale_int(1, GST_SECOND, std::max(1, display_fps_));
    uint64_t preview_index = 0;

    while (keep_running_) {
        auto start_time = std::chrono::steady_clock::now();

        if (frame_available_ && pinned_buffer_ && preview_gray8_gpu_) {
            uint64_t timestamp_us = 0;
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                timestamp_us = latest_timestamp_us_;
            }

            {
                std::lock_guard<std::mutex> gpu_lock(gpu_copy_mutex_);
                cudaMemcpy(pinned_buffer_, preview_gray8_gpu_, size,
                           cudaMemcpyDeviceToHost);
            }

            // Use app-owned memory for each pushed buffer so lifetime is decoupled
            // from our reusable pinned staging buffer.
            GstBuffer* buffer = gst_buffer_new_allocate(NULL, size, NULL);
            if (!buffer) {
                std::cerr << "Warning: previewer failed to allocate GstBuffer" << std::endl;
                continue;
            }

            GstMapInfo map;
            if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
                std::cerr << "Warning: previewer failed to map GstBuffer" << std::endl;
                gst_buffer_unref(buffer);
                continue;
            }
            std::memcpy(map.data, pinned_buffer_, size);
            gst_buffer_unmap(buffer, &map);

            // Drive preview timing from display cadence to avoid sink delays caused
            // by sparse/high-rate camera timestamps.
            const GstClockTime pts = preview_index * frame_duration;
            GST_BUFFER_PTS(buffer) = pts;
            GST_BUFFER_DTS(buffer) = pts;
            GST_BUFFER_DURATION(buffer) = frame_duration;

            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
            if (ret == GST_FLOW_OK) {
                preview_index++;
                frames_displayed_++;
            } else {
                std::cerr << "Warning: previewer push buffer failed: " << ret
                          << " ts=" << timestamp_us << std::endl;
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        int sleep_ms = std::max(0, (int)(interval_ms - elapsed));
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }
}
