#define GST_USE_UNSTABLE_API
#include "encoder_lane.h"

#include <iostream>
#include <sstream>

#include <gst/video/video.h>

#include "time_utils.h"

EncoderLane::EncoderLane(const Config& cfg)
    : cfg_(cfg),
      nv12_pool_(cfg.pool_size, nullptr),
      slot_in_use_(cfg.pool_size),
      pool_events_(cfg.pool_size, nullptr) {
    cudaSetDevice(cfg_.gpu_id);
    const size_t nv12_size = static_cast<size_t>(cfg_.width) * cfg_.height * 3 / 2;
    for (int i = 0; i < cfg_.pool_size; ++i) {
        cudaMalloc(&nv12_pool_[i], nv12_size);
        cudaEventCreateWithFlags(&pool_events_[i], cudaEventDisableTiming);
        slot_in_use_[i].store(false);
    }
    // Every encoder pipeline owns an independent non-blocking CUDA stream.
    // This prevents legacy default-stream synchronization from coupling lanes.
    cudaStreamCreateWithFlags(&convert_stream_, cudaStreamNonBlocking);
}

EncoderLane::~EncoderLane() {
    Stop();
    cudaSetDevice(cfg_.gpu_id);
    for (int i = 0; i < cfg_.pool_size; ++i) {
        if (nv12_pool_[i]) cudaFree(nv12_pool_[i]);
        if (pool_events_[i]) cudaEventDestroy(pool_events_[i]);
    }
    if (convert_stream_) cudaStreamDestroy(convert_stream_);
}

bool EncoderLane::Start(ChunkCallback chunk_cb, ErrorCallback error_cb) {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (started_.load(std::memory_order_relaxed)) return true;

    chunk_callback_ = std::move(chunk_cb);
    error_callback_ = std::move(error_cb);
    submitted_ = 0;
    completed_ = 0;
    dropped_ = 0;
    queue_depth_ = 0;
    pipeline_failed_ = false;
    stopping_ = false;

    cuda_ctx_ = gst_cuda_context_new(cfg_.gpu_id);
    if (!cuda_ctx_) {
        std::cerr << "EncoderLane[" << cfg_.lane_id << "]: Failed to create CUDA context on GPU "
                  << cfg_.gpu_id << std::endl;
        return false;
    }
    cuda_allocator_ = GST_CUDA_ALLOCATOR(g_object_new(GST_TYPE_CUDA_ALLOCATOR, NULL));
    gst_cuda_allocator_set_active(cuda_allocator_, TRUE);
    gst_video_info_set_format(&video_info_, GST_VIDEO_FORMAT_NV12, cfg_.width, cfg_.height);

    std::stringstream ss;
    ss << "appsrc name=mysrc caps=\"video/x-raw(memory:CUDAMemory), format=NV12, width="
       << cfg_.width << ", height=" << cfg_.height << ", framerate=" << cfg_.fps << "/1\" ! "
       << "queue max-size-buffers=" << cfg_.pool_size << " max-size-bytes=0 max-size-time=0 ! "
       << "nvh264enc name=enc cuda-device-id=" << cfg_.gpu_id
       << " preset=p1 rc-mode=cbr gop-size=" << cfg_.gop_size
       << " bitrate=" << cfg_.bitrate_kbps
       << " bframes=0 rc-lookahead=0 zerolatency=true ! "
       << "h264parse config-interval=-1 ! "
       << "appsink name=mysink emit-signals=false sync=false max-buffers=8 drop=false";

    GError* parse_err = nullptr;
    pipeline_ = gst_parse_launch(ss.str().c_str(), &parse_err);
    if (parse_err) {
        std::cerr << "EncoderLane[" << cfg_.lane_id << "]: parse error: " << parse_err->message
                  << std::endl;
        g_error_free(parse_err);
    }
    if (!pipeline_) return false;

    appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "mysrc");
    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "mysink");
    if (!appsrc_ || !appsink_) {
        std::cerr << "EncoderLane[" << cfg_.lane_id << "]: Failed to get appsrc/appsink"
                  << std::endl;
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        return false;
    }
    g_object_set(G_OBJECT(appsrc_), "is-live", TRUE, "format", GST_FORMAT_TIME, "block", FALSE,
                 "max-bytes", static_cast<guint64>(0), NULL);

    GstBus* bus = gst_element_get_bus(pipeline_);
    gst_bus_add_watch(bus, on_bus_message, this);
    gst_object_unref(bus);

    loop_ = g_main_loop_new(NULL, FALSE);
    loop_thread_ = std::thread([this]() { g_main_loop_run(loop_); });

    gst_element_set_state(pipeline_, GST_STATE_PLAYING);

    keep_running_ = true;
    push_thread_ = std::thread(&EncoderLane::PushLoop, this);
    pull_thread_ = std::thread(&EncoderLane::PullLoop, this);

    started_.store(true, std::memory_order_relaxed);
    return true;
}

void EncoderLane::Stop() {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (!started_.load(std::memory_order_relaxed)) return;

    stopping_.store(true, std::memory_order_relaxed);
    keep_running_ = false;
    submit_cv_.notify_all();
    if (push_thread_.joinable()) push_thread_.join();

    if (pipeline_) {
        GstBus* bus = gst_element_get_bus(pipeline_);
        if (appsrc_) gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));
        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus, 10 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
        if (msg) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError* err = NULL;
                gst_message_parse_error(msg, &err, NULL);
                std::cerr << "EncoderLane[" << cfg_.lane_id << "]: error during Stop: "
                          << err->message << std::endl;
                g_error_free(err);
            }
            gst_message_unref(msg);
        }
        gst_object_unref(bus);
    }

    if (pull_thread_.joinable()) pull_thread_.join();

    if (pipeline_) {
        if (loop_) {
            g_main_loop_quit(loop_);
            if (loop_thread_.joinable()) loop_thread_.join();
            g_main_loop_unref(loop_);
            loop_ = nullptr;
        }
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        if (appsrc_) {
            gst_object_unref(appsrc_);
            appsrc_ = nullptr;
        }
        if (appsink_) {
            gst_object_unref(appsink_);
            appsink_ = nullptr;
        }
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    } else if (loop_) {
        g_main_loop_quit(loop_);
        if (loop_thread_.joinable()) loop_thread_.join();
        g_main_loop_unref(loop_);
        loop_ = nullptr;
    }

    if (cuda_allocator_) {
        gst_object_unref(cuda_allocator_);
        cuda_allocator_ = nullptr;
    }
    if (cuda_ctx_) {
        gst_object_unref(cuda_ctx_);
        cuda_ctx_ = nullptr;
    }

    {
        std::lock_guard<std::mutex> lock2(submit_mutex_);
        std::queue<PendingSubmit> empty;
        std::swap(submit_queue_, empty);
    }
    {
        std::lock_guard<std::mutex> lock2(pending_mutex_);
        pending_frames_.clear();
    }
    for (int i = 0; i < cfg_.pool_size; ++i) slot_in_use_[i].store(false);
    next_slot_ = 0;
    queue_depth_ = 0;
    started_.store(false, std::memory_order_relaxed);
}

void* EncoderLane::ReserveSlot(int* out_slot_index) {
    int idx = next_slot_.fetch_add(1, std::memory_order_relaxed) % cfg_.pool_size;
    if (slot_in_use_[idx].load(std::memory_order_acquire)) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    slot_in_use_[idx].store(true, std::memory_order_release);
    *out_slot_index = idx;
    return nv12_pool_[idx];
}

void EncoderLane::SubmitSlot(int slot_index, uint64_t gop_index, uint64_t source_index,
                              uint64_t pts_ns, bool force_idr) {
    cudaEventRecord(pool_events_[slot_index], convert_stream_);
    {
        std::lock_guard<std::mutex> lock(submit_mutex_);
        submit_queue_.push({slot_index, gop_index, source_index, pts_ns, force_idr, now_us()});
        queue_depth_.fetch_add(1, std::memory_order_relaxed);
    }
    submit_cv_.notify_one();
    submitted_.fetch_add(1, std::memory_order_relaxed);
}

void EncoderLane::ReleaseSlot(int index) {
    if (index >= 0 && index < cfg_.pool_size) slot_in_use_[index].store(false, std::memory_order_release);
}

void EncoderLane::RequestKeyframe() {
    if (!pipeline_) return;
    gst_element_send_event(pipeline_,
                            gst_video_event_new_upstream_force_key_unit(GST_CLOCK_TIME_NONE, TRUE, 0));
}

void EncoderLane::PushLoop() {
    while (true) {
        PendingSubmit item;
        {
            std::unique_lock<std::mutex> lock(submit_mutex_);
            submit_cv_.wait(lock, [this] { return !submit_queue_.empty() || !keep_running_; });
            if (submit_queue_.empty() && !keep_running_) break;
            item = submit_queue_.front();
            submit_queue_.pop();
            queue_depth_.fetch_sub(1, std::memory_order_relaxed);
        }

        if (!appsrc_ || pipeline_failed_.load(std::memory_order_relaxed)) {
            ReleaseSlot(item.slot_index);
            dropped_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        cudaEventSynchronize(pool_events_[item.slot_index]);
        encode_latency_.Record(static_cast<double>(now_us() - item.enqueue_time_us));

        if (item.force_idr) RequestKeyframe();

        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_frames_.emplace_back(item.gop_index, item.source_index, item.pts_ns);
        }

        auto* info = new std::pair<EncoderLane*, int>(this, item.slot_index);
        GstMemory* mem = gst_cuda_allocator_alloc_wrapped(
            cuda_allocator_, cuda_ctx_, NULL, &video_info_,
            (CUdeviceptr)nv12_pool_[item.slot_index], info, [](gpointer d) {
                auto i = static_cast<std::pair<EncoderLane*, int>*>(d);
                i->first->ReleaseSlot(i->second);
                delete i;
            });

        GstBuffer* buffer = gst_buffer_new();
        gst_buffer_append_memory(buffer, mem);
        GST_BUFFER_PTS(buffer) = item.pts_ns;
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, cfg_.fps);

        GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
        if (ret == GST_FLOW_FLUSHING) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
        } else if (ret == GST_FLOW_EOS || ret < GST_FLOW_OK) {
            pipeline_failed_.store(true, std::memory_order_relaxed);
            if (!stopping_.load(std::memory_order_relaxed) && error_callback_) {
                error_callback_("EncoderLane pipeline flow error");
            }
        }
    }
}

void EncoderLane::PullLoop() {
    while (true) {
        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink_));
        if (!sample) {
            if (gst_app_sink_is_eos(GST_APP_SINK(appsink_)) || !keep_running_.load(std::memory_order_relaxed))
                break;
            continue;
        }

        GstBuffer* buf = gst_sample_get_buffer(sample);
        GstMapInfo map;
        EncodedChunk chunk;
        chunk.lane_id = cfg_.lane_id;
        if (buf && gst_buffer_map(buf, &map, GST_MAP_READ)) {
            chunk.data.assign(map.data, map.data + map.size);
            chunk.pts_ns = GST_BUFFER_PTS(buf);
            gst_buffer_unmap(buf, &map);
        }

        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            if (!pending_frames_.empty()) {
                auto [gop, src, pts] = pending_frames_.front();
                pending_frames_.pop_front();
                chunk.gop_index = gop;
                chunk.source_index = src;
                if (chunk.pts_ns == 0) chunk.pts_ns = pts;
            }
        }

        completed_.fetch_add(1, std::memory_order_relaxed);
        gst_sample_unref(sample);
        if (chunk_callback_) chunk_callback_(std::move(chunk));
    }
}

EncoderLane::Stats EncoderLane::GetStats() const {
    Stats s;
    s.submitted = submitted_.load(std::memory_order_relaxed);
    s.completed = completed_.load(std::memory_order_relaxed);
    s.dropped = dropped_.load(std::memory_order_relaxed);
    s.queue_depth = queue_depth_.load(std::memory_order_relaxed);
    s.encode_latency = encode_latency_.GetSnapshot();
    return s;
}

gboolean EncoderLane::on_bus_message(GstBus*, GstMessage* msg, gpointer user_data) {
    EncoderLane* self = static_cast<EncoderLane*>(user_data);
    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        GError* err = NULL;
        gchar* debug = NULL;
        gst_message_parse_error(msg, &err, &debug);
        std::string err_msg = "EncoderLane[" + std::to_string(self->cfg_.lane_id) +
                               "] Error: " + std::string(err->message);
        std::cerr << err_msg << std::endl;
        self->pipeline_failed_.store(true, std::memory_order_relaxed);
        if (!self->stopping_.load(std::memory_order_relaxed) && self->error_callback_) {
            self->error_callback_(err_msg);
        }
        g_error_free(err);
        g_free(debug);
    }
    return TRUE;
}
