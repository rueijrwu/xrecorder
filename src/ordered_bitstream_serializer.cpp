#define GST_USE_UNSTABLE_API
#include "ordered_bitstream_serializer.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

#include "time_utils.h"

OrderedBitstreamSerializer::OrderedBitstreamSerializer(const Config& cfg) : cfg_(cfg) {}

OrderedBitstreamSerializer::~OrderedBitstreamSerializer() { Stop(); }

bool OrderedBitstreamSerializer::Start(ErrorCallback error_cb) {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (started_.load(std::memory_order_relaxed)) return true;

    error_callback_ = std::move(error_cb);
    pipeline_failed_ = false;
    stopping_ = false;
    next_expected_gop_ = 0;
    bytes_written_ = 0;
    frames_written_ = 0;
    frame_gaps_ = 0;
    gop_gaps_ = 0;

    std::stringstream ss;
    ss << "appsrc name=finalsrc caps=\"video/x-h264, stream-format=byte-stream, alignment=au\" ! "
       << "h264parse ! matroskamux ! filesink location=\"" << cfg_.output_path << "/"
       << cfg_.output_name << ".mkv\"";

    GError* parse_err = nullptr;
    pipeline_ = gst_parse_launch(ss.str().c_str(), &parse_err);
    if (parse_err) {
        std::cerr << "OrderedBitstreamSerializer: parse error: " << parse_err->message << std::endl;
        g_error_free(parse_err);
    }
    if (!pipeline_) return false;

    appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "finalsrc");
    if (!appsrc_) {
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
    emit_thread_ = std::thread(&OrderedBitstreamSerializer::EmitLoop, this);

    started_.store(true, std::memory_order_relaxed);
    return true;
}

void OrderedBitstreamSerializer::Stop() {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (!started_.load(std::memory_order_relaxed)) return;

    stopping_.store(true, std::memory_order_relaxed);
    keep_running_ = false;
    cv_.notify_all();
    if (emit_thread_.joinable()) emit_thread_.join();

    if (pipeline_) {
        GstBus* bus = gst_element_get_bus(pipeline_);
        if (appsrc_) gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));
        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus, 10 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
        if (msg) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError* err = NULL;
                gst_message_parse_error(msg, &err, NULL);
                std::cerr << "OrderedBitstreamSerializer: error during Stop: " << err->message
                          << std::endl;
                g_error_free(err);
            }
            gst_message_unref(msg);
        }
        gst_object_unref(bus);

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
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    } else if (loop_) {
        g_main_loop_quit(loop_);
        if (loop_thread_.joinable()) loop_thread_.join();
        g_main_loop_unref(loop_);
        loop_ = nullptr;
    }

    {
        std::lock_guard<std::mutex> lock2(mutex_);
        pending_.clear();
    }
    started_.store(false, std::memory_order_relaxed);
}

void OrderedBitstreamSerializer::SetGopExpectedCount(uint64_t gop_index, uint32_t count) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& buf = pending_[gop_index];
        buf.nominal_expected = count;
        buf.expected_set = true;
        if (buf.first_seen_us == 0) buf.first_seen_us = now_us();
    }
    cv_.notify_one();
}

void OrderedBitstreamSerializer::NotifyFramesDropped(uint64_t gop_index, uint32_t count) {
    if (count == 0) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& buf = pending_[gop_index];
        buf.dropped += count;
        if (buf.first_seen_us == 0) buf.first_seen_us = now_us();
        frame_gaps_.fetch_add(count, std::memory_order_relaxed);
    }
    cv_.notify_one();
}

void OrderedBitstreamSerializer::PushChunk(EncodedChunk&& chunk) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& buf = pending_[chunk.gop_index];
        if (buf.first_seen_us == 0) buf.first_seen_us = now_us();
        buf.chunks.push_back(std::move(chunk));
    }
    cv_.notify_one();
}

void OrderedBitstreamSerializer::EmitChunk(const EncodedChunk& chunk) {
    if (!appsrc_ || pipeline_failed_.load(std::memory_order_relaxed)) return;
    if (chunk.data.empty()) return;

    uint8_t* copy = static_cast<uint8_t*>(g_malloc(chunk.data.size()));
    std::memcpy(copy, chunk.data.data(), chunk.data.size());
    GstBuffer* buffer = gst_buffer_new_wrapped(copy, chunk.data.size());
    GST_BUFFER_PTS(buffer) = chunk.pts_ns;
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, cfg_.fps);

    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
    if (ret == GST_FLOW_OK) {
        bytes_written_.fetch_add(chunk.data.size(), std::memory_order_relaxed);
        frames_written_.fetch_add(1, std::memory_order_relaxed);
    } else if (ret == GST_FLOW_EOS || ret < GST_FLOW_OK) {
        pipeline_failed_.store(true, std::memory_order_relaxed);
        if (!stopping_.load(std::memory_order_relaxed) && error_callback_) {
            error_callback_("OrderedBitstreamSerializer pipeline flow error");
        }
    }
}

void OrderedBitstreamSerializer::EmitLoop() {
    auto expected_chunks = [](const GopBuffer& buf) -> uint32_t {
        if (!buf.expected_set) return 0;
        return (buf.nominal_expected > buf.dropped)
                   ? (buf.nominal_expected - buf.dropped)
                   : 0;
    };

    while (true) {
        std::vector<EncodedChunk> to_emit;
        bool have_gop = false;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(100), [this, &expected_chunks] {
                if (!keep_running_) return true;
                auto it = pending_.find(next_expected_gop_);
                if (it == pending_.end() || !it->second.expected_set) return false;
                return it->second.chunks.size() >= expected_chunks(it->second);
            });

            auto it = pending_.find(next_expected_gop_);
            if (it != pending_.end()) {
                uint32_t expected = expected_chunks(it->second);
                bool complete = it->second.expected_set && it->second.chunks.size() >= expected;
                bool stalled = it->second.first_seen_us != 0 &&
                    now_us() - it->second.first_seen_us > cfg_.stall_timeout_ms * 1000ULL;

                if (complete || stalled || !keep_running_) {
                    // Explicit camera/application drops are already included in
                    // frame_gaps_. Only add chunks that disappeared after they
                    // were expected to reach the encoder/serializer.
                    if (!complete && it->second.expected_set &&
                        expected > it->second.chunks.size()) {
                        frame_gaps_.fetch_add(expected - it->second.chunks.size(),
                                              std::memory_order_relaxed);
                    }
                    if (!complete) gop_gaps_.fetch_add(1, std::memory_order_relaxed);
                    to_emit = std::move(it->second.chunks);
                    pending_.erase(it);
                    next_expected_gop_++;
                    have_gop = true;
                }
            } else if (!pending_.empty() &&
                       (now_us() - pending_.begin()->second.first_seen_us >
                            cfg_.stall_timeout_ms * 1000ULL ||
                        !keep_running_)) {
                gop_gaps_.fetch_add(1, std::memory_order_relaxed);
                next_expected_gop_ = pending_.begin()->first;
                continue;
            } else if (!keep_running_ && pending_.empty()) {
                break;
            }
        }

        if (have_gop) {
            std::sort(to_emit.begin(), to_emit.end(),
                      [](const EncodedChunk& a, const EncodedChunk& b) {
                          return a.source_index < b.source_index;
                      });
            for (const auto& chunk : to_emit) EmitChunk(chunk);
        }
    }
}

OrderedBitstreamSerializer::Stats OrderedBitstreamSerializer::GetStats() const {
    Stats s;
    s.bytes_written = bytes_written_.load(std::memory_order_relaxed);
    s.frames_written = frames_written_.load(std::memory_order_relaxed);
    s.frame_gaps = frame_gaps_.load(std::memory_order_relaxed);
    s.gop_gaps = gop_gaps_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        s.pending_gops = pending_.size();
    }
    return s;
}

gboolean OrderedBitstreamSerializer::on_bus_message(GstBus*, GstMessage* msg, gpointer user_data) {
    auto* self = static_cast<OrderedBitstreamSerializer*>(user_data);
    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        GError* err = NULL;
        gchar* debug = NULL;
        gst_message_parse_error(msg, &err, &debug);
        std::string err_msg = "OrderedBitstreamSerializer Error: " + std::string(err->message);
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
