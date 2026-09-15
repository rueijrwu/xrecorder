#define GST_USE_UNSTABLE_API
#include "gst_recorder.h"

#include <iostream>
#include <sstream>

#include "cuda_utils.h"

GstRecorder::GstRecorder(const std::string& path, const std::string& name,
                         int width, int height, int fps, const std::string& codec)
    : output_path_(path),
      output_name_(name),
      width_(width),
      height_(height),
      fps_(fps),
      codec_(codec) {
  const size_t gray8_size = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  const size_t nv12_size = gray8_size * 3 / 2;
  for (int i = 0; i < POOL_SIZE; ++i) {
    cudaMalloc(&gray8_pool_[i], gray8_size);
    cudaMalloc(&gpu_pool_[i], nv12_size);
    if (codec_ == "raw") cudaHostAlloc(&host_pool_[i], gray8_size, cudaHostAllocDefault);
    slot_in_use_[i] = false;
    cudaEventCreateWithFlags(&pool_events_[i], cudaEventDisableTiming);
  }
  cudaStreamCreate(&record_stream_);
}

GstRecorder::~GstRecorder() {
  Stop();
  for (int i = 0; i < POOL_SIZE; ++i) {
    if (gray8_pool_[i]) cudaFree(gray8_pool_[i]);
    if (gpu_pool_[i]) cudaFree(gpu_pool_[i]);
    if (host_pool_[i]) cudaFreeHost(host_pool_[i]);
    if (pool_events_[i]) cudaEventDestroy(pool_events_[i]);
  }
  if (record_stream_) cudaStreamDestroy(record_stream_);
}

bool GstRecorder::Start(ErrorCallback error_cb) {
  std::lock_guard<std::mutex> lock(stop_mutex_);
  if (started_.load(std::memory_order_relaxed)) return true;

  gst_init(NULL, NULL);
  error_callback_ = error_cb;
  frames_encoded_ = 0;
  frames_dropped_ = 0;
  pipeline_failed_ = false;
  stopping_ = false;
  first_ts_ = 0;

  std::stringstream ss;
  if (codec_ == "raw") {
    ss << "appsrc name=mysrc caps=\"video/x-raw, format=GRAY8, width="
       << width_ << ", height=" << height_ << ", framerate=" << fps_ << "/1\" ! "
       << "queue max-size-buffers=0 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
       << "filesink location=\""
       << output_path_ << "/" << output_name_ << ".raw\"";
  } else {
    cuda_ctx_ = gst_cuda_context_new(0);
    if (!cuda_ctx_) {
      std::cerr << "Failed to create GStreamer CUDA context." << std::endl;
      return false;
    }
    cuda_allocator_ = GST_CUDA_ALLOCATOR(g_object_new(GST_TYPE_CUDA_ALLOCATOR, NULL));
    gst_cuda_allocator_set_active(cuda_allocator_, TRUE);
    gst_video_info_set_format(&video_info_, GST_VIDEO_FORMAT_NV12, width_, height_);

    std::string encoder;
    std::string caps_str;
    if (codec_ == "h265" || codec_ == "hevc") {
      encoder = "nvh265enc";
      caps_str = "video/x-h265";
    } else if (codec_ == "av1") {
      encoder = "nvav1enc";
      caps_str = "video/x-av1";
    } else {
      encoder = "nvh264enc";
      caps_str = "video/x-h264";
    }
    ss << "appsrc name=mysrc caps=\"video/x-raw(memory:CUDAMemory), format=NV12, width="
       << width_ << ", height=" << height_ << ", framerate=" << fps_ << "/1\" ! "
       << "queue max-size-buffers=1000 ! "
       << encoder << " preset=p1 rc-mode=cbr gop-size=30 bitrate=100000 zerolatency=true ! "
       << caps_str << " ! matroskamux ! filesink location=\""
       << output_path_ << "/" << output_name_ << ".mkv\"";
  }

  GError* parse_err = nullptr;
  pipeline_ = gst_parse_launch(ss.str().c_str(), &parse_err);
  if (parse_err) {
    std::cerr << "GStreamer parse error: " << parse_err->message << std::endl;
    g_error_free(parse_err);
  }
  if (!pipeline_) {
    std::cerr << "Failed to create GStreamer pipeline for recorder." << std::endl;
    return false;
  }

  appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "mysrc");
  if (!appsrc_) {
    std::cerr << "Failed to get appsrc from recorder pipeline." << std::endl;
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
    return false;
  }
  g_object_set(G_OBJECT(appsrc_), "is-live", TRUE, "format", GST_FORMAT_TIME,
               "block", FALSE, "max-bytes", static_cast<guint64>(0), NULL);

  GstBus* bus = gst_element_get_bus(pipeline_);
  gst_bus_add_watch(bus, on_bus_message, this);
  gst_object_unref(bus);

  loop_ = g_main_loop_new(NULL, FALSE);
  loop_thread_ = std::thread([this]() { g_main_loop_run(loop_); });

  gst_element_set_state(pipeline_, GST_STATE_PLAYING);

  if (codec_ != "raw") {
    keep_running_ = true;
    recorder_thread_ = std::thread(&GstRecorder::RecorderLoop, this);
  }
  started_.store(true, std::memory_order_relaxed);
  return true;
}

void GstRecorder::Stop() {
  std::lock_guard<std::mutex> lock(stop_mutex_);
  if (!started_.load(std::memory_order_relaxed)) return;

  stopping_.store(true, std::memory_order_relaxed);
  keep_running_ = false;
  queue_cv_.notify_all();
  if (recorder_thread_.joinable()) recorder_thread_.join();

  if (pipeline_) {
    GstBus* bus = gst_element_get_bus(pipeline_);
    if (appsrc_) {
      gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));
    }
    GstMessage* msg = gst_bus_timed_pop_filtered(
        bus, 10 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
    if (msg) {
      if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        GError* err = NULL;
        gst_message_parse_error(msg, &err, NULL);
        std::cerr << "[GstRecorder] Pipeline error during Stop: " << err->message << std::endl;
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

  if (cuda_allocator_) {
    gst_object_unref(cuda_allocator_);
    cuda_allocator_ = nullptr;
  }
  if (cuda_ctx_) {
    gst_object_unref(cuda_ctx_);
    cuda_ctx_ = nullptr;
  }

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!frame_queue_.empty()) frame_queue_.pop();
    for (int i = 0; i < POOL_SIZE; ++i) slot_in_use_[i] = false;
    pool_index_ = 0;
  }
  started_.store(false, std::memory_order_relaxed);
}

void GstRecorder::PushFrame(void* gpu_buffer, uint64_t timestamp_us) {
  if (stopping_.load(std::memory_order_relaxed) ||
      pipeline_failed_.load(std::memory_order_relaxed) || !appsrc_) {
    frames_dropped_++;
    return;
  }

  if (slot_in_use_[pool_index_].load()) {
    frames_dropped_++;
    return;
  }
  slot_in_use_[pool_index_].store(true);

  if (codec_ == "raw") {
    cudaMemcpyAsync(host_pool_[pool_index_], gpu_buffer, static_cast<size_t>(width_) * height_,
                    cudaMemcpyDeviceToHost, record_stream_);
    cudaStreamSynchronize(record_stream_);

    auto info = new std::pair<GstRecorder*, int>(this, pool_index_);
    const size_t gray8_size = static_cast<size_t>(width_) * height_;
    GstMemory* mem = gst_memory_new_wrapped(
        (GstMemoryFlags)0, host_pool_[pool_index_], gray8_size, 0, gray8_size, info,
        [](gpointer d) {
          auto i = static_cast<std::pair<GstRecorder*, int>*>(d);
          i->first->ReleaseSlot(i->second);
          delete i;
        });

    GstBuffer* buffer = gst_buffer_new();
    gst_buffer_append_memory(buffer, mem);
    if (first_ts_ == 0) first_ts_ = timestamp_us;
    GST_BUFFER_PTS(buffer) = (timestamp_us - first_ts_) * 1000;
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, fps_);

    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
    if (ret == GST_FLOW_OK) {
      frames_encoded_++;
    } else if (ret == GST_FLOW_FLUSHING) {
      frames_dropped_++;
    } else if (ret == GST_FLOW_EOS || ret < GST_FLOW_OK) {
      pipeline_failed_.store(true, std::memory_order_relaxed);
      if (!stopping_.load(std::memory_order_relaxed) && error_callback_) {
        error_callback_("Recorder pipeline flow error");
      }
    }
    pool_index_ = (pool_index_ + 1) % POOL_SIZE;
    return;
  }

  void* gray8 = gray8_pool_[pool_index_];
  void* target = gpu_pool_[pool_index_];
  cudaEvent_t ev = pool_events_[pool_index_];

  cudaMemcpyAsync(gray8, gpu_buffer, static_cast<size_t>(width_) * height_,
                  cudaMemcpyDeviceToDevice, record_stream_);
  convert_gray8_to_nv12_gpu(static_cast<const uint8_t*>(gray8),
                            static_cast<uint8_t*>(target), width_, height_, record_stream_);
  cudaEventRecord(ev, record_stream_);

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    frame_queue_.push({target, timestamp_us, ev, pool_index_});
    pool_index_ = (pool_index_ + 1) % POOL_SIZE;
    queue_cv_.notify_one();
  }
}

GstRecorder::Stats GstRecorder::GetStats() const {
  uint32_t current_q = 0;
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    current_q = static_cast<uint32_t>(frame_queue_.size());
  }
  return {frames_encoded_, frames_dropped_, current_q};
}

void GstRecorder::ReleaseSlot(int index) {
  if (index >= 0 && index < POOL_SIZE) slot_in_use_[index].store(false);
}

gboolean GstRecorder::on_bus_message(GstBus*, GstMessage* msg, gpointer user_data) {
  GstRecorder* self = static_cast<GstRecorder*>(user_data);
  if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
    GError* err = NULL;
    gchar* debug = NULL;
    gst_message_parse_error(msg, &err, &debug);
    std::string err_msg = "GStreamer Recorder Error: " + std::string(err->message);
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

void GstRecorder::RecorderLoop() {
  while (true) {
    FrameData data;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this] { return !frame_queue_.empty() || !keep_running_; });
      if (frame_queue_.empty() && !keep_running_) break;
      data = frame_queue_.front();
      frame_queue_.pop();
    }

    if (!appsrc_) continue;
    cudaEventSynchronize(data.ready_event);

    auto info = new std::pair<GstRecorder*, int>(this, data.pool_index);
    GstMemory* mem = gst_cuda_allocator_alloc_wrapped(
        cuda_allocator_, cuda_ctx_, NULL, &video_info_, (CUdeviceptr)data.gpu_buffer, info,
        [](gpointer d) {
          auto i = static_cast<std::pair<GstRecorder*, int>*>(d);
          i->first->ReleaseSlot(i->second);
          delete i;
        });

    GstBuffer* buffer = gst_buffer_new();
    gst_buffer_append_memory(buffer, mem);

    if (first_ts_ == 0) first_ts_ = data.timestamp_us;
    GST_BUFFER_PTS(buffer) = (data.timestamp_us - first_ts_) * 1000;
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, fps_);

    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
    if (ret == GST_FLOW_OK) {
      frames_encoded_++;
    } else if (ret == GST_FLOW_FLUSHING) {
      frames_dropped_++;
    } else if (ret == GST_FLOW_EOS || ret < GST_FLOW_OK) {
      pipeline_failed_.store(true, std::memory_order_relaxed);
      if (!stopping_.load(std::memory_order_relaxed) && error_callback_) {
        error_callback_("Recorder pipeline flow error");
      }
      break;
    }
  }
}
