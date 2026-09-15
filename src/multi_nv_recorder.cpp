#include "multi_nv_recorder.h"

#include <iostream>

#include "cuda_utils.h"

MultiNvRecorder::MultiNvRecorder(const Config& cfg, const std::string& output_path,
                                  const std::string& output_name)
    : cfg_(cfg), output_path_(output_path), output_name_(output_name) {
    gray8_size_ = static_cast<size_t>(cfg_.width) * cfg_.height;

    cudaSetDevice(cfg_.capture_gpu_id);
    cudaStreamCreate(&capture_xfer_stream_);

    std::vector<GopScheduler::LaneWeight> weights;
    for (size_t i = 0; i < cfg_.lanes.size(); ++i) {
        const auto& spec = cfg_.lanes[i];
        EncoderLane::Config lane_cfg;
        lane_cfg.lane_id = static_cast<int>(i);
        lane_cfg.gpu_id = spec.gpu_id;
        lane_cfg.width = cfg_.width;
        lane_cfg.height = cfg_.height;
        lane_cfg.fps = cfg_.fps;
        lane_cfg.gop_size = cfg_.gop_size;
        lane_cfg.pool_size = cfg_.pool_size_per_lane;
        lane_cfg.bitrate_kbps = spec.bitrate_kbps;
        lanes_.push_back(std::make_unique<EncoderLane>(lane_cfg));
        weights.push_back({static_cast<int>(i), spec.weight});

        if (spec.gpu_id != cfg_.capture_gpu_id &&
            remote_transfers_.find(spec.gpu_id) == remote_transfers_.end()) {
            SetupRemoteTransfer(spec.gpu_id);
        }
    }
    scheduler_ = std::make_unique<GopScheduler>(weights);

    OrderedBitstreamSerializer::Config ser_cfg;
    ser_cfg.width = cfg_.width;
    ser_cfg.height = cfg_.height;
    ser_cfg.fps = cfg_.fps;
    ser_cfg.output_path = output_path_;
    ser_cfg.output_name = output_name_;
    ser_cfg.stall_timeout_ms = cfg_.stall_timeout_ms;
    serializer_ = std::make_unique<OrderedBitstreamSerializer>(ser_cfg);
}

MultiNvRecorder::~MultiNvRecorder() {
    Stop();
    for (auto& [gpu_id, rt] : remote_transfers_) {
        cudaSetDevice(gpu_id);
        for (void* p : rt.gray8_staging) if (p) cudaFree(p);
        for (cudaEvent_t e : rt.xfer_events) if (e) cudaEventDestroy(e);
        for (void* p : rt.pinned_staging) if (p) cudaFreeHost(p);
    }
    cudaSetDevice(cfg_.capture_gpu_id);
    if (capture_xfer_stream_) cudaStreamDestroy(capture_xfer_stream_);
}

void MultiNvRecorder::SetupRemoteTransfer(int lane_gpu_id) {
    RemoteTransfer rt;

    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, lane_gpu_id, cfg_.capture_gpu_id);
    if (can_access) {
        cudaSetDevice(lane_gpu_id);
        cudaError_t err = cudaDeviceEnablePeerAccess(cfg_.capture_gpu_id, 0);
        rt.peer_ok = (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled);
    }
    std::cout << "MultiNvRecorder: cross-GPU path capture_gpu=" << cfg_.capture_gpu_id
              << " -> lane_gpu=" << lane_gpu_id
              << (rt.peer_ok ? " uses P2P" : " uses pinned-host staging fallback") << std::endl;

    rt.gray8_staging.assign(cfg_.pool_size_per_lane, nullptr);
    cudaSetDevice(lane_gpu_id);
    for (int i = 0; i < cfg_.pool_size_per_lane; ++i) {
        cudaMalloc(&rt.gray8_staging[i], gray8_size_);
    }

    if (!rt.peer_ok) {
        rt.pinned_staging.assign(cfg_.pool_size_per_lane, nullptr);
        rt.xfer_events.assign(cfg_.pool_size_per_lane, nullptr);
        cudaSetDevice(cfg_.capture_gpu_id);
        for (int i = 0; i < cfg_.pool_size_per_lane; ++i) {
            cudaHostAlloc(&rt.pinned_staging[i], gray8_size_, cudaHostAllocDefault);
            cudaEventCreateWithFlags(&rt.xfer_events[i], cudaEventDisableTiming);
        }
    }

    remote_transfers_.emplace(lane_gpu_id, std::move(rt));
}

bool MultiNvRecorder::Start(ErrorCallback error_cb) {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (started_.load(std::memory_order_relaxed)) return true;

    error_callback_ = std::move(error_cb);
    source_index_ = 0;
    first_ts_us_ = 0;
    frames_submitted_ = 0;
    frames_dropped_ = 0;
    cross_gpu_frames_ = 0;
    gop_lane_map_.clear();

    if (!serializer_->Start(error_callback_)) return false;

    for (auto& lane : lanes_) {
        bool ok = lane->Start(
            [this](EncodedChunk&& chunk) { serializer_->PushChunk(std::move(chunk)); },
            error_callback_);
        if (!ok) {
            std::cerr << "MultiNvRecorder: failed to start lane " << lane->lane_id() << std::endl;
            return false;
        }
    }

    started_.store(true, std::memory_order_relaxed);
    return true;
}

void MultiNvRecorder::Stop() {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (!started_.load(std::memory_order_relaxed)) return;

    // Drain every lane (pushes all reserved slots and waits for EOS) before
    // stopping the serializer, so no encoded chunk is lost.
    for (auto& lane : lanes_) lane->Stop();
    serializer_->Stop();

    gop_lane_map_.clear();
    started_.store(false, std::memory_order_relaxed);
}

void MultiNvRecorder::PushFrame(void* xi_gpu_ptr, uint64_t timestamp_us, uint32_t /*acq_nframe*/) {
    if (!started_.load(std::memory_order_relaxed)) return;

    uint64_t source_index = source_index_++;
    uint64_t gop_index = source_index / static_cast<uint64_t>(cfg_.gop_size);
    bool is_gop_start = (source_index % static_cast<uint64_t>(cfg_.gop_size)) == 0;

    int lane_idx;
    if (is_gop_start) {
        lane_idx = scheduler_->AssignLane(
            [this](int id) { return lanes_[static_cast<size_t>(id)]->QueueDepth(); },
            cfg_.max_queue_depth);
        gop_lane_map_[gop_index] = lane_idx;
        serializer_->SetGopExpectedCount(gop_index, static_cast<uint32_t>(cfg_.gop_size));
        // Bound gop_lane_map_ memory: drop stale entries well behind the
        // live GOP window (serializer keeps up within a handful of GOPs).
        if (gop_index >= 8) gop_lane_map_.erase(gop_index - 8);
    } else {
        auto it = gop_lane_map_.find(gop_index);
        if (it == gop_lane_map_.end()) {
            frames_dropped_.fetch_add(1, std::memory_order_relaxed);
            serializer_->NotifyFrameDropped(gop_index);
            return;
        }
        lane_idx = it->second;
    }

    EncoderLane* lane = lanes_[static_cast<size_t>(lane_idx)].get();

    int slot_index = -1;
    void* nv12_dst = lane->ReserveSlot(&slot_index);
    if (!nv12_dst) {
        frames_dropped_.fetch_add(1, std::memory_order_relaxed);
        serializer_->NotifyFrameDropped(gop_index);
        return;
    }

    if (first_ts_us_ == 0) first_ts_us_ = timestamp_us;
    uint64_t pts_ns = (timestamp_us - first_ts_us_) * 1000ULL;

    if (lane->gpu_id() == cfg_.capture_gpu_id) {
        // Local path: convert directly from the XIMEA GPUDirect pointer into
        // the lane's own NV12 slot — no intermediate GRAY8 copy.
        cudaSetDevice(lane->gpu_id());
        convert_gray8_to_nv12_gpu(static_cast<const uint8_t*>(xi_gpu_ptr),
                                   static_cast<uint8_t*>(nv12_dst), cfg_.width, cfg_.height,
                                   lane->ConvertStream());
    } else {
        cross_gpu_frames_.fetch_add(1, std::memory_order_relaxed);
        RemoteTransfer& rt = remote_transfers_[lane->gpu_id()];
        void* gray8_dst = rt.gray8_staging[static_cast<size_t>(slot_index)];

        if (rt.peer_ok) {
            cudaMemcpyPeerAsync(gray8_dst, lane->gpu_id(), xi_gpu_ptr, cfg_.capture_gpu_id,
                                 gray8_size_, lane->ConvertStream());
        } else {
            cudaSetDevice(cfg_.capture_gpu_id);
            cudaMemcpyAsync(rt.pinned_staging[static_cast<size_t>(slot_index)], xi_gpu_ptr,
                             gray8_size_, cudaMemcpyDeviceToHost, capture_xfer_stream_);
            cudaEventRecord(rt.xfer_events[static_cast<size_t>(slot_index)], capture_xfer_stream_);

            cudaSetDevice(lane->gpu_id());
            cudaStreamWaitEvent(lane->ConvertStream(), rt.xfer_events[static_cast<size_t>(slot_index)], 0);
            cudaMemcpyAsync(gray8_dst, rt.pinned_staging[static_cast<size_t>(slot_index)],
                             gray8_size_, cudaMemcpyHostToDevice, lane->ConvertStream());
        }

        cudaSetDevice(lane->gpu_id());
        convert_gray8_to_nv12_gpu(static_cast<const uint8_t*>(gray8_dst),
                                   static_cast<uint8_t*>(nv12_dst), cfg_.width, cfg_.height,
                                   lane->ConvertStream());
    }

    lane->SubmitSlot(slot_index, gop_index, source_index, pts_ns, is_gop_start);
    frames_submitted_.fetch_add(1, std::memory_order_relaxed);
}

MultiNvRecorder::Stats MultiNvRecorder::GetStats() const {
    Stats s;
    s.frames_submitted = frames_submitted_.load(std::memory_order_relaxed);
    s.frames_dropped = frames_dropped_.load(std::memory_order_relaxed);
    s.cross_gpu_frames = cross_gpu_frames_.load(std::memory_order_relaxed);
    for (const auto& lane : lanes_) s.lane_stats.push_back(lane->GetStats());
    s.serializer_stats = serializer_->GetStats();
    return s;
}
