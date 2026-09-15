#include "multi_nv_recorder.h"

#include <algorithm>
#include <iostream>

#include "cuda_utils.h"

MultiNvRecorder::MultiNvRecorder(const Config& cfg, const std::string& output_path,
                                  const std::string& output_name)
    : cfg_(cfg), output_path_(output_path), output_name_(output_name) {
    gray8_size_ = static_cast<size_t>(cfg_.width) * cfg_.height;

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

        if (spec.gpu_id != cfg_.capture_gpu_id) {
            // Transfer ownership and fallback transfer stream are per encoder
            // lane. Two 5070 lanes therefore remain independent even though
            // they share the same physical destination GPU.
            SetupRemoteTransfer(static_cast<int>(i), spec.gpu_id);
        }
    }
    scheduler_ = std::make_unique<GopScheduler>(weights);
}

MultiNvRecorder::~MultiNvRecorder() {
    Stop();
    for (auto& [lane_id, rt] : remote_transfers_) {
        (void)lane_id;

        cudaSetDevice(rt.gpu_id);
        for (void* p : rt.gray8_staging) {
            if (p) cudaFree(p);
        }

        if (!rt.peer_ok) {
            cudaSetDevice(cfg_.capture_gpu_id);
            for (cudaEvent_t e : rt.xfer_events) {
                if (e) cudaEventDestroy(e);
            }
            if (rt.capture_stream) cudaStreamDestroy(rt.capture_stream);
            for (void* p : rt.pinned_staging) {
                if (p) cudaFreeHost(p);
            }
        }
    }
}

void MultiNvRecorder::SetupRemoteTransfer(int lane_id, int lane_gpu_id) {
    RemoteTransfer rt;
    rt.gpu_id = lane_gpu_id;

    // P2P work is issued on this destination lane's ConvertStream(), so the
    // destination GPU needs peer access to the capture GPU.
    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, lane_gpu_id, cfg_.capture_gpu_id);
    if (can_access) {
        cudaSetDevice(lane_gpu_id);
        cudaError_t err = cudaDeviceEnablePeerAccess(cfg_.capture_gpu_id, 0);
        rt.peer_ok = (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled);
        if (err == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
    }

    std::cout << "MultiNvRecorder: lane " << lane_id
              << " cross-GPU path capture_gpu=" << cfg_.capture_gpu_id
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
        // Every remote encoder pipeline gets its own capture-side transfer
        // stream. This avoids serializing both 5070 lanes through one D2H
        // stream when P2P is unavailable.
        cudaStreamCreateWithFlags(&rt.capture_stream, cudaStreamNonBlocking);
        for (int i = 0; i < cfg_.pool_size_per_lane; ++i) {
            cudaHostAlloc(&rt.pinned_staging[i], gray8_size_, cudaHostAllocDefault);
            cudaEventCreateWithFlags(&rt.xfer_events[i], cudaEventDisableTiming);
        }
    }

    remote_transfers_.emplace(lane_id, std::move(rt));
}

void MultiNvRecorder::BuildSerializer() {
    OrderedBitstreamSerializer::Config ser_cfg;
    ser_cfg.width = cfg_.width;
    ser_cfg.height = cfg_.height;
    ser_cfg.fps = cfg_.fps;
    ser_cfg.output_path = output_path_;
    ser_cfg.output_name = output_name_;
    ser_cfg.stall_timeout_ms = cfg_.stall_timeout_ms;
    serializer_ = std::make_unique<OrderedBitstreamSerializer>(ser_cfg);
}

void MultiNvRecorder::RegisterMissingFrames(uint64_t first_source_index, uint64_t count) {
    // Split a camera-side sequence gap by GOP instead of iterating once per
    // missing frame. This keeps capture-thread work bounded by the number of
    // GOPs crossed by the gap.
    const uint64_t gop_size = static_cast<uint64_t>(cfg_.gop_size);
    uint64_t source = first_source_index;
    uint64_t remaining = count;

    while (remaining > 0) {
        uint64_t gop_index = source / gop_size;
        uint64_t offset = source % gop_size;
        uint32_t in_this_gop = static_cast<uint32_t>(
            std::min<uint64_t>(remaining, gop_size - offset));

        serializer_->SetGopExpectedCount(gop_index, static_cast<uint32_t>(cfg_.gop_size));
        serializer_->NotifyFramesDropped(gop_index, in_this_gop);

        source += in_this_gop;
        remaining -= in_this_gop;
    }
}

bool MultiNvRecorder::Start(ErrorCallback error_cb) {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (started_.load(std::memory_order_relaxed)) return true;

    error_callback_ = std::move(error_cb);
    have_acq_sequence_ = false;
    first_acq_nframe_ = 0;
    last_acq_nframe_ = 0;
    last_source_index_ = 0;
    first_ts_us_ = 0;
    frames_submitted_ = 0;
    frames_dropped_ = 0;
    cross_gpu_frames_ = 0;
    gop_routes_.clear();

    BuildSerializer();
    if (!serializer_->Start(error_callback_)) return false;

    for (auto& lane : lanes_) {
        bool ok = lane->Start(
            [this](EncodedChunk&& chunk) { serializer_->PushChunk(std::move(chunk)); },
            error_callback_);
        if (!ok) {
            std::cerr << "MultiNvRecorder: failed to start lane " << lane->lane_id() << std::endl;
            for (auto& started_lane : lanes_) {
                if (started_lane.get() == lane.get()) break;
                started_lane->Stop();
            }
            serializer_->Stop();
            return false;
        }
    }

    started_.store(true, std::memory_order_relaxed);
    return true;
}

void MultiNvRecorder::Stop() {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (!started_.load(std::memory_order_relaxed)) return;

    for (auto& lane : lanes_) lane->Stop();
    if (serializer_) serializer_->Stop();

    gop_routes_.clear();
    started_.store(false, std::memory_order_relaxed);
}

void MultiNvRecorder::PushFrame(void* xi_gpu_ptr, uint64_t timestamp_us, uint32_t acq_nframe) {
    if (!started_.load(std::memory_order_relaxed)) return;

    uint64_t source_index = 0;
    if (!have_acq_sequence_) {
        have_acq_sequence_ = true;
        first_acq_nframe_ = acq_nframe;
        last_acq_nframe_ = acq_nframe;
        last_source_index_ = 0;
        source_index = 0;
    } else {
        // Unsigned subtraction naturally handles the uint32 acq_nframe wrap.
        uint32_t step = acq_nframe - last_acq_nframe_;

        // step==0 is a duplicate. A delta > 2^31 is treated as an unexpected
        // backwards/reset sequence rather than a multi-billion-frame gap.
        if (step == 0 || step > 0x80000000u) {
            frames_dropped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        if (step > 1) {
            RegisterMissingFrames(last_source_index_ + 1,
                                  static_cast<uint64_t>(step) - 1);
        }

        source_index = last_source_index_ + static_cast<uint64_t>(step);
        last_source_index_ = source_index;
        last_acq_nframe_ = acq_nframe;
    }

    uint64_t gop_index = source_index / static_cast<uint64_t>(cfg_.gop_size);

    // Assign the GOP when its first ACTUAL frame arrives. If the nominal GOP
    // boundary frame was missing, this first actual frame must still become
    // the IDR for that independently encoded GOP.
    auto route_it = gop_routes_.find(gop_index);
    if (route_it == gop_routes_.end()) {
        int lane_idx = scheduler_->AssignLane(
            [this](int id) { return lanes_[static_cast<size_t>(id)]->QueueDepth(); },
            cfg_.max_queue_depth);
        route_it = gop_routes_.emplace(gop_index, GopRoute{lane_idx, true}).first;
        serializer_->SetGopExpectedCount(gop_index, static_cast<uint32_t>(cfg_.gop_size));

        if (gop_index >= 8) gop_routes_.erase(gop_index - 8);
    }

    GopRoute& route = route_it->second;
    EncoderLane* lane = lanes_[static_cast<size_t>(route.lane_idx)].get();

    int slot_index = -1;
    void* nv12_dst = lane->ReserveSlot(&slot_index);
    if (!nv12_dst) {
        frames_dropped_.fetch_add(1, std::memory_order_relaxed);
        serializer_->NotifyFramesDropped(gop_index, 1);
        // Keep needs_idr=true if this was the first actual frame of the GOP.
        return;
    }

    if (first_ts_us_ == 0) first_ts_us_ = timestamp_us;
    uint64_t pts_ns = (timestamp_us - first_ts_us_) * 1000ULL;

    if (lane->gpu_id() == cfg_.capture_gpu_id) {
        cudaSetDevice(lane->gpu_id());
        convert_gray8_to_nv12_gpu(static_cast<const uint8_t*>(xi_gpu_ptr),
                                   static_cast<uint8_t*>(nv12_dst), cfg_.width, cfg_.height,
                                   lane->ConvertStream());
    } else {
        cross_gpu_frames_.fetch_add(1, std::memory_order_relaxed);
        RemoteTransfer& rt = remote_transfers_.at(route.lane_idx);
        void* gray8_dst = rt.gray8_staging[static_cast<size_t>(slot_index)];

        if (rt.peer_ok) {
            // Each lane uses its own destination ConvertStream, so P2P copy
            // and conversion for lane 1 and lane 2 can progress concurrently.
            cudaSetDevice(lane->gpu_id());
            cudaMemcpyPeerAsync(gray8_dst, lane->gpu_id(),
                                xi_gpu_ptr, cfg_.capture_gpu_id,
                                gray8_size_, lane->ConvertStream());
        } else {
            // Each remote lane owns a separate capture-side stream. D2H work
            // therefore does not serialize between the two 5070 pipelines.
            cudaSetDevice(cfg_.capture_gpu_id);
            cudaMemcpyAsync(rt.pinned_staging[static_cast<size_t>(slot_index)], xi_gpu_ptr,
                            gray8_size_, cudaMemcpyDeviceToHost, rt.capture_stream);
            cudaEventRecord(rt.xfer_events[static_cast<size_t>(slot_index)], rt.capture_stream);

            cudaSetDevice(lane->gpu_id());
            cudaStreamWaitEvent(lane->ConvertStream(),
                                rt.xfer_events[static_cast<size_t>(slot_index)], 0);
            cudaMemcpyAsync(gray8_dst, rt.pinned_staging[static_cast<size_t>(slot_index)],
                            gray8_size_, cudaMemcpyHostToDevice, lane->ConvertStream());
        }

        cudaSetDevice(lane->gpu_id());
        convert_gray8_to_nv12_gpu(static_cast<const uint8_t*>(gray8_dst),
                                   static_cast<uint8_t*>(nv12_dst), cfg_.width, cfg_.height,
                                   lane->ConvertStream());
    }

    bool force_idr = route.needs_idr;
    lane->SubmitSlot(slot_index, gop_index, source_index, pts_ns, force_idr);
    route.needs_idr = false;
    frames_submitted_.fetch_add(1, std::memory_order_relaxed);
}

MultiNvRecorder::Stats MultiNvRecorder::GetStats() const {
    Stats s;
    s.frames_submitted = frames_submitted_.load(std::memory_order_relaxed);
    s.frames_dropped = frames_dropped_.load(std::memory_order_relaxed);
    s.cross_gpu_frames = cross_gpu_frames_.load(std::memory_order_relaxed);
    for (const auto& lane : lanes_) s.lane_stats.push_back(lane->GetStats());
    if (serializer_) s.serializer_stats = serializer_->GetStats();
    return s;
}
