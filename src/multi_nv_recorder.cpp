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

        // Every lane gets an application-owned GRAY8 ring and an independent
        // copy stream, including the local PRO lane. This establishes one
        // identical ownership boundary for all three pipelines.
        SetupLaneTransfer(static_cast<int>(i), spec.gpu_id);
    }

    scheduler_ = std::make_unique<GopScheduler>(weights);
}

MultiNvRecorder::~MultiNvRecorder() {
    Stop();

    for (auto& [lane_id, transfer] : lane_transfers_) {
        (void)lane_id;

        cudaSetDevice(transfer.gpu_id);
        for (void* p : transfer.gray8_staging) {
            if (p) cudaFree(p);
        }
        for (cudaEvent_t e : transfer.copy_done_events) {
            if (e) cudaEventDestroy(e);
        }
        if (transfer.copy_stream) cudaStreamDestroy(transfer.copy_stream);

        if (!transfer.peer_ok && transfer.gpu_id != cfg_.capture_gpu_id) {
            cudaSetDevice(cfg_.capture_gpu_id);
            for (cudaEvent_t e : transfer.d2h_done_events) {
                if (e) cudaEventDestroy(e);
            }
            if (transfer.capture_stream) cudaStreamDestroy(transfer.capture_stream);
            for (void* p : transfer.pinned_staging) {
                if (p) cudaFreeHost(p);
            }
        }
    }
}

void MultiNvRecorder::SetupLaneTransfer(int lane_id, int lane_gpu_id) {
    LaneTransfer transfer;
    transfer.gpu_id = lane_gpu_id;

    const bool local = (lane_gpu_id == cfg_.capture_gpu_id);
    transfer.peer_ok = local;

    if (!local) {
        int can_access = 0;
        cudaDeviceCanAccessPeer(&can_access, lane_gpu_id, cfg_.capture_gpu_id);
        if (can_access) {
            cudaSetDevice(lane_gpu_id);
            cudaError_t err = cudaDeviceEnablePeerAccess(cfg_.capture_gpu_id, 0);
            transfer.peer_ok =
                (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled);
            if (err == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        }
    }

    cudaSetDevice(lane_gpu_id);
    cudaStreamCreateWithFlags(&transfer.copy_stream, cudaStreamNonBlocking);

    transfer.gray8_staging.assign(cfg_.pool_size_per_lane, nullptr);
    transfer.copy_done_events.assign(cfg_.pool_size_per_lane, nullptr);
    for (int i = 0; i < cfg_.pool_size_per_lane; ++i) {
        cudaMalloc(&transfer.gray8_staging[i], gray8_size_);
        cudaEventCreateWithFlags(&transfer.copy_done_events[i], cudaEventDisableTiming);
    }

    if (!local && !transfer.peer_ok) {
        transfer.pinned_staging.assign(cfg_.pool_size_per_lane, nullptr);
        transfer.d2h_done_events.assign(cfg_.pool_size_per_lane, nullptr);

        cudaSetDevice(cfg_.capture_gpu_id);
        cudaStreamCreateWithFlags(&transfer.capture_stream, cudaStreamNonBlocking);
        for (int i = 0; i < cfg_.pool_size_per_lane; ++i) {
            cudaHostAlloc(&transfer.pinned_staging[i], gray8_size_, cudaHostAllocDefault);
            cudaEventCreateWithFlags(&transfer.d2h_done_events[i], cudaEventDisableTiming);
        }
    }

    std::cout << "MultiNvRecorder: lane " << lane_id
              << " copy path capture_gpu=" << cfg_.capture_gpu_id
              << " -> lane_gpu=" << lane_gpu_id;
    if (local) {
        std::cout << " uses local D2D ownership copy";
    } else if (transfer.peer_ok) {
        std::cout << " uses direct P2P ownership copy";
    } else {
        std::cout << " uses pinned-host ownership copy fallback";
    }
    std::cout << std::endl;

    lane_transfers_.emplace(lane_id, std::move(transfer));
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
    // Logical FrameGroup boundaries are still the encoder GOP boundaries.
    // Split camera-side sequence gaps by those boundaries so missing frames
    // never collapse the source timeline.
    const uint64_t group_size = static_cast<uint64_t>(cfg_.gop_size);
    uint64_t source = first_source_index;
    uint64_t remaining = count;

    while (remaining > 0) {
        const uint64_t group_index = source / group_size;
        const uint64_t offset = source % group_size;
        const uint32_t in_this_group = static_cast<uint32_t>(
            std::min<uint64_t>(remaining, group_size - offset));

        serializer_->SetGopExpectedCount(group_index,
                                         static_cast<uint32_t>(cfg_.gop_size));
        serializer_->NotifyFramesDropped(group_index, in_this_group);

        source += in_this_group;
        remaining -= in_this_group;
    }
}

bool MultiNvRecorder::Start(ErrorCallback error_cb) {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (started_.load(std::memory_order_relaxed)) return true;

    error_callback_ = std::move(error_cb);
    have_acq_sequence_ = false;
    last_acq_nframe_ = 0;
    last_source_index_ = 0;
    first_ts_us_ = 0;
    frames_submitted_ = 0;
    frames_dropped_ = 0;
    cross_gpu_frames_ = 0;
    frame_group_routes_.clear();

    BuildSerializer();
    if (!serializer_->Start(error_callback_)) return false;

    for (auto& lane : lanes_) {
        bool ok = lane->Start(
            [this](EncodedChunk&& chunk) { serializer_->PushChunk(std::move(chunk)); },
            error_callback_);
        if (!ok) {
            std::cerr << "MultiNvRecorder: failed to start lane " << lane->lane_id()
                      << std::endl;
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

    frame_group_routes_.clear();
    started_.store(false, std::memory_order_relaxed);
}

void MultiNvRecorder::PushFrame(void* xi_gpu_ptr, uint64_t timestamp_us,
                                uint32_t acq_nframe) {
    if (!started_.load(std::memory_order_relaxed)) return;

    uint64_t source_index = 0;
    if (!have_acq_sequence_) {
        have_acq_sequence_ = true;
        last_acq_nframe_ = acq_nframe;
        last_source_index_ = 0;
        source_index = 0;
    } else {
        uint32_t step = acq_nframe - last_acq_nframe_;

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

    // FrameGroup is a logical routing interval only. We do NOT wait for all
    // frames in the group. Every frame streams immediately after its ownership
    // copy into the one lane selected for this interval.
    const uint64_t group_index =
        source_index / static_cast<uint64_t>(cfg_.gop_size);

    auto route_it = frame_group_routes_.find(group_index);
    if (route_it == frame_group_routes_.end()) {
        const int lane_idx = scheduler_->AssignLane(
            [this](int id) { return lanes_[static_cast<size_t>(id)]->QueueDepth(); },
            cfg_.max_queue_depth);
        route_it = frame_group_routes_
                       .emplace(group_index, FrameGroupRoute{lane_idx, true})
                       .first;

        // The logical FrameGroup and encoded H.264 GOP use the same index and
        // nominal frame count, which lets the serializer restore group order.
        serializer_->SetGopExpectedCount(group_index,
                                         static_cast<uint32_t>(cfg_.gop_size));

        if (group_index >= 8) frame_group_routes_.erase(group_index - 8);
    }

    FrameGroupRoute& route = route_it->second;
    EncoderLane* lane = lanes_[static_cast<size_t>(route.lane_idx)].get();
    LaneTransfer& transfer = lane_transfers_.at(route.lane_idx);

    int slot_index = -1;
    void* nv12_dst = lane->ReserveSlot(&slot_index);
    if (!nv12_dst) {
        frames_dropped_.fetch_add(1, std::memory_order_relaxed);
        serializer_->NotifyFramesDropped(group_index, 1);
        return;
    }

    void* gray8_dst = transfer.gray8_staging[static_cast<size_t>(slot_index)];
    cudaEvent_t copy_done =
        transfer.copy_done_events[static_cast<size_t>(slot_index)];

    const bool local = (lane->gpu_id() == cfg_.capture_gpu_id);
    cudaError_t copy_submit = cudaSuccess;

    if (local) {
        cudaSetDevice(cfg_.capture_gpu_id);
        copy_submit = cudaMemcpyAsync(gray8_dst, xi_gpu_ptr, gray8_size_,
                                      cudaMemcpyDeviceToDevice,
                                      transfer.copy_stream);
        if (copy_submit == cudaSuccess) {
            copy_submit = cudaEventRecord(copy_done, transfer.copy_stream);
        }
    } else if (transfer.peer_ok) {
        cross_gpu_frames_.fetch_add(1, std::memory_order_relaxed);
        cudaSetDevice(lane->gpu_id());
        copy_submit = cudaMemcpyPeerAsync(gray8_dst, lane->gpu_id(),
                                          xi_gpu_ptr, cfg_.capture_gpu_id,
                                          gray8_size_, transfer.copy_stream);
        if (copy_submit == cudaSuccess) {
            copy_submit = cudaEventRecord(copy_done, transfer.copy_stream);
        }
    } else {
        cross_gpu_frames_.fetch_add(1, std::memory_order_relaxed);

        cudaSetDevice(cfg_.capture_gpu_id);
        copy_submit = cudaMemcpyAsync(
            transfer.pinned_staging[static_cast<size_t>(slot_index)],
            xi_gpu_ptr, gray8_size_, cudaMemcpyDeviceToHost,
            transfer.capture_stream);
        if (copy_submit == cudaSuccess) {
            copy_submit = cudaEventRecord(
                transfer.d2h_done_events[static_cast<size_t>(slot_index)],
                transfer.capture_stream);
        }

        if (copy_submit == cudaSuccess) {
            cudaSetDevice(lane->gpu_id());
            copy_submit = cudaStreamWaitEvent(
                transfer.copy_stream,
                transfer.d2h_done_events[static_cast<size_t>(slot_index)], 0);
        }
        if (copy_submit == cudaSuccess) {
            copy_submit = cudaMemcpyAsync(
                gray8_dst,
                transfer.pinned_staging[static_cast<size_t>(slot_index)],
                gray8_size_, cudaMemcpyHostToDevice,
                transfer.copy_stream);
        }
        if (copy_submit == cudaSuccess) {
            copy_submit = cudaEventRecord(copy_done, transfer.copy_stream);
        }
    }

    if (copy_submit != cudaSuccess) {
        lane->CancelReservedSlot(slot_index);
        frames_dropped_.fetch_add(1, std::memory_order_relaxed);
        serializer_->NotifyFramesDropped(group_index, 1);
        if (error_callback_) {
            error_callback_(std::string("Ownership copy submission failed: ") +
                            cudaGetErrorString(copy_submit));
        }
        return;
    }

    // The only capture-side synchronization. We never wait for completion of
    // the rest of the logical FrameGroup, conversion, or NVENC.
    cudaSetDevice(lane->gpu_id());
    cudaError_t copy_status = cudaEventSynchronize(copy_done);
    if (copy_status != cudaSuccess) {
        lane->CancelReservedSlot(slot_index);
        frames_dropped_.fetch_add(1, std::memory_order_relaxed);
        serializer_->NotifyFramesDropped(group_index, 1);
        if (error_callback_) {
            error_callback_(std::string("Ownership copy failed: ") +
                            cudaGetErrorString(copy_status));
        }
        return;
    }

    convert_gray8_to_nv12_gpu(static_cast<const uint8_t*>(gray8_dst),
                               static_cast<uint8_t*>(nv12_dst),
                               cfg_.width, cfg_.height,
                               lane->ConvertStream());

    if (first_ts_us_ == 0) first_ts_us_ = timestamp_us;
    const uint64_t pts_ns = (timestamp_us - first_ts_us_) * 1000ULL;

    const bool force_idr = route.needs_idr;
    lane->SubmitSlot(slot_index, group_index, source_index, pts_ns, force_idr);
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
