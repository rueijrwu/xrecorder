#include "gop_scheduler.h"

#include <algorithm>
#include <limits>
#include <numeric>

GopScheduler::GopScheduler(std::vector<LaneWeight> lanes) {
    SetWeights(std::move(lanes));
}

void GopScheduler::SetWeights(std::vector<LaneWeight> lanes) {
    std::lock_guard<std::mutex> lock(mutex_);
    lanes_.clear();
    lanes_.reserve(lanes.size());
    for (const auto& lw : lanes) lanes_.push_back({lw.lane_id, lw.weight, 0.0});
}

int GopScheduler::AssignLane(const std::function<uint32_t(int lane_id)>& queue_depth_fn,
                              uint32_t max_queue_depth) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lanes_.empty()) return -1;

    double total = 0.0;
    for (auto& l : lanes_) {
        l.current += l.weight;
        total += l.weight;
    }

    std::vector<size_t> order(lanes_.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [this](size_t a, size_t b) { return lanes_[a].current > lanes_[b].current; });

    int chosen = -1;
    for (size_t idx : order) {
        if (queue_depth_fn(lanes_[idx].lane_id) < max_queue_depth) {
            chosen = static_cast<int>(idx);
            break;
        }
    }
    if (chosen < 0) {
        // Every lane is over the bound — fall back to the least-loaded one
        // rather than stalling the scheduler (capture must never block).
        uint32_t best_depth = std::numeric_limits<uint32_t>::max();
        for (size_t i = 0; i < lanes_.size(); ++i) {
            uint32_t d = queue_depth_fn(lanes_[i].lane_id);
            if (d < best_depth) {
                best_depth = d;
                chosen = static_cast<int>(i);
            }
        }
    }

    lanes_[static_cast<size_t>(chosen)].current -= total;
    return lanes_[static_cast<size_t>(chosen)].lane_id;
}
