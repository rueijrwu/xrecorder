#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

// Assigns whole GOPs (never individual frames) to encoder lanes, using
// smooth weighted round-robin biased by each lane's measured capacity, and
// skipping any lane whose queue is currently over a bound (RECORD.md
// "Scheduler" section).
class GopScheduler {
public:
    struct LaneWeight {
        int lane_id;
        double weight;  // relative sustained capacity, e.g. measured fps
    };

    explicit GopScheduler(std::vector<LaneWeight> lanes);

    void SetWeights(std::vector<LaneWeight> lanes);

    // queue_depth_fn(lane_id) must return that lane's current queue depth.
    // Returns the chosen lane_id.
    int AssignLane(const std::function<uint32_t(int lane_id)>& queue_depth_fn,
                    uint32_t max_queue_depth);

private:
    struct State {
        int lane_id;
        double weight;
        double current = 0.0;
    };

    std::mutex mutex_;
    std::vector<State> lanes_;
};
