#pragma once

#include <cstdint>
#include <vector>

// Authoritative capture-side frame descriptor, handed from XimeaCapture to
// the recorder. acq_nframe (not the timestamp) is the authoritative source
// sequence id — see RECORD.md "Use acq_nframe as the authoritative frame
// sequence".
struct CaptureFrame {
    void*    xi_gpu_ptr;
    uint64_t xi_timestamp_us;
    uint32_t acq_nframe;
};

// Per-frame lifecycle metadata, enough to reconstruct/validate the capture
// timeline and localize any loss to a specific pipeline stage.
struct FrameMetadata {
    uint32_t acq_nframe = 0;
    uint64_t xi_timestamp_us = 0;
    uint64_t source_index = 0;   // monotonic app-side sequence, 0-based
    uint64_t gop_index = 0;
    int      lane_id = -1;
    int      gpu_id = -1;
    uint64_t capture_arrival_us = 0;
    uint64_t conversion_submit_us = 0;
    uint64_t encode_submit_us = 0;
    uint64_t encode_complete_us = 0;
    uint64_t serialized_us = 0;
};

// One encoded access unit produced by a lane, tagged with enough info for
// the OrderedBitstreamSerializer to restore strict source order.
struct EncodedChunk {
    uint64_t gop_index = 0;
    uint64_t source_index = 0;   // position within the whole recording
    uint64_t pts_ns = 0;
    int      lane_id = -1;
    std::vector<uint8_t> data;
};
