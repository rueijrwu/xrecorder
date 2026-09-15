# RECORD.md

## Goal

`xrecorder` must sustain **1000 fps** recording from the XIMEA camera at approximately **4096×992 GRAY8**, using all three available NVENC engines while preserving exact camera order in one final recording:

- **RTX PRO 2000 Blackwell**: XIMEA GPUDirect landing GPU + 1 NVENC lane
- **RTX 5070 Ti**: 2 NVENC lanes

The important throughput invariant is:

```text
one new camera frame every 1 ms
!=
one frame must finish encoding within 1 ms
```

If each NVENC lane needs roughly 2–3 ms per frame, three lanes can still provide enough aggregate service rate for a 1 kHz input stream.

The recorder therefore uses **parallel encoding with ordered serialization**.

---

# Hardware topology

The RTX 5070 Ti is not treated as the XIMEA GPUDirect target. The XIMEA camera lands frames in **RTX PRO 2000 VRAM** and the recorder copies only the work assigned to the 5070 Ti across GPUs.

```text
XIMEA @ 1000 fps
      |
      | XiAPI GPUDirect
      v
RTX PRO 2000 VRAM
      |
      +--> PRO EncoderLane 0 --> PRO NVENC
      |
      +-- GRAY8 GPU copy --> RTX 5070 Ti --> EncoderLane 1 --> NVENC A
      |
      +-- GRAY8 GPU copy --> RTX 5070 Ti --> EncoderLane 2 --> NVENC B

three independent GOP streams
      |
      v
OrderedBitstreamSerializer
      |
      v
single H.264/MKV recording
```

The intended default lane topology is therefore:

```text
capture GPU : RTX PRO 2000
lane 0      : RTX PRO 2000
lane 1      : RTX 5070 Ti
lane 2      : RTX 5070 Ti
```

The exact CUDA device IDs depend on enumeration order, so `--capture-gpu` and `--lane-gpus` remain explicit overrides.

Example when PRO 2000 is CUDA device 1 and 5070 Ti is device 0:

```bash
ximea_cli --capture-gpu 1 --lane-gpus 1,0,0
```

---

# Throughput and cross-GPU bandwidth

At 4096×992 GRAY8:

```text
4096 × 992 = 4,063,232 bytes/frame
```

At 1000 fps:

```text
~4.063 GB/s raw camera traffic
```

If the two 5070 Ti lanes handle approximately two-thirds of the GOPs, the PRO→5070 copy load is approximately:

```text
4.063 GB/s × 2/3 ~= 2.71 GB/s
```

The copy must happen while the image is still **GRAY8**.

Do not convert to NV12 on the PRO 2000 before sending the frame to the 5070 Ti, because NV12 is 1.5 bytes/pixel and would increase inter-GPU traffic to roughly:

```text
2.71 GB/s × 1.5 ~= 4.06 GB/s
```

Preferred remote path:

```text
XiAPI GRAY8 in PRO VRAM
      |
      | cudaMemcpyPeerAsync
      v
owned GRAY8 staging slot in 5070 VRAM
      |
      | CUDA GRAY8 -> NV12
      v
owned NV12 encoder slot in 5070 VRAM
      |
      v
NVENC
```

Fallback when CUDA P2P is unavailable:

```text
PRO VRAM
   |
   | D2H async
   v
pinned host staging
   |
   | H2D async
   v
5070 VRAM
```

P2P availability must be checked at startup. PCIe topology, ACS/IOMMU configuration, and driver/platform constraints can prevent peer access even when both GPUs support CUDA.

---

# Critical ownership rule: remote staging is per encoder lane

A destination GPU may contain more than one encoder lane. In this system the RTX 5070 Ti has two lanes.

Therefore GRAY8 staging ownership must be **per lane**, not merely per destination GPU.

The unsafe design is:

```text
remote_transfers_[gpu_id].gray8_staging[slot_index]
```

because lane A and lane B can independently reserve the same local slot number at the same time:

```text
5070 lane A -> slot 7
5070 lane B -> slot 7
```

If both lanes share one GPU-level staging ring, both writes target the same GRAY8 buffer and corrupt each other.

The corrected design is:

```text
remote_transfers_[lane_id].gray8_staging[slot_index]
```

Each remote lane owns its own staging ring. Slot N in a lane's GRAY8 ring is paired with slot N in that lane's NV12 pool, so the staging slot cannot be reused until the corresponding encoder slot becomes free.

With a 48-slot pool:

```text
48 × 4.063 MB ~= 195 MB GRAY8 staging per remote lane
```

Two remote 5070 lanes therefore consume roughly 390 MB of GRAY8 staging in addition to their NV12 pools. This is acceptable for the target hardware and keeps synchronization simple and bounded.

---

# P2P context rule

The GPU-to-GPU copy is issued on the **destination encoder lane's CUDA stream**.

Therefore the destination GPU must be current before calling:

```cpp
cudaMemcpyPeerAsync(..., lane->ConvertStream());
```

The recorder checks destination→capture peer access and enables peer access from the destination GPU to the capture GPU.

This avoids using a CUDA stream that belongs to a different current device/context.

---

# Local PRO lane

Frames scheduled to the PRO 2000 lane do not need a GRAY8 copy.

```text
XiAPI GRAY8 in PRO VRAM
      |
      | CUDA GRAY8 -> NV12
      v
owned PRO NV12 slot
      |
      v
PRO NVENC
```

The GRAY8→NV12 conversion is the ownership handoff into application-managed encoder memory.

The XiAPI acquisition ring remains a short scheduling-jitter buffer; it must not be treated as the encoder backlog.

---

# Split-GOP parallel encoding

Do not round-robin individual inter-predicted frames across independent encoder sessions.

Unsafe for normal inter-frame H.264:

```text
frame 0 -> lane A
frame 1 -> lane B
frame 2 -> lane C
frame 3 -> lane A
```

Each encoder owns a different reference history, so those outputs cannot simply be interleaved into one normal temporal H.264 stream.

Instead assign whole independently decodable GOPs:

```text
GOP 0 -> lane 0
GOP 1 -> lane 1
GOP 2 -> lane 2
GOP 3 -> lane 0
...
```

With `gop-size=30`:

```text
GOP 0 = frames   0-29
GOP 1 = frames  30-59
GOP 2 = frames  60-89
```

Each GOP starts with a forced keyframe/IDR and each lane ends in `appsink`. The single serializer emits completed GOPs strictly in source order into one final muxer/filesink.

---

# Current implementation structure

```text
XimeaCapture
    |
    v
MultiNvRecorder
    |
    +--> GopScheduler
    |
    +--> EncoderLane 0 (capture GPU / PRO)
    +--> EncoderLane 1 (remote / 5070)
    +--> EncoderLane 2 (remote / 5070)
    |
    v
OrderedBitstreamSerializer
    |
    v
h264parse ! matroskamux ! filesink
```

The encoder lane pool is bounded and preallocated. The old 1000-frame GRAY8+NV12 pool design is not used for the multi-NVENC path.

Default per-lane NV12 pool:

```text
48 slots
```

At 4096×992, one NV12 frame is about 6.095 MB, so one 48-slot NV12 pool is roughly 293 MB.

The exact pool size should eventually be tuned from measured p99/p99.9 queue depth and completion latency.

---

# XiAPI acquisition requirements

The capture path uses:

```text
XI_PRM_TRANSPORT_DATA_TARGET = GPU_RAM
XI_PRM_IMAGE_DATA_FORMAT     = XI_FRM_TRANSPORT_DATA
XI_PRM_BUFFER_POLICY         = XI_BP_UNSAFE
```

`image.bp` is XiAPI-managed GPU memory and can eventually be reused by the acquisition ring.

The capture callback must therefore submit the ownership-transfer operation immediately:

```text
local PRO GOP:
    XiAPI GRAY8 -> CUDA conversion -> owned NV12

remote 5070 GOP:
    XiAPI GRAY8 -> GPU copy -> owned 5070 GRAY8 -> owned NV12
```

The recorder must never depend on the XiAPI pointer through the complete NVENC lifetime.

XiAPI acquisition parameter ordering remains:

```text
1. XI_PRM_ACQ_BUFFER_SIZE
2. XI_PRM_BUFFERS_QUEUE_SIZE
3. read both back and verify accepted values
```

---

# Frame identity

`XI_IMG::acq_nframe` is used by `XimeaCapture` to detect exact camera-side sequence gaps.

Current audit note: `acq_nframe` is passed into `MultiNvRecorder::PushFrame()`, but the recorder still builds GOP indices from its own dense `source_index_`. This means camera-side loss is detected, but `acq_nframe` is not yet the authoritative recorder identity.

This remains a follow-up item. A future change should preserve both:

```text
acq_nframe      = authoritative camera sequence
source_index    = recording-relative index
```

without hiding acquisition gaps.

---

# Serializer lifecycle

The serializer must be rebuilt for every recording session.

Reason: the GUI can stop and start recording repeatedly and calls `SetOutputName()` with an indexed filename. Constructing the serializer only once would leave its internal output filename stale.

Current design:

```text
SetOutputName(indexed_name)
      |
      v
MultiNvRecorder::Start()
      |
      v
BuildSerializer() using current output_name_
```

This also prevents reuse of stopped GStreamer serializer state between recording sessions.

---

# Audit of the previous two commits

Baseline commits reviewed:

```text
0d19d88385b58e26f731ba6d3db83b1e2dbe9e23
    Implement multi-NVENC 1 kHz recording architecture per RECORD.md

55246af2cdd90f7e811824abdd680464c6d1c8b4
    Document implementation status and cross-GPU transfer design
```

The architecture was broadly correct, but the audit found these important issues:

1. **Wrong capture-GPU assumption**
   - Previous design assumed the 5070 Ti was the XIMEA GPUDirect landing GPU.
   - Correct target topology uses the RTX PRO 2000 for capture.

2. **Wrong default lane placement for the corrected topology**
   - Previous default: two lanes on capture GPU, one on the other GPU.
   - Corrected default: one lane on capture/PRO GPU, two lanes on the other/5070 GPU.

3. **Remote staging collision**
   - Previous staging was keyed by destination GPU.
   - Two 5070 lanes could use the same staging slot concurrently.
   - Corrected to per-lane remote-transfer ownership.

4. **P2P current-device/context issue**
   - Peer copy used a destination-lane CUDA stream without first guaranteeing the destination GPU was current.
   - Corrected by setting the lane GPU current before `cudaMemcpyPeerAsync`.

5. **Recording filename lifecycle bug**
   - `SetOutputName()` changed `MultiNvRecorder` state after the serializer had already captured the old name.
   - Corrected by rebuilding the serializer on each `Start()`.

---

# Implementation status

Implemented:

- explicit capture GPU ID
- one local + two remote default lane topology
- per-lane NV12 pools
- per-lane remote GRAY8 staging pools
- CUDA P2P path
- pinned-host fallback path
- GRAY8 transfer before NV12 conversion
- split-GOP scheduling
- forced GOP keyframes
- three independent encoder pipelines
- ordered H.264 serialization into one MKV
- serializer rebuild per recording session
- XiAPI acquisition buffer ordering/readback
- camera-side `acq_nframe` gap detection

Still requiring hardware validation or follow-up:

- confirm actual CUDA device IDs for PRO 2000 and 5070 Ti
- confirm PRO→5070 CUDA P2P on the installed motherboard/PCIe topology
- benchmark each NVENC lane independently
- tune scheduler weights from measured sustained fps
- run sustained 1 kHz zero-gap stress tests
- measure true NVENC completion latency (current lane latency metric is not yet a full encode-completion measurement)
- propagate `acq_nframe` as authoritative identity through recorder metadata/serialization
- add cross-GPU transfer latency percentiles

---

# Success criteria

The implementation is successful only when a sustained hardware run demonstrates:

```text
camera acquisition      = 1000 fps
camera acq_nframe gaps  = 0
application drops       = 0
serializer GOP gaps     = 0
final frame order       = exact
queues                   = bounded
PRO->5070 copies         = stable
aggregate NVENC capacity > 1000 fps with useful margin
```

Preferred aggregate encode capacity remains approximately:

```text
1150-1250 fps or higher
```

so transient scheduling, PCIe, muxing, and encoder jitter do not consume the entire throughput margin.
