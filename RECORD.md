# RECORD.md

## Goal

`xrecorder` must sustain **1000 fps** recording from the XIMEA camera at approximately **4096×992 GRAY8**, using all three available NVENC engines while preserving camera sequence identity and final video order:

- **RTX PRO 2000 Blackwell**: XIMEA GPUDirect landing GPU + 1 NVENC lane
- **RTX 5070 Ti**: 2 NVENC lanes

The throughput invariant is:

```text
one new camera frame every 1 ms
!=
one frame must finish encoding within 1 ms
```

Three independent encoder pipelines provide aggregate throughput while the serializer restores source order.

---

# Hardware topology

```text
XIMEA @ 1000 fps
      |
      | XiAPI GPUDirect
      v
RTX PRO 2000 VRAM
      |
      +--> PRO lane 0: GRAY8 -> NV12 -> PRO NVENC
      |
      +-- GRAY8 GPU copy --> RTX 5070 Ti --> lane 1 -> NVENC A
      |
      +-- GRAY8 GPU copy --> RTX 5070 Ti --> lane 2 -> NVENC B

three independent GOP streams
      |
      v
OrderedBitstreamSerializer
      |
      v
single H.264/MKV recording
```

Default topology:

```text
capture GPU : RTX PRO 2000
lane 0      : RTX PRO 2000
lane 1      : RTX 5070 Ti
lane 2      : RTX 5070 Ti
```

CUDA device enumeration is machine-dependent. Example when the 5070 Ti is device 0 and the PRO 2000 is device 1:

```bash
ximea_cli --capture-gpu 1 --lane-gpus 1,0,0
```

---

# Cross-GPU bandwidth

At 4096×992 GRAY8:

```text
4096 × 992 = 4,063,232 bytes/frame
~4.063 GB/s at 1000 fps
```

If the two 5070 lanes process approximately two-thirds of the GOPs:

```text
PRO -> 5070 GRAY8 traffic ~= 2.71 GB/s
```

The copy must happen before NV12 expansion. Copying NV12 would increase traffic by 1.5×.

Preferred path:

```text
XiAPI GRAY8 in PRO VRAM
      |
      | cudaMemcpyPeerAsync
      v
owned GRAY8 staging in 5070 VRAM
      |
      | GRAY8 -> NV12 CUDA kernel
      v
owned NV12 encoder slot
      |
      v
NVENC
```

Fallback when P2P is unavailable:

```text
PRO VRAM
   |
   | D2H async on this lane's capture-side stream
   v
pinned host staging
   |
   | H2D async
   v
5070 VRAM
```

P2P availability still depends on CUDA support and motherboard/PCIe topology.

---

# Per-pipeline CUDA streams

Every encoder pipeline must be independently schedulable.

Each `EncoderLane` owns:

```text
one NVENC/GStreamer pipeline
one bounded NV12 pool
one CUDA conversion/preparation stream
one push thread
one pull thread
```

`EncoderLane::convert_stream_` is created with:

```cpp
cudaStreamCreateWithFlags(&convert_stream_, cudaStreamNonBlocking);
```

This avoids legacy default-stream synchronization coupling otherwise independent lanes.

For P2P operation:

```text
lane 1 P2P copy -> lane 1 ConvertStream -> lane 1 conversion -> lane 1 NVENC
lane 2 P2P copy -> lane 2 ConvertStream -> lane 2 conversion -> lane 2 NVENC
```

For pinned-host fallback, each remote lane also owns a separate capture-GPU stream:

```text
lane 1 capture_stream -> pinned A -> lane 1 ConvertStream
lane 2 capture_stream -> pinned B -> lane 2 ConvertStream
```

There is no shared `capture_xfer_stream_` between the two 5070 pipelines.

CUDA current-device state is thread-local, so each `EncoderLane::PushLoop()` explicitly executes:

```cpp
cudaSetDevice(cfg_.gpu_id);
```

before touching that lane's CUDA events or CUDA-backed GStreamer memory.

---

# Remote staging ownership

Remote GRAY8 staging is **per encoder lane**, not per destination GPU.

Unsafe:

```text
remote_transfers_[gpu_id].gray8_staging[slot]
```

because both 5070 lanes can simultaneously reserve the same slot number.

Correct:

```text
remote_transfers_[lane_id].gray8_staging[slot]
```

Each remote lane has an independent GRAY8 ring, pinned-host fallback ring, CUDA events, and fallback transfer stream.

With 48 slots:

```text
~195 MB GRAY8 staging per remote lane
~390 MB total for two 5070 lanes
```

---

# Authoritative frame identity: acq_nframe

`XI_IMG::acq_nframe` is now the authoritative recorder sequence, not just a camera telemetry field.

The old behavior was wrong:

```text
source_index = source_index_++
```

because a missing camera frame was silently removed from the recorder timeline.

The current behavior derives recording-relative source position from successive `acq_nframe` deltas:

```text
first actual frame             -> source_index 0
next acq_nframe + 1            -> source_index +1
next acq_nframe + N            -> source_index +N
missing camera frames remain missing source indices
```

Unsigned `uint32_t` subtraction is used for the delta, so the normal XiAPI frame-counter wrap is handled naturally. A zero delta is treated as a duplicate, and an implausibly large backward/reset delta is rejected rather than interpreted as billions of missing frames.

Example:

```text
acq_nframe:  100, 101, 104, 105
source_index:  0,   1,   4,   5
missing:                 2,3
```

The missing source positions are registered with the serializer immediately.

---

# Gap-aware GOP bookkeeping

Camera-side gaps may occur before the first actual frame of a GOP. Therefore the serializer no longer stores only one mutable `expected` counter.

Each pending GOP stores:

```text
nominal_expected
explicit_dropped
expected_set
received encoded chunks
```

Effective completion count is:

```text
expected_chunks = nominal_expected - explicit_dropped
```

This allows a camera gap to be registered before the first actual frame of that GOP arrives.

Large acquisition gaps are split by GOP boundaries rather than processed one missing frame at a time, keeping capture-thread work proportional to the number of GOPs crossed.

The serializer counts explicit camera/application drops in `frame_gaps`, while additional chunks that disappear later in the encode path are detected separately by the stall logic.

---

# GOP assignment and IDR behavior

Whole GOPs are assigned to lanes; individual inter-predicted frames are not round-robin distributed across encoders.

```text
GOP 0 -> lane 0
GOP 1 -> lane 1
GOP 2 -> lane 2
GOP 3 -> lane 0
...
```

The route state is:

```text
gop -> { lane_idx, needs_idr }
```

The first **successfully submitted actual frame** of each GOP is forced to a keyframe/IDR.

This matters when:

- the nominal GOP-boundary camera frame is missing; or
- the first actual frame cannot reserve an encoder slot.

`needs_idr` is cleared only after successful submission, so the next actual frame still receives the IDR request if the previous one was dropped.

---

# XiAPI buffer ownership

The capture path uses:

```text
XI_PRM_TRANSPORT_DATA_TARGET = GPU_RAM
XI_PRM_IMAGE_DATA_FORMAT     = XI_FRM_TRANSPORT_DATA
XI_PRM_BUFFER_POLICY         = XI_BP_UNSAFE
```

`image.bp` is XiAPI-owned GPU memory and may later be reused by the acquisition ring.

Ownership handoff must therefore happen immediately:

```text
local PRO lane:
XiAPI GRAY8 -> lane NV12 slot

remote 5070 lane:
XiAPI GRAY8 -> lane-owned 5070 GRAY8 staging -> lane NV12 slot
```

NVENC must never depend on the XiAPI pointer for the complete encode lifetime.

XiAPI acquisition configuration order remains:

```text
1. XI_PRM_ACQ_BUFFER_SIZE
2. XI_PRM_BUFFERS_QUEUE_SIZE
3. read both values back and verify them
```

---

# Memory model

The old 1000-frame GRAY8 + NV12 recorder pool is not used by the multi-NVENC path.

Default per-lane pool:

```text
48 NV12 slots
```

At 4096×992:

```text
NV12 frame ~= 6.095 MB
48-slot NV12 pool ~= 293 MB per lane
```

Remote lanes additionally own 48 GRAY8 staging slots each.

Pools are bounded so encoder backlog cannot grow without limit.

---

# Serializer lifecycle

A new `OrderedBitstreamSerializer` is built for every recording session so `SetOutputName()` is honored and stopped GStreamer state is not reused.

Encoded GOPs from all lanes are emitted in source order into one final pipeline:

```text
encoded appsrc -> h264parse -> matroskamux -> filesink
```

Within each completed GOP, chunks are explicitly sorted by `source_index` before final emission.

---

# Audit history

Baseline implementation commits audited:

```text
0d19d88385b58e26f731ba6d3db83b1e2dbe9e23
55246af2cdd90f7e811824abdd680464c6d1c8b4
```

Important fixes made after that audit:

1. Capture topology changed to PRO 2000 -> 5070 Ti copies.
2. Default lanes changed to one PRO lane + two 5070 lanes.
3. Remote staging changed from per-GPU to per-lane ownership.
4. Destination CUDA device is made current before P2P submission.
5. Serializer is rebuilt for every recording session.
6. `acq_nframe` now controls recording-relative source sequence and GOP identity.
7. Camera gaps are explicitly registered with the serializer.
8. Missing nominal GOP-start frames still produce an IDR on the first actual frame.
9. Every lane has an independent non-blocking CUDA preparation stream.
10. Every remote fallback pipeline has an independent capture-side transfer stream.
11. Encoder worker threads explicitly bind their CUDA device.

---

# Remaining hardware validation

The implementation still requires real hardware validation for:

- actual CUDA device IDs for PRO 2000 and 5070 Ti;
- PRO -> 5070 P2P availability and measured bandwidth;
- sustained throughput of each of the three NVENC sessions;
- scheduler weight tuning;
- long-duration 1000 fps zero-gap testing;
- true NVENC completion-latency measurement;
- cross-GPU transfer latency percentiles.

The current lane latency metric is not yet full NVENC completion latency.

---

# Success criteria

```text
camera acquisition       = 1000 fps
camera acq_nframe gaps   = 0 in a successful run
application drops        = 0
serializer frame gaps    = 0
serializer GOP gaps      = 0
final frame order        = exact
all queues               = bounded
three encoder pipelines  = concurrent
PRO -> 5070 transfers    = stable
aggregate NVENC capacity > 1000 fps with useful margin
```

Preferred aggregate capacity remains approximately:

```text
1150-1250 fps or higher
```

so PCIe, scheduling, muxing, and transient encoder jitter do not consume the entire throughput margin.
