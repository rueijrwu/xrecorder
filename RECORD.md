# RECORD.md

## Goal

`xrecorder` targets **1000 fps** recording from XIMEA using:

- RTX PRO 2000 Blackwell as the XiAPI GPUDirect landing GPU and one NVENC lane
- RTX 5070 Ti as two additional NVENC lanes
- default ROI **2048×1024**, `offset-y=1040`

The recorder is a four-concurrent-path system:

```text
1 capture/RDMA + ownership-copy path
3 independent process/NVENC pipelines
```

The capture path waits only for ownership-copy completion. It never waits for conversion or NVENC completion.

---

# Hardware topology

```text
Camera @ 1000 fps
      |
      | XiAPI GPUDirect RDMA
      v
XiAPI-owned GRAY8 in RTX PRO 2000 VRAM
      |
      | ownership copy for selected lane
      | capture waits ONLY for copy_done
      v
lane-owned GRAY8
      |
      +----------------------+----------------------+
      |                      |                      |
      v                      v                      v
PRO pipeline            5070 pipeline A       5070 pipeline B
process stream 0        process stream 1      process stream 2
GRAY8 -> NV12           GRAY8 -> NV12         GRAY8 -> NV12
      |                      |                      |
      v                      v                      v
PRO NVENC               5070 NVENC A          5070 NVENC B
      |                      |                      |
      +----------------------+----------------------+
                             |
                             v
                  OrderedBitstreamSerializer
                             |
                             v
                        one H.264/MKV
```

Default logical topology:

```text
capture GPU : RTX PRO 2000
lane 0      : RTX PRO 2000
lane 1      : RTX 5070 Ti
lane 2      : RTX 5070 Ti
```

CUDA IDs are machine-dependent and remain configurable.

---

# Ownership-copy boundary

`image.bp` is XiAPI-managed GPU memory. No encoder pipeline may depend on it beyond the capture callback.

Every frame therefore performs exactly one ownership handoff:

```text
XiAPI RDMA buffer
      |
      | copy into lane-owned GRAY8
      v
copy_done
      |
      +-- capture may return/release XiAPI source here
      |
      v
lane process stream
      |
      | GRAY8 -> NV12
      v
NVENC
```

For the local PRO lane this is a D2D copy. For 5070 lanes it is P2P when available, otherwise D2H pinned-host staging followed by H2D.

Each lane owns its own copy resources and its own process stream. The two 5070 lanes do not share transfer streams or staging slots.

---

# Logical FrameGroup design

`--gop-size` defines two matching concepts:

1. a **logical capture FrameGroup** used for routing/scheduling;
2. the independently decodable **H.264 GOP** encoded by that lane.

The default is:

```text
--gop-size 30
```

A FrameGroup is **not** a physical 30-frame prebuffer.

Frames are processed immediately as they arrive:

```text
FrameGroup 0: source indices 0-29   -> lane A
FrameGroup 1: source indices 30-59  -> lane B
FrameGroup 2: source indices 60-89  -> lane C
```

For FrameGroup 0:

```text
frame 0  -> ownership copy -> lane A process immediately
frame 1  -> ownership copy -> lane A process immediately
...
frame 29 -> ownership copy -> lane A process immediately
```

The system does **not** wait until frame 29 arrives before lane A starts work.

This is important because a 30-frame prebuffer at 1000 fps would add an unnecessary 30 ms of latency and would not improve throughput.

The FrameGroup is therefore a **routing interval**, not a storage object.

Internally the route is:

```text
group_index -> { lane_idx, needs_idr }
```

The first actual frame of a group selects the lane. Every later actual frame whose source index falls in the same group uses that route.

---

# Why streaming the FrameGroup is better

At 1000 fps, a 30-frame group spans 30 ms.

With a measured single-pipeline capacity around 340 fps, a lane can process frames while the group is still arriving rather than waiting for the entire group first.

Representative overlap:

```text
0-30 ms:
  capture streams group 0 to lane A
  lane A already processes group 0

30-60 ms:
  capture streams group 1 to lane B
  lane A continues/drains group 0
  lane B processes group 1

60-90 ms:
  capture streams group 2 to lane C
  lane A/B continue their work
  lane C processes group 2
```

This preserves the intended four-way concurrency:

```text
capture/RDMA/copy
+
PRO pipeline
+
5070 pipeline A
+
5070 pipeline B
```

---

# Default camera ROI

The current default ROI is:

```text
width    = 2048
height   = 1024
offset-x = 0
offset-y = 1040
fps      = 1000
```

At 2048×1024 GRAY8:

```text
2,097,152 bytes/frame = 2 MiB/frame
~2.097 GB/s raw camera rate at 1000 fps
```

If roughly two-thirds of groups run on the two 5070 lanes:

```text
PRO -> 5070 GRAY8 traffic ~= 1.40 GB/s
```

This gives substantially more headroom than 4096×1024 while preserving 1024 vertical pixels.

---

# Per-lane memory and streams

Each lane owns:

```text
bounded GRAY8 staging ring
bounded NV12 ring
copy_done events
one non-blocking copy stream
one non-blocking process/conversion stream
one independent GStreamer/NVENC pipeline
```

For remote 5070 fallback operation, each lane also owns:

```text
one capture-side D2H stream
one pinned-host staging ring
D2H completion events
```

No two lanes share these resources.

The process stream is created with:

```cpp
cudaStreamCreateWithFlags(&convert_stream_, cudaStreamNonBlocking);
```

CUDA current-device state is thread-local, so encoder worker threads explicitly bind their lane GPU before accessing CUDA objects.

---

# acq_nframe is authoritative

`XI_IMG::acq_nframe` defines source sequence identity.

Recording-relative `source_index` advances by the actual unsigned XiAPI frame-number delta rather than by a dense local counter.

Example:

```text
acq_nframe:   100  101  104  105
source_index:   0    1    4    5
missing:                  2,3
```

Missing frames remain missing and are registered with the serializer. They do not collapse FrameGroup boundaries.

Thus, with `gop-size=30`, FrameGroup membership is always determined by:

```text
group_index = source_index / 30
```

regardless of camera-side gaps.

---

# IDR behavior

Each logical FrameGroup maps to one independently decodable encoder GOP.

The first **successfully submitted actual frame** of the group receives the forced keyframe/IDR request.

If the nominal first source frame is missing, the first actual frame still becomes the GOP start. If that actual frame cannot reserve a slot, `needs_idr` remains true until a frame is successfully submitted.

---

# Capture critical path

For each camera frame:

```text
xiGetImage already returned image.bp
      |
read acq_nframe + timestamp
      |
compute logical FrameGroup index
      |
lookup/assign one lane for that group
      |
reserve lane slot
      |
submit ownership copy
      |
WAIT ONLY FOR copy_done
      |
enqueue GRAY8->NV12 on lane process stream
      |
submit lane descriptor
      |
return to XiAPI
```

Capture does not wait for:

```text
remaining frames in the FrameGroup
GRAY8->NV12 completion
GStreamer appsrc consumption
NVENC completion
serializer
disk output
```

---

# Serializer

Each encoder lane outputs encoded access units to `OrderedBitstreamSerializer`.

The serializer groups them by the same logical FrameGroup/H.264 GOP index and emits groups in source order into:

```text
encoded appsrc -> h264parse -> matroskamux -> filesink
```

Camera-side and application-side frame gaps are tracked explicitly.

---

# CLI semantics

The intended CLI description is:

```text
--gop-size INT [30]
    Frames per logical capture FrameGroup and independently-decodable
    encoder GOP. All frames in a group are routed to the same encoder
    pipeline and processed immediately as they arrive; no whole-group
    prebuffer is used.
```

Camera ROI defaults:

```text
--width 2048
--height 1024
--offset-y 1040
```

---

# Remaining hardware validation

Required on the target machine:

- verify actual CUDA IDs for PRO 2000 and 5070 Ti;
- verify direct PRO -> 5070 P2P availability;
- measure ownership-copy mean/p95/p99/p99.9 latency;
- verify ownership-copy latency remains below the 1 ms frame period with margin;
- measure sustained throughput of all three NVENC sessions at 2048×1024;
- verify in Nsight Systems that capture/copy overlaps all three pipelines;
- tune scheduler weights if the three lanes have unequal measured throughput;
- perform long-duration 1000 fps zero-gap recording tests;
- measure true NVENC completion latency separately from preparation latency.

---

# Success criteria

```text
camera acquisition        = 1000 fps
capture ownership copy    < 1 ms sustained with margin
camera acq_nframe gaps    = 0 in a successful run
application drops         = 0
serializer frame gaps     = 0
serializer GOP gaps       = 0
final frame order         = exact
all queues                = bounded
capture + 3 pipelines     = concurrent
PRO -> 5070 copies        = stable
aggregate NVENC capacity  > 1000 fps with useful margin
```
