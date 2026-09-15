# RECORD.md

## Goal

`xrecorder` must sustain **1000 fps** recording from the XIMEA camera at approximately **4096×992 GRAY8**, using all three available NVENC engines while preserving camera sequence identity and final video order:

- **RTX PRO 2000 Blackwell**: XIMEA GPUDirect landing GPU + 1 NVENC lane
- **RTX 5070 Ti**: 2 NVENC lanes

The key throughput invariant is:

```text
one new camera frame every 1 ms
!=
one frame must finish encoding within 1 ms
```

The implementation is intentionally a **four-concurrent-path system**:

```text
1 capture/RDMA + ownership-copy path
3 independent processing/encoding pipelines
```

The capture path synchronizes only at the ownership-copy boundary. It never waits for GRAY8→NV12 conversion or NVENC completion.

---

# Four-concurrent-path architecture

```text
Camera @ 1000 fps
      |
      | XiAPI GPUDirect RDMA
      v
XiAPI-owned GRAY8 in RTX PRO 2000 VRAM
      |
      | one ownership copy for the selected lane
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

At steady state, a representative overlap is:

```text
capture/copy : frame N
PRO pipeline : older GOP/frame work
5070 lane A  : older GOP/frame work
5070 lane B  : older GOP/frame work
```

The only host-side synchronization on the capture path is:

```cpp
cudaEventSynchronize(copy_done);
```

After `copy_done`, the XiAPI RDMA pointer is no longer needed by the recorder.

---

# Hardware topology

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

The RTX 5070 Ti is not used as the XiAPI GPUDirect landing GPU. The camera lands in PRO 2000 VRAM; frames assigned to the 5070 Ti are copied there before processing.

---

# Ownership boundary: every lane copies first

The XiAPI capture path uses:

```text
XI_PRM_TRANSPORT_DATA_TARGET = GPU_RAM
XI_PRM_IMAGE_DATA_FORMAT     = XI_FRM_TRANSPORT_DATA
XI_PRM_BUFFER_POLICY         = XI_BP_UNSAFE
```

`image.bp` points to XiAPI-managed GPU memory. That memory can be reused by the acquisition ring, so no processing pipeline is allowed to consume `image.bp` directly.

Every frame follows the same rule:

```text
XiAPI RDMA pointer
      |
      | COPY ONLY
      v
lane-owned GRAY8 slot
      |
      | copy_done
      +---- capture may release/return to XiAPI here
      |
      v
lane-owned processing begins independently
```

This applies even to the local PRO lane.

Old local optimization — no longer used:

```text
XiAPI GRAY8 -> direct GRAY8->NV12 conversion
```

Current local PRO path:

```text
XiAPI GRAY8 in PRO VRAM
      |
      | D2D ownership copy
      v
PRO lane-owned GRAY8
      |
      | capture waits copy_done only
      v
PRO process stream: GRAY8 -> NV12 -> NVENC
```

Remote 5070 path:

```text
XiAPI GRAY8 in PRO VRAM
      |
      | P2P ownership copy
      v
5070 lane-owned GRAY8
      |
      | capture waits copy_done only
      v
5070 process stream: GRAY8 -> NV12 -> NVENC
```

The extra local D2D copy is intentional. It creates one clean and identical ownership boundary for all lanes and completely decouples XiAPI buffer lifetime from conversion/NVENC latency.

---

# Copy streams versus process streams

Each lane has **two conceptual stages**:

```text
copy stage
process/encode stage
```

Each lane owns an independent non-blocking copy stream and an independent non-blocking process stream.

For lane `i`:

```text
copy_stream[i]
    |
    | ownership copy into lane GRAY8
    v
copy_done[i][slot]
    |
    | host capture waits here and nowhere later
    v
process_stream[i]
    |
    | GRAY8 -> NV12
    v
NVENC pipeline i
```

`EncoderLane::convert_stream_` is the lane process stream and is created with:

```cpp
cudaStreamCreateWithFlags(&convert_stream_, cudaStreamNonBlocking);
```

The copy streams are also created with `cudaStreamNonBlocking`.

This prevents legacy default-stream synchronization from coupling independent lanes.

CUDA current-device state is thread-local, so each encoder worker thread explicitly binds its own GPU before touching lane CUDA objects.

---

# Local and P2P copy paths

For the local PRO lane:

```text
PRO XiAPI RDMA memory
      |
      | cudaMemcpyAsync D2D on lane 0 copy stream
      v
PRO lane-owned GRAY8
```

For a remote 5070 lane with peer access:

```text
PRO XiAPI RDMA memory
      |
      | cudaMemcpyPeerAsync on that lane's copy stream
      v
5070 lane-owned GRAY8
```

The destination GPU is current for the P2P submission because the copy stream belongs to the destination lane GPU.

Direct peer access is checked/enabled per destination lane GPU.

---

# Pinned-host fallback

If direct CUDA P2P is unavailable, the ownership copy becomes:

```text
PRO VRAM
   |
   | D2H async on this lane's capture-side stream
   v
lane-owned pinned host slot
   |
   | destination copy stream waits for d2h_done
   | H2D async
   v
5070 lane-owned GRAY8
   |
   | copy_done
   v
process pipeline
```

Capture still performs only one host synchronization: the final `copy_done` event representing completion of the full ownership transfer into destination GPU memory.

Each remote lane owns its own:

```text
capture-side fallback stream
pinned-host ring
d2h_done events
destination copy stream
GRAY8 ring
copy_done events
```

Therefore the two 5070 pipelines do not serialize through shared fallback resources.

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

The ownership copy happens while the image is still GRAY8. Copying NV12 would increase cross-GPU traffic by 1.5×.

---

# Per-lane memory ownership

Every lane now owns a GRAY8 staging ring, not only remote lanes.

Each lane therefore owns:

```text
48 GRAY8 staging slots
48 NV12 encoder slots
per-slot copy_done events
one copy stream
one process stream
one independent NVENC/GStreamer pipeline
```

At 4096×992:

```text
GRAY8 frame ~= 4.063 MB
48 GRAY8 slots ~= 195 MB per lane

NV12 frame ~= 6.095 MB
48 NV12 slots ~= 293 MB per lane
```

Approximate application-owned GPU staging across three lanes:

```text
3 × (195 MB + 293 MB) ~= 1.46 GB
```

This is deliberate and replaces dependence on XiAPI memory after the ownership copy.

The old 1000-frame recorder pool is not used by the multi-NVENC path.

---

# Authoritative frame identity: acq_nframe

`XI_IMG::acq_nframe` is the authoritative recorder sequence.

Recording-relative `source_index` advances by the actual unsigned `acq_nframe` delta rather than by a dense `source_index_++` counter.

Example:

```text
acq_nframe:   100  101  104  105
source_index:   0    1    4    5
missing:                  2,3
```

Missing camera positions remain missing in the recorder timeline and are registered with the serializer immediately.

Normal uint32 frame-counter wrap is handled by unsigned subtraction. Duplicate or implausibly backward/reset sequences are rejected.

---

# Gap-aware GOP bookkeeping

Camera-side gaps can occur before the first actual frame of a GOP.

Each pending GOP therefore tracks:

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

Large camera gaps are split by GOP boundary rather than processed frame-by-frame.

---

# GOP assignment and IDR behavior

Whole GOPs are scheduled to lanes.

```text
GOP 0 -> lane 0
GOP 1 -> lane 1
GOP 2 -> lane 2
GOP 3 -> lane 0
...
```

Per-GOP route state is:

```text
gop -> { lane_idx, needs_idr }
```

The first **successfully submitted actual frame** of each GOP gets the forced keyframe/IDR request.

If the nominal GOP-start frame is missing, or the first actual frame cannot reserve a lane slot, `needs_idr` remains true for the next actual frame.

---

# Capture critical path

Conceptually, one capture callback now performs only:

```text
xiGetImage already returned image.bp
      |
read acq_nframe + timestamp
      |
choose/lookup GOP lane
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

It does **not** wait for:

```text
GRAY8->NV12 completion
GStreamer appsrc consumption
NVENC completion
ordered serialization
disk output
```

This is the central real-time design invariant.

---

# Serializer

Each encoder lane ends in `appsink`. Encoded access units are tagged with GOP and source identity and sent to `OrderedBitstreamSerializer`.

The serializer emits completed GOPs in source order into:

```text
encoded appsrc -> h264parse -> matroskamux -> filesink
```

Within a GOP, chunks are explicitly ordered by `source_index` before emission.

A new serializer is built for every recording session so indexed output filenames are honored and stopped GStreamer state is not reused.

---

# XiAPI configuration

Acquisition configuration order remains:

```text
1. XI_PRM_ACQ_BUFFER_SIZE
2. XI_PRM_BUFFERS_QUEUE_SIZE
3. read both values back and verify accepted values
```

The XiAPI ring is only a short acquisition-jitter buffer. It is not the encoder backlog.

---

# Audit history

Baseline implementation commits audited:

```text
0d19d88385b58e26f731ba6d3db83b1e2dbe9e23
55246af2cdd90f7e811824abdd680464c6d1c8b4
```

Important fixes since that audit:

1. Capture topology changed to PRO 2000 -> 5070 Ti copies.
2. Default lanes changed to one PRO lane + two 5070 lanes.
3. Remote staging changed from per-GPU to per-lane ownership.
4. Serializer is rebuilt for every recording session.
5. `acq_nframe` now controls recorder source identity and gap bookkeeping.
6. Missing nominal GOP-start frames still force an IDR on the first actual submitted frame.
7. Every process pipeline owns a non-blocking CUDA process stream.
8. Every lane now also owns a distinct non-blocking copy stream.
9. Pinned-host fallback resources are per remote lane.
10. The local PRO lane now performs an explicit D2D ownership copy instead of reading XiAPI memory directly.
11. Capture synchronizes only the ownership-copy completion event.
12. Conversion and NVENC lifetime are fully decoupled from XiAPI buffer lifetime.

---

# Remaining hardware validation

The implementation still requires real hardware validation for:

- actual CUDA device IDs for PRO 2000 and 5070 Ti;
- PRO -> 5070 direct P2P availability;
- ownership-copy latency and its p99/p99.9 relative to the 1 ms capture period;
- sustained throughput of all three NVENC sessions;
- proof in Nsight Systems that capture/copy overlaps all three processing pipelines;
- scheduler weight tuning;
- long-duration 1000 fps zero-gap testing;
- true NVENC completion-latency measurement;
- cross-GPU transfer latency percentiles.

The current lane latency metric is not yet full NVENC completion latency.

---

# Success criteria

```text
camera acquisition        = 1000 fps
capture copy handoff      < 1 ms sustained, with margin
camera acq_nframe gaps    = 0 in a successful run
application drops         = 0
serializer frame gaps     = 0
serializer GOP gaps       = 0
final frame order         = exact
all queues                = bounded
capture + 3 pipelines     = concurrent
PRO -> 5070 transfers     = stable
aggregate NVENC capacity  > 1000 fps with useful margin
```

Preferred aggregate encode capacity remains approximately:

```text
1150-1250 fps or higher
```

so PCIe, scheduling, muxing, and transient encoder jitter do not consume the entire throughput margin.
