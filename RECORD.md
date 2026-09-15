# RECORD.md

## Goal

Extend `xrecorder` to sustain **1000 fps recording** from the XIMEA camera using all three available NVENC engines:

- **RTX 5070 Ti**: 2 NVENC engines
- **RTX PRO 2000 Blackwell**: 1 NVENC engine

The current implementation has already demonstrated that one NVENC session can compress a frame in roughly **<3 ms**. The new design should exploit three parallel encoder lanes so that the system can accept approximately one frame every 1 ms while preserving strict source-frame order in the final recording.

The intended meaning of "serialized" is:

- encoding lanes run in parallel;
- encoded output is serialized back into original camera order;
- the final result remains one logically ordered recording.

---

## Throughput model

Three encoders do not reduce a single frame's encode latency to 1 ms. Instead, they provide enough aggregate service rate to accept one new frame every 1 ms.

If each encoder requires about 3 ms per frame:

```text
333 fps + 333 fps + 333 fps ~= 1000 fps
```

More generally:

```text
R_aggregate = 1/T0 + 1/T1 + 1/T2
```

where `Ti` is the average encode service time of encoder lane `i`.

For reliable lossless 1 kHz recording, the target should not merely be slightly above 1000 fps. A practical target is approximately:

```text
aggregate sustained encode capacity >= 1150-1250 fps
```

This leaves margin for jitter, muxing, PCIe transfer, and encoder variability.

The RTX PRO 2000 should be benchmarked independently. Do not assume that its sustained encode throughput equals either NVENC engine on the RTX 5070 Ti.

---

# Recommended Architecture

```text
XIMEA @ 1000 fps
      |
      | GPUDirect -> RTX 5070 Ti
      v
capture / sequencing thread
      |
      | independent GOP chunks
      +-------------> Encoder lane A --+
      |                 RTX 5070 Ti     |
      |                 NVENC           |
      |                                 |
      +-------------> Encoder lane B ---+--> ordered serializer --> one MKV/H264
      |                 RTX 5070 Ti     |
      |                 NVENC           |
      |                                 |
      +-------------> Encoder lane C ---+
                        RTX PRO 2000
                        NVENC
```

The capture path and the encoding path must remain decoupled by bounded preallocated queues.

---

# Important: do not distribute individual inter-predicted frames round-robin

Do **not** use this with normal inter-frame H.264 encoding:

```text
frame 0 -> encoder A
frame 1 -> encoder B
frame 2 -> encoder C
frame 3 -> encoder A
...
```

If each encoder maintains its own reference history, encoder A would encode frame 3 using frame 0 as its previous temporal reference while frames 1 and 2 belong to other bitstreams. Those independent encoder outputs cannot simply be interleaved into one valid temporal H.264 stream.

Round-robin per-frame scheduling is only straightforward if every frame is independently encoded, e.g. all-I/all-IDR, which would significantly reduce compression efficiency.

---

# Preferred solution: split-GOP parallel encoding

Encode independent GOP blocks instead.

Example with `gop-size=30`:

```text
GOP 0: frames   0-29  -> lane A
GOP 1: frames  30-59  -> lane B
GOP 2: frames  60-89  -> lane C

GOP 3: frames  90-119 -> lane A
GOP 4: frames 120-149 -> lane B
GOP 5: frames 150-179 -> lane C
...
```

Each GOP must:

- start with an IDR;
- be independently decodable;
- carry enough parameter information for safe concatenation/serialization;
- preserve original source-frame timestamps and frame IDs.

The output serializer emits GOPs strictly in source order:

```text
GOP0 -> GOP1 -> GOP2 -> GOP3 -> ...
```

This preserves inter-frame compression inside each GOP while allowing the GOPs themselves to be encoded concurrently.

This is conceptually the same multi-instance storage-compression strategy NVIDIA demonstrates with independent GOP sections.

---

# Current xrecorder limitations

## 1. CUDA device is hard-wired to device 0

Current XIMEA setup uses:

```cpp
cudaSetDevice(0);
```

and configures:

```cpp
XI_PRM_TRANSPORT_DATA_TARGET = XI_TRANSPORT_DATA_TARGET_GPU_RAM;
```

The camera therefore targets CUDA device 0.

The recorder also creates its GStreamer CUDA context with:

```cpp
cuda_ctx_ = gst_cuda_context_new(0);
```

The recorder must be changed so GPU ID is explicit per encoder lane.

---

## 2. Only one NVENC pipeline exists

Current architecture contains one `GstRecorder` and one NVENC pipeline.

This must become a multi-lane recorder, approximately:

```text
MultiNvRecorder
  +- EncoderLane 0  -> 5070 Ti session A
  +- EncoderLane 1  -> 5070 Ti session B
  +- EncoderLane 2  -> PRO 2000 session
  +- GopScheduler
  +- OrderedBitstreamSerializer
```

The two sessions on the RTX 5070 Ti should be normal simultaneous encode sessions. NVIDIA's driver can distribute sessions across the physical NVENC engines; the application should not depend on manually selecting "NVENC engine 0" or "NVENC engine 1".

---

# GPU placement

## Capture should preferably land on the RTX 5070 Ti

At the current default resolution:

```text
4096 x 992 x 1 byte = 4,063,232 bytes/frame
```

At 1000 fps:

```text
~4.063 GB/s raw GRAY8 camera traffic
```

Because two of the three NVENC engines are on the RTX 5070 Ti, landing XIMEA GPUDirect frames on that GPU minimizes inter-GPU traffic.

Approximately two-thirds of encoding work can remain local to the RTX 5070 Ti.

Only GOPs assigned to the RTX PRO 2000 need cross-GPU transfer.

If load distribution is approximately 2:1:

```text
4.063 GB/s x ~1/3 ~= 1.35 GB/s
```

of GRAY8 data needs to leave the RTX 5070 Ti.

This is substantially better than capturing on the RTX PRO 2000 and transferring roughly two-thirds of all frames to the RTX 5070 Ti.

---

# Cross-GPU transfer

At startup, test peer access:

```cpp
cudaDeviceCanAccessPeer(...)
```

Preferred path:

```text
5070 VRAM -> PRO 2000 VRAM
```

using CUDA peer-to-peer transfer where supported.

Fallback path:

```text
5070 VRAM
   |
   v
pinned host staging
   |
   v
PRO 2000 VRAM
```

Only the PRO-2000-bound GOPs should pay this transfer cost.

Transfer GRAY8 before color conversion, not NV12, because GRAY8 is smaller.

---

# Avoid the current 1000-frame recorder pool design

Current recorder uses:

```cpp
static constexpr int POOL_SIZE = 1000;
```

and allocates both:

```text
GRAY8 slot
NV12 slot
```

for every entry.

At 4096x992:

```text
GRAY8 ~= 4.063 MB
NV12  ~= 6.095 MB
per slot ~= 10.158 MB
```

Therefore:

```text
1000 slots ~= 10.16 GB ~= 9.46 GiB
```

for one recorder alone.

Replicating this for multiple encoder lanes is not viable, especially on the 16 GB RTX 5070 Ti.

Instead, use a much smaller bounded pool sized from measured worst-case latency, for example initially:

```text
32-64 slots per encoder lane
```

Then tune based on measured p99/p999 queue depth and encode completion time.

Example:

```text
64 NV12 frames ~= 390 MB
```

per lane at the current resolution.

---

# Remove unnecessary local GRAY8 copy

Current recorder path performs:

```text
XIMEA GRAY8
  |
  | cudaMemcpy D2D
  v
owned GRAY8
  |
  | CUDA conversion
  v
owned NV12
```

For encoder lanes on the same GPU as the camera, the intermediate GRAY8 copy is unnecessary.

Preferred local path:

```text
XIMEA GRAY8
   |
   | GRAY8 -> NV12 CUDA kernel
   v
owned NV12 slot
```

This removes one 4 MB D2D copy per local frame.

For frames going to the RTX PRO 2000:

```text
XIMEA GRAY8 on 5070
       |
       | P2P or staged copy
       v
owned GRAY8 on PRO 2000
       |
       | CUDA conversion
       v
owned NV12 on PRO 2000
```

This minimizes cross-GPU bandwidth.

---

# XiAPI buffer ownership

The current recorder uses:

```cpp
XI_PRM_BUFFER_POLICY = XI_BP_UNSAFE;
```

This is appropriate for a high-rate zero-extra-copy capture path, but `image.bp` points into XiAPI-managed circular acquisition memory. The memory can later be reused/overwritten by the camera driver.

Therefore NVENC must never directly depend on an XiAPI buffer whose lifetime extends beyond the immediate capture callback.

Required rule:

```text
xiGetImage()
     |
     v
obtain image.bp
     |
     v
immediately enqueue GPU copy/conversion into application-owned memory
     |
     v
release XiAPI slot as soon as possible
```

For local 5070 encoding, the GRAY8->NV12 CUDA kernel itself can be the ownership-transfer operation: the output NV12 buffer is application-owned.

For PRO-2000 GOPs, first copy the GRAY8 image into an application-owned buffer on the destination GPU, then release dependence on the XiAPI memory.

The XiAPI acquisition ring should be treated only as a short scheduling-jitter safety margin, not as the encoder backlog.

---

# XiAPI parameter ordering

Current code sets:

```cpp
XI_PRM_BUFFERS_QUEUE_SIZE = 129;
...
XI_PRM_ACQ_BUFFER_SIZE = payload_size * 150;
```

XiAPI documents that changing `XI_PRM_ACQ_BUFFER_SIZE` can invalidate/recalculate the queue-size setting.

Preferred order:

```cpp
set XI_PRM_ACQ_BUFFER_SIZE;
set XI_PRM_BUFFERS_QUEUE_SIZE;
```

Then read both parameters back and verify the actual values accepted by the driver.

At 1 kHz this verification is important.

---

# Use acq_nframe as the authoritative frame sequence

Current loss detection estimates missing frames using timestamp differences.

Instead, use:

```cpp
XI_IMG::acq_nframe
```

as the authoritative capture sequence identifier.

Recommended descriptor:

```cpp
struct CaptureFrame {
    void*    xi_gpu_ptr;
    uint64_t xi_timestamp_us;
    uint32_t acq_nframe;
};
```

The recorder should use `acq_nframe` to:

- identify exact frame order;
- detect camera-side losses;
- build GOP ranges;
- serialize output;
- diagnose queue drops;
- verify final recording continuity.

The XIMEA timestamp should still be preserved because it is needed for experimental timing.

---

# Encoder pipeline design

For the first implementation, retain GStreamer because the existing implementation has already demonstrated useful NVENC performance.

Each encoder lane should be approximately:

```text
appsrc CUDAMemory
   |
   v
NVENC H.264 encoder
   - explicit cuda-device-id
   - preset=p1
   - bframes=0
   - rc-lookahead=0
   - zerolatency=true
   - gop-size=30
   - force IDR at GOP start
   - repeat parameter sets as needed
   |
   v
h264parse
   |
   v
appsink
```

Do **not** put an independent `matroskamux` and `filesink` behind every encoder lane.

Instead:

```text
Lane A appsink --+
Lane B appsink --+--> ordered encoded-GOP queue
Lane C appsink --+
                       |
                       v
                single serializer
                       |
                       v
                 encoded appsrc
                       |
                       v
                    h264parse
                       |
                       v
                  matroskamux
                       |
                       v
                    filesink
```

This guarantees one final file and one ordered video timeline.

---

# Scheduler

The scheduler should assign **whole GOPs**, not arbitrary frames.

Initial simple policy:

```text
GOP 0 -> 5070 lane A
GOP 1 -> 5070 lane B
GOP 2 -> PRO 2000
repeat
```

Better production policy:

- measure sustained throughput of every lane;
- use weighted GOP scheduling;
- keep bounded queue depth per lane;
- avoid sending new GOPs to an overloaded lane;
- preserve monotonic GOP sequence IDs;
- never permit the capture thread to block on encoder completion.

Example if measured capacities are:

```text
5070 lane A = 390 fps
5070 lane B = 390 fps
PRO 2000    = 280 fps
```

then scheduling should approximately follow that capacity ratio instead of strict 1:1:1 assignment.

---

# Capture thread requirements

The XIMEA acquisition thread should do as little work as possible.

Its critical path should be approximately:

```text
xiGetImage()
    |
    v
read acq_nframe + timestamp
    |
    v
reserve output slot / submit CUDA work
    |
    v
push bounded descriptor
    |
    v
return immediately
```

Do not perform in the capture thread:

- muxing;
- disk I/O;
- long mutex waits;
- encoder waits;
- frame reordering;
- blocking preview operations;
- heap allocation in the steady-state path.

All queues and frame pools should be preallocated and bounded.

---

# Preview

Preview must remain lower priority than recording.

The previewer should never be allowed to:

- delay XiAPI buffer release;
- block capture;
- consume recording-owned slots;
- create unbounded CUDA work;
- compete with the encoder path unnecessarily.

Preview dropping is acceptable. Recording dropping is not.

---

# Metadata

For every frame or GOP, preserve enough metadata to reconstruct and validate the original capture timeline.

Recommended frame metadata:

```text
acq_nframe
XiAPI timestamp
source frame index
GOP index
encoder lane
GPU ID
capture arrival time
conversion-submit time
encode-submit time
encode-complete time
serialized time
```

This makes it possible to identify whether any loss occurred at:

- camera acquisition;
- XiAPI queue;
- application frame pool;
- GPU conversion;
- inter-GPU transfer;
- encoder queue;
- encoder itself;
- serializer;
- disk output.

---

# Telemetry

Expose at least:

```text
camera captured
camera acq_nframe gaps
camera XiAPI drops/timeouts
capture queue depth
5070 lane A submitted/completed/queue
5070 lane B submitted/completed/queue
PRO 2000 submitted/completed/queue
cross-GPU transfer latency
serializer pending GOP count
serializer frame gaps
bytes written
recording throughput
```

Also record moving and percentile latency statistics:

```text
mean
p95
p99
p99.9
max
```

for each pipeline stage.

---

# Success criteria

The implementation should be considered successful only when long-duration tests show:

```text
camera sustained        = 1000 fps
camera frame gaps       = 0
application drops       = 0
serializer gaps         = 0
final frame order       = exact
aggregate NVENC         >= ~1150 fps preferred margin
queue occupation        = bounded
acq_nframe continuity   = exact
```

The final file must preserve all 1000 source frames per second in order even though individual encoder sessions may each have approximately 2-3 ms encode latency.

---

# Implementation direction

Recommended implementation sequence:

1. Add explicit CUDA/GPU ID support to capture and recorder components.
2. Add `acq_nframe` to the capture callback and telemetry.
3. Fix XiAPI acquisition-buffer parameter ordering and verify values after setting them.
4. Replace the 1000-slot recorder pool with smaller bounded preallocated per-lane pools.
5. Remove the unnecessary local GRAY8 intermediate copy.
6. Introduce `EncoderLane` abstraction with explicit GPU assignment.
7. Run two encoder sessions on the RTX 5070 Ti and one on the RTX PRO 2000.
8. Add cross-GPU GRAY8 transfer for PRO-2000-bound GOPs.
9. Add independent GOP scheduling and forced IDR boundaries.
10. Replace per-lane mux/filesink with appsink output.
11. Add ordered GOP serializer and one final mux/filesink.
12. Add detailed per-stage telemetry and continuity validation.
13. Benchmark each lane independently and tune weighted GOP scheduling.
14. Stress-test for sustained 1 kHz recording with zero gaps.

---

# Core design principle

The recorder should optimize for **throughput pipeline parallelism**, not for reducing single-frame NVENC latency.

The key invariant is:

```text
new camera frame every 1 ms
!=
frame must finish encoding within 1 ms
```

Instead:

```text
camera produces one frame every 1 ms
three NVENC lanes process independent GOP work concurrently
serializer restores strict source order
```

That is the architecture most likely to achieve reliable 1 kHz recording with the RTX 5070 Ti + RTX PRO 2000 Blackwell configuration while preserving reasonable H.264 compression efficiency.

---

# Implementation status

## Completed (Steps 1–12)

**Step 1: Explicit CUDA/GPU ID support**
- `XimeaCapture::SetGpuId()` / `GetGpuId()` (default 0)
- `EncoderLane::Config::gpu_id` per lane
- `MultiNvRecorder::Config::capture_gpu_id` (the GPU that GPUDirect frames land on)
- CLI: `--capture-gpu N`, `--lane-gpus CSV` (e.g., `--lane-gpus 0,0,1`)

**Step 2: acq_nframe as authoritative sequence**
- Threaded from `XI_IMG::acq_nframe` through capture callback signature
- Drop detection now uses frame count, not timestamp deltas
- Passed to `MultiNvRecorder::PushFrame()` for precise GOP boundaries and loss accounting

**Step 3: XiAPI parameter ordering and verification**
- `XI_PRM_ACQ_BUFFER_SIZE` set **before** `XI_PRM_BUFFERS_QUEUE_SIZE` (RECORD.md order)
- Both parameters read back after setting; warning if driver clamps values below request
- Important at 1 kHz where silent clamp would starve the acquisition ring

**Step 4: Bounded per-lane pools**
- Replaced 1000-slot 10 GB pool with 48-slot per-lane (default, configurable via `--lane-pool-size`)
- Per-lane pool at 4096×992: ~390 MB NV12 (vs. 10 GB for 1000-slot design)

**Step 5: Remove local GRAY8 copy**
- Lanes on capture GPU: convert GRAY8 → NV12 directly from XIMEA GPUDirect pointer
- No intermediate GRAY8 copy, saves 4 MB/frame for local lanes

**Step 6–7: EncoderLane + multi-GPU instantiation**
- `EncoderLane` class: one NVENC session per GPU, explicit `cuda-device-id=N`
- Three lanes by default: two on 5070 Ti (NVENC engines A & B), one on PRO 2000
- Auto-detection of available GPUs; override via `--lane-gpus`

**Step 8: Cross-GPU GRAY8 transfer**
- P2P `cudaMemcpyPeerAsync` (preferred) with automatic fallback to pinned-host staging
- Transfer happens on the capture GPU's stream; convert happens on destination GPU's stream
- See "Cross-GPU Transfer Design" section below

**Step 9–10: GOP scheduling and appsink**
- `GopScheduler`: weighted round-robin, skips overloaded lanes
- Each lane's pipeline ends in `appsink` (no mux/filesink per lane)

**Step 11: OrderedBitstreamSerializer**
- Buffers encoded chunks from all lanes keyed by GOP index
- Emits GOPs strictly in source order into final pipeline: `h264parse ! matroskamux ! filesink`
- Force-flushes incomplete/stalled GOPs (2 s timeout) rather than blocking forever
- Counts frame/GOP gaps in telemetry

**Step 12: Telemetry and metadata**
- Per-lane stats: submitted, completed, dropped, queue depth, encode latency percentiles (p95/p99/p999)
- Cross-GPU frame counter, pending/gap GOP counts, bytes written
- `XimeaTelemetry` extended; pybind11 bindings updated
- CLI telemetry line includes GOP gaps and cross-GPU frame count

## Deferred (Steps 13–14)

**Step 13: Per-lane benchmarking**
- Requires hardware test run
- Once measured, update `RecorderConfig::lanes[i].weight` with actual sustained fps
- Current default: equal weights (1.0) for all lanes

**Step 14: 1 kHz stress test**
- Long-duration test to validate zero frame/GOP gaps and bounded queue depth
- Measure p99/p999 latencies, identify jitter sources
- Tune pool sizes, stall timeout, and scheduling thresholds

---

# Cross-GPU Transfer Design

## Why cross-GPU transfer is necessary

The XIMEA GPUDirect RDMA can only stream to **one GPU at a time**. Three design choices:

1. **Capture to GPU with most NVENC capacity** (current)
   - 5070 Ti receives all frames (4.063 GB/s at 1000 fps, 4096×992)
   - 2/3 encoding local (5070 Ti lanes) — no transfer
   - 1/3 frames cross to PRO 2000 — **~1.35 GB/s** (acceptable)

2. **Capture to PRO 2000, transfer back**
   - 2/3 of frames must cross to 5070 Ti — **~2.7 GB/s** (worse)

3. **Use only one GPU**
   - Wastes an NVENC engine, cannot sustain 1000 fps

Option 1 minimizes cross-GPU bandwidth. RECORD.md "GPU placement" section justifies this trade-off.

## Current implementation: 48-slot staging pool

For each remote GPU (e.g., PRO 2000), the code allocates:

```cpp
rt.gray8_staging.assign(cfg_.pool_size_per_lane, nullptr);  // 48 slots on PRO 2000
```

**Why 48 slots:**
- Matches the bounded pool of the destination lane
- Allows transfer to be fully **pipelined** with encoding: while one frame is converting/encoding, the next transfer can be in-flight
- With 1 ms per frame and ~3 ms encode latency, a lane can have ~3 frames in-flight; 48 slots provide ample safety margin
- P2P and pinned-host transfers are asynchronous; pipelining hides transfer latency

**Trade-off:**
- GPU memory: 48 × 4 MB ≈ 192 MB on each remote GPU
- For PRO 2000 with 24 GB VRAM, this is negligible
- For lower-VRAM GPUs, could reduce to a 2–4 slot ring buffer with synchronous transfer (but loses parallelism)

## Alternative: 2-slot ring buffer

A lighter approach for memory-constrained deployments:

```cpp
// Allocate just 2 staging buffers instead of 48
rt.gray8_staging.assign(2, nullptr);
```

**Pros:**
- Saves ~188 MB per remote GPU
- Still allows some pipelining (transfer N while encoding N-1)

**Cons:**
- More complex synchronization (must wait for previous transfer to complete before reusing slot)
- Slightly higher per-frame latency (serialized dependency)
- Not recommended unless VRAM is truly constrained

For the current RTX 5070 Ti + PRO 2000 setup, **48-slot is fine**. If deploying to lower-VRAM devices, measure the impact and tune accordingly.

---

# Known limitations and future work

1. **Lane weight tuning requires benchmarking** (step 13)
   - Current weights are equal; scheduler distributes GOPs uniformly
   - Real GPUs may have different sustained fps; once measured, update weights for better load balance

2. **Capture thread allocations**
   - `gop_lane_map_` uses `std::map::insert` ~33 Hz (one per GOP)
   - RECORD.md capture-thread rule discourages steady-state allocations
   - Low frequency makes this unlikely to cause jitter, but a fixed-size ring array would close the gap

3. **Cross-GPU transfer latency not in telemetry**
   - Only a frame counter (`rec_cross_gpu_frames`), not percentile latency
   - Adding event-based timing would require capture-thread changes (careful to avoid stalls)

4. **No automatic lane redistribution**
   - If one lane dies, GOP scheduler skips to the least-loaded live lane, but does not rebalance weights
   - Manual reconfiguration or dynamic weight adjustment could help, but would complicate restart logic