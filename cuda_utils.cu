// cuda_utils.cu — Production GRAY8→NV12 kernel
// Strategy from ncu analysis (k_vec_ldg, 84% DRAM throughput, 16.8µs, 0.24% L2 hit):
//  1. Issue UV store FIRST  — no data dependency, hides behind load latency
//  2. Then issue both Y loads — two in-flight 128-bit reads per thread (max ILP)
//  3. __stcs for ALL writes  — streaming store bypasses L1, avoids cache pollution
//  4. ld.cg for reads        — L2-only (L1 hit = 0% anyway), frees L1 for other uses
//  5. 2-row-per-thread       — halves kernel launches overhead, fuses UV cleanly
#include "cuda_utils.h"

// Load 128-bit via "cache global" (L2 only, bypass L1)
__device__ __forceinline__ uint4 ld_cg(const uint4* p)
{
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p) : "memory");
    return v;
}

__global__ void gray8_to_nv12_prod(
        const uint8_t* __restrict__ src,
        uint8_t*       __restrict__ dst,
        int w, int h)
{
    int x   = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (x >= w || row >= h) return;

    // Step 1: UV write first — constant, no data dep, overlaps with DRAM load latency
    __stcs(reinterpret_cast<uint4*>(dst + w * h + (row >> 1) * w + x),
           make_uint4(0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u));

    // Step 2: Issue both Y loads simultaneously (2x in-flight 128-bit loads)
    uint4 v0 = ld_cg(reinterpret_cast<const uint4*>(src + row       * w + x));
    uint4 v1 = ld_cg(reinterpret_cast<const uint4*>(src + (row + 1) * w + x));

    // Step 3: Write Y planes with streaming store (bypass L1)
    __stcs(reinterpret_cast<uint4*>(dst + row       * w + x), v0);
    __stcs(reinterpret_cast<uint4*>(dst + (row + 1) * w + x), v1);
}

void convert_gray8_to_nv12_gpu(
        const uint8_t* src, uint8_t* dst, int width, int height, cudaStream_t stream)
{
    dim3 blk(128, 1);
    dim3 grd((width / 16 + 127) / 128, height / 2);
    gray8_to_nv12_prod<<<grd, blk, 0, stream>>>(src, dst, width, height);
}
