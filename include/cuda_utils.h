#pragma once
#include <cstdint>
#include <cuda_runtime.h>

void convert_gray8_to_nv12_gpu(const uint8_t* src, uint8_t* dst, int width, int height, cudaStream_t stream = 0);
