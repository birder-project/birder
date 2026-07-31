/*!
**************************************************************************************************
* TransNeXt
* Taken from:
* https://github.com/DaiShiResearch/TransNeXt/blob/main/swattention_extension
* Licensed under the Apache License, Version 2.0
*
* Modified by:
* Ofer Hasson - 2026-07-20
**************************************************************************************************
*/

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>

template <typename scalar_t>
__global__ void av_fw_kernel(const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> attn_weight,
                             const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> values,
                             at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> output, int height,
                             int width, int kernel_size) {
    using acc_t = at::opmath_type<scalar_t>;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (values.size(0) * values.size(1))) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < values.size(2)) {
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < values.size(3)) {
                const int b = x / values.size(1);
                const int h = x - b * values.size(1);
                const int i = y / width;
                const int j = y - i * width;
                const int start_i = i - (kernel_size - 1) / 2;
                const int start_j = j - (kernel_size - 1) / 2;
                acc_t updt = acc_t(0);
                int k_offset = 0;

#pragma unroll
                for (int current_i = start_i; current_i < (start_i + kernel_size); ++current_i) {
#pragma unroll
                    for (int current_j = start_j; current_j < (start_j + kernel_size); ++current_j) {
                        if (((current_i >= 0) && (current_i < height)) && ((current_j >= 0) && (current_j < width))) {
                            const int current_offset = current_i * width + current_j;
                            updt += static_cast<acc_t>(attn_weight[b][h][y][k_offset]) *
                                    static_cast<acc_t>(values[b][h][current_offset][z]);
                        }
                        ++k_offset;
                    }
                }
                output[b][h][y][z] = static_cast<scalar_t>(updt);
            }
        }
    }
}

at::Tensor av_fw_cu(const at::Tensor &attn_weight, const at::Tensor &values, int height, int width, int kernel_size,
                    int cuda_threads) {
    at::DeviceGuard guard(values.device());

    const int batch = values.size(0), num_heads = values.size(1), num_tokens = values.size(2),
              channels = values.size(3);

    const int dim_threads = std::min(cuda_threads, channels);
    const int pixel_threads = std::min(int(cuda_threads / dim_threads), num_tokens);
    const int batch_threads = std::max(1, cuda_threads / (pixel_threads * dim_threads));

    at::Tensor output = at::empty({batch, num_heads, num_tokens, channels}, values.options());

    const dim3 threads(batch_threads, pixel_threads, dim_threads);
    const dim3 blocks(((batch * num_heads) + threads.x - 1) / threads.x, (num_tokens + threads.y - 1) / threads.y,
                      (channels + threads.z - 1) / threads.z);
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, attn_weight.scalar_type(), "av_fw_cu", ([&] {
            av_fw_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
                attn_weight.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                values.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(), height, width, kernel_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));

    return output;
}
