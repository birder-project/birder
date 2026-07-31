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
#include <vector>

template <typename scalar_t>
__global__ void qk_bw_kernel(const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> d_attn_weight,
                             const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> keys,
                             at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> d_queries, int height,
                             int width, int kernel_size) {
    using acc_t = at::opmath_type<scalar_t>;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (keys.size(0) * keys.size(1))) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < keys.size(2)) {
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < keys.size(3)) {
                const int b = x / keys.size(1);
                const int h = x - b * keys.size(1);
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
                            updt += static_cast<acc_t>(d_attn_weight[b][h][y][k_offset]) *
                                    static_cast<acc_t>(keys[b][h][current_offset][z]);
                        }
                        ++k_offset;
                    }
                }
                d_queries[b][h][y][z] = static_cast<scalar_t>(updt);
            }
        }
    }
}

template <typename scalar_t>
__global__ void qk_inverse_bw_kernel(const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> d_attn_weight,
                                     const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> queries,
                                     at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> d_keys, int height,
                                     int width, int kernel_size) {
    using acc_t = at::opmath_type<scalar_t>;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (d_keys.size(0) * d_keys.size(1))) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < d_keys.size(2)) {
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < d_keys.size(3)) {
                const int b = x / d_keys.size(1);
                const int h = x - b * d_keys.size(1);
                const int i = y / width;
                const int j = y - i * width;
                const int q_start_i = i - kernel_size / 2;
                const int q_end_i = i + 1 + (kernel_size - 1) / 2;
                const int q_start_j = j - kernel_size / 2;
                const int q_end_j = j + 1 + (kernel_size - 1) / 2;
                acc_t updt = acc_t(0);
                int k_offset = kernel_size * kernel_size;
#pragma unroll
                for (int current_i = q_start_i; current_i < q_end_i; ++current_i) {
#pragma unroll
                    for (int current_j = q_start_j; current_j < q_end_j; ++current_j) {
                        --k_offset;
                        if (((current_i >= 0) && (current_i < height)) && ((current_j >= 0) && (current_j < width))) {
                            const int current_offset = current_i * width + current_j;
                            updt += static_cast<acc_t>(d_attn_weight[b][h][current_offset][k_offset]) *
                                    static_cast<acc_t>(queries[b][h][current_offset][z]);
                        }
                    }
                }
                d_keys[b][h][y][z] = static_cast<scalar_t>(updt);
            }
        }
    }
}

std::vector<at::Tensor> qk_bw_cu(const at::Tensor &d_attn_weight, const at::Tensor &queries, const at::Tensor &keys,
                                 int height, int width, int kernel_size, int cuda_threads) {
    at::DeviceGuard guard(queries.device());

    TORCH_CHECK(queries.size(0) == keys.size(0), "Query and Key should have same Batch_Size");
    TORCH_CHECK(queries.size(1) == keys.size(1), "Query and Key should have same Head Nums");
    TORCH_CHECK(queries.size(2) == keys.size(2), "Query and Key should have same Pixel Nums");
    TORCH_CHECK(queries.size(3) == keys.size(3), "Query and Key should have same Head Dims");
    const int batch = queries.size(0), num_heads = queries.size(1), num_tokens = queries.size(2),
              channels = queries.size(3);

    const int dim_threads = std::min(cuda_threads, channels);
    const int pixel_threads = std::min(int(cuda_threads / dim_threads), num_tokens);
    const int batch_threads = std::max(1, cuda_threads / (pixel_threads * dim_threads));

    at::Tensor d_queries = at::empty({batch, num_heads, num_tokens, channels}, queries.options());
    at::Tensor d_keys = at::empty({batch, num_heads, num_tokens, channels}, keys.options());

    const dim3 threads(batch_threads, pixel_threads, dim_threads);
    const dim3 blocks(((batch * num_heads) + threads.x - 1) / threads.x, (num_tokens + threads.y - 1) / threads.y,
                      (channels + threads.z - 1) / threads.z);
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, queries.scalar_type(), "qk_bw_cu", ([&] {
            qk_bw_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
                d_attn_weight.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                keys.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                d_queries.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(), height, width, kernel_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            qk_inverse_bw_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
                d_attn_weight.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                queries.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                d_keys.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(), height, width, kernel_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));

    return {d_queries, d_keys};
}
