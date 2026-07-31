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
__global__ void av_bw_kernel(const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> d_output,
                             const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> values,
                             at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> d_attn_weight, int height,
                             int width, int kernel_size) {
    using acc_t = at::opmath_type<scalar_t>;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (d_output.size(0) * d_output.size(1))) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < d_output.size(2)) {
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < kernel_size * kernel_size) {
                const int b = x / d_output.size(1);
                const int h = x - b * d_output.size(1);
                const int ki = z / kernel_size;
                const int kj = z - ki * kernel_size;
                const int i = y / width;
                const int j = y - i * width;
                const int ni = i + ki - (kernel_size - 1) / 2;
                const int nj = j + kj - (kernel_size - 1) / 2;

                acc_t updt = acc_t(0);
                if (((ni >= 0) && (ni < height)) && ((nj >= 0) && (nj < width))) {
                    const int key_y = ni * width + nj;
#pragma unroll
                    for (int dim_offset = 0; dim_offset < d_output.size(3); ++dim_offset)
                        updt += static_cast<acc_t>(d_output[b][h][y][dim_offset]) *
                                static_cast<acc_t>(values[b][h][key_y][dim_offset]);
                }
                d_attn_weight[b][h][y][z] = static_cast<scalar_t>(updt);
            }
        }
    }
}

template <typename scalar_t>
__global__ void av_inverse_bw_kernel(const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> attn_weight,
                                     const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> d_output,
                                     at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> d_values,
                                     int height, int width, int kernel_size) {
    using acc_t = at::opmath_type<scalar_t>;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (d_values.size(0) * d_values.size(1))) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < d_values.size(2)) {
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < d_values.size(3)) {
                const int b = x / d_values.size(1);
                const int h = x - b * d_values.size(1);
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
                            updt += static_cast<acc_t>(attn_weight[b][h][current_offset][k_offset]) *
                                    static_cast<acc_t>(d_output[b][h][current_offset][z]);
                        }
                    }
                }
                d_values[b][h][y][z] = static_cast<scalar_t>(updt);
            }
        }
    }
}

std::vector<at::Tensor> av_bw_cu(const at::Tensor &d_output, const at::Tensor &attn_weight, const at::Tensor &values,
                                 int height, int width, int kernel_size, int cuda_threads) {
    at::DeviceGuard guard(values.device());

    const int batch = values.size(0), num_heads = values.size(1), num_tokens = values.size(2),
              channels = values.size(3);
    const int attention_span = kernel_size * kernel_size;

    const int attn_kernel_threads = std::min(cuda_threads, attention_span);
    const int attn_pixel_threads = std::min(int(cuda_threads / attn_kernel_threads), num_tokens);
    const int attn_batch_threads = std::max(1, cuda_threads / (attn_pixel_threads * attn_kernel_threads));
    const dim3 attn_threads(attn_batch_threads, attn_pixel_threads, attn_kernel_threads);
    const dim3 attn_blocks(((batch * num_heads) + attn_threads.x - 1) / attn_threads.x,
                           (num_tokens + attn_threads.y - 1) / attn_threads.y,
                           (attention_span + attn_threads.z - 1) / attn_threads.z);

    const int value_dim_threads = std::min(cuda_threads, channels);
    const int value_pixel_threads = std::min(int(cuda_threads / value_dim_threads), num_tokens);
    const int value_batch_threads = std::max(1, cuda_threads / (value_pixel_threads * value_dim_threads));
    const dim3 value_threads(value_batch_threads, value_pixel_threads, value_dim_threads);
    const dim3 value_blocks(((batch * num_heads) + value_threads.x - 1) / value_threads.x,
                            (num_tokens + value_threads.y - 1) / value_threads.y,
                            (channels + value_threads.z - 1) / value_threads.z);

    at::Tensor d_attn_weight = at::empty({batch, num_heads, num_tokens, attention_span}, attn_weight.options());
    at::Tensor d_values = at::empty({batch, num_heads, num_tokens, channels}, values.options());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, attn_weight.scalar_type(), "av_bw_cu", ([&] {
            av_bw_kernel<scalar_t><<<attn_blocks, attn_threads, 0, stream.stream()>>>(
                d_output.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                values.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                d_attn_weight.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(), height, width, kernel_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            av_inverse_bw_kernel<scalar_t><<<value_blocks, value_threads, 0, stream.stream()>>>(
                attn_weight.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                d_output.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                d_values.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(), height, width, kernel_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));

    return {d_attn_weight, d_values};
}
