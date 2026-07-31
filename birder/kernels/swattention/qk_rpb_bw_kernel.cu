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
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <vector>

namespace {

constexpr int rpb_reduction_threads = 256;
constexpr int rpb_reduction_partitions = 4;

template <typename scalar_t, typename acc_t>
__global__ void rpb_bw_partial_kernel(const scalar_t *__restrict__ d_attn_weight, acc_t *__restrict__ partial_sums,
                                      int batch, int num_heads, int num_tokens, int attention_span, int height,
                                      int width, int kernel_size) {
    __shared__ acc_t block_sums[rpb_reduction_threads];

    const int partition = static_cast<int>(blockIdx.x);
    const int offset = static_cast<int>(blockIdx.y);
    const int head = static_cast<int>(blockIdx.z);
    const int kernel_row = offset / kernel_size;
    const int kernel_column = offset - kernel_row * kernel_size;
    const int radius = (kernel_size - 1) / 2;
    const int item_count = batch * num_tokens;

    acc_t sum = acc_t(0);
    for (int item = partition * blockDim.x + threadIdx.x; item < item_count;
         item += rpb_reduction_partitions * blockDim.x) {
        const int batch_index = item / num_tokens;
        const int token = item - batch_index * num_tokens;
        const int row = token / width;
        const int column = token - row * width;
        const int neighbor_row = row + kernel_row - radius;
        const int neighbor_column = column + kernel_column - radius;
        if (neighbor_row >= 0 && neighbor_row < height && neighbor_column >= 0 && neighbor_column < width) {
            const int input_index = ((batch_index * num_heads + head) * num_tokens + token) * attention_span + offset;
            sum += static_cast<acc_t>(d_attn_weight[input_index]);
        }
    }

    block_sums[threadIdx.x] = sum;
    __syncthreads();
    for (int reduction_offset = rpb_reduction_threads / 2; reduction_offset > 0; reduction_offset /= 2) {
        if (threadIdx.x < reduction_offset) {
            block_sums[threadIdx.x] += block_sums[threadIdx.x + reduction_offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int output_index = (head * attention_span + offset) * rpb_reduction_partitions + partition;
        partial_sums[output_index] = block_sums[0];
    }
}

template <typename scalar_t, typename acc_t>
__global__ void rpb_bw_finalize_kernel(const acc_t *__restrict__ partial_sums, scalar_t *__restrict__ d_rpb,
                                       int output_count) {
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index < output_count) {
        acc_t sum = acc_t(0);
#pragma unroll
        for (int partition = 0; partition < rpb_reduction_partitions; ++partition) {
            sum += partial_sums[output_index * rpb_reduction_partitions + partition];
        }
        d_rpb[output_index] = static_cast<scalar_t>(sum);
    }
}

} // namespace

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

std::vector<at::Tensor> qk_rpb_bw_cu(const at::Tensor &d_attn_weight, const at::Tensor &queries, const at::Tensor &keys,
                                     int height, int width, int kernel_size, int cuda_threads) {
    at::DeviceGuard guard(queries.device());

    TORCH_CHECK(queries.size(0) == keys.size(0), "Query and Key should have same Batch_Size");
    TORCH_CHECK(queries.size(1) == keys.size(1), "Query and Key should have same Head Nums");
    TORCH_CHECK(queries.size(2) == keys.size(2), "Query and Key should have same Pixel Nums");
    TORCH_CHECK(queries.size(3) == keys.size(3), "Query and Key should have same Head Dims");
    const int batch = queries.size(0), num_heads = queries.size(1), num_tokens = queries.size(2),
              channels = queries.size(3);

    const int attention_span = kernel_size * kernel_size;
    const int qk_dim_threads = std::min(cuda_threads, channels);
    const int qk_pixel_threads = std::min(int(cuda_threads / qk_dim_threads), num_tokens);
    const int qk_batch_threads = std::max(1, cuda_threads / (qk_pixel_threads * qk_dim_threads));

    at::Tensor d_queries = at::empty({batch, num_heads, num_tokens, channels}, queries.options());
    at::Tensor d_keys = at::empty({batch, num_heads, num_tokens, channels}, keys.options());
    at::Tensor d_rpb = at::empty({num_heads, attention_span}, keys.options());
    const c10::ScalarType accumulation_type =
        keys.scalar_type() == c10::ScalarType::Half || keys.scalar_type() == c10::ScalarType::BFloat16
            ? c10::ScalarType::Float
            : keys.scalar_type();
    at::Tensor rpb_partial_sums =
        at::empty({num_heads, attention_span, rpb_reduction_partitions}, keys.options().dtype(accumulation_type));

    const dim3 qk_threads(qk_batch_threads, qk_pixel_threads, qk_dim_threads);
    const dim3 qk_blocks(((batch * num_heads) + qk_threads.x - 1) / qk_threads.x,
                         (num_tokens + qk_threads.y - 1) / qk_threads.y, (channels + qk_threads.z - 1) / qk_threads.z);
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, queries.scalar_type(), "qk_bw_cu", ([&] {
            using acc_t = at::opmath_type<scalar_t>;
            const dim3 rpb_blocks(rpb_reduction_partitions, attention_span, num_heads);
            rpb_bw_partial_kernel<scalar_t, acc_t><<<rpb_blocks, rpb_reduction_threads, 0, stream.stream()>>>(
                d_attn_weight.data_ptr<scalar_t>(), rpb_partial_sums.data_ptr<acc_t>(), batch, num_heads, num_tokens,
                attention_span, height, width, kernel_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            const int rpb_output_count = num_heads * attention_span;
            const int rpb_finalize_blocks = (rpb_output_count + rpb_reduction_threads - 1) / rpb_reduction_threads;
            rpb_bw_finalize_kernel<scalar_t, acc_t><<<rpb_finalize_blocks, rpb_reduction_threads, 0, stream.stream()>>>(
                rpb_partial_sums.data_ptr<acc_t>(), d_rpb.data_ptr<scalar_t>(), rpb_output_count);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            qk_bw_kernel<scalar_t><<<qk_blocks, qk_threads, 0, stream.stream()>>>(
                d_attn_weight.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                keys.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                d_queries.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(), height, width, kernel_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            qk_inverse_bw_kernel<scalar_t><<<qk_blocks, qk_threads, 0, stream.stream()>>>(
                d_attn_weight.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                queries.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                d_keys.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(), height, width, kernel_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));

    return {d_queries, d_keys, d_rpb};
}
