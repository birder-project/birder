/*!
**************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************
* Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
* Copyright (c) 2018 Microsoft
**************************************************************************
* Modified from https://github.com/huggingface/transformers/pull/35979/files
**************************************************************************
* Packed per-level point-count support added by:
* Ofer Hasson - 2026-07-17
**************************************************************************
*/

#include <cuda_runtime.h>

#include <ATen/OpMathType.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

constexpr int cuda_num_threads = 1024;
constexpr int im2col_num_threads = 256;
constexpr int warp_reduce_num_threads = 256;
inline int get_blocks(const int n, const int num_threads) {
    return (n + num_threads - 1) / num_threads;
}

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(const scalar_t *bottom_data, int height, int width, int nheads,
                                                   int channels, scalar_t h, scalar_t w, int m, int c) {
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh, hw = 1 - lw;

    const int w_stride = nheads * channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels + c;

    scalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
    }
    scalar_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
    }
    scalar_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
    }
    scalar_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
    }

    const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear(const scalar_t *bottom_data, int height, int width, int nheads,
                                               int channels, scalar_t h, scalar_t w, int m, int c, scalar_t top_grad,
                                               scalar_t attn_weight, scalar_t *grad_value, scalar_t *grad_sampling_loc,
                                               scalar_t *grad_attn_weight) {
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh, hw = 1 - lw;

    const int w_stride = nheads * channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels + c;

    const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    const scalar_t top_grad_value = top_grad * attn_weight;
    scalar_t grad_h_weight = 0, grad_w_weight = 0;

    scalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
        grad_h_weight -= hw * v1;
        grad_w_weight -= hh * v1;
        atomicAdd(grad_value + ptr1, w1 * top_grad_value);
    }
    scalar_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
        grad_h_weight -= lw * v2;
        grad_w_weight += hh * v2;
        atomicAdd(grad_value + ptr2, w2 * top_grad_value);
    }
    scalar_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
        grad_h_weight += hw * v3;
        grad_w_weight -= lh * v3;
        atomicAdd(grad_value + ptr3, w3 * top_grad_value);
    }
    scalar_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
        grad_h_weight += lw * v4;
        grad_w_weight += lh * v4;
        atomicAdd(grad_value + ptr4, w4 * top_grad_value);
    }

    const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    *grad_attn_weight = top_grad * val;
    *grad_sampling_loc = width * grad_w_weight * top_grad_value;
    *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear_gm(const scalar_t *bottom_data, int height, int width, int nheads,
                                                  int channels, scalar_t h, scalar_t w, int m, int c, scalar_t top_grad,
                                                  scalar_t attn_weight, scalar_t *grad_value,
                                                  scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh, hw = 1 - lw;

    const int w_stride = nheads * channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels + c;

    const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    const scalar_t top_grad_value = top_grad * attn_weight;
    scalar_t grad_h_weight = 0, grad_w_weight = 0;

    scalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
        grad_h_weight -= hw * v1;
        grad_w_weight -= hh * v1;
        atomicAdd(grad_value + ptr1, w1 * top_grad_value);
    }
    scalar_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
        grad_h_weight -= lw * v2;
        grad_w_weight += hh * v2;
        atomicAdd(grad_value + ptr2, w2 * top_grad_value);
    }
    scalar_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
        grad_h_weight += hw * v3;
        grad_w_weight -= lh * v3;
        atomicAdd(grad_value + ptr3, w3 * top_grad_value);
    }
    scalar_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
        grad_h_weight += lw * v4;
        grad_w_weight += lh * v4;
        atomicAdd(grad_value + ptr4, w4 * top_grad_value);
    }

    const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    atomicAdd(grad_attn_weight, top_grad * val);
    atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
    atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
}

template <unsigned int width, typename acc_t>
__device__ __forceinline__ acc_t ms_deform_attn_subgroup_sum(acc_t value, const unsigned int active_mask) {
    static_assert(width == 8 || width == 16 || width == 32,
                  "warp reduction only supports 8-, 16-, or 32-channel subgroups");

#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(active_mask, value, offset, width);
    }

    return value;
}

template <typename scalar_t>
__global__ void
ms_deformable_im2col_gpu_kernel(const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
                                const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
                                const scalar_t *data_attn_weight, const int batch_size, const int spatial_size,
                                const int num_heads, const int channels, const int num_levels, const int num_query,
                                const int num_point, scalar_t *data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        [[maybe_unused]] const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        scalar_t *data_col_ptr = data_col + index;
        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        scalar_t col = 0;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const scalar_t *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t weight = data_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;

                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    col += ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                                                          h_im, w_im, m_col, c_col) *
                           weight;
                }

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
            }
        }
        *data_col_ptr = col;
    }
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel_packed(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight, const int64_t *data_num_points_per_level,
    const int batch_size, const int spatial_size, const int num_heads, const int channels, const int num_levels,
    const int num_query, const int total_points, scalar_t *data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        [[maybe_unused]] const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        scalar_t *data_col_ptr = data_col + index;
        int data_weight_ptr = sampling_index * total_points;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        scalar_t col = 0;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int num_point = data_num_points_per_level[l_col];
            const scalar_t *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t weight = data_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;

                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    col += ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                                                          h_im, w_im, m_col, c_col) *
                           weight;
                }

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
            }
        }
        *data_col_ptr = col;
    }
}

template <typename scalar_t, unsigned int channels, bool packed>
__global__ void ms_deformable_col2im_gpu_kernel_warp_reduce(
    const int num_groups, const scalar_t *grad_col, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t *data_num_points_per_level, const int spatial_size, const int num_heads, const int num_levels,
    const int num_query, const int num_point, const int total_points, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
    static_assert(warp_reduce_num_threads % channels == 0, "channel subgroups must divide the thread block");

    constexpr int groups_per_block = warp_reduce_num_threads / channels;
    const int group_in_block = threadIdx.x / channels;
    const int channel = threadIdx.x % channels;
    const int group_index = blockIdx.x * groups_per_block + group_in_block;
    const bool group_valid = group_index < num_groups;

    // Every physical warp executes the ballot before invalid logical subgroups exit. The resulting mask therefore
    // names exactly the lanes that later participate in each shuffle reduction, including a partially used last warp.
    const unsigned int active_mask = __ballot_sync(0xffffffffU, group_valid);
    if (group_valid == false) {
        return;
    }

    using acc_t = at::opmath_type<scalar_t>;
    const int m_col = group_index % num_heads;
    const int b_col = group_index / (num_query * num_heads);
    const scalar_t top_grad = grad_col[group_index * channels + channel];
    const int points_per_group = packed ? total_points : num_levels * num_point;
    int data_weight_ptr = group_index * points_per_group;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
        const int level_start_id = data_level_start_index[l_col];
        const int spatial_h_ptr = l_col << 1;
        const int spatial_h = data_spatial_shapes[spatial_h_ptr];
        const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
        const int level_num_point = packed ? data_num_points_per_level[l_col] : num_point;
        const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
        const scalar_t *data_value_ptr = data_value + value_ptr_offset;
        scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

        for (int p_col = 0; p_col < level_num_point; ++p_col) {
            const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
            const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
            const scalar_t weight = data_attn_weight[data_weight_ptr];
            const scalar_t h_im = loc_h * spatial_h - 0.5;
            const scalar_t w_im = loc_w * spatial_w - 0.5;
            scalar_t local_grad_sampling_loc[2] = {0, 0};
            scalar_t local_grad_attn_weight = 0;
            if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                ms_deform_attn_col2im_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im,
                                               m_col, channel, top_grad, weight, grad_value_ptr,
                                               local_grad_sampling_loc, &local_grad_attn_weight);
            }

            acc_t grad_w = static_cast<acc_t>(local_grad_sampling_loc[0]);
            acc_t grad_h = static_cast<acc_t>(local_grad_sampling_loc[1]);
            acc_t grad_a = static_cast<acc_t>(local_grad_attn_weight);
            grad_w = ms_deform_attn_subgroup_sum<channels>(grad_w, active_mask);
            grad_h = ms_deform_attn_subgroup_sum<channels>(grad_h, active_mask);
            grad_a = ms_deform_attn_subgroup_sum<channels>(grad_a, active_mask);

            if (channel == 0) {
                grad_sampling_loc[data_loc_w_ptr] = static_cast<scalar_t>(grad_w);
                grad_sampling_loc[data_loc_w_ptr + 1] = static_cast<scalar_t>(grad_h);
                grad_attn_weight[data_weight_ptr] = static_cast<scalar_t>(grad_a);
            }

            data_weight_ptr += 1;
            data_loc_w_ptr += 2;
        }
    }
}

template <typename scalar_t, unsigned int block_size>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(
    const int n, const scalar_t *grad_col, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads, const int channels, const int num_levels,
    const int num_query, const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
    CUDA_KERNEL_LOOP(index, n) {
        __shared__ scalar_t cache_grad_sampling_loc[block_size * 2];
        __shared__ scalar_t cache_grad_attn_weight[block_size];
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        [[maybe_unused]] const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_attn_weight += grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t *data_value_ptr = data_value + value_ptr_offset;
            scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t weight = data_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
                *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
                *(cache_grad_attn_weight + threadIdx.x) = 0;
                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    ms_deform_attn_col2im_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
                                                   w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
                                                   cache_grad_sampling_loc + (threadIdx.x << 1),
                                                   cache_grad_attn_weight + threadIdx.x);
                }

                __syncthreads();
                if (tid == 0) {
                    scalar_t _grad_w = cache_grad_sampling_loc[0], _grad_h = cache_grad_sampling_loc[1],
                             _grad_a = cache_grad_attn_weight[0];
                    int sid = 2;
                    for (unsigned int tid = 1; tid < block_size; ++tid) {
                        _grad_w += cache_grad_sampling_loc[sid];
                        _grad_h += cache_grad_sampling_loc[sid + 1];
                        _grad_a += cache_grad_attn_weight[tid];
                        sid += 2;
                    }

                    *grad_sampling_loc = _grad_w;
                    *(grad_sampling_loc + 1) = _grad_h;
                    *grad_attn_weight = _grad_a;
                }
                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_attn_weight += grad_weight_stride;
                grad_sampling_loc += grad_loc_stride;
            }
        }
    }
}

template <typename scalar_t, unsigned int block_size>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(
    const int n, const scalar_t *grad_col, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads, const int channels, const int num_levels,
    const int num_query, const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
    CUDA_KERNEL_LOOP(index, n) {
        __shared__ scalar_t cache_grad_sampling_loc[block_size * 2];
        __shared__ scalar_t cache_grad_attn_weight[block_size];
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        [[maybe_unused]] const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_attn_weight += grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t *data_value_ptr = data_value + value_ptr_offset;
            scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t weight = data_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
                *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
                *(cache_grad_attn_weight + threadIdx.x) = 0;
                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    ms_deform_attn_col2im_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
                                                   w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
                                                   cache_grad_sampling_loc + (threadIdx.x << 1),
                                                   cache_grad_attn_weight + threadIdx.x);
                }

                __syncthreads();

                for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        const unsigned int xid1 = tid << 1;
                        const unsigned int xid2 = (tid + s) << 1;
                        cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
                        cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
                        cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    *grad_sampling_loc = cache_grad_sampling_loc[0];
                    *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
                    *grad_attn_weight = cache_grad_attn_weight[0];
                }
                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_attn_weight += grad_weight_stride;
                grad_sampling_loc += grad_loc_stride;
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v1(
    const int n, const scalar_t *grad_col, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads, const int channels, const int num_levels,
    const int num_query, const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
    CUDA_KERNEL_LOOP(index, n) {
        extern __shared__ int _s[];
        scalar_t *cache_grad_sampling_loc = (scalar_t *) _s;
        scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        [[maybe_unused]] const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_attn_weight += grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t *data_value_ptr = data_value + value_ptr_offset;
            scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t weight = data_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
                *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
                *(cache_grad_attn_weight + threadIdx.x) = 0;
                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    ms_deform_attn_col2im_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
                                                   w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
                                                   cache_grad_sampling_loc + (threadIdx.x << 1),
                                                   cache_grad_attn_weight + threadIdx.x);
                }

                __syncthreads();
                if (tid == 0) {
                    scalar_t _grad_w = cache_grad_sampling_loc[0], _grad_h = cache_grad_sampling_loc[1],
                             _grad_a = cache_grad_attn_weight[0];
                    int sid = 2;
                    for (unsigned int tid = 1; tid < blockDim.x; ++tid) {
                        _grad_w += cache_grad_sampling_loc[sid];
                        _grad_h += cache_grad_sampling_loc[sid + 1];
                        _grad_a += cache_grad_attn_weight[tid];
                        sid += 2;
                    }

                    *grad_sampling_loc = _grad_w;
                    *(grad_sampling_loc + 1) = _grad_h;
                    *grad_attn_weight = _grad_a;
                }
                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_attn_weight += grad_weight_stride;
                grad_sampling_loc += grad_loc_stride;
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2(
    const int n, const scalar_t *grad_col, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads, const int channels, const int num_levels,
    const int num_query, const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
    CUDA_KERNEL_LOOP(index, n) {
        extern __shared__ int _s[];
        scalar_t *cache_grad_sampling_loc = (scalar_t *) _s;
        scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        [[maybe_unused]] const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_attn_weight += grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t *data_value_ptr = data_value + value_ptr_offset;
            scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t weight = data_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
                *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
                *(cache_grad_attn_weight + threadIdx.x) = 0;
                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    ms_deform_attn_col2im_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
                                                   w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
                                                   cache_grad_sampling_loc + (threadIdx.x << 1),
                                                   cache_grad_attn_weight + threadIdx.x);
                }

                __syncthreads();

                for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0; s >>= 1, spre >>= 1) {
                    if (tid < s) {
                        const unsigned int xid1 = tid << 1;
                        const unsigned int xid2 = (tid + s) << 1;
                        cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
                        cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
                        cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
                        if (tid + (s << 1) < spre) {
                            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s << 1)];
                            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2 + (s << 1)];
                            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
                        }
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    *grad_sampling_loc = cache_grad_sampling_loc[0];
                    *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
                    *grad_attn_weight = cache_grad_attn_weight[0];
                }
                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_attn_weight += grad_weight_stride;
                grad_sampling_loc += grad_loc_stride;
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(
    const int n, const scalar_t *grad_col, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads, const int channels, const int num_levels,
    const int num_query, const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
    CUDA_KERNEL_LOOP(index, n) {
        extern __shared__ int _s[];
        scalar_t *cache_grad_sampling_loc = (scalar_t *) _s;
        scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        [[maybe_unused]] const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_attn_weight += grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t *data_value_ptr = data_value + value_ptr_offset;
            scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t weight = data_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
                *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
                *(cache_grad_attn_weight + threadIdx.x) = 0;
                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    ms_deform_attn_col2im_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
                                                   w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
                                                   cache_grad_sampling_loc + (threadIdx.x << 1),
                                                   cache_grad_attn_weight + threadIdx.x);
                }

                __syncthreads();

                for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0; s >>= 1, spre >>= 1) {
                    if (tid < s) {
                        const unsigned int xid1 = tid << 1;
                        const unsigned int xid2 = (tid + s) << 1;
                        cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
                        cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
                        cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
                        if (tid + (s << 1) < spre) {
                            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s << 1)];
                            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2 + (s << 1)];
                            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
                        }
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    atomicAdd(grad_sampling_loc, cache_grad_sampling_loc[0]);
                    atomicAdd(grad_sampling_loc + 1, cache_grad_sampling_loc[1]);
                    atomicAdd(grad_attn_weight, cache_grad_attn_weight[0]);
                }
                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_attn_weight += grad_weight_stride;
                grad_sampling_loc += grad_loc_stride;
            }
        }
    }
}

template <typename scalar_t>
__global__ void
ms_deformable_col2im_gpu_kernel_gm(const int n, const scalar_t *grad_col, const scalar_t *data_value,
                                   const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
                                   const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
                                   const int batch_size, const int spatial_size, const int num_heads,
                                   const int channels, const int num_levels, const int num_query, const int num_point,
                                   scalar_t *grad_value, scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
    CUDA_KERNEL_LOOP(index, n) {
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        [[maybe_unused]] const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_attn_weight += grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t *data_value_ptr = data_value + value_ptr_offset;
            scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t weight = data_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    ms_deform_attn_col2im_bilinear_gm(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
                                                      w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
                                                      grad_sampling_loc, grad_attn_weight);
                }
                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_attn_weight += grad_weight_stride;
                grad_sampling_loc += grad_loc_stride;
            }
        }
    }
}

template <typename scalar_t, unsigned int block_size>
__global__ void ms_deformable_col2im_gpu_kernel_packed_shm_blocksize_aware_reduce(
    const int n, const scalar_t *grad_col, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t *data_num_points_per_level, const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query, const int total_points, scalar_t *grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
    CUDA_KERNEL_LOOP(index, n) {
        __shared__ scalar_t cache_grad_sampling_loc[block_size * 2];
        __shared__ scalar_t cache_grad_attn_weight[block_size];
        const unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        [[maybe_unused]] const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * total_points;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_attn_weight += grad_sampling_ptr;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int num_point = data_num_points_per_level[l_col];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t *data_value_ptr = data_value + value_ptr_offset;
            scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t weight = data_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                cache_grad_sampling_loc[threadIdx.x << 1] = 0;
                cache_grad_sampling_loc[(threadIdx.x << 1) + 1] = 0;
                cache_grad_attn_weight[threadIdx.x] = 0;
                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    ms_deform_attn_col2im_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
                                                   w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
                                                   cache_grad_sampling_loc + (threadIdx.x << 1),
                                                   cache_grad_attn_weight + threadIdx.x);
                }

                __syncthreads();
                if (tid == 0) {
                    scalar_t grad_w = cache_grad_sampling_loc[0];
                    scalar_t grad_h = cache_grad_sampling_loc[1];
                    scalar_t grad_a = cache_grad_attn_weight[0];
                    int shared_idx = 2;
                    for (unsigned int channel_idx = 1; channel_idx < block_size; ++channel_idx) {
                        grad_w += cache_grad_sampling_loc[shared_idx];
                        grad_h += cache_grad_sampling_loc[shared_idx + 1];
                        grad_a += cache_grad_attn_weight[channel_idx];
                        shared_idx += 2;
                    }

                    *grad_sampling_loc = grad_w;
                    *(grad_sampling_loc + 1) = grad_h;
                    *grad_attn_weight = grad_a;
                }
                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_attn_weight += 1;
                grad_sampling_loc += 2;
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_packed_gm(
    const int n, const scalar_t *grad_col, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t *data_num_points_per_level, const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query, const int total_points, scalar_t *grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
    CUDA_KERNEL_LOOP(index, n) {
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        [[maybe_unused]] const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        const scalar_t top_grad = grad_col[index];

        int data_weight_ptr = sampling_index * total_points;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_sampling_loc += grad_sampling_ptr << 1;
        grad_attn_weight += grad_sampling_ptr;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const int num_point = data_num_points_per_level[l_col];
            const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
            const scalar_t *data_value_ptr = data_value + value_ptr_offset;
            scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t weight = data_attn_weight[data_weight_ptr];

                const scalar_t h_im = loc_h * spatial_h - 0.5;
                const scalar_t w_im = loc_w * spatial_w - 0.5;
                if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                    ms_deform_attn_col2im_bilinear_gm(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
                                                      w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
                                                      grad_sampling_loc, grad_attn_weight);
                }
                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_attn_weight += 1;
                grad_sampling_loc += 2;
            }
        }
    }
}

template <typename scalar_t>
void ms_deformable_im2col_cuda(cudaStream_t stream, const scalar_t *data_value, const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight, const int batch_size, const int spatial_size,
                               const int num_heads, const int channels, const int num_levels, const int num_query,
                               const int num_point, scalar_t *data_col) {
    const int num_kernels = batch_size * num_query * num_heads * channels;
    const int num_actual_kernels = batch_size * num_query * num_heads * channels;
    const int num_threads = im2col_num_threads;
    ms_deformable_im2col_gpu_kernel<scalar_t><<<get_blocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
        num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight,
        batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, data_col);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void ms_deformable_im2col_packed_cuda(cudaStream_t stream, const scalar_t *data_value,
                                      const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
                                      const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
                                      const int64_t *data_num_points_per_level, const int batch_size,
                                      const int spatial_size, const int num_heads, const int channels,
                                      const int num_levels, const int num_query, const int total_points,
                                      scalar_t *data_col) {
    const int num_kernels = batch_size * num_query * num_heads * channels;
    const int num_threads = im2col_num_threads;
    ms_deformable_im2col_gpu_kernel_packed<scalar_t><<<get_blocks(num_kernels, num_threads), num_threads, 0, stream>>>(
        num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight,
        data_num_points_per_level, batch_size, spatial_size, num_heads, channels, num_levels, num_query, total_points,
        data_col);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, unsigned int block_size>
void ms_deformable_col2im_packed_cuda_blocksize_aware(
    cudaStream_t stream, const scalar_t *grad_col, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t *data_num_points_per_level, const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query, const int total_points, scalar_t *grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
    const int num_kernels = batch_size * num_query * num_heads * channels;
    ms_deformable_col2im_gpu_kernel_packed_shm_blocksize_aware_reduce<scalar_t, block_size>
        <<<get_blocks(num_kernels, block_size), block_size, 0, stream>>>(
            num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
            data_attn_weight, data_num_points_per_level, batch_size, spatial_size, num_heads, channels, num_levels,
            num_query, total_points, grad_value, grad_sampling_loc, grad_attn_weight);
}

template <typename scalar_t, unsigned int channels, bool packed>
void ms_deformable_col2im_cuda_warp_reduce(cudaStream_t stream, const scalar_t *grad_col, const scalar_t *data_value,
                                           const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
                                           const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
                                           const int64_t *data_num_points_per_level, const int batch_size,
                                           const int spatial_size, const int num_heads, const int num_levels,
                                           const int num_query, const int num_point, const int total_points,
                                           scalar_t *grad_value, scalar_t *grad_sampling_loc,
                                           scalar_t *grad_attn_weight) {
    constexpr int groups_per_block = warp_reduce_num_threads / channels;
    const int num_groups = batch_size * num_query * num_heads;
    ms_deformable_col2im_gpu_kernel_warp_reduce<scalar_t, channels, packed>
        <<<get_blocks(num_groups, groups_per_block), warp_reduce_num_threads, 0, stream>>>(
            num_groups, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
            data_attn_weight, data_num_points_per_level, spatial_size, num_heads, num_levels, num_query, num_point,
            total_points, grad_value, grad_sampling_loc, grad_attn_weight);
}

template <typename scalar_t>
void ms_deformable_col2im_packed_cuda(cudaStream_t stream, const scalar_t *grad_col, const scalar_t *data_value,
                                      const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
                                      const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
                                      const int64_t *data_num_points_per_level, const int batch_size,
                                      const int spatial_size, const int num_heads, const int channels,
                                      const int num_levels, const int num_query, const int total_points,
                                      scalar_t *grad_value, scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
    switch (channels) {
    case 1:
        ms_deformable_col2im_packed_cuda_blocksize_aware<scalar_t, 1>(
            stream, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
            data_attn_weight, data_num_points_per_level, batch_size, spatial_size, num_heads, channels, num_levels,
            num_query, total_points, grad_value, grad_sampling_loc, grad_attn_weight);
        break;
    case 2:
        ms_deformable_col2im_packed_cuda_blocksize_aware<scalar_t, 2>(
            stream, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
            data_attn_weight, data_num_points_per_level, batch_size, spatial_size, num_heads, channels, num_levels,
            num_query, total_points, grad_value, grad_sampling_loc, grad_attn_weight);
        break;
    case 4:
        ms_deformable_col2im_packed_cuda_blocksize_aware<scalar_t, 4>(
            stream, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
            data_attn_weight, data_num_points_per_level, batch_size, spatial_size, num_heads, channels, num_levels,
            num_query, total_points, grad_value, grad_sampling_loc, grad_attn_weight);
        break;
    case 8:
        ms_deformable_col2im_cuda_warp_reduce<scalar_t, 8, true>(
            stream, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
            data_attn_weight, data_num_points_per_level, batch_size, spatial_size, num_heads, num_levels, num_query, 0,
            total_points, grad_value, grad_sampling_loc, grad_attn_weight);
        break;
    case 16:
        ms_deformable_col2im_cuda_warp_reduce<scalar_t, 16, true>(
            stream, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
            data_attn_weight, data_num_points_per_level, batch_size, spatial_size, num_heads, num_levels, num_query, 0,
            total_points, grad_value, grad_sampling_loc, grad_attn_weight);
        break;
    case 32:
        ms_deformable_col2im_cuda_warp_reduce<scalar_t, 32, true>(
            stream, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
            data_attn_weight, data_num_points_per_level, batch_size, spatial_size, num_heads, num_levels, num_query, 0,
            total_points, grad_value, grad_sampling_loc, grad_attn_weight);
        break;
    default: {
        const int num_threads = std::min(channels, cuda_num_threads);
        const int num_kernels = batch_size * num_query * num_heads * channels;
        ms_deformable_col2im_gpu_kernel_packed_gm<scalar_t>
            <<<get_blocks(num_kernels, num_threads), num_threads, 0, stream>>>(
                num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                data_attn_weight, data_num_points_per_level, batch_size, spatial_size, num_heads, channels, num_levels,
                num_query, total_points, grad_value, grad_sampling_loc, grad_attn_weight);
        break;
    }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void ms_deformable_col2im_cuda(cudaStream_t stream, const scalar_t *grad_col, const scalar_t *data_value,
                               const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
                               const int batch_size, const int spatial_size, const int num_heads, const int channels,
                               const int num_levels, const int num_query, const int num_point, scalar_t *grad_value,
                               scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
    const int num_threads = (channels > cuda_num_threads) ? cuda_num_threads : channels;
    const int num_kernels = batch_size * num_query * num_heads * channels;
    const int num_actual_kernels = batch_size * num_query * num_heads * channels;
    if (channels > 1024) {
        if ((channels & 1023) == 0) {
            ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
                <<<get_blocks(num_actual_kernels, num_threads), num_threads, num_threads * 3 * sizeof(scalar_t),
                   stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index,
                             data_sampling_loc, data_attn_weight, batch_size, spatial_size, num_heads, channels,
                             num_levels, num_query, num_point, grad_value, grad_sampling_loc, grad_attn_weight);
        } else {
            ms_deformable_col2im_gpu_kernel_gm<scalar_t>
                <<<get_blocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                    num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                    data_attn_weight, batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                    grad_value, grad_sampling_loc, grad_attn_weight);
        }
    } else {
        switch (channels) {
        case 1:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
                <<<get_blocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                    num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                    data_attn_weight, batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                    grad_value, grad_sampling_loc, grad_attn_weight);
            break;
        case 2:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
                <<<get_blocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                    num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                    data_attn_weight, batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                    grad_value, grad_sampling_loc, grad_attn_weight);
            break;
        case 4:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
                <<<get_blocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                    num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                    data_attn_weight, batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                    grad_value, grad_sampling_loc, grad_attn_weight);
            break;
        case 8:
            ms_deformable_col2im_cuda_warp_reduce<scalar_t, 8, false>(
                stream, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                data_attn_weight, nullptr, batch_size, spatial_size, num_heads, num_levels, num_query, num_point, 0,
                grad_value, grad_sampling_loc, grad_attn_weight);
            break;
        case 16:
            ms_deformable_col2im_cuda_warp_reduce<scalar_t, 16, false>(
                stream, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                data_attn_weight, nullptr, batch_size, spatial_size, num_heads, num_levels, num_query, num_point, 0,
                grad_value, grad_sampling_loc, grad_attn_weight);
            break;
        case 32:
            ms_deformable_col2im_cuda_warp_reduce<scalar_t, 32, false>(
                stream, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                data_attn_weight, nullptr, batch_size, spatial_size, num_heads, num_levels, num_query, num_point, 0,
                grad_value, grad_sampling_loc, grad_attn_weight);
            break;
        case 64:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
                <<<get_blocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                    num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                    data_attn_weight, batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                    grad_value, grad_sampling_loc, grad_attn_weight);
            break;
        case 128:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
                <<<get_blocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                    num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                    data_attn_weight, batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                    grad_value, grad_sampling_loc, grad_attn_weight);
            break;
        case 256:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
                <<<get_blocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                    num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                    data_attn_weight, batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                    grad_value, grad_sampling_loc, grad_attn_weight);
            break;
        case 512:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
                <<<get_blocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                    num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                    data_attn_weight, batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                    grad_value, grad_sampling_loc, grad_attn_weight);
            break;
        case 1024:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
                <<<get_blocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
                    num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc,
                    data_attn_weight, batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                    grad_value, grad_sampling_loc, grad_attn_weight);
            break;
        default:
            if (channels < 64) {
                ms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
                    <<<get_blocks(num_actual_kernels, num_threads), num_threads, num_threads * 3 * sizeof(scalar_t),
                       stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index,
                                 data_sampling_loc, data_attn_weight, batch_size, spatial_size, num_heads, channels,
                                 num_levels, num_query, num_point, grad_value, grad_sampling_loc, grad_attn_weight);
            } else {
                ms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
                    <<<get_blocks(num_actual_kernels, num_threads), num_threads, num_threads * 3 * sizeof(scalar_t),
                       stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes, data_level_start_index,
                                 data_sampling_loc, data_attn_weight, batch_size, spatial_size, num_heads, channels,
                                 num_levels, num_query, num_point, grad_value, grad_sampling_loc, grad_attn_weight);
            }
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
