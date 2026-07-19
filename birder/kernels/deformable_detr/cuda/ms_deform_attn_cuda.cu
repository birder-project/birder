/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
* Modified from https://github.com/huggingface/transformers/pull/35979/files
**************************************************************************************************
* Packed per-level point-count support added by:
* Ofer Hasson — 2026-07-17
**************************************************************************************************
*/

#include "cuda/ms_deform_im2col_cuda.cuh"
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#pragma once
#include <torch/extension.h>

at::Tensor ms_deform_attn_cuda_forward(const at::Tensor &value, const at::Tensor &spatial_shapes,
                                       const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
                                       const at::Tensor &attn_weight, const int im2col_step) {
    at::DeviceGuard guard(value.device());

    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

    AT_ASSERTM(value.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    AT_ASSERTM(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto output = at::empty({batch, num_query, num_heads, channels}, value.options());

    const int batch_n = im2col_step_;
    auto output_n = output.view({batch / im2col_step_, batch_n, num_query, num_heads, channels});
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    for (int n = 0; n < batch / im2col_step_; ++n) {
        auto columns = output_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
                ms_deformable_im2col_cuda(
                    at::cuda::getCurrentCUDAStream(), value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                    spatial_shapes.data_ptr<int64_t>(), level_start_index.data_ptr<int64_t>(),
                    sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                    attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size, batch_n, spatial_size,
                    num_heads, channels, num_levels, num_query, num_point, columns.data_ptr<scalar_t>());
            }));
    }

    output = output.view({batch, num_query, num_heads * channels});

    return output;
}

at::Tensor ms_deform_attn_cuda_packed_forward(const at::Tensor &value, const at::Tensor &spatial_shapes,
                                              const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
                                              const at::Tensor &attn_weight, const at::Tensor &num_points_per_level,
                                              const int im2col_step) {
    at::DeviceGuard guard(value.device());

    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    AT_ASSERTM(num_points_per_level.is_contiguous(), "num_points_per_level tensor has to be contiguous");

    AT_ASSERTM(value.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    AT_ASSERTM(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
    AT_ASSERTM(num_points_per_level.is_cuda(), "num_points_per_level must be a CUDA tensor");

    AT_ASSERTM(sampling_loc.dim() == 5, "packed sampling_loc must have shape [B, Q, H, total_points, 2]");
    AT_ASSERTM(attn_weight.dim() == 4, "packed attn_weight must have shape [B, Q, H, total_points]");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);
    const int num_levels = spatial_shapes.size(0);
    const int num_query = sampling_loc.size(1);
    const int total_points = sampling_loc.size(3);

    AT_ASSERTM(num_points_per_level.scalar_type() == at::ScalarType::Long,
               "num_points_per_level must have dtype int64");
    AT_ASSERTM(num_points_per_level.numel() == num_levels,
               "num_points_per_level must contain one entry per feature level");
    AT_ASSERTM(attn_weight.size(3) == total_points,
               "sampling_loc and attn_weight must have the same total point count");

    AT_ASSERTM(im2col_step > 0, "im2col_step must be positive");
    const int im2col_step_ = std::min(batch, im2col_step);

    auto output = at::empty({batch, num_query, num_heads, channels}, value.options());

    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * total_points * 2;
    auto per_attn_weight_size = num_query * num_heads * total_points;
    for (int batch_start = 0; batch_start < batch; batch_start += im2col_step_) {
        const int batch_n = std::min(im2col_step_, batch - batch_start);
        auto columns = output.narrow(0, batch_start, batch_n);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(), "ms_deform_attn_packed_forward_cuda",
            ([&] {
                ms_deformable_im2col_packed_cuda(
                    at::cuda::getCurrentCUDAStream(), value.data_ptr<scalar_t>() + batch_start * per_value_size,
                    spatial_shapes.data_ptr<int64_t>(), level_start_index.data_ptr<int64_t>(),
                    sampling_loc.data_ptr<scalar_t>() + batch_start * per_sample_loc_size,
                    attn_weight.data_ptr<scalar_t>() + batch_start * per_attn_weight_size,
                    num_points_per_level.data_ptr<int64_t>(), batch_n, spatial_size, num_heads, channels, num_levels,
                    num_query, total_points, columns.data_ptr<scalar_t>());
            }));
    }

    return output.view({batch, num_query, num_heads * channels});
}

std::vector<at::Tensor> ms_deform_attn_cuda_backward(const at::Tensor &value, const at::Tensor &spatial_shapes,
                                                     const at::Tensor &level_start_index,
                                                     const at::Tensor &sampling_loc, const at::Tensor &attn_weight,
                                                     const at::Tensor &grad_output, const int im2col_step) {
    at::DeviceGuard guard(value.device());

    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    AT_ASSERTM(value.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    AT_ASSERTM(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
    AT_ASSERTM(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    const int batch_n = im2col_step_;
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    auto grad_output_n = grad_output.view({batch / im2col_step_, batch_n, num_query, num_heads, channels});

    for (int n = 0; n < batch / im2col_step_; ++n) {
        auto grad_output_g = grad_output_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
                ms_deformable_col2im_cuda(
                    at::cuda::getCurrentCUDAStream(), grad_output_g.data_ptr<scalar_t>(),
                    value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size, spatial_shapes.data_ptr<int64_t>(),
                    level_start_index.data_ptr<int64_t>(),
                    sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                    attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size, batch_n, spatial_size,
                    num_heads, channels, num_levels, num_query, num_point,
                    grad_value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                    grad_sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                    grad_attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size);
            }));
    }

    return {grad_value, grad_sampling_loc, grad_attn_weight};
}

std::vector<at::Tensor> ms_deform_attn_cuda_packed_backward(const at::Tensor &value, const at::Tensor &spatial_shapes,
                                                            const at::Tensor &level_start_index,
                                                            const at::Tensor &sampling_loc,
                                                            const at::Tensor &attn_weight,
                                                            const at::Tensor &num_points_per_level,
                                                            const at::Tensor &grad_output, const int im2col_step) {
    at::DeviceGuard guard(value.device());

    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    AT_ASSERTM(num_points_per_level.is_contiguous(), "num_points_per_level tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    AT_ASSERTM(value.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    AT_ASSERTM(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
    AT_ASSERTM(num_points_per_level.is_cuda(), "num_points_per_level must be a CUDA tensor");
    AT_ASSERTM(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);
    const int num_levels = spatial_shapes.size(0);
    const int num_query = sampling_loc.size(1);
    const int total_points = sampling_loc.size(3);

    AT_ASSERTM(im2col_step > 0, "im2col_step must be positive");
    const int im2col_step_ = std::min(batch, im2col_step);

    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * total_points * 2;
    auto per_attn_weight_size = num_query * num_heads * total_points;
    auto grad_output_reshaped = grad_output.view({batch, num_query, num_heads, channels});

    for (int batch_start = 0; batch_start < batch; batch_start += im2col_step_) {
        const int batch_n = std::min(im2col_step_, batch - batch_start);
        auto grad_output_g = grad_output_reshaped.narrow(0, batch_start, batch_n);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(), "ms_deform_attn_packed_backward_cuda",
            ([&] {
                ms_deformable_col2im_packed_cuda(
                    at::cuda::getCurrentCUDAStream(), grad_output_g.data_ptr<scalar_t>(),
                    value.data_ptr<scalar_t>() + batch_start * per_value_size, spatial_shapes.data_ptr<int64_t>(),
                    level_start_index.data_ptr<int64_t>(),
                    sampling_loc.data_ptr<scalar_t>() + batch_start * per_sample_loc_size,
                    attn_weight.data_ptr<scalar_t>() + batch_start * per_attn_weight_size,
                    num_points_per_level.data_ptr<int64_t>(), batch_n, spatial_size, num_heads, channels, num_levels,
                    num_query, total_points, grad_value.data_ptr<scalar_t>() + batch_start * per_value_size,
                    grad_sampling_loc.data_ptr<scalar_t>() + batch_start * per_sample_loc_size,
                    grad_attn_weight.data_ptr<scalar_t>() + batch_start * per_attn_weight_size);
            }));
    }

    return {grad_value, grad_sampling_loc, grad_attn_weight};
}
