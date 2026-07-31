/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
* Packed per-level point-count support added by:
* Ofer Hasson - 2026-07-17
**************************************************************************************************
*/

#pragma once

#include <cstdint>
#include <vector>

#include <ATen/core/Tensor.h>

at::Tensor ms_deform_attn_cuda_forward(const at::Tensor &value, const at::Tensor &spatial_shapes,
                                       const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
                                       const at::Tensor &attn_weight, int64_t im2col_step);

at::Tensor ms_deform_attn_cuda_packed_forward(const at::Tensor &value, const at::Tensor &spatial_shapes,
                                              const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
                                              const at::Tensor &attn_weight, const at::Tensor &num_points_per_level,
                                              int64_t im2col_step);

std::vector<at::Tensor> ms_deform_attn_cuda_backward(const at::Tensor &value, const at::Tensor &spatial_shapes,
                                                     const at::Tensor &level_start_index,
                                                     const at::Tensor &sampling_loc, const at::Tensor &attn_weight,
                                                     const at::Tensor &grad_output, int64_t im2col_step);

std::vector<at::Tensor> ms_deform_attn_cuda_packed_backward(const at::Tensor &value, const at::Tensor &spatial_shapes,
                                                            const at::Tensor &level_start_index,
                                                            const at::Tensor &sampling_loc,
                                                            const at::Tensor &attn_weight,
                                                            const at::Tensor &num_points_per_level,
                                                            const at::Tensor &grad_output, int64_t im2col_step);
