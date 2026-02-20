/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "ms_deform_attn.h"
#include <torch/library.h>

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("ms_deform_attn_forward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor sampling_loc, Tensor attn_weight, int im2col_step) -> Tensor");
  ops.impl("ms_deform_attn_forward", torch::kCUDA, &ms_deform_attn_forward);

  ops.def("ms_deform_attn_backward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor sampling_loc, Tensor attn_weight, Tensor grad_output, int im2col_step) -> Tensor[]");
  ops.impl("ms_deform_attn_backward", torch::kCUDA, &ms_deform_attn_backward);
}
