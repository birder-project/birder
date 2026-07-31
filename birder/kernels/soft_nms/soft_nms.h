/*!
**************************************************************************************************
* Soft-NMS
* Taken from:
* https://github.com/MrParosk/soft_nms
* Licensed under the MIT License
*
* Class-aware CUDA implementation added by:
* Ofer Hasson - 2026-07-22
**************************************************************************************************
*/

#pragma once

#include <tuple>

#include <ATen/core/Tensor.h>

std::tuple<at::Tensor, at::Tensor> soft_nms_cuda(const at::Tensor &boxes, const at::Tensor &scores,
                                                 const at::Tensor &class_ids, double sigma, double score_threshold);
