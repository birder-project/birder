/*!
**************************************************************************************************
* Linear assignment
* Adapted from:
* https://github.com/ivan-chai/torch-linear-assignment
* Licensed under the Apache License, Version 2.0
*
* Modified by:
* Ofer Hasson - 2026-07-20
**************************************************************************************************
*/

#include <ATen/core/Tensor.h>
#include <c10/core/DeviceType.h>
#include <torch/library.h>

#include <vector>

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

std::vector<at::Tensor> batch_linear_assignment_cuda(const at::Tensor &cost);

std::vector<at::Tensor> batch_linear_assignment(const at::Tensor &cost) {
    return batch_linear_assignment_cuda(cost);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("batch_linear_assignment(Tensor cost) -> Tensor[]");
    ops.impl("batch_linear_assignment", c10::kCUDA, &batch_linear_assignment);
}
