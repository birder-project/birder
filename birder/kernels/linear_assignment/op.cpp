/*!
**************************************************************************************************
* Linear assignment
* Adapted from:
* https://github.com/ivan-chai/torch-linear-assignment
* Licensed under the Apache License, Version 2.0
**************************************************************************************************
*/

#include <torch/extension.h>
#include <torch/library.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

std::vector<torch::Tensor> batch_linear_assignment_cuda(const torch::Tensor& cost);

std::vector<torch::Tensor> batch_linear_assignment(const torch::Tensor& cost) {
  CHECK_INPUT(cost);
  return batch_linear_assignment_cuda(cost);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("batch_linear_assignment(Tensor cost) -> Tensor[]");
  ops.impl("batch_linear_assignment", torch::kCUDA, &batch_linear_assignment);
}
