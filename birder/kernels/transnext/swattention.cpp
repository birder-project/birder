/*!
**************************************************************************************************
* TransNeXt
* Taken from:
* https://github.com/DaiShiResearch/TransNeXt/blob/main/swattention_extension
* Licensed under the Apache License, Version 2.0
**************************************************************************************************
*/

#include <torch/extension.h>
#include <torch/library.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif


torch::Tensor qk_fw_cu(
    const torch::Tensor queries,
    const torch::Tensor keys,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
);

torch::Tensor qk_forward(
    const torch::Tensor queries,
    const torch::Tensor keys,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t cuda_threads
){
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);

    return qk_fw_cu(queries, keys, static_cast<int>(height), static_cast<int>(width), static_cast<int>(kernel_size), static_cast<int>(cuda_threads));
}


std::vector<torch::Tensor> qk_bw_cu(
    const torch::Tensor d_attn_weight,
    const torch::Tensor queries,
    const torch::Tensor keys,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
);

std::vector<torch::Tensor> qk_backward(
    const torch::Tensor d_attn_weight,
    const torch::Tensor queries,
    const torch::Tensor keys,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t cuda_threads
){
    CHECK_INPUT(d_attn_weight);
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);

    return qk_bw_cu(d_attn_weight, queries, keys, static_cast<int>(height), static_cast<int>(width), static_cast<int>(kernel_size), static_cast<int>(cuda_threads));
}


std::vector<torch::Tensor> qk_rpb_bw_cu(
    const torch::Tensor d_attn_weight,
    const torch::Tensor queries,
    const torch::Tensor keys,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
);

std::vector<torch::Tensor> qk_rpb_backward(
    const torch::Tensor d_attn_weight,
    const torch::Tensor queries,
    const torch::Tensor keys,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t cuda_threads
){
    CHECK_INPUT(d_attn_weight);
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);

    return qk_rpb_bw_cu(d_attn_weight, queries, keys, static_cast<int>(height), static_cast<int>(width), static_cast<int>(kernel_size), static_cast<int>(cuda_threads));
}


torch::Tensor qk_rpb_fw_cu(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor rpb,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
);

torch::Tensor qk_rpb_forward(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor rpb,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t cuda_threads
){
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);
    CHECK_INPUT(rpb);

    return qk_rpb_fw_cu(queries, keys, rpb, static_cast<int>(height), static_cast<int>(width), static_cast<int>(kernel_size), static_cast<int>(cuda_threads));
}

torch::Tensor av_fw_cu(
    const torch::Tensor attn_weight,
    const torch::Tensor values,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
);

torch::Tensor av_forward(
    const torch::Tensor attn_weight,
    const torch::Tensor values,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t cuda_threads
){
    CHECK_INPUT(attn_weight);
    CHECK_INPUT(values);

    return av_fw_cu(attn_weight, values, static_cast<int>(height), static_cast<int>(width), static_cast<int>(kernel_size), static_cast<int>(cuda_threads));
}


std::vector<torch::Tensor> av_bw_cu(
    const torch::Tensor d_output,
    const torch::Tensor attn_weight,
    const torch::Tensor values,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
);

std::vector<torch::Tensor> av_backward(
    const torch::Tensor d_output,
    const torch::Tensor attn_weight,
    const torch::Tensor values,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t cuda_threads
){
    CHECK_INPUT(d_output);
    CHECK_INPUT(attn_weight);
    CHECK_INPUT(values);

    return av_bw_cu(d_output, attn_weight, values, static_cast<int>(height), static_cast<int>(width), static_cast<int>(kernel_size), static_cast<int>(cuda_threads));
}


TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("qk_forward(Tensor queries, Tensor keys, int height, int width, int kernel_size, int cuda_threads) -> Tensor");
    ops.impl("qk_forward", torch::kCUDA, &qk_forward);

    ops.def("qk_backward(Tensor d_attn_weight, Tensor queries, Tensor keys, int height, int width, int kernel_size, int cuda_threads) -> Tensor[]");
    ops.impl("qk_backward", torch::kCUDA, &qk_backward);

    ops.def("qk_rpb_forward(Tensor queries, Tensor keys, Tensor rpb, int height, int width, int kernel_size, int cuda_threads) -> Tensor");
    ops.impl("qk_rpb_forward", torch::kCUDA, &qk_rpb_forward);

    ops.def("qk_rpb_backward(Tensor d_attn_weight, Tensor queries, Tensor keys, int height, int width, int kernel_size, int cuda_threads) -> Tensor[]");
    ops.impl("qk_rpb_backward", torch::kCUDA, &qk_rpb_backward);

    ops.def("av_forward(Tensor attn_weight, Tensor values, int height, int width, int kernel_size, int cuda_threads) -> Tensor");
    ops.impl("av_forward", torch::kCUDA, &av_forward);

    ops.def("av_backward(Tensor d_output, Tensor attn_weight, Tensor values, int height, int width, int kernel_size, int cuda_threads) -> Tensor[]");
    ops.impl("av_backward", torch::kCUDA, &av_backward);
}
