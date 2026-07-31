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

#include <ATen/core/Tensor.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cstdint>
#include <vector>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SAME_DEVICE(x, y) TORCH_CHECK(x.device() == y.device(), #x " and " #y " must be on the same device")
#define CHECK_SAME_DTYPE(x, y)                                                                                         \
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), #x " and " #y " must have the same dtype")

// CUDA dispatch guarantees at least one CUDA input; same-device checks make per-tensor CUDA checks redundant.

namespace {
void check_token_grid(const at::Tensor &tensor, int64_t height, int64_t width) {
    TORCH_CHECK(tensor.size(2) == height * width, "number of tokens must equal height * width");
}

void check_attention_shape(const at::Tensor &attn_weight, const at::Tensor &reference, int64_t kernel_size) {
    TORCH_CHECK(attn_weight.size(0) == reference.size(0) && attn_weight.size(1) == reference.size(1) &&
                    attn_weight.size(2) == reference.size(2) && attn_weight.size(3) == kernel_size * kernel_size,
                "attention weights must have shape [batch, heads, tokens, kernel_size^2]");
}
} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

at::Tensor qk_fw_cu(const at::Tensor &queries, const at::Tensor &keys, int height, int width, int kernel_size,
                    int cuda_threads);

at::Tensor qk_forward(const at::Tensor &queries, const at::Tensor &keys, int64_t height, int64_t width,
                      int64_t kernel_size, int64_t cuda_threads) {
    CHECK_CONTIGUOUS(queries);
    CHECK_CONTIGUOUS(keys);
    CHECK_SAME_DEVICE(queries, keys);
    CHECK_SAME_DTYPE(queries, keys);
    check_token_grid(queries, height, width);

    return qk_fw_cu(queries, keys, static_cast<int>(height), static_cast<int>(width), static_cast<int>(kernel_size),
                    static_cast<int>(cuda_threads));
}

std::vector<at::Tensor> qk_bw_cu(const at::Tensor &d_attn_weight, const at::Tensor &queries, const at::Tensor &keys,
                                 int height, int width, int kernel_size, int cuda_threads);

std::vector<at::Tensor> qk_backward(const at::Tensor &d_attn_weight, const at::Tensor &queries, const at::Tensor &keys,
                                    int64_t height, int64_t width, int64_t kernel_size, int64_t cuda_threads) {
    CHECK_CONTIGUOUS(d_attn_weight);
    CHECK_CONTIGUOUS(queries);
    CHECK_CONTIGUOUS(keys);
    CHECK_SAME_DEVICE(queries, d_attn_weight);
    CHECK_SAME_DEVICE(queries, keys);
    CHECK_SAME_DTYPE(queries, d_attn_weight);
    CHECK_SAME_DTYPE(queries, keys);
    check_token_grid(queries, height, width);
    check_attention_shape(d_attn_weight, queries, kernel_size);

    return qk_bw_cu(d_attn_weight, queries, keys, static_cast<int>(height), static_cast<int>(width),
                    static_cast<int>(kernel_size), static_cast<int>(cuda_threads));
}

std::vector<at::Tensor> qk_rpb_bw_cu(const at::Tensor &d_attn_weight, const at::Tensor &queries, const at::Tensor &keys,
                                     int height, int width, int kernel_size, int cuda_threads);

std::vector<at::Tensor> qk_rpb_backward(const at::Tensor &d_attn_weight, const at::Tensor &queries,
                                        const at::Tensor &keys, int64_t height, int64_t width, int64_t kernel_size,
                                        int64_t cuda_threads) {
    CHECK_CONTIGUOUS(d_attn_weight);
    CHECK_CONTIGUOUS(queries);
    CHECK_CONTIGUOUS(keys);
    CHECK_SAME_DEVICE(queries, d_attn_weight);
    CHECK_SAME_DEVICE(queries, keys);
    CHECK_SAME_DTYPE(queries, d_attn_weight);
    CHECK_SAME_DTYPE(queries, keys);
    check_token_grid(queries, height, width);
    check_attention_shape(d_attn_weight, queries, kernel_size);

    return qk_rpb_bw_cu(d_attn_weight, queries, keys, static_cast<int>(height), static_cast<int>(width),
                        static_cast<int>(kernel_size), static_cast<int>(cuda_threads));
}

at::Tensor qk_rpb_fw_cu(const at::Tensor &queries, const at::Tensor &keys, const at::Tensor &rpb, int height, int width,
                        int kernel_size, int cuda_threads);

at::Tensor qk_rpb_forward(const at::Tensor &queries, const at::Tensor &keys, const at::Tensor &rpb, int64_t height,
                          int64_t width, int64_t kernel_size, int64_t cuda_threads) {
    CHECK_CONTIGUOUS(queries);
    CHECK_CONTIGUOUS(keys);
    CHECK_CONTIGUOUS(rpb);
    CHECK_SAME_DEVICE(queries, keys);
    CHECK_SAME_DEVICE(queries, rpb);
    CHECK_SAME_DTYPE(queries, keys);
    CHECK_SAME_DTYPE(queries, rpb);
    check_token_grid(queries, height, width);

    return qk_rpb_fw_cu(queries, keys, rpb, static_cast<int>(height), static_cast<int>(width),
                        static_cast<int>(kernel_size), static_cast<int>(cuda_threads));
}

at::Tensor av_fw_cu(const at::Tensor &attn_weight, const at::Tensor &values, int height, int width, int kernel_size,
                    int cuda_threads);

at::Tensor av_forward(const at::Tensor &attn_weight, const at::Tensor &values, int64_t height, int64_t width,
                      int64_t kernel_size, int64_t cuda_threads) {
    CHECK_CONTIGUOUS(attn_weight);
    CHECK_CONTIGUOUS(values);
    CHECK_SAME_DEVICE(values, attn_weight);
    CHECK_SAME_DTYPE(values, attn_weight);
    check_token_grid(values, height, width);
    check_attention_shape(attn_weight, values, kernel_size);

    return av_fw_cu(attn_weight, values, static_cast<int>(height), static_cast<int>(width),
                    static_cast<int>(kernel_size), static_cast<int>(cuda_threads));
}

std::vector<at::Tensor> av_bw_cu(const at::Tensor &d_output, const at::Tensor &attn_weight, const at::Tensor &values,
                                 int height, int width, int kernel_size, int cuda_threads);

std::vector<at::Tensor> av_backward(const at::Tensor &d_output, const at::Tensor &attn_weight, const at::Tensor &values,
                                    int64_t height, int64_t width, int64_t kernel_size, int64_t cuda_threads) {
    CHECK_CONTIGUOUS(d_output);
    CHECK_CONTIGUOUS(attn_weight);
    CHECK_CONTIGUOUS(values);
    CHECK_SAME_DEVICE(values, d_output);
    CHECK_SAME_DEVICE(values, attn_weight);
    CHECK_SAME_DTYPE(values, d_output);
    CHECK_SAME_DTYPE(values, attn_weight);
    check_token_grid(values, height, width);
    check_attention_shape(attn_weight, values, kernel_size);
    TORCH_CHECK(d_output.sizes() == values.sizes(), "d_output and values must have the same shape");

    return av_bw_cu(d_output, attn_weight, values, static_cast<int>(height), static_cast<int>(width),
                    static_cast<int>(kernel_size), static_cast<int>(cuda_threads));
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def(
        "qk_forward(Tensor queries, Tensor keys, int height, int width, int kernel_size, int cuda_threads) -> Tensor");
    ops.impl("qk_forward", c10::kCUDA, &qk_forward);

    ops.def("qk_backward(Tensor d_attn_weight, Tensor queries, Tensor keys, int height, int width, int kernel_size, "
            "int cuda_threads) -> Tensor[]");
    ops.impl("qk_backward", c10::kCUDA, &qk_backward);

    ops.def("qk_rpb_forward(Tensor queries, Tensor keys, Tensor rpb, int height, int width, int kernel_size, int "
            "cuda_threads) -> Tensor");
    ops.impl("qk_rpb_forward", c10::kCUDA, &qk_rpb_forward);

    ops.def("qk_rpb_backward(Tensor d_attn_weight, Tensor queries, Tensor keys, int height, int width, int "
            "kernel_size, int cuda_threads) -> Tensor[]");
    ops.impl("qk_rpb_backward", c10::kCUDA, &qk_rpb_backward);

    ops.def("av_forward(Tensor attn_weight, Tensor values, int height, int width, int kernel_size, int cuda_threads) "
            "-> Tensor");
    ops.impl("av_forward", c10::kCUDA, &av_forward);

    ops.def("av_backward(Tensor d_output, Tensor attn_weight, Tensor values, int height, int width, int kernel_size, "
            "int cuda_threads) -> Tensor[]");
    ops.impl("av_backward", c10::kCUDA, &av_backward);
}
