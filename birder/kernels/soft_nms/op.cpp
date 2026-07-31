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

#include "soft_nms.h"

#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
#include <torch/library.h>

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

#ifndef TORCH_LIBRARY_IMPL_EXPAND
#define TORCH_LIBRARY_IMPL_EXPAND(NAME, KEY, MODULE) TORCH_LIBRARY_IMPL(NAME, KEY, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("soft_nms(Tensor boxes, Tensor scores, Tensor class_ids, float sigma, float score_threshold) -> (Tensor, "
            "Tensor)");
    ops.impl("soft_nms", c10::kCUDA, &soft_nms_cuda);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Autograd, ops) {
    ops.impl("soft_nms", torch::autograd::autogradNotImplementedFallback());
}
