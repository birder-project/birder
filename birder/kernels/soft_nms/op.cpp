/*!
**************************************************************************************************
* Soft-NMS
* Taken from:
* https://github.com/MrParosk/soft_nms
* Licensed under the MIT License
**************************************************************************************************
*/

#include <torch/extension.h>
#include "soft_nms.h"

// Wrapper function to match Python expectations
std::vector<torch::Tensor> soft_nms_wrapper(
    const torch::Tensor& boxes,
    const torch::Tensor& scores,
    double sigma,
    double score_threshold) {

    auto [updated_scores, keep] = soft_nms(boxes, scores, sigma, score_threshold);
    return {updated_scores, keep};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("soft_nms", &soft_nms_wrapper, "soft_nms",
          py::arg("boxes"), py::arg("scores"), py::arg("sigma"), py::arg("score_threshold"));
}
