/*!
**************************************************************************************************
* Soft-NMS
* Taken from:
* https://github.com/MrParosk/soft_nms
* Licensed under the MIT License
**************************************************************************************************
*/

#include "soft_nms.h"
#include <torch/torch.h>

using namespace torch::indexing;

torch::Tensor calculate_area(const torch::Tensor& boxes) {
    /*
    Computes the area of the boxes.

    Boxes are expected to have the shape [N, 4] and of the format [x_min, y_min, x_max, y_max].
    */

    auto x1 = boxes.index({Slice(), 0});
    auto y1 = boxes.index({Slice(), 1});
    auto x2 = boxes.index({Slice(), 2});
    auto y2 = boxes.index({Slice(), 3});
    auto areas = (x2 - x1) * (y2 - y1);
    return areas;
}

torch::Tensor calculate_iou(const torch::Tensor& boxes, const torch::Tensor& areas, const int& idx) {
    /*
    Computes the IOU between the box at index idx and all boxes "below" it (i.e. idx+1 until the end of the tensor).

    Boxes are expected to have the shape [N, 4] and of the format [x_min, y_min, x_max, y_max].
    Area are expected to have the shape [N].
    */

    auto xx1 = torch::maximum(boxes.index({idx, 0}), boxes.index({Slice(idx + 1, None), 0}));
    auto yy1 = torch::maximum(boxes.index({idx, 1}), boxes.index({Slice(idx + 1, None), 1}));
    auto xx2 = torch::minimum(boxes.index({idx, 2}), boxes.index({Slice(idx + 1, None), 2}));
    auto yy2 = torch::minimum(boxes.index({idx, 3}), boxes.index({Slice(idx + 1, None), 3}));

    auto w = torch::maximum(torch::zeros_like(xx1), xx2 - xx1);
    auto h = torch::maximum(torch::zeros_like(yy1), yy2 - yy1);

    auto intersection = w * h;
    auto union_ = areas.index({idx}) + areas.index({Slice(idx + 1, None)}) - intersection;
    auto iou = torch::div(intersection, union_);
    return iou;
}

void update_sorting_order(torch::Tensor& boxes, torch::Tensor& scores, torch::Tensor& areas, const int& idx) {
    /*
    Since the scores get updated with soft-nms we need to "re-sort" them and their corresponding boxes.

    Boxes are expected to have the shape [N, 4] and of the format [x_min, y_min, x_max, y_max].
    Scores are expected to have the shape [N].
    */

    torch::Tensor max_score, t_max_idx;
    std::tie(max_score, t_max_idx) = torch::max(scores.index({Slice(idx + 1, None)}), 0);

    // max_idx is computed from sliced data, therefore need to convert it to "global" max idx
    auto max_idx = t_max_idx.item<int>() + idx + 1;

    if (scores.index({idx}).item<float>() < max_score.item<float>()) {
        auto boxes_idx = boxes.index({idx}).clone();
        auto boxes_max = boxes.index({max_idx}).clone();
        boxes.index({idx}) = boxes_max;
        boxes.index({max_idx}) = boxes_idx;

        auto scores_idx = scores.index({idx}).clone();
        auto scores_max = scores.index({max_idx}).clone();
        scores.index({idx}) = scores_max;
        scores.index({max_idx}) = scores_idx;

        auto areas_idx = areas.index({idx}).clone();
        auto areas_max = areas.index({max_idx}).clone();
        areas.index({idx}) = areas_max;
        areas.index({max_idx}) = areas_idx;
    }
}

std::tuple<torch::Tensor, torch::Tensor> soft_nms(
    const torch::Tensor& boxes,
    const torch::Tensor& scores,
    const double sigma,
    const double score_threshold) {
    /*
    Performs soft-nms on the boxes (with the Gaussian function).

    Args:
        boxes (Tensor[N, 4]): Boxes to perform NMS on. They are expected to be in (x_min, y_min, x_max, y_max) format.
        scores (Tensor[N]): Scores for each one of the boxes.
        sigma (double): The sigma parameter described in the paper which controls how much the score is decreased on overlap.
        score_threshold (double): Will filter out all updated-scores which has value than score_threshold.

    Returns:
        updated_scores (Tensor): float tensor with the updated scores, i.e.
            the scores after they have been decreased according to the overlap, sorted in decreasing order of scores.
        keep (Tensor): int64 tensor with the indices of the elements that have been kept by soft-nms,
            sorted in decreasing order of scores.
    */

    int num_element = boxes.size(0);
    auto indicies = torch::arange(0, num_element, torch::dtype(torch::kI32)).view({num_element, 1});
    indicies = indicies.to(boxes.device());
    auto boxes_indicies = torch::cat({boxes, indicies}, 1);

    auto scores_updated = scores.clone();
    auto areas = calculate_area(boxes_indicies);

    for (int i = 0; i < num_element - 1; i++) {
        update_sorting_order(boxes_indicies, scores_updated, areas, i);
        auto iou = calculate_iou(boxes_indicies, areas, i);
        auto weight = torch::exp(-(iou * iou) / sigma);
        scores_updated.index({Slice(i + 1, None)}) = weight * scores_updated.index({Slice(i + 1, None)});
    }

    auto keep = boxes_indicies.index({scores_updated > score_threshold, 4});
    keep = keep.to(torch::kLong);

    scores_updated = scores_updated.index({scores_updated > score_threshold});
    return std::make_tuple(scores_updated, keep);
}
