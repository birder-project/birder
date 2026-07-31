"""
Weighted Boxes Fusion, adapted from
https://github.com/ZFTurbo/Weighted-Boxes-Fusion

Paper "Weighted boxes fusion: Ensembling boxes from different object detection models",
https://arxiv.org/abs/1910.13302
"""

# Reference license: MIT

from typing import Literal
from typing import Optional

import torch

ConfType = Literal["avg", "max", "box_and_model_avg", "absent_model_aware_avg", "cluster_avg", "cluster_max"]


def _box_iou_single(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (box_area + boxes_area - inter)


def _fuse_label_boxes(
    label_boxes: torch.Tensor,
    label_scores: torch.Tensor,
    label_weights: torch.Tensor,
    label_source_ids: torch.Tensor,
    source_weights: torch.Tensor,
    iou_thr: float,
    conf_type: ConfType,
    allows_overflow: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(label_scores * label_weights, descending=True)
    num_boxes = label_boxes.size(0)
    cluster_boxes = torch.empty((num_boxes, 4), dtype=label_boxes.dtype, device=label_boxes.device)
    score_weight_sums = torch.empty((num_boxes,), dtype=label_scores.dtype, device=label_scores.device)
    weight_sums = torch.empty((num_boxes,), dtype=label_weights.dtype, device=label_weights.device)
    max_scores = torch.empty((num_boxes,), dtype=label_scores.dtype, device=label_scores.device)
    max_weighted_scores = torch.empty((num_boxes,), dtype=label_scores.dtype, device=label_scores.device)
    boxes_counts = torch.empty((num_boxes,), dtype=label_scores.dtype, device=label_scores.device)
    track_sources = conf_type in ("box_and_model_avg", "absent_model_aware_avg")
    source_presence: Optional[torch.Tensor] = None
    if track_sources is True:
        source_presence = torch.zeros((num_boxes, source_weights.numel()), dtype=torch.bool, device=label_boxes.device)

    cluster_count = 0

    for idx in order:
        box = label_boxes[idx]
        score = label_scores[idx]
        weight = label_weights[idx]
        score_weight = score * weight
        source_id = label_source_ids[idx]

        if cluster_count == 0:
            cluster_boxes[0] = box
            score_weight_sums[0] = score_weight
            weight_sums[0] = weight
            max_scores[0] = score
            max_weighted_scores[0] = score_weight
            boxes_counts[0] = 1
            if source_presence is not None:
                source_presence[0, source_id] = True

            cluster_count = 1
            continue

        ious = _box_iou_single(box, cluster_boxes[:cluster_count])
        max_iou, cluster_idx = torch.max(ious, dim=0)
        if max_iou > iou_thr:
            total_score_weight = score_weight_sums[cluster_idx] + score_weight
            cluster_boxes[cluster_idx] = (
                cluster_boxes[cluster_idx] * score_weight_sums[cluster_idx] + box * score_weight
            ) / total_score_weight
            score_weight_sums[cluster_idx] = total_score_weight
            weight_sums[cluster_idx] += weight
            max_scores[cluster_idx] = torch.maximum(max_scores[cluster_idx], score)
            max_weighted_scores[cluster_idx] = torch.maximum(max_weighted_scores[cluster_idx], score_weight)
            boxes_counts[cluster_idx] += 1
            if source_presence is not None:
                source_presence[cluster_idx, source_id] = True

        else:
            cluster_boxes[cluster_count] = box
            score_weight_sums[cluster_count] = score_weight
            weight_sums[cluster_count] = weight
            max_scores[cluster_count] = score
            max_weighted_scores[cluster_count] = score_weight
            boxes_counts[cluster_count] = 1
            if source_presence is not None:
                source_presence[cluster_count, source_id] = True

            cluster_count += 1

    active_score_weight_sums = score_weight_sums[:cluster_count]
    active_weight_sums = weight_sums[:cluster_count]
    active_boxes_counts = boxes_counts[:cluster_count]
    total_source_weight = source_weights.sum()
    num_sources = source_weights.numel()
    unique_source_weight_sums: Optional[torch.Tensor] = None
    absent_source_weight_sums: Optional[torch.Tensor] = None
    if source_presence is not None:
        active_source_presence = source_presence[:cluster_count]
        source_weights_row = source_weights.unsqueeze(0)
        unique_source_weight_sums = (active_source_presence.to(source_weights.dtype) * source_weights_row).sum(dim=1)
        absent_source_weight_sums = ((~active_source_presence).to(source_weights.dtype) * source_weights_row).sum(dim=1)

    if conf_type == "avg":
        scores = active_score_weight_sums / active_boxes_counts
        if allows_overflow is True:
            scores = scores * active_boxes_counts / total_source_weight
        else:
            scores = scores * torch.clamp(active_boxes_counts, max=num_sources) / total_source_weight

    elif conf_type == "max":
        scores = max_weighted_scores[:cluster_count] / source_weights.max()

    elif conf_type == "box_and_model_avg":
        assert unique_source_weight_sums is not None
        scores = (active_score_weight_sums / active_weight_sums) * (unique_source_weight_sums / total_source_weight)

    elif conf_type == "absent_model_aware_avg":
        assert absent_source_weight_sums is not None
        scores = active_score_weight_sums / (active_weight_sums + absent_source_weight_sums)

    elif conf_type == "cluster_avg":
        scores = active_score_weight_sums / active_weight_sums

    elif conf_type == "cluster_max":
        scores = max_scores[:cluster_count]

    else:
        raise ValueError(f"Unsupported conf_type: {conf_type}")

    if allows_overflow is False and conf_type in ("avg", "cluster_avg", "cluster_max"):
        scores = scores.clamp(max=1.0)

    return (cluster_boxes[:cluster_count], scores)


def weighted_boxes_fusion(
    boxes_list: list[torch.Tensor],
    scores_list: list[torch.Tensor],
    labels_list: list[torch.Tensor],
    weights: Optional[list[float]] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    conf_type: ConfType = "avg",
    allows_overflow: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weights is None:
        weights = [1.0] * len(boxes_list)
    if len(weights) != len(boxes_list):
        raise ValueError("weights must match number of box sets")

    if len(boxes_list) > 0:
        device = boxes_list[0].device
    else:
        device = torch.device("cpu")

    boxes_all: list[torch.Tensor] = []
    scores_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    weights_all: list[torch.Tensor] = []
    source_ids_all: list[torch.Tensor] = []
    for source_idx, (boxes, scores, labels, weight) in enumerate(zip(boxes_list, scores_list, labels_list, weights)):
        if boxes.numel() == 0 or weight == 0:
            continue

        boxes_tensor = boxes.detach().to(dtype=torch.float32)
        scores_tensor = scores.detach().to(dtype=torch.float32)
        labels_tensor = labels.detach().to(dtype=torch.int64)

        keep = torch.logical_and(scores_tensor >= skip_box_thr, scores_tensor > 0)
        if not keep.any():
            continue

        boxes_tensor = boxes_tensor[keep]
        scores_tensor = scores_tensor[keep]
        labels_tensor = labels_tensor[keep]
        weights_tensor = scores_tensor.new_full(scores_tensor.shape, weight)

        boxes_all.append(boxes_tensor)
        scores_all.append(scores_tensor)
        labels_all.append(labels_tensor)
        weights_all.append(weights_tensor)
        source_ids_all.append(labels_tensor.new_full(labels_tensor.shape, source_idx))

    if len(boxes_all) == 0:
        empty_boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
        empty_scores = torch.zeros((0,), dtype=torch.float32, device=device)
        empty_labels = torch.zeros((0,), dtype=torch.int64, device=device)
        return (empty_boxes, empty_scores, empty_labels)

    boxes_tensor = torch.concat(boxes_all, dim=0)
    scores_tensor = torch.concat(scores_all, dim=0)
    labels_tensor = torch.concat(labels_all, dim=0)
    weights_tensor = torch.concat(weights_all, dim=0)
    source_ids_tensor = torch.concat(source_ids_all, dim=0)
    source_weights_tensor = weights_tensor.new_tensor(weights)
    label_order = torch.argsort(labels_tensor)
    boxes_tensor = boxes_tensor[label_order]
    scores_tensor = scores_tensor[label_order]
    labels_tensor = labels_tensor[label_order]
    weights_tensor = weights_tensor[label_order]
    source_ids_tensor = source_ids_tensor[label_order]
    labels_unique, label_counts = torch.unique_consecutive(labels_tensor, return_counts=True)

    fused_boxes: list[torch.Tensor] = []
    fused_scores: list[torch.Tensor] = []
    fused_labels: list[torch.Tensor] = []

    start = 0
    for label, count in zip(labels_unique, label_counts):
        end = start + int(count)
        label_boxes = boxes_tensor[start:end]
        label_scores = scores_tensor[start:end]
        label_weights = weights_tensor[start:end]
        label_source_ids = source_ids_tensor[start:end]
        boxes, scores = _fuse_label_boxes(
            label_boxes,
            label_scores,
            label_weights,
            label_source_ids,
            source_weights_tensor,
            iou_thr,
            conf_type,
            allows_overflow,
        )
        fused_boxes.append(boxes)
        fused_scores.append(scores)
        fused_labels.append(label.expand(scores.shape[0]))
        start = end

    fused_boxes_tensor = torch.concat(fused_boxes, dim=0)
    fused_scores_tensor = torch.concat(fused_scores, dim=0)
    fused_labels_tensor = torch.concat(fused_labels, dim=0)
    order = torch.argsort(fused_scores_tensor, descending=True)
    fused_boxes_tensor = fused_boxes_tensor[order]
    fused_scores_tensor = fused_scores_tensor[order]
    fused_labels_tensor = fused_labels_tensor[order]

    return (fused_boxes_tensor, fused_scores_tensor, fused_labels_tensor)


def fuse_detections_wbf_single(
    detections: list[dict[str, torch.Tensor]],
    weights: Optional[list[float]] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    conf_type: ConfType = "avg",
    allows_overflow: bool = False,
) -> dict[str, torch.Tensor]:
    if len(detections) == 0:
        return {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    boxes_list = [detection["boxes"] for detection in detections]
    scores_list = [detection["scores"] for detection in detections]
    labels_list = [detection["labels"] for detection in detections]

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        conf_type=conf_type,
        allows_overflow=allows_overflow,
    )

    return {"boxes": boxes, "scores": scores, "labels": labels}


def fuse_detections_wbf(
    detections_list: list[list[dict[str, torch.Tensor]]],
    weights: Optional[list[float]] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    conf_type: ConfType = "avg",
    allows_overflow: bool = False,
) -> list[dict[str, torch.Tensor]]:
    if len(detections_list) == 0:
        return []

    # Outer list is the augmentations, inner is the batch
    batch_size = len(detections_list[0])
    fused: list[dict[str, torch.Tensor]] = []
    for idx in range(batch_size):
        per_image = [detections[idx] for detections in detections_list]
        fused.append(
            fuse_detections_wbf_single(
                per_image,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
                conf_type=conf_type,
                allows_overflow=allows_overflow,
            )
        )

    return fused
