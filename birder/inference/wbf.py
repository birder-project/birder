"""
Weighted Boxes Fusion, adapted from
https://github.com/ZFTurbo/Weighted-Boxes-Fusion

Paper "Weighted boxes fusion: Ensembling boxes from different object detection models",
https://arxiv.org/abs/1910.13302
"""

# Reference license: MIT

from dataclasses import dataclass
from typing import Literal
from typing import Optional

import torch
from torchvision.ops import box_iou

ConfType = Literal["avg", "max", "box_and_model_avg", "absent_model_aware_avg"]


@dataclass
class BoxCluster:
    box: torch.Tensor
    score_weight_sum: torch.Tensor
    weight_sum: torch.Tensor
    max_score: torch.Tensor
    boxes_count: int

    @classmethod
    def from_entry(cls, box: torch.Tensor, score: torch.Tensor, weight: torch.Tensor) -> "BoxCluster":
        score_weight = score * weight
        return cls(
            box=box.clone(),
            score_weight_sum=score_weight,
            weight_sum=weight,
            max_score=score,
            boxes_count=1,
        )

    def add(self, box: torch.Tensor, score: torch.Tensor, weight: torch.Tensor) -> None:
        score_weight = score * weight
        total_weight = self.score_weight_sum + score_weight
        self.box = (self.box * self.score_weight_sum + box * score_weight) / total_weight
        self.score_weight_sum = total_weight
        self.weight_sum += weight
        self.max_score = torch.maximum(self.max_score, score)
        self.boxes_count += 1


# pylint: disable=too-many-locals,too-many-branches
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
    for boxes, scores, labels, weight in zip(boxes_list, scores_list, labels_list, weights):
        if boxes.numel() == 0:
            continue

        boxes_tensor = boxes.detach().to(dtype=torch.float32)
        scores_tensor = scores.detach().to(dtype=torch.float32)
        labels_tensor = labels.detach().to(dtype=torch.int64)

        keep = scores_tensor >= skip_box_thr
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

    if len(boxes_all) == 0:
        empty_boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
        empty_scores = torch.zeros((0,), dtype=torch.float32, device=device)
        empty_labels = torch.zeros((0,), dtype=torch.int64, device=device)
        return (empty_boxes, empty_scores, empty_labels)

    boxes_tensor = torch.concat(boxes_all, dim=0)
    scores_tensor = torch.concat(scores_all, dim=0)
    labels_tensor = torch.concat(labels_all, dim=0)
    weights_tensor = torch.concat(weights_all, dim=0)
    labels_unique = torch.unique(labels_tensor)

    total_weight = float(sum(weights))
    num_models = len(weights)
    fused_boxes: list[torch.Tensor] = []
    fused_scores: list[torch.Tensor] = []
    fused_labels: list[torch.Tensor] = []

    for label in labels_unique:
        label_mask = labels_tensor == label
        label_boxes = boxes_tensor[label_mask]
        label_scores = scores_tensor[label_mask]
        label_weights = weights_tensor[label_mask]
        order = torch.argsort(label_scores, descending=True)
        clusters: list[BoxCluster] = []
        for idx in order:
            box = label_boxes[idx]
            score = label_scores[idx]
            weight = label_weights[idx]
            if len(clusters) == 0:
                clusters.append(BoxCluster.from_entry(box, score, weight))
                continue

            cluster_boxes = torch.stack([cluster.box for cluster in clusters], dim=0)
            ious = box_iou(box.unsqueeze(0), cluster_boxes).squeeze(0)
            max_iou, max_idx = torch.max(ious, dim=0)
            if max_iou > iou_thr:
                clusters[int(max_idx)].add(box, score, weight)
            else:
                clusters.append(BoxCluster.from_entry(box, score, weight))

        for cluster in clusters:
            if conf_type == "avg":
                score = cluster.score_weight_sum / cluster.weight_sum
            elif conf_type == "max":
                score = cluster.max_score
            elif conf_type == "box_and_model_avg":
                score = (cluster.score_weight_sum / cluster.weight_sum) * (cluster.boxes_count / num_models)
            elif conf_type == "absent_model_aware_avg":
                score = cluster.score_weight_sum / total_weight
            else:
                raise ValueError(f"Unsupported conf_type: {conf_type}")

            if allows_overflow is False:
                score = score.clamp(max=1.0)

            fused_boxes.append(cluster.box)
            fused_scores.append(score)
            fused_labels.append(label)

    fused_scores_tensor = torch.stack(fused_scores)
    order = torch.argsort(fused_scores_tensor, descending=True)
    fused_boxes_tensor = torch.stack(fused_boxes, dim=0)[order]
    fused_scores_tensor = fused_scores_tensor[order]
    fused_labels_tensor = torch.stack(fused_labels)[order]

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

    (boxes, scores, labels) = weighted_boxes_fusion(
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
