"""Hungarian matching shared by DETR-family detectors."""

import math

import torch
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from birder.ops.linear_assignment import LinearAssignment

Assignment = tuple[torch.Tensor, torch.Tensor]


class HungarianMatcher:
    def __init__(
        self,
        class_weight: float,
        bbox_weight: float,
        giou_weight: float,
        use_focal_cost: bool = True,
        use_giou: bool = True,
        clamp_predicted_box_sizes: bool = False,
    ) -> None:
        weights = (class_weight, bbox_weight, giou_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError(f"Matching weights must be finite and non-negative, got {weights}")
        if all(weight == 0 for weight in weights):
            raise ValueError("At least one matching weight must be positive")

        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.use_focal_cost = use_focal_cost
        self.use_giou = use_giou
        self.clamp_predicted_box_sizes = clamp_predicted_box_sizes
        self._linear_assignment = LinearAssignment()

    def match(
        self, class_logits: torch.Tensor, box_predictions: torch.Tensor, targets: list[dict[str, torch.Tensor]]
    ) -> list[Assignment]:
        """
        Match one prediction set per image
        """

        if class_logits.ndim != 3:
            raise ValueError(f"Expected class logits with shape [B, Q, C], got {tuple(class_logits.size())}")

        return self.match_grouped(  # type: ignore[no-any-return]
            class_logits.unsqueeze(1), box_predictions.unsqueeze(1), targets
        )[0]

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def match_grouped(
        self,
        class_logits: torch.Tensor,
        box_predictions: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        group_chunk_size: int | None = None,
    ) -> list[list[Assignment]]:
        """
        Match independent prediction groups against the same image targets

        Inputs have shapes [B, G, Q, C] and [B, G, Q, 4]. Results are
        group-major, then image-major, and contain group-local query indices.
        'group_chunk_size' bounds the number of groups whose cost tensors are materialized together.
        """

        if class_logits.ndim != 4:
            raise ValueError(f"Expected class logits with shape [B, G, Q, C], got {tuple(class_logits.size())}")

        batch_size, num_groups, num_queries, num_classes = class_logits.size()
        if group_chunk_size is None:
            group_chunk_size = num_groups

        size_buckets: dict[int, list[int]] = {}
        for batch_index, target in enumerate(targets):
            target_count = target["boxes"].size(0)
            if target_count > 0:
                size_buckets.setdefault(target_count, []).append(batch_index)

        grouped_assignments: list[list[Assignment]] = []
        for group_start in range(0, num_groups, group_chunk_size):
            group_end = min(group_start + group_chunk_size, num_groups)
            chunk_logits = class_logits[:, group_start:group_end]
            chunk_boxes = box_predictions[:, group_start:group_end]
            chunk_groups = group_end - group_start
            empty_indices = torch.empty((chunk_groups, 0), dtype=torch.int64, device=class_logits.device)
            image_assignments: list[Assignment] = [(empty_indices, empty_indices) for _ in range(batch_size)]

            for target_count, batch_indices in size_buckets.items():
                if len(batch_indices) == 1:
                    batch_index = batch_indices[0]
                    bucket_logits = chunk_logits[batch_index : batch_index + 1]
                    bucket_boxes = chunk_boxes[batch_index : batch_index + 1]
                    target_labels = targets[batch_index]["labels"].unsqueeze(0)
                    target_boxes = targets[batch_index]["boxes"].unsqueeze(0)
                else:
                    bucket_logits = torch.stack([chunk_logits[index] for index in batch_indices], dim=0)
                    bucket_boxes = torch.stack([chunk_boxes[index] for index in batch_indices], dim=0)
                    target_labels = torch.stack([targets[index]["labels"] for index in batch_indices], dim=0)
                    target_boxes = torch.stack([targets[index]["boxes"] for index in batch_indices], dim=0)

                target_index = target_labels[:, None, None, :].expand(-1, chunk_groups, num_queries, target_count)
                if self.use_focal_cost is True:
                    gather_before_focal = target_count <= num_classes
                    if gather_before_focal is True:
                        focal_logits = torch.gather(bucket_logits, dim=-1, index=target_index)
                    else:
                        focal_logits = bucket_logits

                    probabilities = focal_logits.sigmoid()
                    alpha = 0.25
                    gamma = 2.0
                    negative_class_cost = (
                        (1 - alpha)
                        * (probabilities**gamma)
                        * (-F.logsigmoid(-focal_logits))  # pylint: disable=not-callable
                    )
                    class_cost = (
                        alpha
                        * ((1 - probabilities) ** gamma)
                        * (-F.logsigmoid(focal_logits))  # pylint: disable=not-callable
                    )
                    class_cost.sub_(negative_class_cost)
                    if gather_before_focal is False:
                        class_cost = torch.gather(class_cost, dim=-1, index=target_index)

                    # These tensors can be much larger than the final cost
                    # matrix when the classifier has many classes.
                    del focal_logits, probabilities, negative_class_cost

                else:
                    class_cost = -torch.gather(bucket_logits.softmax(dim=-1), dim=-1, index=target_index)

                expanded_target_boxes = target_boxes[:, None, :, :]
                cost = torch.cdist(bucket_boxes, expanded_target_boxes, p=1.0)
                cost.mul_(self.bbox_weight)
                cost.add_(class_cost, alpha=self.class_weight)

                overlap_boxes = bucket_boxes
                if self.clamp_predicted_box_sizes is True:
                    overlap_boxes = torch.concat((bucket_boxes[..., :2], bucket_boxes[..., 2:].clamp(min=0)), dim=-1)

                predicted_xyxy = box_ops.box_convert(overlap_boxes, in_fmt="cxcywh", out_fmt="xyxy")
                target_xyxy = box_ops.box_convert(expanded_target_boxes, in_fmt="cxcywh", out_fmt="xyxy")
                if self.use_giou is True:
                    overlap_cost = -box_ops.generalized_box_iou(predicted_xyxy, target_xyxy)
                else:
                    overlap_cost = -box_ops.box_iou(predicted_xyxy, target_xyxy)

                cost.add_(overlap_cost, alpha=self.giou_weight)
                cost = self._replace_non_finite_costs(cost)

                col4row, row4col = self._linear_assignment(cost.flatten(0, 1))
                bucket_size = len(batch_indices)
                if num_queries >= target_count:
                    row4col = row4col.view(bucket_size, chunk_groups, target_count)
                    prediction_indices, matched_target_indices = torch.sort(row4col, dim=-1)
                else:
                    col4row = col4row.view(bucket_size, chunk_groups, num_queries)
                    prediction_indices = torch.arange(
                        num_queries, dtype=torch.int64, device=class_logits.device
                    ).expand(bucket_size, chunk_groups, -1)
                    matched_target_indices = col4row

                for bucket_index, batch_index in enumerate(batch_indices):
                    image_assignments[batch_index] = (
                        prediction_indices[bucket_index],
                        matched_target_indices[bucket_index],
                    )

            for group_index in range(chunk_groups):
                grouped_assignments.append(
                    [
                        (image_prediction_indices[group_index], image_target_indices[group_index])
                        for image_prediction_indices, image_target_indices in image_assignments
                    ]
                )

        return grouped_assignments

    @staticmethod
    def _replace_non_finite_costs(cost: torch.Tensor) -> torch.Tensor:
        """
        Replace invalid edges with a penalty dominating every finite assignment
        """

        finite = torch.isfinite(cost)
        matrix_dims = (-2, -1)
        has_finite = finite.any(dim=matrix_dims, keepdim=True)
        finite_min = cost.masked_fill(~finite, torch.inf).amin(dim=matrix_dims, keepdim=True)
        finite_max = cost.masked_fill(~finite, -torch.inf).amax(dim=matrix_dims, keepdim=True)

        assignment_size = min(cost.shape[-2:])
        penalty_bound = assignment_size * finite_max - (assignment_size - 1) * finite_min
        penalty = torch.nextafter(penalty_bound, torch.full_like(penalty_bound, torch.inf))
        max_value = torch.full_like(penalty, torch.finfo(cost.dtype).max)
        penalty = torch.where(torch.isfinite(penalty), penalty, max_value)
        penalty = torch.where(has_finite, penalty, torch.ones_like(penalty))
        return torch.where(finite, cost, penalty)
