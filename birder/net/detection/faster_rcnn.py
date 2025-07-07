"""
Faster R-CNN, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py

Paper "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
https://arxiv.org/abs/1506.01497
"""

# Reference license: BSD 3-Clause

from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from birder.net.base import DetectorBackbone
from birder.net.detection.base import AnchorGenerator
from birder.net.detection.base import BackboneWithFPN
from birder.net.detection.base import BoxCoder
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.base import ImageList
from birder.net.detection.base import Matcher


class BalancedPositiveNegativeSampler:
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image: int, positive_fraction: float) -> None:
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Parameters
        ----------
        matched_idxs
            list of tensors containing -1, 0 or positive values.
            Each tensor corresponds to a specific image.
            -1 values are ignored, 0 are considered as negatives and > 0 as positives

        Returns
        -------
        returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example
        """

        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(matched_idxs_per_image >= 1)[0]
            negative = torch.where(matched_idxs_per_image == 0)[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)

            # Protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos

            # Protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # Randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # Create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return (pos_idx, neg_idx)


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, in_channels: int, num_anchors: int, conv_depth: int = 1) -> None:
        super().__init__()
        conv_list = []
        for _ in range(conv_depth):
            conv_list.append(
                Conv2dNormActivation(
                    in_channels,
                    in_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    norm_layer=None,
                    bias=True,
                )
            )

        self.conv = nn.Sequential(*conv_list)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Weights initialization
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer: torch.Tensor, N: int, C: int, H: int, W: int) -> torch.Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)

    return layer


def concat_box_prediction_layers(
    box_cls: list[torch.Tensor], box_regression: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    box_cls_flattened = []
    box_regression_flattened = []

    # For each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        (N, AxC, H, W) = box_cls_per_level.shape  # pylint: disable=invalid-name
        Ax4 = box_regression_per_level.shape[1]  # pylint: disable=invalid-name
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    # Concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.concat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.concat(box_regression_flattened, dim=1).reshape(-1, 4)

    return (box_cls, box_regression)


class RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: nn.Module,
        # Faster-RCNN Training
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_image: int,
        positive_fraction: float,
        # Faster-RCNN Inference
        pre_nms_top_n: dict[str, int],
        post_nms_top_n: dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # Used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        # Used during inference
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self) -> int:
        if self.training is True:
            return self._pre_nms_top_n["training"]

        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        if self.training is True:
            return self._post_nms_top_n["training"]

        return self._post_nms_top_n["testing"]

    def assign_targets_to_anchors(
        self, anchors: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)

            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)

                # Get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # Discard indices that are between thresholds
                idxs_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[idxs_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)

        return (labels, matched_gt_boxes)

    def _get_top_n_idx(self, objectness: torch.Tensor, num_anchors_per_level: list[int]) -> torch.Tensor:
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), int(ob.size(1)))
            (_, top_n_idx) = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors

        return torch.concat(r, dim=1)

    def filter_proposals(
        self,
        proposals: torch.Tensor,
        objectness: torch.Tensor,
        image_shapes: list[tuple[int, int]],
        num_anchors_per_level: list[int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        num_images = proposals.shape[0]
        device = proposals.device

        # Do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        level_list: list[torch.Tensor] = [
            torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.concat(level_list, dim=0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # Select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # Remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            (boxes, scores, lvl) = boxes[keep], scores[keep], lvl[keep]

            # Remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            (boxes, scores, lvl) = boxes[keep], scores[keep], lvl[keep]

            # Non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # Keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            (boxes, scores) = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)

        return (final_boxes, final_scores)

    def compute_loss(
        self,
        objectness: torch.Tensor,
        pred_bbox_deltas: torch.Tensor,
        labels: list[torch.Tensor],
        regression_targets: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (sampled_pos_idxs, sampled_neg_idxs) = self.fg_bg_sampler(labels)
        sampled_pos_idxs = torch.where(torch.concat(sampled_pos_idxs, dim=0))[0]
        sampled_neg_idxs = torch.where(torch.concat(sampled_neg_idxs, dim=0))[0]

        sampled_idxs = torch.concat([sampled_pos_idxs, sampled_neg_idxs], dim=0)

        objectness = objectness.flatten()
        labels = torch.concat(labels, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_idxs],
            torch.concat(regression_targets, dim=0)[sampled_pos_idxs],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_idxs.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_idxs], labels[sampled_idxs])

        return (objectness_loss, box_loss)

    def forward(
        self,
        images: ImageList,
        features: dict[str, torch.Tensor],
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        # RPN uses all feature maps that are available
        features_list = list(features.values())
        (objectness, pred_bbox_deltas) = self.head(features_list)
        anchors = self.anchor_generator(images, features_list)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        (objectness, pred_bbox_deltas) = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        # Apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        (boxes, _scores) = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses: dict[str, torch.Tensor] = {}
        if self.training is True:
            if targets is None:
                raise ValueError("targets should not be None")

            (labels, matched_gt_boxes) = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            (loss_objectness, loss_rpn_box_reg) = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }

        return (boxes, losses)


class FastRCNNConvFCHead(nn.Sequential):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        conv_layers: list[int],
        fc_layers: list[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        (in_channels, in_height, in_width) = input_size

        blocks = []
        previous_channels = in_channels
        for current_channels in conv_layers:
            blocks.append(
                Conv2dNormActivation(
                    previous_channels,
                    current_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    norm_layer=norm_layer,
                    bias=False,
                )
            )
            previous_channels = current_channels

        blocks.append(nn.Flatten())
        previous_channels = previous_channels * in_height * in_width
        for current_channels in fc_layers:
            blocks.append(nn.Linear(previous_channels, current_channels))
            blocks.append(nn.ReLU(inplace=True))
            previous_channels = current_channels

        super().__init__(*blocks)

        # Weights initialization
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers for Fast R-CNN
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            torch._assert(  # pylint: disable=protected-access
                list(x.shape[2:]) == [1, 1],
                "x has the wrong shape, expecting the last two dimensions "
                f"to be [1,1] instead of {list(x.shape[2:])}",
            )

        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return (scores, bbox_deltas)


def faster_rcnn_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    label_list: list[torch.Tensor],
    regression_targets: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.concat(label_list, dim=0)
    regression_targets = torch.concat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # Get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_idxs_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_idxs_subset]
    (N, _num_classes) = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_idxs_subset, labels_pos],
        regression_targets[sampled_pos_idxs_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return (classification_loss, box_loss)


class RoIHeads(nn.Module):
    def __init__(
        self,
        box_roi_pool: MultiScaleRoIAlign,
        box_head: nn.Module,
        box_predictor: FastRCNNPredictor,
        # Faster R-CNN training
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_image: int,
        positive_fraction: float,
        bbox_reg_weights: tuple[float, float, float, float],
        # Faster R-CNN inference
        score_thresh: float,
        nms_thresh: float,
        detections_per_img: int,
    ) -> None:
        super().__init__()

        self.box_similarity = box_ops.box_iou

        # Assign ground-truth boxes for each proposal
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self.box_coder = BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def assign_targets_to_proposals(
        self, proposals: list[torch.Tensor], gt_boxes: list[torch.Tensor], gt_labels: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)

            else:
                # Set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_idxs = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_idxs] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_idxs = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_idxs] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)

        return (matched_idxs, labels)

    def subsample(self, labels: list[torch.Tensor]) -> list[torch.Tensor]:
        (sampled_pos_idxs, sampled_neg_idxs) = self.fg_bg_sampler(labels)
        sampled_idxs = []
        for pos_idxs_img, neg_idxs_img in zip(sampled_pos_idxs, sampled_neg_idxs):
            img_sampled_idxs = torch.where(pos_idxs_img | neg_idxs_img)[0]
            sampled_idxs.append(img_sampled_idxs)

        return sampled_idxs

    def add_gt_proposals(self, proposals: list[torch.Tensor], gt_boxes: list[torch.Tensor]) -> list[torch.Tensor]:
        proposals = [torch.concat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    @torch.compiler.disable()  # type: ignore[misc]
    def select_training_samples(
        self,
        proposals: list[torch.Tensor],
        targets: Optional[list[dict[str, torch.Tensor]]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if targets is None:
            raise ValueError("targets should not be None")
        if all("boxes" in t for t in targets) is False:
            raise ValueError("Every element of targets should have a boxes key")
        if all("labels" in t for t in targets) is False:
            raise ValueError("Every element of targets should have a labels key")

        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # Append ground-truth bounding boxes to proposals
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # Get matching gt indices for each proposal
        (matched_idxs, labels) = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        # Sample a fixed proportion of positive-negative proposals
        sampled_idxs = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_idxs = sampled_idxs[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_idxs]
            labels[img_id] = labels[img_id][img_sampled_idxs]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_idxs]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)

            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        return (proposals, matched_idxs, labels, regression_targets)

    def postprocess_detections(
        self,
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # Create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # Remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # Batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # Remove low scoring boxes
            idxs = torch.where(scores > self.score_thresh)[0]
            boxes = boxes[idxs]
            scores = scores[idxs]
            labels = labels[idxs]

            # Remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # Non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # Keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return (all_boxes, all_scores, all_labels)

    @torch.compiler.disable(recursive=False)  # type: ignore[misc]
    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if t["boxes"].dtype not in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if t["labels"].dtype != torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

        if self.training is True:
            (proposals, _matched_idxs, labels, regression_targets) = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            _matched_idxs = None  # noqa: F841

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        (class_logits, box_regression) = self.box_predictor(box_features)

        losses = {}
        result: list[dict[str, torch.Tensor]] = []
        if self.training is True:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")

            (loss_classifier, loss_box_reg) = faster_rcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        else:
            (boxes, scores, labels) = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return (result, losses)


# pylint: disable=invalid-name
class Faster_RCNN(DetectionBaseNet):
    default_size = (640, 640)
    auto_register = True

    # pylint: disable=too-many-locals
    def __init__(
        self,
        num_classes: int,
        backbone: DetectorBackbone,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(num_classes, backbone, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is None, "config not supported"

        fpn_width = 256
        rpn_pre_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 1000
        rpn_post_nms_top_n_train = 2000
        rpn_post_nms_top_n_test = 1000
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_nms_thresh = 0.7
        rpn_score_thresh = 0.0
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        box_score_thresh = 0.05
        box_nms_thresh = 0.5
        box_detections_per_img = 100
        canonical_scale = 224

        self.backbone_with_fpn = BackboneWithFPN(self.backbone, fpn_width, extra_blocks=LastLevelMaxPool())

        anchor_sizes = [[32], [64], [128], [256], [512]]
        anchor_sizes = anchor_sizes[-len(self.backbone.return_stages) - 1 :]
        aspect_ratios = [[0.5, 1.0, 2.0]] * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = RPNHead(
            self.backbone_with_fpn.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2
        )

        rpn_pre_nms_top_n = {"training": rpn_pre_nms_top_n_train, "testing": rpn_pre_nms_top_n_test}
        rpn_post_nms_top_n = {"training": rpn_post_nms_top_n_train, "testing": rpn_post_nms_top_n_test}

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=self.backbone.return_stages,
            output_size=canonical_scale // 32,
            sampling_ratio=2,
            canonical_scale=canonical_scale,
        )
        box_head = FastRCNNConvFCHead(
            (self.backbone_with_fpn.out_channels, canonical_scale // 32, canonical_scale // 32),
            [256, 256, 256, 256],
            [1024],
            norm_layer=nn.BatchNorm2d,
        )

        self.representation_size = 1024
        box_predictor = FastRCNNPredictor(self.representation_size, self.num_classes)

        self.roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes + 1
        box_predictor = FastRCNNPredictor(self.representation_size, self.num_classes)
        self.roi_heads.box_predictor = box_predictor

    def freeze(self, freeze_classifier: bool = True) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.roi_heads.box_predictor.parameters():
                param.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
        masks: Optional[torch.Tensor] = None,
        image_sizes: Optional[list[list[int]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        self._input_check(targets)
        images = self._to_img_list(x, image_sizes)

        features = self.backbone_with_fpn(x)
        (proposals, proposal_losses) = self.rpn(images, features, targets)
        (detections, detector_losses) = self.roi_heads(features, proposals, images.image_sizes, targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return (detections, losses)
