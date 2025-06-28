import math
from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from typing import Optional
from typing import TypedDict

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock

from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import DataShapeType
from birder.net.base import DetectorBackbone

DetectorResultType = TypedDict(
    "DetectorResultType", {"boxes": torch.Tensor, "labels": torch.Tensor, "scores": torch.Tensor}
)
DetectionSignatureType = TypedDict(
    "DetectionSignatureType",
    {
        "dynamic": bool,
        "inputs": list[DataShapeType],
        "outputs": tuple[list[DetectorResultType], dict[str, torch.Tensor]],
        "num_labels": int,
    },
)


def get_detection_signature(input_shape: tuple[int, ...], num_outputs: int, dynamic: bool) -> DetectionSignatureType:
    return {
        "dynamic": dynamic,
        "inputs": [{"data_shape": [0, *input_shape[1:]]}],
        "outputs": ([{"boxes": [0, 4], "labels": [0], "scores": [0]}], {}),
        "num_labels": num_outputs,
    }


class DetectionBaseNet(nn.Module):
    default_size: tuple[int, int]
    auto_register = False
    scriptable = True
    task = str(Task.OBJECT_DETECTION)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.auto_register is False:
            # Exclude networks with custom config (initialized only with aliases)
            return

        registry.register_model(cls.__name__.lower(), cls)

    def __init__(
        self,
        num_classes: int,
        backbone: DetectorBackbone,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.input_channels = backbone.input_channels
        self.num_classes = num_classes + 1  # Background always at index 0
        self.backbone = backbone
        self.backbone.transform_to_backbone()
        if hasattr(self, "net_param") is False:  # Avoid overriding aliases
            self.net_param = net_param
        if hasattr(self, "config") is False:  # Avoid overriding aliases
            self.config = config
        elif config is not None:
            assert self.config is not None
            self.config.update(config)  # Override with custom config

        if size is not None:
            self.size = size
        else:
            self.size = self.default_size

        assert isinstance(self.size, tuple)
        assert isinstance(self.size[0], int)
        assert isinstance(self.size[1], int)

        self.dynamic_size = False

    def reset_classifier(self, num_classes: int) -> None:
        raise NotImplementedError

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        """
        Override this when one time adjustments for different resolutions is required.
        This should run after load_state_dict.
        """

        self.size = new_size
        self.backbone.adjust_size(new_size)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        self.backbone.set_dynamic_size(dynamic_size)
        self.dynamic_size = dynamic_size

    def freeze(self, freeze_classifier: bool = True) -> None:
        raise NotImplementedError

    # pylint: disable=protected-access
    def _input_check(self, targets: Optional[list[dict[str, torch.Tensor]]]) -> None:
        if self.training is True:
            if targets is None:
                torch._assert(False, "targets should not be None when in training mode")

            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )

                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # Print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degenerate_bb: list[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degenerate_bb} for target at index {target_idx}.",
                    )

    # pylint: disable=protected-access
    def _to_img_list(self, x: torch.Tensor, image_sizes: Optional[list[list[int]]] = None) -> "ImageList":
        if image_sizes is None:
            image_sizes = [img.shape[-2:] for img in x]

        image_sizes_list: list[tuple[int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        return ImageList(x, image_sizes_list)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
        masks: Optional[torch.Tensor] = None,
        image_sizes: Optional[list[list[int]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        # TypedDict not supported for TorchScript - avoid returning DetectorResultType
        raise NotImplementedError


class ImageList:
    def __init__(self, tensors: torch.Tensor, image_sizes: list[tuple[int, int]]) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> "ImageList":
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


###############################################################################
# Backbone Classes
###############################################################################


class BackboneWithFPN(nn.Module):
    def __init__(
        self, backbone: DetectorBackbone, out_channels: int, extra_blocks: Optional[ExtraFPNBlock] = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.size = self.backbone.size
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.backbone.return_channels,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=nn.BatchNorm2d,
        )

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = self.backbone.detection_features(x)
        out = self.fpn(out)
        return out


# Code adapted from https://github.com/fizyr-forks/torchvision/blob/vitdet2/torchvision/ops/feature_pyramid_network.py
# Reference license: BSD 3-Clause


class BackboneWithSimpleFPN(nn.Module):
    def __init__(
        self,
        backbone: DetectorBackbone,
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        num_stages: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.size = self.backbone.size
        self.fpn = SimpleFeaturePyramidNetwork(
            in_channels=self.backbone.return_channels[-1],
            out_channels=out_channels,
            norm_layer=nn.BatchNorm2d,
            extra_blocks=extra_blocks,
            num_stages=num_stages,
        )

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = self.backbone.detection_features(x)
        x = out[self.backbone.return_stages[-1]]
        out = self.fpn(x)
        return out


class SimpleFeaturePyramidNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Callable[..., nn.Module],
        extra_blocks: Optional[ExtraFPNBlock] = None,
        num_stages: int = 4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for block_index in range(4 - num_stages, 4):
            layers = []

            current_in_channels = in_channels
            if block_index == 0:
                layers.extend(
                    [
                        nn.ConvTranspose2d(
                            in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)
                        ),
                        norm_layer(in_channels // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            in_channels // 2, in_channels // 4, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)
                        ),
                    ]
                )
                current_in_channels = in_channels // 4
            elif block_index == 1:
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)
                    ),
                )
                current_in_channels = in_channels // 2
            elif block_index == 2:
                pass
            elif block_index == 3:
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

            layers.extend(
                [
                    Conv2dNormActivation(
                        current_in_channels,
                        out_channels,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0),
                        norm_layer=norm_layer,
                        activation_layer=None,
                    ),
                    Conv2dNormActivation(
                        out_channels,
                        out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        norm_layer=norm_layer,
                        activation_layer=None,
                    ),
                ]
            )

            self.blocks.append(nn.Sequential(*layers))

        if extra_blocks is not None:
            if not isinstance(extra_blocks, ExtraFPNBlock):
                raise TypeError(f"extra_blocks should be of type ExtraFPNBlock not {type(extra_blocks)}")

        self.extra_blocks = extra_blocks

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        results: list[torch.Tensor] = []
        names: list[str] = []
        for idx, block in enumerate(self.blocks):
            results.append(block(x))
            names.append(f"stage{idx+1}")

        if self.extra_blocks is not None:
            (results, names) = self.extra_blocks(results, [x], names)

        out = OrderedDict(list(zip(names, results)))

        return out


###############################################################################
# General Detection
###############################################################################

# Code adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/_utils.py
# Reference license: BSD 3-Clause


# pylint: disable=protected-access,too-many-locals
@torch.jit._script_if_tracing  # type: ignore
def encode_boxes(reference_boxes: torch.Tensor, proposals: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Encode a set of proposals with respect to some reference boxes
    """

    # Perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # Implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.concat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

    return targets


class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(
        self, weights: tuple[float, float, float, float], bbox_xform_clip: float = math.log(1000.0 / 16)
    ) -> None:
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes: list[torch.Tensor], proposals: list[torch.Tensor]) -> list[torch.Tensor]:
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.concat(reference_boxes, dim=0)
        proposals = torch.concat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        target_list: list[torch.Tensor] = targets.split(boxes_per_image, 0)

        return target_list

    def encode_single(self, reference_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        """
        Encode a set of proposals with respect to some reference boxes
        """

        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    # pylint: disable=protected-access
    def decode(self, rel_codes: torch.Tensor, boxes: list[torch.Tensor]) -> torch.Tensor:
        torch._assert(isinstance(boxes, (list, tuple)), "This function expects boxes of type list or tuple.")
        torch._assert(isinstance(rel_codes, torch.Tensor), "This function expects rel_codes of type torch.Tensor.")

        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.concat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)

        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

        return pred_boxes

    def decode_single(self, rel_codes: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        From a set of original boxes and encoded relative box offsets, get the decoded boxes
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        (wx, wy, ww, wh) = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # Distance from center to box's corner.
        c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

        pred_boxes1 = pred_ctr_x - c_to_c_w
        pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_x + c_to_c_w
        pred_boxes4 = pred_ctr_y + c_to_c_h
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)

        return pred_boxes


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes: list[list[int]],
        aspect_ratios: list[list[float]],
    ) -> None:
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio) for size, aspect_ratio in zip(sizes, aspect_ratios)
        ]

    def generate_anchors(
        self,
        scales_list: list[int],
        aspect_ratios_list: list[float],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        scales = torch.as_tensor(scales_list, dtype=dtype)
        aspect_ratios = torch.as_tensor(aspect_ratios_list, dtype=dtype)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors.round()

    def num_anchors_per_location(self) -> list[int]:
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    # pylint: disable=protected-access
    def grid_anchors(self, grid_sizes: list[list[int]], strides: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        anchors = []
        cell_anchors = self.cell_anchors
        torch._assert(cell_anchors is not None, "cell_anchors should not be None")
        torch._assert(
            len(grid_sizes) == len(strides) == len(cell_anchors),
            "Anchors should be Tuple[Tuple[int]] because each feature "
            "map could potentially have different sizes and aspect ratios. "
            "There needs to be a match between the number of "
            "feature maps passed and the number of sizes / aspect ratios specified.",
        )

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            (grid_height, grid_width) = size
            (stride_height, stride_width) = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def forward(self, image_list: ImageList, feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype = feature_maps[0].dtype
        device = feature_maps[0].device
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),
            ]
            for g in grid_sizes
        ]
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device) for cell_anchor in self.cell_anchors]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors: list[list[torch.Tensor]] = []
        for _ in range(len(image_list.image_sizes)):
            anchors.append(anchors_over_all_feature_maps)

        anchors = [torch.concat(anchors_per_image, dim=0) for anchors_per_image in anchors]
        return anchors


class Matcher(nn.Module):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False) -> None:
        """
        Parameters
        ----------
        high_threshold
            quality values greater than or equal to this value are candidate matches
        low_threshold
            a lower quality threshold used to stratify matches into three levels:
            1) matches >= high_threshold
            2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
            3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
        allow_low_quality_matches
            if True, produce additional matches
            for predictions that have only low-quality match candidates.
        """

        super().__init__()
        self.BELOW_LOW_THRESHOLD = Matcher.BELOW_LOW_THRESHOLD  # pylint: disable=invalid-name
        self.BETWEEN_THRESHOLDS = Matcher.BETWEEN_THRESHOLDS  # pylint: disable=invalid-name
        torch._assert(low_threshold <= high_threshold, "low_threshold should be <= high_threshold")
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def forward(self, match_quality_matrix: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        match_quality_matrix
            an MxN tensor, containing the pairwise quality
            between M ground-truth elements and N predicted elements

        Returns
        -------
        an N tensor where N[i] is a matched gt in [0, M - 1] or a negative
        value indicating that prediction i could not be matched
        """

        if match_quality_matrix.numel() == 0:
            # Empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No ground-truth boxes available for one of the images during training")

            raise ValueError("No proposal boxes available for one of the images during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches is True:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches is True:
            if all_matches is None:
                torch._assert(False, "all_matches should not be None")  # pylint: disable=protected-access
            else:
                self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(
        self, matches: torch.Tensor, all_matches: torch.Tensor, match_quality_matrix: torch.Tensor
    ) -> None:
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """

        # For each gt, find the prediction with which it has the highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)

        # Find the highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None])
        # Example gt_pred_pairs_of_highest_quality:
        # (tensor([0, 1, 1, 2, 2, 3, 3, 4, 5, 5]),
        #  tensor([39796, 32055, 32070, 39190, 40255, 40390, 41455, 45470, 45325, 46390]))
        # Each element in the first tensor is a gt index,
        # and each element in second tensor is a prediction index
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_idx_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_idx_to_update] = all_matches[pred_idx_to_update]
