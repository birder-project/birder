import torch

from birder.kernels.load_kernel import load_soft_nms


class SoftNMS:
    """
    Soft-NMS -- Improving Object Detection With One Line of Code: https://arxiv.org/abs/1704.04503

    Lazy-loading Soft-NMS operator with batched processing support.

    The custom kernel is loaded on first instantiation, not at import time.
    Falls back to pure PyTorch implementation if kernel loading fails.

    Example:
        >>> soft_nms = SoftNMS()  # Kernel loads here
        >>> (keep, scores) = soft_nms(boxes, scores, idxs, sigma=0.5)
    """

    def __init__(self) -> None:
        if not torch.jit.is_tracing() and not torch.jit.is_scripting():
            self.soft_nms = load_soft_nms()
        else:
            self.soft_nms = None

    @property
    def is_available(self) -> bool:
        return self.soft_nms is not None

    def __call__(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        idxs: torch.Tensor,
        sigma: float = 0.5,
        score_threshold: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if boxes.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.int64, device=boxes.device),
                torch.empty((0,), dtype=torch.float32, device=scores.device),
            )

        # Offset boxes by category to prevent inter-category suppression
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        if self.soft_nms is not None:
            return self.soft_nms.soft_nms(boxes_for_nms, scores, sigma, score_threshold)  # type: ignore[no-any-return]

        return _soft_nms(boxes_for_nms, scores, sigma, score_threshold)

    def soft_nms_single(
        self, boxes: torch.Tensor, scores: torch.Tensor, sigma: float = 0.5, score_threshold: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.soft_nms is not None:
            return self.soft_nms.soft_nms(boxes, scores, sigma, score_threshold)  # type: ignore[no-any-return]

        return _soft_nms(boxes, scores, sigma, score_threshold)


def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N,]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M,]

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter + 1e-8),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )

    return iou


def _soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    score_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Soft non-max suppression algorithm

    Taken from: https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/soft_nms.py
    Reference license: Apache-2.0
    """

    device = boxes.device
    boxes_remain = boxes.clone()
    scores_remain = scores.clone()
    num_elem = scores_remain.size()[0]
    idxs = torch.arange(num_elem, device=device)
    idxs_out = torch.zeros(num_elem, dtype=torch.int64, device=device)
    scores_out = torch.zeros(num_elem, dtype=torch.float32, device=device)
    count: int = 0

    while scores_remain.numel() > 0:
        top_idx = torch.argmax(scores_remain)
        idxs_out[count] = idxs[top_idx]
        scores_out[count] = scores_remain[top_idx]
        count += 1

        top_box = boxes_remain[top_idx]
        ious = _pairwise_iou(top_box.unsqueeze(0), boxes_remain)[0]

        decay = torch.exp(-torch.pow(ious, 2) / sigma)

        scores_remain *= decay
        keep = scores_remain > score_threshold
        keep[top_idx] = torch.tensor(False, device=device)

        boxes_remain = boxes_remain[keep]
        scores_remain = scores_remain[keep]
        idxs = idxs[keep]

    return (scores_out[:count], idxs_out[:count])


def batched_soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    sigma: float = 0.5,
    score_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=boxes.device),
            torch.empty((0,), dtype=torch.float32, device=scores.device),
        )

    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]

    return _soft_nms(
        boxes_for_nms,
        scores,
        sigma=sigma,
        score_threshold=score_threshold,
    )
