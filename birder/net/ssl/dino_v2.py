"""
DINO v2, adapted from
https://github.com/facebookresearch/dinov2/tree/main/dinov2

Paper "DINOv2: Learning Robust Visual Features without Supervision", https://arxiv.org/abs/2304.07193
"""

# Reference license: Apache-2.0

from typing import Any
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from birder.common import training_utils


class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, student_temp: float, center_momentum: float) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center = nn.Buffer(torch.zeros(1, out_dim))

        self.updated = True
        self.reduce_handle: Any = None
        self.len_teacher_output: Optional[int] = None
        self.async_batch_center: Optional[torch.Tensor] = None

    def forward(
        self, student_output_list: list[torch.Tensor], teacher_out_softmax_centered_list: list[torch.Tensor]
    ) -> float:
        total_loss = 0.0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmax_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()

        return total_loss

    @torch.no_grad()  # type: ignore[misc]
    def softmax_center_teacher(self, teacher_output: torch.Tensor, teacher_temp: float) -> torch.Tensor:
        self.apply_center_update()
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()  # type: ignore[misc]
    def sinkhorn_knopp_teacher(
        self, teacher_output: torch.Tensor, teacher_temp: float, n_iterations: int = 3
    ) -> torch.Tensor:
        if training_utils.is_dist_available_and_initialized() is True:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        teacher_output = teacher_output.float()
        q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from the paper
        B = q.size(1) * world_size  # Number of samples to assign
        k = q.size(0)  # How many prototypes

        sum_q = torch.sum(q)
        if training_utils.is_dist_available_and_initialized() is True:
            dist.all_reduce(sum_q)

        q /= sum_q

        for _ in range(n_iterations):
            # Normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(q, dim=1, keepdim=True)
            if training_utils.is_dist_available_and_initialized() is True:
                dist.all_reduce(sum_of_rows)

            q /= sum_of_rows
            q /= k

            # Normalize each column: total weight per sample must be 1/B
            q /= torch.sum(q, dim=0, keepdim=True)
            q /= B

        q *= B  # The columns must sum to 1 so that Q is an assignment

        return q.t()

    @torch.no_grad()  # type: ignore[misc]
    def update_center(self, teacher_output: torch.Tensor) -> None:
        self.reduce_center_update(teacher_output)

    @torch.no_grad()  # type: ignore[misc]
    def reduce_center_update(self, teacher_output: torch.Tensor) -> None:
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()  # type: ignore[misc]
    def apply_center_update(self) -> None:
        if self.updated is False:
            if training_utils.is_dist_available_and_initialized() is True:
                world_size = dist.get_world_size()
            else:
                world_size = 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()

            _t = self.async_batch_center / (self.len_teacher_output * world_size)
            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)
            self.updated = True


# pylint: disable=invalid-name
class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim: int, student_temp: float, center_momentum: float) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center = nn.Buffer(torch.zeros(1, 1, patch_out_dim))
        self.updated = True
        self.reduce_handle: Any = None
        self.len_teacher_patch_tokens: Optional[int] = None
        self.async_batch_center: Optional[torch.Tensor] = None

    def forward(
        self, student_patch_tokens: torch.Tensor, teacher_patch_tokens: torch.Tensor, student_masks_flat: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.sum(teacher_patch_tokens * F.log_softmax(student_patch_tokens / self.student_temp, dim=-1), dim=-1)
        loss = torch.sum(loss * student_masks_flat.float(), dim=-1) / student_masks_flat.sum(dim=-1).clamp(min=1.0)

        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked: torch.Tensor,
        teacher_patch_tokens_masked: torch.Tensor,
        student_masks_flat: torch.Tensor,
        n_masked_patches: Optional[int] = None,
        masks_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]

        loss = loss * masks_weight

        return -loss.sum() / student_masks_flat.size(0)

    @torch.no_grad()  # type: ignore[misc]
    def softmax_center_teacher(self, teacher_patch_tokens: torch.Tensor, teacher_temp: float) -> torch.Tensor:
        self.apply_center_update()
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()  # type: ignore[misc]
    def sinkhorn_knopp_teacher(
        self,
        teacher_output: torch.Tensor,
        teacher_temp: float,
        n_masked_patches_tensor: torch.Tensor,
        n_iterations: int = 3,
    ) -> torch.Tensor:
        teacher_output = teacher_output.float()
        q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from the paper
        B = n_masked_patches_tensor
        if training_utils.is_dist_available_and_initialized() is True:
            dist.all_reduce(B)

        K = q.size(0)  # How many prototypes

        sum_q = torch.sum(q)
        if training_utils.is_dist_available_and_initialized() is True:
            dist.all_reduce(sum_q)

        q /= sum_q

        for _ in range(n_iterations):
            # Normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(q, dim=1, keepdim=True)
            if training_utils.is_dist_available_and_initialized() is True:
                dist.all_reduce(sum_of_rows)

            q /= sum_of_rows
            q /= K

            # Normalize each column: total weight per sample must be 1/B
            q /= torch.sum(q, dim=0, keepdim=True)
            q /= B

        q *= B  # The columns must sum to 1 so that Q is an assignment

        return q.t()

    @torch.no_grad()  # type: ignore[misc]
    def update_center(self, teacher_patch_tokens: torch.Tensor) -> None:
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()  # type: ignore[misc]
    def reduce_center_update(self, teacher_patch_tokens: torch.Tensor) -> None:
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        if training_utils.is_dist_available_and_initialized() is True:
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()  # type: ignore[misc]
    def apply_center_update(self) -> None:
        if self.updated is False:
            if training_utils.is_dist_available_and_initialized() is True:
                world_size = dist.get_world_size()
            else:
                world_size = 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()

            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)
            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)
            self.updated = True


class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko entropic loss regularizer from:
    Spreading vectors for similarity search - https://arxiv.org/abs/1806.03198
    """

    def __init__(self) -> None:
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_nn_inner(self, x: torch.Tensor) -> torch.Tensor:
        # Pairwise dot products
        dots = torch.mm(x, x.t())
        n = x.size(0)
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1

        # Max inner prod -> min distance
        ind = torch.argmax(dots, dim=1)

        return ind

    def forward(self, student_output: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        with torch.amp.autocast(student_output.device.type, enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            ind = self.pairwise_nn_inner(student_output)
            distances = self.pdist(student_output, student_output[ind])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()

        return loss
