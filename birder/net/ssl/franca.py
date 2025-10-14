"""
DINO v2, adapted from
https://github.com/valeoai/Franca/tree/main/franca

Paper "Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning",
https://arxiv.org/abs/2507.14137
"""

# Reference license: Apache-2.0

from typing import Any
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from birder.common import training_utils
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.ssl.base import SSLBaseNet


def _build_mlp(num_layers: int, in_dim: int, bottleneck_dim: int, hidden_dim: int, use_bn: bool) -> nn.Module:
    if num_layers == 1:
        return nn.Linear(in_dim, bottleneck_dim)

    layers = [nn.Linear(in_dim, hidden_dim)]
    if use_bn is True:
        layers.append(nn.BatchNorm1d(hidden_dim))

    layers.append(nn.GELU())
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn is True:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(nn.GELU())

    layers.append(nn.Linear(hidden_dim, bottleneck_dim))
    if use_bn is True:
        layers.append(nn.BatchNorm1d(bottleneck_dim))

    layers.append(nn.GELU())

    return nn.Sequential(*layers)


class DINOHeadMRL(nn.Module):
    # DINO Head with Matryoshka Representation Learning

    def __init__(
        self,
        out_dim: int,
        use_bn: bool,
        num_layers: int,
        hidden_dim: int,
        bottleneck_dim: int,
        nesting_list: list[int],
    ) -> None:
        super().__init__()
        self.nesting_list = nesting_list
        self.matryoshka_projections = nn.ModuleList([nn.Linear(dim, dim, bias=True) for dim in self.nesting_list])

        self.mlps = nn.ModuleList(
            [
                _build_mlp(num_layers, dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn)
                for dim in self.nesting_list
            ]
        )

        self.last_layers = nn.ModuleList(
            [
                nn.utils.parametrizations.weight_norm(
                    nn.Linear(
                        bottleneck_dim,
                        int(out_dim * (dim / self.nesting_list[-1])),
                        bias=False,
                    )
                )
                for dim in self.nesting_list
            ]
        )
        for layer in self.last_layers:
            layer.parametrizations.weight.original0.data.fill_(1)

        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        outputs = []

        for i, dim in enumerate(self.nesting_list):
            # Project input to the appropriate nesting dimension
            h = self.matryoshka_projections[i](x[..., :dim])

            h = self.mlps[i](h)
            out = self.last_layers[i](h)
            outputs.append(out)

        return tuple(outputs)


class DINOLossMRL(nn.Module):
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
        self,
        student_output_list: list[torch.Tensor],
        teacher_out_softmax_centered_list: list[torch.Tensor],
        n_crops: int | tuple[int, int],
        teacher_global: bool,
    ) -> float:
        total_loss = 0
        if teacher_global is False:
            for student_outputs, teacher_outputs in zip(student_output_list, teacher_out_softmax_centered_list):
                student_feat = student_outputs.chunk(n_crops[0])  # type: ignore[index]
                teacher_feat = teacher_outputs.view(n_crops[1], -1, teacher_outputs.shape[-1])  # type: ignore[index]
                for s in student_feat:
                    lsm = F.log_softmax(s / self.student_temp, dim=-1)
                    for t in teacher_feat:
                        loss = torch.sum(t * lsm, dim=-1)
                        total_loss -= loss.mean()

        else:
            for student_outputs, teacher_outputs in zip(student_output_list, teacher_out_softmax_centered_list):
                teacher_outputs = teacher_outputs.view(n_crops, -1, teacher_outputs.shape[-1])
                lsm = F.log_softmax(student_outputs / self.student_temp, dim=-1)
                loss = torch.sum(teacher_outputs.flatten(0, 1) * lsm, dim=-1)
                total_loss -= loss.mean()

        return total_loss

    @torch.no_grad()  # type: ignore[misc]
    def softmax_center_teacher(self, teacher_output: tuple[torch.Tensor], teacher_temp: float) -> tuple[torch.Tensor]:
        self.apply_center_update()
        return tuple(F.softmax((t - self.center) / teacher_temp, dim=-1) for t in teacher_output)

    @torch.no_grad()  # type: ignore[misc]
    def sinkhorn_knopp_teacher(
        self, teacher_output: tuple[torch.Tensor], teacher_temp: float, n_iterations: int = 3
    ) -> tuple[torch.Tensor]:
        world_size = training_utils.get_world_size()

        results = []
        for t_out in teacher_output:
            t_out = t_out.float()
            q = torch.exp(t_out / teacher_temp).t()
            B = q.size(1) * world_size  # Number of samples to assign
            k = q.size(0)  # How many prototypes

            for _ in range(n_iterations):
                sum_of_rows = torch.sum(q, dim=1, keepdim=True)
                if training_utils.is_dist_available_and_initialized() is True:
                    dist.all_reduce(sum_of_rows)

                q /= sum_of_rows
                q /= k
                q /= torch.sum(q, dim=0, keepdim=True)
                q /= B

            q *= B
            results.append(q.t())

        return tuple(results)

    @torch.no_grad()  # type: ignore[misc]
    def update_center(self, teacher_output: tuple[torch.Tensor]) -> None:
        self.reduce_center_update(teacher_output[0])

    @torch.no_grad()  # type: ignore[misc]
    def reduce_center_update(self, teacher_output: torch.Tensor) -> None:
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if training_utils.is_dist_available_and_initialized() is True:
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()  # type: ignore[misc]
    def apply_center_update(self) -> None:
        if self.updated is False:
            world_size = training_utils.get_world_size()
            if self.reduce_handle is not None:
                self.reduce_handle.wait()

            _t = self.async_batch_center / (self.len_teacher_output * world_size)  # type: ignore[operator]
            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)
            self.updated = True


# pylint: disable=invalid-name
class iBOTPatchLossMRL(nn.Module):
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
        self,
        student_patch_tokens_masked: tuple[torch.Tensor],
        teacher_patch_tokens_masked: tuple[torch.Tensor],
        student_masks_flat: torch.Tensor,
        masks_weight: torch.Tensor,
        n_masked_patches: Optional[int] = None,
    ) -> torch.Tensor:
        total_loss = 0.0
        for s, t in zip(student_patch_tokens_masked, teacher_patch_tokens_masked):
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
            total_loss -= loss.sum() / student_masks_flat.shape[0]

        return total_loss

    @torch.no_grad()  # type: ignore[misc]
    def softmax_center_teacher(
        self, teacher_patch_tokens: tuple[torch.Tensor], teacher_temp: float
    ) -> tuple[torch.Tensor]:
        self.apply_center_update()
        return tuple(F.softmax((t - self.center) / teacher_temp, dim=-1) for t in teacher_patch_tokens)

    @torch.no_grad()  # type: ignore[misc]
    def sinkhorn_knopp_teacher(
        self,
        teacher_outputs: tuple[torch.Tensor],
        teacher_temp: float,
        n_masked_patches_tensor: torch.Tensor,
        n_iterations: int = 3,
    ) -> tuple[torch.Tensor]:
        result = []
        for teacher_output in teacher_outputs:
            teacher_output = teacher_output.float()
            q = torch.exp(teacher_output / teacher_temp).t()
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

            q *= B
            result.append(q.t())

        return tuple(result)

    @torch.no_grad()  # type: ignore[misc]
    def update_center(self, teacher_patch_tokens: tuple[torch.Tensor]) -> None:
        self.reduce_center_update(teacher_patch_tokens[0])

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
            world_size = training_utils.get_world_size()
            if self.reduce_handle is not None:
                self.reduce_handle.wait()

            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)  # type: ignore[operator]
            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)
            self.updated = True


class FrancaStudent(SSLBaseNet):
    default_size = (224, 224)

    def __init__(
        self,
        backbone: PreTrainEncoder,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(backbone, config=config, size=size)
        assert self.config is not None, "must set config"
        assert isinstance(self.backbone, MaskedTokenRetentionMixin)

    def forward(self, x: torch.Tensor) -> Any:
        raise NotImplementedError
