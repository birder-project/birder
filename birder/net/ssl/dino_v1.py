"""
DINO v1 (self-DIstillation with NO labels), adapted from
https://github.com/facebookresearch/dino/blob/main/main_dino.py

Paper "Emerging Properties in Self-Supervised Vision Transformers", https://arxiv.org/abs/2104.14294
"""

# Reference license: Apache-2.0

from typing import Any
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from birder.common import training_utils
from birder.net.base import BaseNet
from birder.net.ssl.base import SSLBaseNet


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        num_crops: int,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_temp_epochs: int,
        num_epochs: int,
        student_temp: float,
        center_momentum: float,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.num_crops = num_crops
        self.center = nn.Buffer(torch.zeros(1, out_dim))

        # We apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = torch.concat(
            (
                torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                torch.ones(num_epochs - warmup_teacher_temp_epochs) * teacher_temp,
            ),
            dim=0,
        )

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor, epoch: int) -> torch.Tensor:
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.num_crops)

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0.0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):  # pylint: disable=consider-using-enumerate
                if v == iq:
                    # Skip cases where student and teacher operate on the same view
                    continue

                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()  # type: ignore[misc]
    def update_center(self, teacher_output: torch.Tensor) -> None:
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if training_utils.is_dist_available_and_initialized() is False:
            batch_center = batch_center / len(teacher_output)
        else:
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool,
        norm_last_layer: bool,
        num_layers: int,
        hidden_dim: int,
        bottleneck_dim: int,
    ) -> None:
        super().__init__()
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
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
            self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer is True:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)

        return x

    def cancel_last_layer_gradients(self) -> None:
        self.last_layer.zero_grad()


# pylint: disable=invalid-name
class DINO_v1(SSLBaseNet):
    def __init__(
        self,
        backbone: BaseNet,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(backbone, config=config, size=size)
        assert self.config is not None, "must set config"

        out_dim: int = self.config["out_dim"]
        use_bn: bool = self.config["use_bn"]
        norm_last_layer: bool = self.config["norm_last_layer"]
        num_layers: int = self.config["num_layers"]
        hidden_dim: int = self.config["hidden_dim"]
        bottleneck_dim: int = self.config["bottleneck_dim"]

        self.head = DINOHead(
            self.backbone.embedding_size,
            out_dim,
            use_bn=use_bn,
            norm_last_layer=norm_last_layer,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:  # pylint: disable=arguments-renamed
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([x.size(-1) for x in xs]),
                return_counts=True,
            )[1],
            0,
        )

        start_idx = 0
        output = torch.empty(0).to(xs[0].device)
        for end_idx in idx_crops:
            out = self.backbone.embedding(torch.concat(xs[start_idx:end_idx], dim=0))

            # Accumulate outputs
            output = torch.concat((output, out), dim=0)
            start_idx = end_idx

        return self.head(output)
