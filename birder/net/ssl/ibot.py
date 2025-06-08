"""
iBOT, adapted from
https://github.com/bytedance/ibot/blob/main/models/head.py

Paper "iBOT: Image BERT Pre-Training with Online Tokenizer", https://arxiv.org/abs/2111.07832
"""

# Reference license: Apache-2.0


from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from birder.common import training_utils
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.ssl.base import SSLBaseNet
from birder.net.ssl.dino_v1 import DINOHead


# pylint: disable=invalid-name
class iBOTLoss(nn.Module):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        out_dim: int,
        patch_out_dim: int,
        num_global_crops: int,
        num_local_crops: int,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_temp2: float,
        teacher_temp2: float,
        warmup_teacher_temp_epochs: int,
        epochs: int,
        student_temp: float,
        center_momentum: float,
        center_momentum2: float,
        lambda1: float,
        lambda2: float,
        mim_start_epoch: int,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.num_global_crops = num_global_crops
        self.num_crops = num_global_crops + num_local_crops
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.center = nn.Buffer(torch.zeros(1, out_dim))
        self.center2 = nn.Buffer(torch.zeros(1, 1, patch_out_dim))

        # We apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = torch.concat(
            (
                torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                torch.ones(epochs - warmup_teacher_temp_epochs) * teacher_temp,
            ),
            dim=0,
        )
        if mim_start_epoch == 0:
            self.teacher_temp2_schedule = torch.concat(
                (
                    torch.linspace(warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs),
                    torch.ones(epochs - warmup_teacher_temp_epochs) * teacher_temp2,
                ),
                dim=0,
            )
        else:
            self.teacher_temp2_schedule = torch.concat(
                (
                    torch.ones(mim_start_epoch) * warmup_teacher_temp2,
                    torch.linspace(warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs),
                    torch.ones(epochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2,
                ),
                dim=0,
            )

    # pylint: disable=too-many-locals
    def forward(
        self,
        student_embedding: torch.Tensor,
        student_features: torch.Tensor,
        teacher_embedding: torch.Tensor,
        teacher_features: torch.Tensor,
        student_local_embedding: torch.Tensor,
        student_mask: Optional[torch.Tensor],
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        student_embedding = torch.concat([student_embedding, student_local_embedding], dim=0)

        # Embedding and features for global patches
        student_embedding = student_embedding / self.student_temp
        student_embedding_n = student_embedding.chunk(self.num_crops)
        student_features = student_features / self.student_temp
        student_features_n = student_features.chunk(self.num_global_crops)

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_embedding_center = F.softmax((teacher_embedding - self.center) / temp, dim=-1)
        teacher_embedding_center = teacher_embedding_center.detach().chunk(self.num_global_crops)
        teacher_features_center = F.softmax((teacher_features - self.center2) / temp2, dim=-1)
        teacher_features_center = teacher_features_center.detach().chunk(self.num_global_crops)

        total_loss1 = torch.tensor(0.0, device=student_embedding.device)
        total_loss2 = torch.tensor(0.0, device=student_embedding.device)
        n_loss_terms1 = 0
        n_loss_terms2 = 0
        for q, teacher_embedding_c in enumerate(teacher_embedding_center):
            for v, student_embedding_c in enumerate(student_embedding_n):
                if v == q:
                    loss2 = torch.sum(
                        -teacher_features_center[q] * F.log_softmax(student_features_n[v], dim=-1), dim=-1
                    )
                    if student_mask is not None:
                        mask = student_mask[v]
                        loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                        total_loss2 += loss2.mean()

                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_embedding_c * F.log_softmax(student_embedding_c, dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1

        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = {"embedding": total_loss1, "features": total_loss2, "all": total_loss1 + total_loss2}
        self.update_center(teacher_embedding, teacher_features)

        return total_loss

    @torch.no_grad()  # type: ignore[misc]
    def update_center(self, teacher_embedding: torch.Tensor, teacher_features: torch.Tensor) -> None:
        embedding_center = torch.sum(teacher_embedding, dim=0, keepdim=True)
        if training_utils.is_dist_available_and_initialized() is False:
            embedding_center = embedding_center / (len(teacher_embedding))
        else:
            dist.all_reduce(embedding_center)
            embedding_center = embedding_center / (len(teacher_embedding) * dist.get_world_size())

        self.center = self.center * self.center_momentum + embedding_center * (1 - self.center_momentum)

        features_center = torch.sum(teacher_features.mean(1), dim=0, keepdim=True)
        if training_utils.is_dist_available_and_initialized() is False:
            features_center = features_center / (len(teacher_features))
        else:
            dist.all_reduce(features_center)
            features_center = features_center / (len(teacher_features) * dist.get_world_size())

        self.center2 = self.center2 * self.center_momentum2 + features_center * (1 - self.center_momentum2)


# pylint: disable=invalid-name
class iBOTHead(DINOHead):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        norm_last_layer: bool,
        num_layers: int,
        hidden_dim: int,
        bottleneck_dim: int,
        patch_out_dim: int,
        shared_head: bool,
    ) -> None:
        super().__init__(
            in_dim,
            out_dim,
            use_bn=False,
            norm_last_layer=norm_last_layer,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )

        if shared_head is False:
            self.last_layer2 = nn.utils.parametrizations.weight_norm(
                nn.Linear(bottleneck_dim, patch_out_dim, bias=False)
            )
            self.last_layer2.parametrizations.weight.original0.data.fill_(1)
            if norm_last_layer is True:
                self.last_layer2.parametrizations.weight.original0.requires_grad = False

        else:
            assert out_dim == patch_out_dim
            self.last_layer2 = self.last_layer

    def forward(  # type: ignore[override]  # pylint: disable=arguments-differ
        self, embedding: torch.Tensor, features: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Embedding and features are as coming out of "masked_encoding_retention" function
        This means embedding shape is (B, D) while features shape is (B, C, H, W)

        NOTE: Only models with encoding size == embedding size are supported here
        """

        embedding = embedding.unsqueeze(1)  # (B, 1, D)
        if features is not None:
            features = features.flatten(2).transpose(1, 2)  # (B, L, C)
            x = torch.concat([embedding, features], dim=1)
        else:
            x = embedding

        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x1 = self.last_layer(x[:, 0])
        if features is not None:
            x2 = self.last_layer2(x[:, 1:])
        else:
            x2 = None

        return (x1, x2)


# pylint: disable=invalid-name
class iBOT(SSLBaseNet):
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

        out_dim: int = self.config["out_dim"]
        norm_last_layer: bool = self.config["norm_last_layer"]
        num_layers: int = self.config["num_layers"]
        hidden_dim: int = self.config["hidden_dim"]
        bottleneck_dim: int = self.config["bottleneck_dim"]
        patch_out_dim: int = self.config["patch_out_dim"]
        shared_head: bool = self.config["shared_head"]

        self.head = iBOTHead(
            self.backbone.embedding_size,
            out_dim,
            norm_last_layer=norm_last_layer,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            patch_out_dim=patch_out_dim,
            shared_head=shared_head,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.backbone.stem_width), requires_grad=True)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-differ
        self, x: torch.Tensor, masks: Optional[torch.Tensor], return_keys: Literal["all", "embedding"] = "all"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if masks is not None:
            input_mask = masks
        else:
            seq_len = (x.size(2) // self.backbone.max_stride) * (x.size(3) // self.backbone.max_stride)
            input_mask = torch.zeros([x.size(0), seq_len], device=x.device)

        outs = self.backbone.masked_encoding_retention(
            x, input_mask, mask_token=self.mask_token, return_keys=return_keys
        )
        if return_keys == "embedding":
            outs["features"] = None

        return self.head(outs["embedding"], outs["features"])  # type: ignore[no-any-return]
