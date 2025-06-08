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
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.ssl.base import SSLBaseNet


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool,
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

        # Weight initialization
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)

        return x


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
        masks_weight: torch.Tensor,
        n_masked_patches: Optional[int] = None,
    ) -> torch.Tensor:
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
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


class DINOv2Student(SSLBaseNet):
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

        dino_out_dim: int = self.config["dino_out_dim"]
        use_bn: bool = self.config["use_bn"]
        num_layers: int = self.config["num_layers"]
        hidden_dim: int = self.config["hidden_dim"]
        head_bottleneck_dim: int = self.config["head_bottleneck_dim"]
        ibot_separate_head: bool = self.config["ibot_separate_head"]
        ibot_out_dim: int = self.config.get("ibot_out_dim", dino_out_dim)

        self.dino_head = DINOHead(
            self.backbone.embedding_size,
            dino_out_dim,
            use_bn=use_bn,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
        )
        if ibot_separate_head is False:
            self.ibot_head = None
        else:
            self.ibot_head = DINOHead(
                self.backbone.embedding_size,
                ibot_out_dim,
                use_bn=use_bn,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                bottleneck_dim=head_bottleneck_dim,
            )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.backbone.stem_width), requires_grad=True)

    # pylint: disable=arguments-differ
    def forward(  # type: ignore[override]
        self,
        global_crops: torch.Tensor,
        local_crops: torch.Tensor,
        mask: torch.Tensor,
        upper_bound: int,
        mask_indices_list: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_masked_patches = mask_indices_list.size(0)

        global_out = self.backbone.masked_encoding_retention(global_crops, mask, self.mask_token, return_keys="all")
        global_features = global_out["features"]
        global_features = global_features.flatten(2).transpose(1, 2)
        global_embedding = global_out["embedding"]

        local_embedding = self.backbone.embedding(local_crops)

        global_embedding_after_head = self.dino_head(global_embedding)
        local_embedding_after_head = self.dino_head(local_embedding)

        embed_dim = global_embedding.size(-1)
        buffer_tensor_patch_tokens = global_features.new_zeros(upper_bound, embed_dim)
        buffer_tensor_patch_tokens[:n_masked_patches].copy_(
            torch.index_select(global_features.flatten(0, 1), dim=0, index=mask_indices_list)
        )

        if self.ibot_head is None:
            global_masked_patch_tokens_after_head = self.dino_head(buffer_tensor_patch_tokens)[:n_masked_patches]
        else:
            global_masked_patch_tokens_after_head = self.ibot_head(buffer_tensor_patch_tokens)[:n_masked_patches]

        return (
            global_embedding,
            global_embedding_after_head,
            local_embedding_after_head,
            global_masked_patch_tokens_after_head,
        )


class DINOv2Teacher(SSLBaseNet):
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

        dino_out_dim: int = self.config["dino_out_dim"]
        use_bn: bool = self.config["use_bn"]
        num_layers: int = self.config["num_layers"]
        hidden_dim: int = self.config["hidden_dim"]
        head_bottleneck_dim: int = self.config["head_bottleneck_dim"]
        ibot_separate_head: bool = self.config["ibot_separate_head"]
        ibot_out_dim: int = self.config.get("ibot_out_dim", dino_out_dim)

        self.dino_head = DINOHead(
            self.backbone.embedding_size,
            dino_out_dim,
            use_bn=use_bn,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
        )
        if ibot_separate_head is False:
            self.ibot_head = None
        else:
            self.ibot_head = DINOHead(
                self.backbone.embedding_size,
                ibot_out_dim,
                use_bn=use_bn,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                bottleneck_dim=head_bottleneck_dim,
            )

        # Unused, Makes for an easier EMA update
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.backbone.stem_width), requires_grad=True)

    # pylint: disable=arguments-differ
    def forward(  # type: ignore[override]
        self, x: torch.Tensor, n_crops: int, upper_bound: int, mask_indices_list: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_masked_patches = mask_indices_list.size(0)

        seq_len = (x.size(2) // self.backbone.max_stride) * (x.size(3) // self.backbone.max_stride)
        mask = torch.zeros([x.size(0), seq_len], device=x.device)

        out = self.backbone.masked_encoding_retention(x, mask=mask, return_keys="all")
        features = out["features"]
        features = features.flatten(2).transpose(1, 2)
        embedding = out["embedding"]

        embedding = embedding.chunk(n_crops)
        # NOTE: These are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
        embedding = torch.concat((embedding[1], embedding[0]))

        patch_dim = features.size(-1)
        n = embedding.size(0)

        if self.ibot_head is None:
            buffer_tensor = features.new_zeros(upper_bound + n, patch_dim)
            buffer_tensor[:n].copy_(embedding)
            torch.index_select(
                features.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_tensor[n : n + n_masked_patches],
            )
            tokens_after_head = self.dino_head(buffer_tensor)
            embedding_after_head = tokens_after_head[:n]
            masked_patch_tokens_after_head = tokens_after_head[n : n + n_masked_patches]
        else:
            buffer_teacher = features.new_zeros(upper_bound, patch_dim)
            torch.index_select(
                features.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_teacher[:n_masked_patches],
            )
            embedding_after_head = self.dino_head(embedding)
            masked_patch_tokens_after_head = self.ibot_head(buffer_teacher)[:n_masked_patches]

        return (embedding_after_head, masked_patch_tokens_after_head)
