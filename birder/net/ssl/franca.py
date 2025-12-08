"""
Franca, adapted from
https://github.com/valeoai/Franca/tree/main/franca

Paper "Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning",
https://arxiv.org/abs/2507.14137

Changes from original:
* Removed centering (not supported upstream), left only Sinkhorn Knopp
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


def _get_nesting_list(embedding_size: int, nesting_levels: int) -> list[int]:
    return [embedding_size // (2 ** (nesting_levels - 1 - i)) for i in range(nesting_levels)]


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

        # Weight initialization
        for m in self.matryoshka_projections:
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for m in self.mlps.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []

        for i, dim in enumerate(self.nesting_list):
            # Project input to the appropriate nesting dimension
            h = self.matryoshka_projections[i](x[..., :dim])

            h = self.mlps[i](h)
            out = self.last_layers[i](h)
            outputs.append(out)

        return tuple(outputs)

    def cancel_last_layer_gradients(self) -> None:
        for layer in self.last_layers:
            layer.zero_grad()


class DINOLossMRL(nn.Module):
    def __init__(self, student_temp: float) -> None:
        super().__init__()
        self.student_temp = student_temp

    def forward(
        self,
        student_output_list: list[torch.Tensor],
        teacher_out_softmax_centered_list: list[torch.Tensor],
        n_crops: int | tuple[int, int],
        teacher_global: bool,
    ) -> float:
        total_loss = 0.0
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

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def sinkhorn_knopp_teacher(
        self, teacher_output: tuple[torch.Tensor, ...], teacher_temp: float, n_iterations: int = 3
    ) -> tuple[torch.Tensor, ...]:
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


# pylint: disable=invalid-name
class iBOTPatchLossMRL(nn.Module):
    def __init__(self, student_temp: float) -> None:
        super().__init__()
        self.student_temp = student_temp

    def forward(
        self,
        student_patch_tokens_masked: tuple[torch.Tensor, ...],
        teacher_patch_tokens_masked: tuple[torch.Tensor, ...],
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

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def sinkhorn_knopp_teacher(
        self,
        teacher_outputs: tuple[torch.Tensor, ...],
        teacher_temp: float,
        n_masked_patches_tensor: torch.Tensor,
        n_iterations: int = 3,
    ) -> tuple[torch.Tensor, ...]:
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

        dino_out_dim: int = self.config["dino_out_dim"]
        use_bn: bool = self.config["use_bn"]
        num_layers: int = self.config["num_layers"]
        hidden_dim: int = self.config["hidden_dim"]
        head_bottleneck_dim: int = self.config["head_bottleneck_dim"]
        ibot_separate_head: bool = self.config["ibot_separate_head"]
        ibot_out_dim: int = self.config.get("ibot_out_dim", dino_out_dim)
        nesting_levels: int = self.config.get("nesting_levels", 5)

        nesting_list = _get_nesting_list(self.backbone.embedding_size, nesting_levels)
        self.dino_head = DINOHeadMRL(
            dino_out_dim,
            use_bn=use_bn,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
            nesting_list=nesting_list,
        )
        if ibot_separate_head is False:
            self.ibot_head = None
        else:
            self.ibot_head = DINOHeadMRL(
                ibot_out_dim,
                use_bn=use_bn,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                bottleneck_dim=head_bottleneck_dim,
                nesting_list=nesting_list,
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
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        n_masked_patches = mask_indices_list.size(0)

        global_out = self.backbone.masked_encoding_retention(global_crops, mask, self.mask_token, return_keys="all")
        global_features = global_out["features"]
        global_features = global_features.flatten(2).transpose(1, 2)
        global_embedding = global_out["embedding"]

        local_embedding = self.backbone.embedding(local_crops)

        # DINO head returns tuple of outputs for each nesting level
        global_embedding_after_head = self.dino_head(global_embedding)
        local_embedding_after_head = self.dino_head(local_embedding)

        embed_dim = global_embedding.size(-1)
        buffer_tensor_patch_tokens = global_features.new_zeros(upper_bound, embed_dim)
        buffer_tensor_patch_tokens[:n_masked_patches].copy_(
            torch.index_select(global_features.flatten(0, 1), dim=0, index=mask_indices_list)
        )

        if self.ibot_head is None:
            global_masked_patch_tokens_after_head = tuple(
                t[:n_masked_patches] for t in self.dino_head(buffer_tensor_patch_tokens)
            )
        else:
            global_masked_patch_tokens_after_head = tuple(
                t[:n_masked_patches] for t in self.ibot_head(buffer_tensor_patch_tokens)
            )

        return (
            global_embedding,
            global_embedding_after_head,
            local_embedding_after_head,
            global_masked_patch_tokens_after_head,
        )


class FrancaTeacher(SSLBaseNet):
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
        nesting_levels: int = self.config.get("nesting_levels", 5)

        nesting_list = _get_nesting_list(self.backbone.embedding_size, nesting_levels)
        self.dino_head = DINOHeadMRL(
            dino_out_dim,
            use_bn=use_bn,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
            nesting_list=nesting_list,
        )
        if ibot_separate_head is False:
            self.ibot_head = None
        else:
            self.ibot_head = DINOHeadMRL(
                ibot_out_dim,
                use_bn=use_bn,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                bottleneck_dim=head_bottleneck_dim,
                nesting_list=nesting_list,
            )

        # Unused, Makes for an easier EMA update
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.backbone.stem_width), requires_grad=True)

    # pylint: disable=arguments-differ
    def forward(  # type: ignore[override]
        self, x: torch.Tensor, n_crops: int, upper_bound: int, mask_indices_list: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
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
            embedding_after_head = tuple(t[:n] for t in tokens_after_head)
            masked_patch_tokens_after_head = tuple(t[n : n + n_masked_patches] for t in tokens_after_head)
        else:
            buffer_teacher = features.new_zeros(upper_bound, patch_dim)
            torch.index_select(
                features.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_teacher[:n_masked_patches],
            )
            embedding_after_head = self.dino_head(embedding)
            masked_patch_tokens_after_head = tuple(t[:n_masked_patches] for t in self.ibot_head(buffer_teacher))

        return (embedding_after_head, masked_patch_tokens_after_head)
