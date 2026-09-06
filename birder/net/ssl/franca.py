"""
Franca, adapted from
https://github.com/valeoai/Franca/tree/main/franca

Paper "Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning",
https://arxiv.org/abs/2507.14137

Changes from original:
* Removed centering (not supported upstream), left only Sinkhorn Knopp
* Added optional SinkhornQueue to improve stability when using smaller batch sizes
"""

# Reference license: Apache-2.0

from typing import Any
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from birder.common import training_utils
from birder.layers.moe import MoETrainingOutputType
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.ssl.base import SSLBaseNet
from birder.net.ssl.base import combine_moe_training_outputs


def _get_nesting_list(embedding_size: int, nesting_levels: int) -> list[int]:
    return [embedding_size // (2 ** (nesting_levels - 1 - i)) for i in range(nesting_levels)]


def _build_mlp(num_layers: int, in_dim: int, bottleneck_dim: int, hidden_dim: int, use_bn: bool) -> nn.Module:
    if num_layers == 1:
        return nn.Linear(in_dim, bottleneck_dim)

    layers = [nn.Linear(in_dim, hidden_dim, bias=not use_bn)]
    if use_bn is True:
        layers.append(nn.BatchNorm1d(hidden_dim))

    layers.append(nn.GELU())
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=not use_bn))
        if use_bn is True:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(nn.GELU())

    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=not use_bn))
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
        self.matryoshka_projections = nn.ModuleList([nn.Linear(dim, dim) for dim in self.nesting_list])

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

        self.grad_checkpointing = False
        self.grad_checkpointing_preserve_rng_state = True
        self.grad_checkpointing_use_reentrant = False

    def set_grad_checkpointing(
        self, enable: bool = True, *, preserve_rng_state: bool = True, use_reentrant: bool = False
    ) -> None:
        self.grad_checkpointing = enable
        self.grad_checkpointing_preserve_rng_state = preserve_rng_state
        self.grad_checkpointing_use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []

        for i, dim in enumerate(self.nesting_list):
            if self.grad_checkpointing is True and torch.is_grad_enabled() is True and not torch.jit.is_scripting():

                def head_body(input_tensor: torch.Tensor, *, idx: int = i, nesting_dim: int = dim) -> torch.Tensor:
                    h = self.matryoshka_projections[idx](input_tensor[..., :nesting_dim])
                    return self.mlps[idx](h)

                h = checkpoint(
                    head_body,
                    x,
                    use_reentrant=self.grad_checkpointing_use_reentrant,
                    preserve_rng_state=self.grad_checkpointing_preserve_rng_state,
                )
            else:
                # Project input to the appropriate nesting dimension
                h = self.matryoshka_projections[i](x[..., :dim])
                h = self.mlps[i](h)

            out = self.last_layers[i](h)
            outputs.append(out)

        return tuple(outputs)

    def cancel_last_layer_gradients(self) -> None:
        for layer in self.last_layers:
            layer.zero_grad()


class SinkhornQueue(nn.Module):
    def __init__(self, queue_size: int, dim: int) -> None:
        super().__init__()
        self.queue_size = queue_size
        self.active = True
        self.queue = nn.Buffer(torch.empty(queue_size, dim))
        self._queue_ptr = 0
        self._queue_full = False

    def get_extra_state(self) -> dict[str, int | bool]:
        return {
            "version": 1,
            "queue_ptr": self._queue_ptr,
            "queue_full": self._queue_full,
        }

    def set_extra_state(self, state: dict[str, int | bool]) -> None:
        if state["version"] != 1:
            raise ValueError(f"Unsupported SinkhornQueue state version: {state['version']}")

        self._queue_ptr = int(state["queue_ptr"])
        self._queue_full = bool(state["queue_full"])

    def set_active(self, active: bool) -> None:
        self.active = active

    def get(self) -> Optional[torch.Tensor]:
        if self.active is False:
            return None
        if self._queue_full is False:
            return None

        return self.queue

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def forward(self, values: torch.Tensor) -> None:
        if self.active is False:
            return
        if values.numel() == 0:
            return
        if values.dim() != 2:
            raise ValueError("SinkhornQueue expects a 2D tensor")

        values = values.detach()
        if values.size(0) >= self.queue_size:
            self.queue.copy_(values[-self.queue_size :])
            self._queue_ptr = 0
            self._queue_full = True
            return

        ptr = self._queue_ptr
        end = ptr + values.size(0)
        if end <= self.queue_size:
            self.queue[ptr:end].copy_(values)
        else:
            first = self.queue_size - ptr
            self.queue[ptr:].copy_(values[:first])
            self.queue[: end - self.queue_size].copy_(values[first:])

        self._queue_ptr = end % self.queue_size
        self._queue_full |= end >= self.queue_size


class DINOLossMRL(nn.Module):
    def __init__(
        self, student_temp: float, nesting_levels: int, queue_size: Optional[int] = None, out_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.queue_active = True
        self.queue_size = queue_size
        if queue_size is None:
            self.sinkhorn_queue = None
        else:
            assert out_dim is not None, "out_dim is required when queue_size is set"

            queue_dims = _get_nesting_list(out_dim, nesting_levels)
            self.sinkhorn_queue = nn.ModuleList()
            for dim in queue_dims:
                queue = SinkhornQueue(queue_size, dim)
                queue.set_active(self.queue_active)
                self.sinkhorn_queue.append(queue)

    def forward(
        self,
        student_output_list: list[torch.Tensor],
        teacher_out_softmax_centered_list: list[torch.Tensor],
        n_crops: int | tuple[int, int],
        teacher_global: bool,
    ) -> torch.Tensor:
        total_loss = 0.0
        if teacher_global is False:
            n_student_crops, n_teacher_crops = n_crops  # type: ignore[misc]
            for student_outputs, teacher_outputs in zip(student_output_list, teacher_out_softmax_centered_list):
                s = student_outputs.view(n_student_crops, -1, student_outputs.size(-1)).float()
                t = teacher_outputs.view(n_teacher_crops, -1, teacher_outputs.size(-1)).float()
                lsm = F.log_softmax(s / self.student_temp, dim=-1)
                total_loss -= torch.sum(t.sum(dim=0) * lsm.sum(dim=0), dim=-1).mean()

        else:
            for student_outputs, teacher_outputs in zip(student_output_list, teacher_out_softmax_centered_list):
                teacher_outputs = teacher_outputs.view(n_crops, -1, teacher_outputs.size(-1)).float()
                lsm = F.log_softmax(student_outputs.float() / self.student_temp, dim=-1)
                loss = torch.sum(teacher_outputs.flatten(0, 1) * lsm, dim=-1)
                total_loss -= loss.mean()

        return total_loss

    def forward_reference(
        self,
        student_output_list: list[torch.Tensor],
        teacher_out_softmax_centered_list: list[torch.Tensor],
        n_crops: int | tuple[int, int],
        teacher_global: bool,
    ) -> torch.Tensor:
        total_loss = 0.0
        if teacher_global is False:
            for student_outputs, teacher_outputs in zip(student_output_list, teacher_out_softmax_centered_list):
                student_feat = student_outputs.chunk(n_crops[0])  # type: ignore[index]
                teacher_feat = teacher_outputs.view(n_crops[1], -1, teacher_outputs.size(-1))  # type: ignore[index]
                for s in student_feat:
                    lsm = F.log_softmax(s.float() / self.student_temp, dim=-1)
                    for t in teacher_feat:
                        loss = torch.sum(t.float() * lsm, dim=-1)
                        total_loss -= loss.mean()

        else:
            for student_outputs, teacher_outputs in zip(student_output_list, teacher_out_softmax_centered_list):
                teacher_outputs = teacher_outputs.view(n_crops, -1, teacher_outputs.size(-1)).float()
                lsm = F.log_softmax(student_outputs.float() / self.student_temp, dim=-1)
                loss = torch.sum(teacher_outputs.flatten(0, 1) * lsm, dim=-1)
                total_loss -= loss.mean()

        return total_loss

    def set_queue_active(self, active: bool) -> None:
        self.queue_active = active
        if self.sinkhorn_queue is not None:
            for queue in self.sinkhorn_queue:
                queue.set_active(active)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def sinkhorn_knopp_teacher(
        self, teacher_output: tuple[torch.Tensor, ...], teacher_temp: float, n_iterations: int = 3
    ) -> tuple[torch.Tensor, ...]:
        results = []
        for idx, t_out in enumerate(teacher_output):
            current_output = t_out
            if self.sinkhorn_queue is not None:
                queue = self.sinkhorn_queue[idx].get()
            else:
                queue = None

            if queue is not None:
                # NOTE: Concat created a new tensor, can modify in-place
                t_out = torch.concat([t_out, queue], dim=0)
                t_out = t_out.float()

            else:
                t_out = t_out.float().clone()

            t_out.div_(teacher_temp).exp_()
            q = t_out.t()

            for _ in range(n_iterations):
                # Normalize each prototype globally
                sum_of_rows = torch.sum(q, dim=1, keepdim=True)
                if training_utils.is_dist_available_and_initialized() is True:
                    dist.all_reduce(sum_of_rows)

                q /= sum_of_rows

                # Normalize each sample locally
                q /= torch.sum(q, dim=0, keepdim=True)

            out = q.t()
            if queue is not None:
                out = out[: current_output.size(0)]

            if self.sinkhorn_queue is not None:
                self.sinkhorn_queue[idx](current_output.detach())

            results.append(out)

        return tuple(results)


class iBOTPatchLossMRL(nn.Module):
    def __init__(
        self, student_temp: float, nesting_levels: int, queue_size: Optional[int] = None, out_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.queue_active = True
        self.queue_size = queue_size
        if queue_size is None:
            self.sinkhorn_queue = None
        else:
            assert out_dim is not None, "out_dim is required when queue_size is set"

            queue_dims = _get_nesting_list(out_dim, nesting_levels)
            self.sinkhorn_queue = nn.ModuleList()
            for dim in queue_dims:
                queue = SinkhornQueue(queue_size, dim)
                queue.set_active(self.queue_active)
                self.sinkhorn_queue.append(queue)

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
            loss = torch.sum(t.float() * F.log_softmax(s.float() / self.student_temp, dim=-1), dim=-1)
            if masks_weight is None:
                masks_weight = (
                    (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                    .unsqueeze(-1)
                    .expand_as(student_masks_flat)[student_masks_flat]
                )
            if n_masked_patches is not None:
                loss = loss[:n_masked_patches]

            loss = loss * masks_weight
            total_loss -= loss.sum() / student_masks_flat.size(0)

        return total_loss

    def set_queue_active(self, active: bool) -> None:
        self.queue_active = active
        if self.sinkhorn_queue is not None:
            for queue in self.sinkhorn_queue:
                queue.set_active(active)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def sinkhorn_knopp_teacher(
        self,
        teacher_outputs: tuple[torch.Tensor, ...],
        teacher_temp: float,
        n_iterations: int = 3,
    ) -> tuple[torch.Tensor, ...]:
        result = []
        for idx, teacher_output in enumerate(teacher_outputs):
            current_output = teacher_output
            if self.sinkhorn_queue is not None:
                queue = self.sinkhorn_queue[idx].get()
            else:
                queue = None

            if queue is not None:
                # NOTE: Concat created a new tensor, can modify in-place
                teacher_output = torch.concat([teacher_output, queue], dim=0)
                teacher_output = teacher_output.float()
            else:
                teacher_output = teacher_output.float().clone()

            teacher_output.div_(teacher_temp).exp_()
            q = teacher_output.t()

            for _ in range(n_iterations):
                # Normalize each prototype globally
                sum_of_rows = torch.sum(q, dim=1, keepdim=True)
                if training_utils.is_dist_available_and_initialized() is True:
                    dist.all_reduce(sum_of_rows)

                q /= sum_of_rows

                # Normalize each sample locally
                q /= torch.sum(q, dim=0, keepdim=True)

            out = q.t()
            if queue is not None:
                out = out[: current_output.size(0)]

            if self.sinkhorn_queue is not None:
                self.sinkhorn_queue[idx](current_output.detach())

            result.append(out)

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

        assert ibot_separate_head is True or self.backbone.feature_dim == self.backbone.embedding_size

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
            ibot_nesting_list = _get_nesting_list(self.backbone.feature_dim, nesting_levels)
            self.ibot_head = DINOHeadMRL(
                ibot_out_dim,
                use_bn=use_bn,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                bottleneck_dim=head_bottleneck_dim,
                nesting_list=ibot_nesting_list,
            )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.backbone.stem_width))

    def set_grad_checkpointing(
        self,
        enable: bool = True,
        *,
        segments: Optional[int] = None,
        preserve_rng_state: bool = True,
        use_reentrant: bool = False,
    ) -> None:
        self.backbone.set_grad_checkpointing(
            enable=enable, segments=segments, preserve_rng_state=preserve_rng_state, use_reentrant=use_reentrant
        )
        self.dino_head.set_grad_checkpointing(
            enable=enable, preserve_rng_state=preserve_rng_state, use_reentrant=use_reentrant
        )
        if self.ibot_head is not None:
            self.ibot_head.set_grad_checkpointing(
                enable=enable, preserve_rng_state=preserve_rng_state, use_reentrant=use_reentrant
            )

    # pylint: disable=arguments-differ
    def forward(  # type: ignore[override]
        self,
        global_crops: torch.Tensor,
        local_crops: torch.Tensor,
        mask: torch.Tensor,
        upper_bound: int,
        mask_indices_list: torch.Tensor,
        *,
        return_moe_training_output: bool = False,
    ) -> dict[str, Any]:
        n_masked_patches = mask_indices_list.size(0)

        if return_moe_training_output is True:
            global_out = self.backbone.masked_encoding_retention(
                global_crops, mask, self.mask_token, return_keys="all", return_moe_training_output=True
            )
        else:
            global_out = self.backbone.masked_encoding_retention(global_crops, mask, self.mask_token, return_keys="all")

        global_features = global_out["features"]
        global_features = global_features.flatten(2).transpose(1, 2)
        global_embedding = global_out["embedding"]

        moe_training_output: Optional[MoETrainingOutputType] = None
        if "moe_training_output" in global_out:
            moe_training_output = global_out["moe_training_output"]

        if return_moe_training_output is True:
            local_features, local_moe_training_output = self.backbone.forward_features(
                local_crops, return_moe_training_output=True  # type: ignore[call-arg]
            )
            moe_training_output = combine_moe_training_outputs(
                moe_training_output, local_moe_training_output, additional_loss_weight=0.1
            )
        else:
            local_features = self.backbone.forward_features(local_crops)

        local_embedding = self.backbone.embedding_from_features(local_features)

        # DINO head returns tuple of outputs for each nesting level
        global_embedding_after_head = self.dino_head(global_embedding)
        local_embedding_after_head = self.dino_head(local_embedding)

        patch_dim = global_features.size(-1)
        buffer_tensor_patch_tokens = global_features.new_zeros(upper_bound, patch_dim)
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

        outputs = {
            "global_embedding": global_embedding,
            "global_embedding_after_head": global_embedding_after_head,
            "local_embedding_after_head": local_embedding_after_head,
            "global_masked_patch_tokens_after_head": global_masked_patch_tokens_after_head,
        }
        if moe_training_output is not None:
            outputs["moe_training_output"] = moe_training_output

        return outputs


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

        assert ibot_separate_head is True or self.backbone.feature_dim == self.backbone.embedding_size

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
            ibot_nesting_list = _get_nesting_list(self.backbone.feature_dim, nesting_levels)
            self.ibot_head = DINOHeadMRL(
                ibot_out_dim,
                use_bn=use_bn,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                bottleneck_dim=head_bottleneck_dim,
                nesting_list=ibot_nesting_list,
            )

        # Unused, Makes for an easier EMA update
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.backbone.stem_width))

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
