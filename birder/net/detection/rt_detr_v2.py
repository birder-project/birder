"""
RT-DETR v2 (Real-Time DEtection TRansformer), adapted from
https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch

Paper "RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer",
https://arxiv.org/abs/2407.17140
"""

# Reference license: Apache-2.0

import copy
import math
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import boxes as box_ops

from birder.common import training_utils
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.deformable_detr import HungarianMatcher
from birder.net.detection.deformable_detr import inverse_sigmoid
from birder.net.detection.rt_detr_v1 import HybridEncoder
from birder.net.detection.rt_detr_v1 import get_contrastive_denoising_training_group
from birder.net.detection.rt_detr_v1 import varifocal_loss
from birder.ops.msda import MultiScaleDeformableAttention as MSDA


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, l_q, C = query.shape
        q = self.q_proj(query).reshape(B, l_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, key.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, value.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        if attn_mask is not None:
            # attn_mask is (L, S) boolean where True = masked
            # SDPA expects True = attend, so we invert
            mask = ~attn_mask
        else:
            mask = None

        attn = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, attn_mask=mask, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
        )

        attn = attn.transpose(1, 2).reshape(B, l_q, C)
        x = self.proj(attn)
        x = self.proj_drop(x)

        return x


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention with per-level point counts
    """

    def __init__(
        self,
        d_model: int,
        n_levels: int,
        n_heads: int,
        n_points: list[int],
        method: Literal["default", "discrete"] = "default",
        offset_scale: float = 0.5,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert len(n_points) == n_levels, f"n_points list length must equal n_levels ({n_levels})"
        assert method in ("default", "discrete"), "method must be 'default' or 'discrete'"

        dim_per_head = d_model // n_heads
        if ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0) is False:
            raise ValueError(
                "Set d_model in MultiScaleDeformableAttention to make the dimension of each attention head a power of 2"
            )

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.method = method
        self.offset_scale = offset_scale

        self.num_points = n_points
        num_points_scale = [1.0 / n for n in self.num_points for _ in range(n)]
        self.num_points_scale = nn.Buffer(torch.tensor(num_points_scale, dtype=torch.float32))
        self.total_points = sum(self.num_points)
        self.uniform_points = len(set(self.num_points)) == 1

        self.msda = MSDA()

        self.sampling_offsets = nn.Linear(d_model, n_heads * self.total_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * self.total_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self.reset_parameters()

        if method == "discrete":
            for param in self.sampling_offsets.parameters():
                param.requires_grad_(False)

    def reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 2).repeat(1, self.total_points, 1)
        scaling = torch.concat([torch.arange(1, n + 1, dtype=torch.float32) for n in self.num_points]).view(1, -1, 1)
        grid_init = grid_init * scaling

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N, num_queries, _ = query.size()
        N, sequence_length, _ = input_flatten.size()
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == sequence_length

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.view(N, sequence_length, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(N, num_queries, self.n_heads, self.total_points, 2)
        attention_weights = self.attention_weights(query).view(N, num_queries, self.n_heads, self.total_points)
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[2] != self.n_levels:
            if reference_points.shape[2] == 1:
                reference_points = reference_points.expand(-1, -1, self.n_levels, -1)
            else:
                raise ValueError(
                    f"reference_points must have {self.n_levels} levels, but got {reference_points.shape[2]}"
                )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations_list = []
            offset_idx = 0
            for lvl in range(self.n_levels):
                n_pts = self.num_points[lvl]
                ref = reference_points[:, :, None, lvl : lvl + 1, :].expand(-1, -1, self.n_heads, n_pts, -1)
                off = sampling_offsets[:, :, :, offset_idx : offset_idx + n_pts, :]
                norm = offset_normalizer[lvl : lvl + 1].view(1, 1, 1, 1, 2)
                sampling_locations_list.append(ref + off / norm)
                offset_idx += n_pts

            sampling_locations = torch.concat(sampling_locations_list, dim=3)

        elif reference_points.shape[-1] == 4:
            sampling_locations_list = []
            offset_idx = 0
            num_points_scale = self.num_points_scale.to(dtype=query.dtype)
            for lvl in range(self.n_levels):
                n_pts = self.num_points[lvl]
                ref = reference_points[:, :, None, lvl : lvl + 1, :].expand(-1, -1, self.n_heads, n_pts, -1)
                off = sampling_offsets[:, :, :, offset_idx : offset_idx + n_pts, :]
                scale = num_points_scale[offset_idx : offset_idx + n_pts].view(1, 1, 1, n_pts, 1)
                sampling_locations_list.append(ref[..., :2] + off * scale * ref[..., 2:] * self.offset_scale)
                offset_idx += n_pts

            sampling_locations = torch.concat(sampling_locations_list, dim=3)

        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead"
            )

        if self.method == "discrete":
            output = self._forward_fallback(
                value, input_spatial_shapes, sampling_locations, attention_weights, method="discrete"
            )
        else:
            if self.uniform_points is True:
                n_pts = self.num_points[0]
                sampling_locations = sampling_locations.view(N, num_queries, self.n_heads, self.n_levels, n_pts, 2)
                attention_weights = attention_weights.view(N, num_queries, self.n_heads, self.n_levels, n_pts)
                output = self.msda(
                    value,
                    input_spatial_shapes,
                    input_level_start_index,
                    sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )
            else:
                output = self._forward_fallback(
                    value, input_spatial_shapes, sampling_locations, attention_weights, method="default"
                )

        output = self.output_proj(output)
        return output

    def _forward_fallback(
        self,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        method: str = "default",
    ) -> torch.Tensor:
        B, _, n_heads, head_dim = value.size()
        num_queries = sampling_locations.size(1)

        sampling_grids = 2 * sampling_locations - 1
        split_shape: list[int] = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)
        sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampling_locations_list = sampling_grids.split(self.num_points, dim=-2)

        sampling_value_list = []
        spatial_shapes_list: list[list[int]] = spatial_shapes.tolist()
        for level, (H, W) in enumerate(spatial_shapes_list):
            value_l = value_list[level].reshape(B * n_heads, head_dim, H, W)
            sampling_grid_l = sampling_locations_list[level]

            if method == "default":
                sampling_value_l = F.grid_sample(
                    value_l,
                    sampling_grid_l,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            else:
                sampling_grid_l = sampling_grid_l.clone()
                sampling_grid_l[..., 0] += 1.0 / W
                sampling_grid_l[..., 1] += 1.0 / H
                sampling_value_l = F.grid_sample(
                    value_l,
                    sampling_grid_l,
                    mode="nearest",
                    padding_mode="border",
                    align_corners=False,
                )

                # Original upstream code (expected grid of [0, 1])
                # e.g. without the 'sampling_grids = 2 * sampling_locations - 1'
                #
                # n_pts = self.num_points[level]
                # sampling_coord = (sampling_grid_l * torch.tensor([[W, H]], device=value.device) + 0.5).to(torch.int64)
                # sampling_coord[..., 0] = sampling_coord[..., 0].clamp(0, W - 1)
                # sampling_coord[..., 1] = sampling_coord[..., 1].clamp(0, H - 1)
                # sampling_coord = sampling_coord.reshape(B * n_heads, num_queries * n_pts, 2)
                # s_idx = (
                #     torch.arange(sampling_coord.shape[0], device=value.device)
                #     .unsqueeze(-1)
                #     .repeat(1, sampling_coord.shape[1])
                # )
                # sampling_value_l = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
                # ... = sampling_value_l.permute(0, 2, 1).reshape(B * n_heads, head_dim, num_queries, n_pts)

            sampling_value_list.append(sampling_value_l)

        attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(B * n_heads, 1, num_queries, sum(self.num_points))
        output = torch.concat(sampling_value_list, dim=-1) * attn_weights
        output = output.sum(-1).reshape(B, n_heads * head_dim, num_queries)

        return output.permute(0, 2, 1)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float,
        n_levels: int,
        n_heads: int,
        n_points: list[int],
        method: Literal["default", "discrete"] = "default",
        offset_scale: float = 0.5,
    ) -> None:
        super().__init__()

        # Self attention
        self.self_attn = MultiheadAttention(d_model, n_heads, attn_drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MultiScaleDeformableAttention(
            d_model, n_levels, n_heads, n_points, method=method, offset_scale=offset_scale
        )
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        src: torch.Tensor,
        src_spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor],
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention
        q = tgt + query_pos
        k = tgt + query_pos

        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        tgt2 = self.cross_attn(
            tgt + query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt


# pylint: disable=invalid-name
class RT_DETRDecoder(nn.Module):
    """
    RT-DETR v2 Decoder with top-k query selection
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        num_queries: int,
        num_decoder_layers: int,
        num_levels: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_decoder_points: list[int],
        method: Literal["default", "discrete"] = "default",
        offset_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_levels = num_levels

        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, [hidden_dim, hidden_dim, 4], activation_layer=nn.ReLU)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            dim_feedforward,
            dropout,
            num_levels,
            num_heads,
            num_decoder_points,
            method=method,
            offset_scale=offset_scale,
        )
        self.layers = _get_clones(decoder_layer, num_decoder_layers)

        self.query_pos_head = MLP(4, [2 * hidden_dim, hidden_dim], activation_layer=nn.ReLU)
        self.class_embed = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)])
        self.bbox_embed = nn.ModuleList(
            [MLP(hidden_dim, [hidden_dim, hidden_dim, 4], activation_layer=nn.ReLU) for _ in range(num_decoder_layers)]
        )
        self.use_cache = True
        self._anchor_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Weights initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.xavier_uniform_(self.enc_output[0].weight)
        nn.init.xavier_uniform_(self.enc_score_head.weight)
        nn.init.constant_(self.enc_score_head.bias, bias_value)
        nn.init.zeros_(self.enc_bbox_head[-2].weight)
        nn.init.zeros_(self.enc_bbox_head[-2].bias)
        for class_embed in self.class_embed:
            nn.init.constant_(class_embed.bias, bias_value)

        for bbox_embed in self.bbox_embed:
            nn.init.zeros_(bbox_embed[-2].weight)
            nn.init.zeros_(bbox_embed[-2].bias)

    def set_cache_enabled(self, enabled: bool) -> None:
        self.use_cache = enabled
        if enabled is False:
            self.clear_cache()

    def clear_cache(self) -> None:
        self._anchor_cache.clear()

    def _generate_anchors(
        self,
        spatial_shapes: list[list[int]],
        grid_size: float = 0.05,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key: Optional[str] = None
        use_cache = self.use_cache is True and torch.jit.is_tracing() is False and torch.jit.is_scripting() is False
        if use_cache is True:
            spatial_key = ",".join(f"{int(h)}x{int(w)}" for h, w in spatial_shapes)
            cache_key = f"{spatial_key}_{grid_size}_{device}_{dtype}"
            cached = self._anchor_cache.get(cache_key)
            if cached is not None:
                return cached

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, dtype=dtype, device=device),
                torch.arange(w, dtype=dtype, device=device),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            valid_wh = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, dim=1)
        eps = 0.01
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(dim=-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        if cache_key is not None:
            self._anchor_cache[cache_key] = (anchors, valid_mask)

        return (anchors, valid_mask)

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes: list[list[int]],
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device, dtype=memory.dtype)
        if memory_padding_mask is not None:
            valid_mask = valid_mask & ~memory_padding_mask.unsqueeze(-1)

        memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        if memory_padding_mask is not None:
            enc_outputs_class = enc_outputs_class.masked_fill(memory_padding_mask[..., None], float("-inf"))

        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        # Select top-k queries based on classification confidence
        _, topk_ind = torch.topk(enc_outputs_class.max(dim=-1).values, self.num_queries, dim=1)

        # Gather reference points
        reference_points_unact = enc_outputs_coord_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1])
        )

        enc_topk_bboxes = reference_points_unact.sigmoid()

        # Gather encoder logits for loss computation
        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        # Extract region features
        target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        target = target.detach()

        return (target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits)

    def forward(  # pylint: disable=too-many-locals
        self,
        feats: list[torch.Tensor],
        spatial_shapes: list[list[int]],
        level_start_index: list[int],
        denoising_class: Optional[torch.Tensor] = None,
        denoising_bbox_unact: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        memory = []
        mask_flatten = []
        for idx, feat in enumerate(feats):
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            memory.append(feat_flat)
            if padding_mask is not None:
                mask_flatten.append(padding_mask[idx].flatten(1))

        memory = torch.concat(memory, dim=1)
        memory_padding_mask = torch.concat(mask_flatten, dim=1) if mask_flatten else None

        # Get decoder input (query selection)
        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = self._get_decoder_input(
            memory, spatial_shapes, memory_padding_mask
        )

        # Concatenate denoising queries if provided
        if denoising_class is not None and denoising_bbox_unact is not None:
            target = torch.concat([denoising_class, target], dim=1)
            init_ref_points_unact = torch.concat([denoising_bbox_unact, init_ref_points_unact], dim=1)

        # Prepare spatial shapes and level start index as tensors
        spatial_shapes_tensor = torch.tensor(spatial_shapes, dtype=torch.long, device=memory.device)
        level_start_index_tensor = torch.tensor(level_start_index, dtype=torch.long, device=memory.device)

        # Decoder forward
        out_bboxes = []
        out_logits = []
        reference_points = init_ref_points_unact.sigmoid()
        for decoder_layer, bbox_head, class_head in zip(self.layers, self.bbox_embed, self.class_embed):
            query_pos = self.query_pos_head(reference_points)
            reference_points_input = reference_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1)
            target = decoder_layer(
                target,
                query_pos,
                reference_points_input,
                memory,
                spatial_shapes_tensor,
                level_start_index_tensor,
                memory_padding_mask,
                attn_mask,
            )

            bbox_delta = bbox_head(target)
            new_reference_points = inverse_sigmoid(reference_points) + bbox_delta
            new_reference_points = new_reference_points.sigmoid()

            # Classification
            class_logits = class_head(target)

            out_bboxes.append(new_reference_points)
            out_logits.append(class_logits)

            # Update reference points for next layer
            reference_points = new_reference_points.detach()

        out_bboxes = torch.stack(out_bboxes)
        out_logits = torch.stack(out_logits)

        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits)


# pylint: disable=invalid-name
class RT_DETR_v2(DetectionBaseNet):
    default_size = (640, 640)

    def __init__(
        self,
        num_classes: int,
        backbone: DetectorBackbone,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
        export_mode: bool = False,
    ) -> None:
        super().__init__(num_classes, backbone, config=config, size=size, export_mode=export_mode)
        assert self.config is not None, "must set config"

        self.reparameterized = False

        # Sigmoid based classification (no background class in predictions)
        self.num_classes = self.num_classes - 1

        hidden_dim = self.config.get("hidden_dim", 256)
        num_heads = self.config.get("num_heads", 8)
        dim_feedforward = self.config.get("dim_feedforward", 1024)
        dropout: float = self.config.get("dropout", 0.0)
        num_encoder_layers: int = self.config.get("num_encoder_layers", 1)
        num_decoder_layers: int = self.config["num_decoder_layers"]
        num_queries: int = self.config.get("num_queries", 300)
        expansion: float = self.config.get("expansion", 1.0)
        depth_multiplier: float = self.config.get("depth_multiplier", 1.0)
        use_giou: bool = self.config.get("use_giou", True)
        num_denoising: int = self.config.get("num_denoising", 100)
        label_noise_ratio: float = self.config.get("label_noise_ratio", 0.5)
        box_noise_scale: float = self.config.get("box_noise_scale", 1.0)
        num_decoder_points: list[int] = self.config.get("num_decoder_points", [4, 4, 4])
        method: Literal["default", "discrete"] = self.config.get("method", "default")
        offset_scale: float = self.config.get("offset_scale", 0.5)

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        self.backbone.return_channels = self.backbone.return_channels[-3:]
        self.backbone.return_stages = self.backbone.return_stages[-3:]
        self.num_levels = len(self.backbone.return_channels)

        self.encoder = HybridEncoder(
            in_channels=self.backbone.return_channels,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_heads=num_heads,
            expansion=expansion,
            depth_multiplier=depth_multiplier,
        )
        self.decoder = RT_DETRDecoder(
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_levels=self.num_levels,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_decoder_points=num_decoder_points,
            method=method,
            offset_scale=offset_scale,
        )

        self.matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, use_giou=use_giou)

        # Denoising class embedding for Contrastive denoising (CDN) training
        if self.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(self.num_classes + 1, hidden_dim, padding_idx=self.num_classes)

        if self.export_mode is False:
            self.forward = torch.compiler.disable(recursive=False)(self.forward)  # type: ignore[method-assign]

    def _set_cache_enabled(self, enabled: bool) -> None:
        self.encoder.set_cache_enabled(enabled)
        self.decoder.set_cache_enabled(enabled)

    def clear_cache(self) -> None:
        self.encoder.clear_cache()
        self.decoder.clear_cache()

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)
        self.clear_cache()

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        super().set_dynamic_size(dynamic_size)
        self._set_cache_enabled(dynamic_size is False)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes

        self.decoder.enc_score_head = nn.Linear(self.hidden_dim, num_classes)
        self.decoder.class_embed = nn.ModuleList(
            [nn.Linear(self.hidden_dim, num_classes) for _ in range(len(self.decoder.layers))]
        )

        if self.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, self.hidden_dim, padding_idx=num_classes)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.decoder.enc_score_head.bias, bias_value)
        for class_embed in self.decoder.class_embed:
            nn.init.constant_(class_embed.bias, bias_value)

    def freeze(self, freeze_classifier: bool = True) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

        if freeze_classifier is False:
            for param in self.decoder.class_embed.parameters():
                param.requires_grad_(True)
            for param in self.decoder.enc_score_head.parameters():
                param.requires_grad_(True)
            if self.num_denoising > 0:
                for param in self.denoising_class_embed.parameters():
                    param.requires_grad_(True)

    def _get_src_permutation_idx(self, indices: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.concat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.concat([src for (src, _) in indices])
        return (batch_idx, src_idx)

    def _class_loss(
        self,
        cls_logits: torch.Tensor,
        box_output: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        indices: list[torch.Tensor],
        num_boxes: float,
    ) -> torch.Tensor:
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.concat([t["labels"][J] for t, (_, J) in zip(targets, indices)], dim=0)
        target_classes = torch.full(cls_logits.shape[:2], self.num_classes, dtype=torch.int64, device=cls_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [cls_logits.shape[0], cls_logits.shape[1], cls_logits.shape[2] + 1],
            dtype=cls_logits.dtype,
            layout=cls_logits.layout,
            device=cls_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        src_boxes = box_output[idx]
        target_boxes = torch.concat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious = torch.diag(
            box_ops.box_iou(
                box_ops.box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
                box_ops.box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
            )
        ).detach()

        target_score_o = torch.zeros(cls_logits.shape[:2], dtype=cls_logits.dtype, device=cls_logits.device)
        target_score_o[idx] = ious.to(cls_logits.dtype)
        target_score = target_score_o.unsqueeze(-1) * target_classes_onehot

        loss = varifocal_loss(cls_logits, target_score, target_classes_onehot, alpha=0.75, gamma=2.0)
        loss_ce = (loss.mean(1).sum() / num_boxes) * cls_logits.shape[1]

        return loss_ce

    def _box_loss(
        self,
        box_output: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        indices: list[torch.Tensor],
        num_boxes: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self._get_src_permutation_idx(indices)
        src_boxes = box_output[idx]
        target_boxes = torch.concat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
                box_ops.box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
            )
        )
        loss_giou = loss_giou.sum() / num_boxes

        return (loss_bbox, loss_giou)

    def _compute_denoising_loss(
        self,
        dn_out_bboxes: torch.Tensor,
        dn_out_logits: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        dn_meta: dict[str, Any],
        num_boxes: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dn_positive_idx = dn_meta["dn_positive_idx"]
        num_groups = dn_meta["dn_num_group"]

        loss_ce_list = []
        loss_bbox_list = []
        loss_giou_list = []

        dn_num_boxes = max(num_boxes * num_groups, 1.0)
        for layer_idx in range(dn_out_logits.shape[0]):
            # Construct indices from denoising metadata
            indices = []
            for batch_idx, pos_idx in enumerate(dn_positive_idx):
                if len(pos_idx) > 0:
                    src_idx = pos_idx
                    num_gt = len(targets[batch_idx]["labels"])
                    tgt_idx = torch.arange(num_gt, device=pos_idx.device).repeat(num_groups)
                    indices.append((src_idx, tgt_idx))
                else:
                    indices.append(
                        (
                            torch.tensor([], dtype=torch.long, device=dn_out_logits.device),
                            torch.tensor([], dtype=torch.long, device=dn_out_logits.device),
                        )
                    )

            loss_ce = self._class_loss(
                dn_out_logits[layer_idx], dn_out_bboxes[layer_idx], targets, indices, dn_num_boxes
            )
            loss_bbox, loss_giou = self._box_loss(dn_out_bboxes[layer_idx], targets, indices, dn_num_boxes)

            loss_ce_list.append(loss_ce)
            loss_bbox_list.append(loss_bbox)
            loss_giou_list.append(loss_giou)

        loss_ce_dn = torch.stack(loss_ce_list).sum()
        loss_bbox_dn = torch.stack(loss_bbox_list).sum()
        loss_giou_dn = torch.stack(loss_giou_list).sum()

        return (loss_ce_dn, loss_bbox_dn, loss_giou_dn)

    @torch.jit.unused  # type: ignore[untyped-decorator]
    @torch.compiler.disable()  # type: ignore[untyped-decorator]
    def _compute_loss_from_outputs(  # pylint: disable=too-many-locals
        self,
        targets: list[dict[str, torch.Tensor]],
        out_bboxes: torch.Tensor,
        out_logits: torch.Tensor,
        enc_topk_bboxes: torch.Tensor,
        enc_topk_logits: torch.Tensor,
        dn_out_bboxes: Optional[torch.Tensor] = None,
        dn_out_logits: Optional[torch.Tensor] = None,
        dn_meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, torch.Tensor]:
        # Compute the average number of target boxes across all nodes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=out_logits.device)
        if training_utils.is_dist_available_and_initialized() is True:
            torch.distributed.all_reduce(num_boxes)

        num_boxes = torch.clamp(num_boxes / training_utils.get_world_size(), min=1).item()

        loss_ce_list = []
        loss_bbox_list = []
        loss_giou_list = []

        # Decoder losses (all layers)
        for layer_idx in range(out_logits.shape[0]):
            indices = self.matcher(out_logits[layer_idx], out_bboxes[layer_idx], targets)
            loss_ce = self._class_loss(out_logits[layer_idx], out_bboxes[layer_idx], targets, indices, num_boxes)
            loss_bbox, loss_giou = self._box_loss(out_bboxes[layer_idx], targets, indices, num_boxes)
            loss_ce_list.append(loss_ce)
            loss_bbox_list.append(loss_bbox)
            loss_giou_list.append(loss_giou)

        # Encoder auxiliary loss
        enc_indices = self.matcher(enc_topk_logits, enc_topk_bboxes, targets)
        loss_ce_enc = self._class_loss(enc_topk_logits, enc_topk_bboxes, targets, enc_indices, num_boxes)
        loss_bbox_enc, loss_giou_enc = self._box_loss(enc_topk_bboxes, targets, enc_indices, num_boxes)
        loss_ce_list.append(loss_ce_enc)
        loss_bbox_list.append(loss_bbox_enc)
        loss_giou_list.append(loss_giou_enc)

        loss_ce = torch.stack(loss_ce_list).sum()  # VFL weight is 1
        loss_bbox = torch.stack(loss_bbox_list).sum() * 5
        loss_giou = torch.stack(loss_giou_list).sum() * 2

        # Add denoising loss if available
        if dn_out_bboxes is not None and dn_out_logits is not None and dn_meta is not None:
            loss_ce_dn, loss_bbox_dn, loss_giou_dn = self._compute_denoising_loss(
                dn_out_bboxes, dn_out_logits, targets, dn_meta, num_boxes
            )
            loss_ce = loss_ce + loss_ce_dn
            loss_bbox = loss_bbox + loss_bbox_dn * 5
            loss_giou = loss_giou + loss_giou_dn * 2

        losses = {
            "labels": loss_ce,
            "boxes": loss_bbox,
            "giou": loss_giou,
        }

        return losses

    @torch.jit.unused  # type: ignore[untyped-decorator]
    @torch.compiler.disable()  # type: ignore[untyped-decorator]
    def compute_loss(
        self,
        encoder_features: list[torch.Tensor],
        spatial_shapes: list[list[int]],
        level_start_index: list[int],
        targets: list[dict[str, torch.Tensor]],
        images: Any,
        masks: Optional[list[torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        device = encoder_features[0].device
        for idx, target in enumerate(targets):
            boxes = target["boxes"]
            boxes = box_ops.box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
            boxes = boxes / torch.tensor(images.image_sizes[idx][::-1] * 2, dtype=torch.float32, device=device)
            targets[idx]["boxes"] = boxes
            targets[idx]["labels"] = target["labels"] - 1  # No background

        denoising_class, denoising_bbox_unact, attn_mask, dn_meta = self._prepare_cdn_queries(targets)

        out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits = self.decoder(
            encoder_features,
            spatial_shapes,
            level_start_index,
            denoising_class,
            denoising_bbox_unact,
            attn_mask,
            masks,
        )

        if dn_meta is not None:
            dn_num_split, _num_queries = dn_meta["dn_num_split"]
            dn_out_bboxes = out_bboxes[:, :, :dn_num_split]
            dn_out_logits = out_logits[:, :, :dn_num_split]
            out_bboxes = out_bboxes[:, :, dn_num_split:]
            out_logits = out_logits[:, :, dn_num_split:]
        else:
            dn_out_bboxes = None
            dn_out_logits = None

        losses: dict[str, torch.Tensor] = self._compute_loss_from_outputs(
            targets, out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits, dn_out_bboxes, dn_out_logits, dn_meta
        )

        return losses

    def postprocess_detections(
        self, class_logits: torch.Tensor, box_regression: torch.Tensor, image_shapes: list[tuple[int, int]]
    ) -> list[dict[str, torch.Tensor]]:
        prob = class_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(class_logits.shape[0], -1), k=self.decoder.num_queries, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // class_logits.shape[2]
        labels = topk_indexes % class_logits.shape[2]
        labels += 1  # Background offset

        target_sizes = torch.tensor(image_shapes, device=class_logits.device)

        # Convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_convert(box_regression, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        detections: list[dict[str, torch.Tensor]] = []
        for s, l, b in zip(scores, labels, boxes):
            detections.append(
                {
                    "boxes": b,
                    "scores": s,
                    "labels": l,
                }
            )

        return detections

    @torch.jit.unused  # type: ignore[untyped-decorator]
    def _prepare_cdn_queries(
        self, targets: list[dict[str, torch.Tensor]]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[dict[str, Any]]]:
        if self.num_denoising > 0:
            result: tuple[
                Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[dict[str, Any]]
            ] = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising_queries=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )
            return result

        return (None, None, None, None)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
        masks: Optional[torch.Tensor] = None,
        image_sizes: Optional[list[list[int]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        self._input_check(targets)
        images = self._to_img_list(x, image_sizes)

        # Backbone features
        features: dict[str, torch.Tensor] = self.backbone.detection_features(x)
        feature_list = list(features.values())

        # Hybrid encoder
        mask_list: list[torch.Tensor] = []
        for feat in feature_list:
            if masks is not None:
                mask_size = feat.shape[-2:]
                m = F.interpolate(masks[None].float(), size=mask_size, mode="nearest").to(torch.bool)[0]
            else:
                B, _, H, W = feat.size()
                m = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)
            mask_list.append(m)

        encoder_features = self.encoder(feature_list, masks=mask_list)

        # Prepare spatial shapes and level start index
        spatial_shapes: list[list[int]] = []
        level_start_index: list[int] = [0]
        for feat in encoder_features:
            H = feat.shape[2]
            W = feat.shape[3]
            spatial_shapes.append([H, W])
            level_start_index.append(H * W + level_start_index[-1])

        level_start_index.pop()

        detections: list[dict[str, torch.Tensor]] = []
        losses: dict[str, torch.Tensor] = {}
        if self.training is True:
            assert targets is not None, "targets should not be None when in training mode"
            losses = self.compute_loss(encoder_features, spatial_shapes, level_start_index, targets, images, mask_list)
        else:
            # Inference path - no CDN
            out_bboxes, out_logits, _, _ = self.decoder(
                encoder_features, spatial_shapes, level_start_index, padding_mask=mask_list
            )
            detections = self.postprocess_detections(out_logits[-1], out_bboxes[-1], images.image_sizes)

        return (detections, losses)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def reparameterize_model(self) -> None:
        if self.reparameterized is True:
            return

        for module in self.modules():
            if hasattr(module, "reparameterize") is True:
                module.reparameterize()

        self.reparameterized = True


registry.register_model_config(
    "rt_detr_v2_s",
    RT_DETR_v2,
    config={
        "num_decoder_layers": 3,
        "expansion": 0.5,
    },
)
registry.register_model_config(
    "rt_detr_v2_s_dsp",
    RT_DETR_v2,
    config={
        "num_decoder_layers": 3,
        "expansion": 0.5,
        "method": "discrete",
    },
)
registry.register_model_config(
    "rt_detr_v2",
    RT_DETR_v2,
    config={
        "num_decoder_layers": 6,
    },
)
registry.register_model_config(
    "rt_detr_v2_dsp",
    RT_DETR_v2,
    config={
        "num_decoder_layers": 6,
        "method": "discrete",
    },
)
registry.register_model_config(
    "rt_detr_v2_l",
    RT_DETR_v2,
    config={
        "num_decoder_layers": 6,
        "expansion": 1.0,
        "depth_multiplier": 1.0,
        "num_heads": 12,  # Deviates from upstream to keep head_dim=32 (power of 2) for MSDA kernel
        "hidden_dim": 384,
        "dim_feedforward": 2048,
    },
)
registry.register_model_config(
    "rt_detr_v2_l_dsp",
    RT_DETR_v2,
    config={
        "num_decoder_layers": 6,
        "expansion": 1.0,
        "depth_multiplier": 1.0,
        "num_heads": 12,  # Deviates from upstream to keep head_dim=32 (power of 2) for MSDA kernel
        "hidden_dim": 384,
        "dim_feedforward": 2048,
        "method": "discrete",
    },
)
