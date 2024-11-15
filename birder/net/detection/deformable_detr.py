"""
Deformable DETR, adapted from
https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_detr.py
and
https://github.com/huggingface/transformers/blob/main/src/transformers/models/deformable_detr/modeling_deformable_detr.py

Paper "Deformable DETR: Deformable Transformers for End-to-End Object Detection",
https://arxiv.org/abs/2010.04159

Changes from original:
* Removed masking / padding and nested tensors (images are resized at the dataloader)
* Removed support for two stage / box refinement
* Zero cost matrix elements on overflow (HungarianMatcher)
"""

# Reference license: Apache-2.0 (both)

import copy
import math
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torchvision.ops import MLP
from torchvision.ops import boxes as box_ops
from torchvision.ops import sigmoid_focal_loss

from birder.common import training_utils
from birder.kernels.load_kernel import load_msda
from birder.net.base import DetectorBackbone
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.detr import PositionEmbeddingSine

MSDA = None


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network
    """

    def __init__(self, cost_class: float, cost_bbox: float, cost_giou: float):
        super().__init__()
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.jit.unused  # type: ignore[misc]
    def forward(
        self, class_logits: torch.Tensor, box_regression: torch.Tensor, targets: list[dict[str, torch.Tensor]]
    ) -> list[torch.Tensor]:
        with torch.no_grad():
            (B, num_queries) = class_logits.shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = class_logits.flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
            out_bbox = box_regression.flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.concat([v["labels"] for v in targets], dim=0)
            tgt_bbox = torch.concat([v["boxes"] for v in targets], dim=0)

            # Compute the classification cost
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1.0)

            # Compute the giou cost between boxes
            cost_giou = -box_ops.generalized_box_iou(
                box_ops.box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
                box_ops.box_convert(tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
            )

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(B, num_queries, -1).cpu()
            C[C.isnan() | C.isinf()] = 0.0

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)

    return torch.log(x1 / x2)


def multi_scale_deformable_attention(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    (batch_size, _, num_heads, hidden_dim) = value.size()
    (_, num_queries, num_heads, num_levels, num_points, _) = sampling_locations.size()
    areas: list[int] = value_spatial_shapes.prod(dim=1).tolist()
    value_list = value.split(areas, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, spatial_shape in enumerate(value_spatial_shapes):
        # (batch_size, height*width, num_heads, hidden_dim)
        # -> (batch_size, height*width, num_heads*hidden_dim)
        # -> (batch_size, num_heads*hidden_dim, height*width)
        # -> (batch_size*num_heads, hidden_dim, height, width)
        height = spatial_shape[0]
        width = spatial_shape[1]
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )

        # (batch_size, num_queries, num_heads, num_points, 2)
        # -> (batch_size, num_heads, num_queries, num_points, 2)
        # -> (batch_size*num_heads, num_queries, num_points, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)

        # (batch_size*num_heads, hidden_dim, num_queries, num_points)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )

    return output.transpose(1, 2).contiguous()


# pylint: disable=abstract-method,arguments-differ
class MSDAFunction(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(  # type: ignore
        ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
    ):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(  # type: ignore[attr-defined]
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step
        )
        ctx.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output):  # type: ignore
        (value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights) = (
            ctx.saved_tensors
        )
        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(  # type: ignore[attr-defined]
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return (grad_value, None, None, grad_sampling_loc, grad_attn_weight, None)


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, d_model: int, n_levels: int, n_heads: int, n_points: int) -> None:
        super().__init__()

        global MSDA  # pylint: disable=global-statement
        if MSDA is None and torch.jit.is_tracing() is False and torch.jit.is_scripting() is False:
            MSDA = load_msda()

        self.custom_kernel = MSDA is not None
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")

        # Ensure dim_per_head is power of 2
        dim_per_head = d_model // n_heads
        if ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0) is False:
            raise ValueError(
                "Set d_model in MultiScaleDeformableAttention to make the dimension of each attention head a power of 2"
            )

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        (N, num_queries, _) = query.size()
        (N, sequence_length, _) = input_flatten.size()
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == sequence_length

        value = self.value_proj(input_flatten)
        value = value.view(N, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, num_queries, self.n_heads, self.n_levels, self.n_points
        )

        # N, num_queries, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead"
            )

        # Pure PyTorch
        if self.custom_kernel is False or value.is_cuda is False:
            output = multi_scale_deformable_attention(
                value, input_spatial_shapes, sampling_locations, attention_weights
            )

        # Custom kernel
        else:
            output = MSDAFunction.apply(
                value,
                input_spatial_shapes,
                input_level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )

        output = self.output_proj(output)

        return output


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float, n_levels: int, n_heads: int, n_points: int) -> None:
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(d_model, n_levels, n_heads, n_points)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float, n_levels: int, n_heads: int, n_points: int) -> None:
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MultiScaleDeformableAttention(d_model, n_levels, n_heads, n_points)
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
    ) -> torch.Tensor:
        # Self attention
        q = tgt + query_pos
        k = tgt + query_pos

        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        tgt2 = self.cross_attn(tgt + query_pos, reference_points, src, src_spatial_shapes, level_start_index)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    @staticmethod
    def get_reference_points(spatial_shapes: torch.Tensor, device: torch.device) -> torch.Tensor:
        reference_points_list = []
        for spatial_shape in spatial_shapes:
            H = spatial_shape[0]
            W = spatial_shape[1]
            (ref_y, ref_x) = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)

        reference_points = torch.concat(reference_points_list, dim=1)
        reference_points = reference_points[:, :, None]

        return reference_points

    def forward(
        self, src: torch.Tensor, spatial_shapes: torch.Tensor, level_start_index: torch.Tensor, pos: torch.Tensor
    ) -> torch.Tensor:
        out = src
        reference_points = self.get_reference_points(spatial_shapes, device=src.device)
        for layer in self.layers:
            out = layer(out, pos, reference_points, spatial_shapes, level_start_index)

        return out


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, return_intermediate: bool) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: torch.Tensor,
        reference_points: torch.Tensor,
        src: torch.Tensor,
        src_spatial_shapes: torch.Tensor,
        src_level_start_index: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for layer in self.layers:
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None]

            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
            )

            if self.return_intermediate is True:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate is True:
            return (torch.stack(intermediate), torch.stack(intermediate_reference_points))

        return (output, reference_points)


class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        return_intermediate_dec: bool,
        num_feature_levels: int,
        dec_n_points: int,
        enc_n_points: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, num_feature_levels, nhead, enc_n_points
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, num_feature_levels, nhead, dec_n_points
        )
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)

        # Weights initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m._reset_parameters()

            nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            nn.init.zeros_(self.reference_points.bias.data)

        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask: torch.Tensor) -> torch.Tensor:
        (_, H, W) = mask.size()
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / H
        valid_ratio_w = valid_w.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], dim=-1)

        return valid_ratio

    def forward(
        self, srcs: list[torch.Tensor], pos_embeds: list[torch.Tensor], query_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Prepare input for encoder
        src_list = []
        lvl_pos_embed_list = []
        spatial_shape_list: list[tuple[int, int]] = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            (bs, c, h, w) = src.size()
            spatial_shape_list.append((h, w))
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_list.append(lvl_pos_embed)
            src_list.append(src)

        src_flatten = torch.concat(src_list, dim=1)
        lvl_pos_embed_flatten = torch.concat(lvl_pos_embed_list, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shape_list, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.concat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]), dim=0)

        # Encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, lvl_pos_embed_flatten)

        # Prepare input for decoder
        (bs, _, c) = memory.size()
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()

        # Decoder
        (hs, inter_references) = self.decoder(
            tgt, reference_points, memory, spatial_shapes, level_start_index, query_embed
        )

        return (hs, reference_points, inter_references)


# pylint: disable=invalid-name
class Deformable_DETR(DetectionBaseNet):
    default_size = 640
    auto_register = True

    def __init__(
        self,
        num_classes: int,
        backbone: DetectorBackbone,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(num_classes, backbone, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is None, "config not supported"

        # Sigmoid based classification (like multi-label networks)
        self.num_classes = self.num_classes - 1

        hidden_dim = 256
        nhead = 8
        num_encoder_layers = 6
        num_decoder_layers = 6
        dim_feedforward = 1024
        dropout = 0.1
        num_feature_levels = 4
        dec_n_points = 4
        enc_n_points = 4
        num_queries = 300

        self.hidden_dim = hidden_dim
        input_proj_list = []
        for ch in self.backbone.return_channels:
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(ch, hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                    nn.GroupNorm(32, hidden_dim),
                )
            )

        self.input_proj = nn.ModuleList(input_proj_list)

        self.transformer = DeformableTransformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            return_intermediate_dec=True,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            enc_n_points=enc_n_points,
        )

        self.class_embed = nn.Linear(hidden_dim, self.num_classes)
        self.bbox_embed = MLP(hidden_dim, [hidden_dim, hidden_dim, 4], activation_layer=nn.ReLU)
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        self.matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)

        # Weights initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.zeros_(proj[0].bias)

        nn.init.zeros_(self.bbox_embed[-2].weight.data)
        nn.init.zeros_(self.bbox_embed[-2].bias.data)
        nn.init.constant_(self.bbox_embed[-2].bias.data[2:], -2.0)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.class_embed = nn.Linear(self.hidden_dim, num_classes)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value

    def _get_src_permutation_idx(self, indices: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.concat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.concat([src for (src, _) in indices])
        return (batch_idx, src_idx)

    def _class_loss(
        self,
        cls_logits: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        indices: list[torch.Tensor],
        num_boxes: int,
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
        loss = sigmoid_focal_loss(cls_logits, target_classes_onehot, alpha=0.25, gamma=2.0)
        loss_ce = (loss.mean(1).sum() / num_boxes) * cls_logits.shape[1]

        return loss_ce

    def _box_loss(
        self,
        box_output: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        indices: list[torch.Tensor],
        num_boxes: int,
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

    @torch.jit.unused  # type: ignore[misc]
    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        cls_logits: torch.Tensor,
        box_output: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=cls_logits.device)
        if training_utils.is_dist_available_and_initialized() is True:
            torch.distributed.all_reduce(num_boxes)

        num_boxes = torch.clamp(num_boxes / training_utils.get_world_size(), min=1).item()

        loss_ce_list = []
        loss_bbox_list = []
        loss_giou_list = []
        for idx in range(cls_logits.size(0)):
            indices = self.matcher(cls_logits[idx], box_output[idx], targets)
            loss_ce_i = self._class_loss(cls_logits[idx], targets, indices, num_boxes)
            (loss_bbox_i, loss_giou_i) = self._box_loss(box_output[idx], targets, indices, num_boxes)
            loss_ce_list.append(loss_ce_i * 2)
            loss_bbox_list.append(loss_bbox_i * 5)
            loss_giou_list.append(loss_giou_i * 2)

        loss_ce = torch.stack(loss_ce_list).sum()
        loss_bbox = torch.stack(loss_bbox_list).sum()
        loss_giou = torch.stack(loss_giou_list).sum()
        losses = {
            "labels": loss_ce,
            "boxes": loss_bbox,
            "giou": loss_giou,
        }

        return losses

    def postprocess_detections(
        self, class_logits: torch.Tensor, box_regression: torch.Tensor, image_shapes: list[tuple[int, int]]
    ) -> list[dict[str, torch.Tensor]]:
        prob = class_logits.sigmoid()
        (topk_values, topk_indexes) = torch.topk(prob.view(class_logits.shape[0], -1), k=100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // class_logits.shape[2]
        labels = topk_indexes % class_logits.shape[2]
        labels += 1  # Background offset

        target_sizes = torch.tensor(image_shapes, device=class_logits.device)

        # Convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_convert(box_regression, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        (img_h, img_w) = target_sizes.unbind(1)
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

    # pylint: disable=protected-access
    def forward(  # type: ignore[override]
        self, x: torch.Tensor, targets: Optional[list[dict[str, torch.Tensor]]] = None
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        self._input_check(targets)

        image_sizes = [img.shape[-2:] for img in x]
        image_sizes_list: list[tuple[int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        features: dict[str, torch.Tensor] = self.backbone.detection_features(x)
        feature_list = list(features.values())
        pos_list = []
        for idx, proj in enumerate(self.input_proj):
            feature_list[idx] = proj(feature_list[idx])
            pos_list.append(self.pos_enc(feature_list[idx]))

        (hs, init_reference, inter_references) = self.transformer(feature_list, pos_list, self.query_embed.weight)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed(hs[lvl])
            tmp = self.bbox_embed(hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        losses = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training:
            assert targets is not None, "targets should not be none when in training mode"

            # Convert target boxes and classes
            for idx, target in enumerate(targets):
                boxes = target["boxes"]
                boxes = box_ops.box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
                boxes = boxes / torch.tensor(image_sizes[idx] * 2, dtype=torch.float32, device=x.device)
                targets[idx]["boxes"] = boxes
                targets[idx]["labels"] = target["labels"] - 1  # No background

            losses = self.compute_loss(targets, outputs_class, outputs_coord)

        else:
            detections = self.postprocess_detections(outputs_class[-1], outputs_coord[-1], image_sizes_list)

        return (detections, losses)
