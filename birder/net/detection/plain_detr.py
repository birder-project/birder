"""
Plain DETR, adapted from
https://github.com/impiga/Plain-DETR

Paper "DETR Doesn't Need Multi-Scale or Locality Design", https://arxiv.org/abs/2308.01904

Changes from original:
* Move background index to first from last (to be inline with the rest of Birder detectors)
* Removed two stage support
* Only support pre-norm (original supports both pre- and post-norm)
"""

# Reference license: MIT

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
from torchvision.ops import sigmoid_focal_loss

from birder.common import training_utils
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.deformable_detr import HungarianMatcher
from birder.net.detection.deformable_detr import inverse_sigmoid
from birder.net.detection.detr import PositionEmbeddingSine
from birder.ops.soft_nms import SoftNMS


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
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, l_q, C = query.size()
        q = self.q_proj(query).reshape(B, l_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, key.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, value.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        if key_padding_mask is not None:
            # key_padding_mask is expected to be boolean (True = masked)
            # SDPA expects True = attend, so we invert
            attn_mask = ~key_padding_mask[:, None, None, :]
        else:
            attn_mask = None

        attn = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
        )

        attn = attn.transpose(1, 2).reshape(B, l_q, C)
        x = self.proj(attn)
        x = self.proj_drop(x)

        return x


class GlobalCrossAttention(nn.Module):
    """
    Global cross-attention with Box-to-Pixel Relative Position Bias (BoxRPB)

    This utilizes Box-to-Pixel Relative Position Bias (BoxRPB) to guide attention
    using the spatial relationship between query boxes and image features.
    The bias calculation is decomposed into axial (x and y) components.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        rpe_hidden_dim: int,
        feature_stride: int,
        rpe_type: Literal["linear", "log"],
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.feature_stride = feature_stride
        self.rpe_type = rpe_type

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.cpb_mlp_x = nn.Sequential(
            nn.Linear(2, rpe_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(rpe_hidden_dim, num_heads, bias=False),
        )
        self.cpb_mlp_y = nn.Sequential(
            nn.Linear(2, rpe_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(rpe_hidden_dim, num_heads, bias=False),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shape: tuple[int, int],
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, num_queries, _ = query.size()
        H, W = spatial_shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(B, num_queries, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        rpe = self._compute_box_rpe(reference_points, H, W, query.device)
        attn = attn + rpe

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, num_queries, self.embed_dim)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out

    # pylint: disable=too-many-locals
    def _compute_box_rpe(self, reference_points: torch.Tensor, H: int, W: int, device: torch.device) -> torch.Tensor:
        B, n_q, _ = reference_points.size()
        stride = self.feature_stride

        # cxcywh to xyxy
        cx, cy, bw, bh = reference_points.unbind(-1)
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Scale to pixel coordinates
        x1 = x1 * (W * stride)
        y1 = y1 * (H * stride)
        x2 = x2 * (W * stride)
        y2 = y2 * (H * stride)

        # Pixel grid (cell centers)
        pos_x = torch.linspace(0.5, W - 0.5, W, device=device) * stride
        pos_y = torch.linspace(0.5, H - 0.5, H, device=device) * stride

        # Box edge to pixel distances
        delta_x1 = x1[:, :, None] - pos_x[None, None, :]
        delta_x2 = x2[:, :, None] - pos_x[None, None, :]
        delta_y1 = y1[:, :, None] - pos_y[None, None, :]
        delta_y2 = y2[:, :, None] - pos_y[None, None, :]

        if self.rpe_type == "log":
            delta_x1 = torch.sign(delta_x1) * torch.log2(torch.abs(delta_x1) + 1.0) / 3.0
            delta_x2 = torch.sign(delta_x2) * torch.log2(torch.abs(delta_x2) + 1.0) / 3.0
            delta_y1 = torch.sign(delta_y1) * torch.log2(torch.abs(delta_y1) + 1.0) / 3.0
            delta_y2 = torch.sign(delta_y2) * torch.log2(torch.abs(delta_y2) + 1.0) / 3.0

        delta_x = torch.stack([delta_x1, delta_x2], dim=-1)
        delta_y = torch.stack([delta_y1, delta_y2], dim=-1)

        rpe_x = self.cpb_mlp_x(delta_x)
        rpe_y = self.cpb_mlp_y(delta_y)

        # Axial decomposition: rpe[h,w] = rpe_y[h] + rpe_x[w]
        rpe = rpe_y[:, :, :, None, :] + rpe_x[:, :, None, :, :]
        rpe = rpe.reshape(B, n_q, H * W, self.num_heads)
        rpe = rpe.permute(0, 3, 1, 2)

        return rpe


class GlobalDecoderLayer(nn.Module):
    """
    Transformer decoder layer with global cross-attention and BoxRPB
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        rpe_hidden_dim: int,
        feature_stride: int,
        rpe_type: Literal["linear", "log"],
    ) -> None:
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, num_heads, attn_drop=dropout)
        self.cross_attn = GlobalCrossAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rpe_hidden_dim=rpe_hidden_dim,
            feature_stride=feature_stride,
            rpe_type=rpe_type,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        memory_pos: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shape: tuple[int, int],
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt2 = self.norm1(tgt)
        qk = tgt2 + query_pos
        tgt2 = self.self_attn(qk, qk, tgt2)
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.cross_attn(
            query=self.norm2(tgt) + query_pos,
            key=memory + memory_pos,
            value=memory,
            reference_points=reference_points,
            spatial_shape=spatial_shape,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tgt)))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


class GlobalDecoder(nn.Module):
    def __init__(
        self, decoder_layer: nn.Module, num_layers: int, norm: nn.Module, return_intermediate: bool, d_model: int
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.d_model = d_model

        self.bbox_embed: Optional[nn.ModuleList] = None
        self.class_embed: Optional[nn.ModuleList] = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

        for m in self.modules():
            if m is not self and hasattr(m, "reset_parameters") is True and callable(m.reset_parameters) is True:
                m.reset_parameters()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        memory_pos: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shape: tuple[int, int],
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = tgt
        intermediate = []
        intermediate_reference_points = []

        if self.bbox_embed is not None:
            for layer, bbox_embed in zip(self.layers, self.bbox_embed):
                reference_points_input = reference_points.detach().clamp(0, 1)

                output = layer(
                    output,
                    memory,
                    query_pos=query_pos,
                    memory_pos=memory_pos,
                    reference_points=reference_points_input,
                    spatial_shape=spatial_shape,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

                output_for_pred = self.norm(output)
                tmp = bbox_embed(output_for_pred)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

                if self.return_intermediate is True:
                    intermediate.append(output_for_pred)
                    intermediate_reference_points.append(new_reference_points)

            if self.return_intermediate is True:
                return torch.stack(intermediate), torch.stack(intermediate_reference_points)

            return output_for_pred.unsqueeze(0), new_reference_points.unsqueeze(0)

        for layer in self.layers:
            reference_points_input = reference_points.detach().clamp(0, 1)

            output = layer(
                output,
                memory,
                query_pos=query_pos,
                memory_pos=memory_pos,
                reference_points=reference_points_input,
                spatial_shape=spatial_shape,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            output_for_pred = self.norm(output)

            if self.return_intermediate is True:
                intermediate.append(output_for_pred)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate is True:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output_for_pred.unsqueeze(0), reference_points.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, attn_drop=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(
        self, src: torch.Tensor, pos: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        src2 = self.norm1(src)
        q = src2 + pos
        k = src2 + pos

        src2 = self.self_attn(q, k, src2, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out, pos=pos, src_key_padding_mask=mask)

        return out


# pylint: disable=invalid-name
class Plain_DETR(DetectionBaseNet):
    default_size = (640, 640)
    block_group_regex = r"encoder\.layers\.(\d+)|decoder\.layers\.(\d+)"

    # pylint: disable=too-many-locals
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

        # Sigmoid based classification (like multi-label networks)
        self.num_classes = self.num_classes - 1

        hidden_dim = 256
        num_heads = 8
        dropout = 0.0
        return_intermediate = True
        dim_feedforward: int = self.config.get("dim_feedforward", 2048)
        num_encoder_layers: int = self.config["num_encoder_layers"]
        num_decoder_layers: int = self.config["num_decoder_layers"]
        num_queries_one2one: int = self.config.get("num_queries_one2one", 300)
        num_queries_one2many: int = self.config.get("num_queries_one2many", 0)
        k_one2many: int = self.config.get("k_one2many", 6)
        lambda_one2many: float = self.config.get("lambda_one2many", 1.0)
        rpe_hidden_dim: int = self.config.get("rpe_hidden_dim", 512)
        rpe_type: Literal["linear", "log"] = self.config.get("rpe_type", "linear")
        box_refine: bool = self.config.get("box_refine", True)
        soft_nms: bool = self.config.get("soft_nms", False)

        self.soft_nms = None
        if soft_nms is True:
            self.soft_nms = SoftNMS()

        self.hidden_dim = hidden_dim
        self.num_queries_one2one = num_queries_one2one
        self.num_queries_one2many = num_queries_one2many
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many
        self.box_refine = box_refine
        self.num_queries = self.num_queries_one2one + self.num_queries_one2many
        if hasattr(self.backbone, "max_stride") is True:
            self.feature_stride = self.backbone.max_stride
        else:
            self.feature_stride = 32

        if num_encoder_layers == 0:
            self.encoder = None
        else:
            encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward, dropout)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = GlobalDecoderLayer(
            hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            rpe_hidden_dim=rpe_hidden_dim,
            feature_stride=self.feature_stride,
            rpe_type=rpe_type,
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = GlobalDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate,
            d_model=hidden_dim,
        )

        self.class_embed = nn.Linear(hidden_dim, self.num_classes)
        self.bbox_embed = MLP(hidden_dim, [hidden_dim, hidden_dim, 4], activation_layer=nn.ReLU)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim * 2)
        self.reference_point_head = MLP(hidden_dim, [hidden_dim, hidden_dim, 4], activation_layer=nn.ReLU)
        self.input_proj = nn.Conv2d(
            self.backbone.return_channels[-1], hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)

        if box_refine is True:
            self.class_embed = _get_clones(self.class_embed, num_decoder_layers)
            self.bbox_embed = _get_clones(self.bbox_embed, num_decoder_layers)
            self.decoder.bbox_embed = self.bbox_embed
        else:
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_decoder_layers)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_decoder_layers)])

        if self.export_mode is False:
            self.forward = torch.compiler.disable(recursive=False)(self.forward)  # type: ignore[method-assign]

        # Weights initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_embed in self.class_embed:
            nn.init.constant_(class_embed.bias, bias_value)

        for idx, bbox_embed in enumerate(self.bbox_embed):
            last_linear = [m for m in bbox_embed.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)
            if idx == 0:
                nn.init.constant_(last_linear.bias[2:], -2.0)  # Small initial wh

        ref_last_linear = [m for m in self.reference_point_head.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(ref_last_linear.weight)
        nn.init.zeros_(ref_last_linear.bias)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        num_decoder_layers = len(self.class_embed)
        self.class_embed = nn.ModuleList([nn.Linear(self.hidden_dim, num_classes) for _ in range(num_decoder_layers)])

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_embed in self.class_embed:
            nn.init.constant_(class_embed.bias, bias_value)

    def freeze(self, freeze_classifier: bool = True) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

        if freeze_classifier is False:
            for param in self.class_embed.parameters():
                param.requires_grad_(True)

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

        target_classes_onehot = torch.zeros(
            cls_logits.size(0),
            cls_logits.size(1),
            cls_logits.size(2) + 1,
            dtype=cls_logits.dtype,
            device=cls_logits.device,
        )
        target_classes_onehot[idx[0], idx[1], target_classes_o] = 1
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss = sigmoid_focal_loss(cls_logits, target_classes_onehot, alpha=0.25, gamma=2.0)
        loss_ce = (loss.mean(1).sum() / num_boxes) * cls_logits.size(1)

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

    @torch.jit.unused  # type: ignore[untyped-decorator]
    @torch.compiler.disable()  # type: ignore[untyped-decorator]
    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        cls_logits: torch.Tensor,
        box_output: torch.Tensor,
        cls_logits_one2many: Optional[torch.Tensor] = None,
        box_output_one2many: Optional[torch.Tensor] = None,
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
            loss_bbox_i, loss_giou_i = self._box_loss(box_output[idx], targets, indices, num_boxes)
            loss_ce_list.append(loss_ce_i)
            loss_bbox_list.append(loss_bbox_i)
            loss_giou_list.append(loss_giou_i)

        loss_ce = torch.stack(loss_ce_list).sum() * 2
        loss_bbox = torch.stack(loss_bbox_list).sum() * 5
        loss_giou = torch.stack(loss_giou_list).sum() * 2

        # One2many loss (hybrid matching)
        if cls_logits_one2many is not None and box_output_one2many is not None:
            targets_one2many = [
                {"boxes": t["boxes"].repeat(self.k_one2many, 1), "labels": t["labels"].repeat(self.k_one2many)}
                for t in targets
            ]
            num_boxes_one2many = num_boxes * self.k_one2many

            loss_ce_list_one2many = []
            loss_bbox_list_one2many = []
            loss_giou_list_one2many = []
            for idx in range(cls_logits_one2many.size(0)):
                indices = self.matcher(cls_logits_one2many[idx], box_output_one2many[idx], targets_one2many)
                loss_ce_i = self._class_loss(cls_logits_one2many[idx], targets_one2many, indices, num_boxes_one2many)
                loss_bbox_i, loss_giou_i = self._box_loss(
                    box_output_one2many[idx], targets_one2many, indices, num_boxes_one2many
                )
                loss_ce_list_one2many.append(loss_ce_i)
                loss_bbox_list_one2many.append(loss_bbox_i)
                loss_giou_list_one2many.append(loss_giou_i)

            loss_ce += torch.stack(loss_ce_list_one2many).sum() * 2 * self.lambda_one2many
            loss_bbox += torch.stack(loss_bbox_list_one2many).sum() * 5 * self.lambda_one2many
            loss_giou += torch.stack(loss_giou_list_one2many).sum() * 2 * self.lambda_one2many

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
        scores, labels = prob.max(-1)
        labels = labels + 1  # Background offset

        # TorchScript doesn't support creating tensor from tuples, convert everything to lists
        target_sizes = torch.tensor([list(s) for s in image_shapes], device=class_logits.device)

        # Convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_convert(box_regression, in_fmt="cxcywh", out_fmt="xyxy")

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        detections: list[dict[str, torch.Tensor]] = []
        for s, l, b in zip(scores, labels, boxes):
            # Non-maximum suppression
            if self.soft_nms is not None:
                soft_scores, keep = self.soft_nms(b, s, l, score_threshold=0.001)
                s[keep] = soft_scores

                b = b[keep]
                s = s[keep]
                l = l[keep]  # noqa: E741

            detections.append(
                {
                    "boxes": b,
                    "scores": s,
                    "labels": l,
                }
            )

        return detections

    # pylint: disable=too-many-locals
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
        masks: Optional[torch.Tensor] = None,
        image_sizes: Optional[list[list[int]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        self._input_check(targets)
        images = self._to_img_list(x, image_sizes)

        features: dict[str, torch.Tensor] = self.backbone.detection_features(x)
        src = features[self.backbone.return_stages[-1]]
        src = self.input_proj(src)
        B, _, H, W = src.size()

        if masks is not None:
            masks = F.interpolate(masks[None].float(), size=(H, W), mode="nearest").to(torch.bool)[0]
            mask_flatten = masks.flatten(1)
        else:
            mask_flatten = None

        pos = self.pos_enc(src, masks)
        src = src.flatten(2).permute(0, 2, 1)
        pos = pos.flatten(2).permute(0, 2, 1)

        if self.encoder is not None:
            memory = self.encoder(src, pos=pos, mask=mask_flatten)
        else:
            memory = src

        # Use all queries during training, only one2one during inference
        if self.training is True and self.num_queries_one2many > 0:
            num_queries_to_use = self.num_queries_one2one + self.num_queries_one2many
        else:
            num_queries_to_use = self.num_queries_one2one

        query_embed = self.query_embed.weight[:num_queries_to_use]
        query_embed, query_pos = torch.split(query_embed, self.hidden_dim, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)
        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)

        reference_points = self.reference_point_head(query_pos).sigmoid()

        hs, inter_references = self.decoder(
            tgt=query_embed,
            memory=memory,
            query_pos=query_pos,
            memory_pos=pos,
            reference_points=reference_points,
            spatial_shape=(H, W),
            memory_key_padding_mask=mask_flatten,
        )

        outputs_classes = []
        outputs_coords = []
        for lvl, (class_embed, bbox_embed) in enumerate(zip(self.class_embed, self.bbox_embed)):
            outputs_class = class_embed(hs[lvl])
            outputs_classes.append(outputs_class)

            if self.box_refine is True:
                outputs_coord = inter_references[lvl]
            else:
                tmp = bbox_embed(hs[lvl])
                tmp = tmp + inverse_sigmoid(reference_points)
                outputs_coord = tmp.sigmoid()

            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        losses = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training is True:
            assert targets is not None, "targets should not be none when in training mode"

            for idx, target in enumerate(targets):
                boxes = target["boxes"]
                boxes = box_ops.box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
                boxes = boxes / torch.tensor(images.image_sizes[idx][::-1] * 2, dtype=torch.float32, device=x.device)
                targets[idx]["boxes"] = boxes
                targets[idx]["labels"] = target["labels"] - 1  # No background

            # Split outputs for one2one and one2many
            outputs_class_one2one = outputs_class[:, :, : self.num_queries_one2one]
            outputs_coord_one2one = outputs_coord[:, :, : self.num_queries_one2one]

            if self.num_queries_one2many > 0:
                outputs_class_one2many = outputs_class[:, :, self.num_queries_one2one :]
                outputs_coord_one2many = outputs_coord[:, :, self.num_queries_one2one :]
            else:
                outputs_class_one2many = None
                outputs_coord_one2many = None

            losses = self.compute_loss(
                targets, outputs_class_one2one, outputs_coord_one2one, outputs_class_one2many, outputs_coord_one2many
            )

        else:
            detections = self.postprocess_detections(outputs_class[-1], outputs_coord[-1], images.image_sizes)

        return (detections, losses)


registry.register_model_config(
    "plain_detr_lite",
    Plain_DETR,
    config={"num_encoder_layers": 1, "num_decoder_layers": 3, "box_refine": False},
)
registry.register_model_config(
    "plain_detr",
    Plain_DETR,
    config={"num_encoder_layers": 0, "num_decoder_layers": 6, "num_queries_one2many": 1500},
)
