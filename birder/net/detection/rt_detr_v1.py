"""
RT-DETR (Real-Time DEtection TRansformer), adapted from
https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch

Paper "DETRs Beat YOLOs on Real-time Object Detection", https://arxiv.org/abs/2304.08069

Note: The paper describes uncertainty-minimal query selection, but the upstream repo
selects top-K queries by classification confidence. We follow the repo behavior here.
"""

# Reference license: Apache-2.0

import copy
import math
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import boxes as box_ops

from birder.common import training_utils
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import pos_embedding_sin_cos_2d
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.deformable_detr import DeformableTransformerDecoderLayer
from birder.net.detection.deformable_detr import HungarianMatcher
from birder.net.detection.deformable_detr import inverse_sigmoid
from birder.net.repvgg import RepVggBlock


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


@torch.jit.unused  # type: ignore[untyped-decorator]
def get_contrastive_denoising_training_group(  # pylint: disable=too-many-locals
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    num_queries: int,
    class_embed: nn.Embedding,
    num_denoising_queries: int,
    label_noise_ratio: float,
    box_noise_scale: float,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[dict[str, Any]]]:
    if num_denoising_queries <= 0:
        return (None, None, None, None)

    num_ground_truths = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device

    max_gt_num = max(num_ground_truths)
    if max_gt_num == 0:
        return (None, None, None, None)

    num_groups = num_denoising_queries // max_gt_num
    num_groups = 1 if num_groups == 0 else num_groups

    # Pad ground truth to max_num of a batch
    batch_size = len(num_ground_truths)

    input_query_class = torch.full([batch_size, max_gt_num], num_classes, dtype=torch.long, device=device)
    input_query_bbox = torch.zeros([batch_size, max_gt_num, 4], dtype=targets[0]["boxes"].dtype, device=device)
    pad_gt_mask = torch.zeros([batch_size, max_gt_num], dtype=torch.bool, device=device)

    for i in range(batch_size):
        num_gt = num_ground_truths[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"].long()
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = True

    input_query_class = input_query_class.tile([1, 2 * num_groups])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_groups, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_groups])

    # Positive and negative mask
    negative_gt_mask = torch.zeros([batch_size, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_groups, 1])
    positive_gt_mask = 1 - negative_gt_mask

    # Contrastive denoising training positive index
    positive_gt_mask_squeezed = positive_gt_mask.squeeze(-1) * pad_gt_mask
    denoise_positive_idx = torch.nonzero(positive_gt_mask_squeezed)[:, 1]
    denoise_positive_idx = torch.split(denoise_positive_idx, [n * num_groups for n in num_ground_truths])

    # Total denoising queries
    total_denoising_queries = int(max_gt_num * 2 * num_groups)

    # Apply label noise
    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint(0, num_classes, input_query_class.shape, dtype=torch.long, device=device)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    # Apply box noise
    if box_noise_scale > 0:
        known_bbox = box_ops.box_convert(input_query_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox = known_bbox + rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = box_ops.box_convert(known_bbox, in_fmt="xyxy", out_fmt="cxcywh")
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    # Embed class labels
    input_query_class = class_embed(input_query_class)

    # Create attention mask
    target_size = total_denoising_queries + num_queries
    attn_mask = torch.zeros([target_size, target_size], dtype=torch.bool, device=device)
    attn_mask[total_denoising_queries:, :total_denoising_queries] = True
    for i in range(num_groups):
        idx_block_start = max_gt_num * 2 * i
        idx_block_end = max_gt_num * 2 * (i + 1)
        attn_mask[idx_block_start:idx_block_end, :idx_block_start] = True
        attn_mask[idx_block_start:idx_block_end, idx_block_end:total_denoising_queries] = True

    denoising_meta_values = {
        "dn_positive_idx": denoise_positive_idx,
        "dn_num_group": num_groups,
        "dn_num_split": [total_denoising_queries, num_queries],
    }

    return (input_query_class, input_query_bbox, attn_mask, denoising_meta_values)


def varifocal_loss(
    pred_logits: torch.Tensor,
    target_score: torch.Tensor,
    target_onehot: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    VariFocal Loss - uses IoU as soft target for positives, focal-like weighting for negatives
    """

    pred_score = pred_logits.sigmoid().detach()
    weight = alpha * pred_score.pow(gamma) * (1 - target_onehot) + target_score
    loss = F.binary_cross_entropy_with_logits(pred_logits, target_score, weight=weight, reduction="none")

    return loss


class CSPRepLayer(nn.Module):
    """
    Cross Stage Partial layer with RepVGG blocks
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, expansion: float) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = Conv2dNormActivation(
            in_channels,
            hidden_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=nn.SiLU,
        )
        self.conv2 = Conv2dNormActivation(
            in_channels,
            hidden_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=nn.SiLU,
        )
        self.bottlenecks = nn.Sequential(
            *[
                RepVggBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=1,
                    use_se=False,
                    reparameterized=False,
                    act_layer=nn.SiLU,
                    skip_identity=True,
                )
                for _ in range(num_blocks)
            ]
        )

        if hidden_channels != out_channels:
            self.conv3: nn.Module = Conv2dNormActivation(
                hidden_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation_layer=nn.SiLU,
            )
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.bottlenecks(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(x1 + x2)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(
        self, src: torch.Tensor, pos: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q = src + pos
        k = src + pos

        (src2, _) = self.self_attn(q, k, value=src, key_padding_mask=key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class AIFI(nn.Module):
    """
    Attention-based Intra-scale Feature Interaction.
    Applies transformer encoder to a single feature scale.
    """

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        self.d_model = d_model
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.use_cache = True
        self._pos_cache: dict[str, torch.Tensor] = {}

    def set_cache_enabled(self, enabled: bool) -> None:
        self.use_cache = enabled
        if enabled is False:
            self.clear_cache()

    def clear_cache(self) -> None:
        self._pos_cache.clear()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        (B, C, H, W) = x.size()
        x = x.flatten(2).permute(0, 2, 1)

        use_cache = self.use_cache is True and torch.jit.is_tracing() is False and torch.jit.is_scripting() is False
        if use_cache is True:
            cache_key = f"{H}x{W}x{C}_{x.device}_{x.dtype}"
            pos = self._pos_cache.get(cache_key)
            if pos is None:
                pos = pos_embedding_sin_cos_2d(H, W, C, num_special_tokens=0, device=x.device).unsqueeze(0)
                self._pos_cache[cache_key] = pos
        else:
            pos = pos_embedding_sin_cos_2d(H, W, C, num_special_tokens=0, device=x.device).unsqueeze(0)

        key_padding_mask = mask.flatten(1) if mask is not None else None
        for layer in self.layers:
            x = layer(x, pos, key_padding_mask=key_padding_mask)

        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        return x


class HybridEncoder(nn.Module):
    """
    RT-DETR Hybrid Encoder: AIFI + CCFM (CNN-based Cross-scale Feature-fusion Module).

    Architecture:
    1. Input projection for each scale
    2. AIFI (transformer) on the highest resolution feature
    3. Top-down FPN path with CSPRepLayers
    4. Bottom-up PAN path with CSPRepLayers
    """

    def __init__(
        self,
        in_channels: list[int],
        hidden_dim: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_heads: int,
        expansion: float,
        depth_multiplier: float,
    ) -> None:
        super().__init__()
        self.num_levels = len(in_channels)

        # Input projection layers
        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(ch, hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
            )

        # AIFI encoder
        self.encoder = AIFI(
            hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_encoder_layers,
        )

        # CCFM: Top-down FPN path
        num_blocks = max(round(3 * depth_multiplier), 1)
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(self.num_levels - 1, 0, -1):
            self.lateral_convs.append(
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    activation_layer=nn.SiLU,
                )
            )
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, num_blocks=num_blocks, expansion=expansion))

        # CCFM: Bottom-up PAN path
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.downsample_convs.append(
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    activation_layer=nn.SiLU,
                )
            )
            self.pan_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, num_blocks=num_blocks, expansion=expansion))

    def forward(self, feats: list[torch.Tensor], masks: Optional[list[torch.Tensor]] = None) -> list[torch.Tensor]:
        # Project all features to hidden_dim
        proj_feats = []
        for idx, proj in enumerate(self.input_proj):
            proj_feats.append(proj(feats[idx]))

        # Apply AIFI to highest scale feature
        encoder_mask = masks[-1] if masks is not None else None
        proj_feats[-1] = self.encoder(proj_feats[-1], mask=encoder_mask)

        # Top-down FPN path
        inner_outs = [proj_feats[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            feat_high = inner_outs[0]
            feat_low = proj_feats[self.num_levels - 2 - idx]

            feat_high = lateral_conv(feat_high)
            inner_outs[0] = feat_high
            feat_high = F.interpolate(feat_high, size=feat_low.shape[-2:], mode="nearest")

            inner_out = fpn_block(torch.concat([feat_high, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # Bottom-up PAN path
        outs = [inner_outs[0]]
        for idx, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            feat_low = downsample_conv(feat_low)
            out = pan_block(torch.concat([feat_low, feat_high], dim=1))
            outs.append(out)

        return outs

    def set_cache_enabled(self, enabled: bool) -> None:
        self.encoder.set_cache_enabled(enabled)

    def clear_cache(self) -> None:
        self.encoder.clear_cache()


# pylint: disable=invalid-name
class RT_DETRDecoder(nn.Module):
    """
    RT-DETR Decoder with uncertainty-minimal query selection.

    Key difference from Deformable DETR: queries are selected from encoder output
    based on classification confidence, rather than using learned query embeddings.
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
        num_decoder_points: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, [hidden_dim, hidden_dim, 4], activation_layer=nn.ReLU)

        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, dim_feedforward, dropout, num_levels, num_heads, num_decoder_points
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
            nn.init.zeros_(bbox_embed[-2].weight.data)
            nn.init.zeros_(bbox_embed[-2].bias.data)

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
        (anchors, valid_mask) = self._generate_anchors(spatial_shapes, device=memory.device, dtype=memory.dtype)
        if memory_padding_mask is not None:
            valid_mask = valid_mask & ~memory_padding_mask.unsqueeze(-1)

        memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        if memory_padding_mask is not None:
            enc_outputs_class = enc_outputs_class.masked_fill(memory_padding_mask[..., None], float("-inf"))

        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        # Select top-k queries based on classification confidence
        (_, topk_ind) = torch.topk(enc_outputs_class.max(dim=-1).values, self.num_queries, dim=1)

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
            feat = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            memory.append(feat)
            if padding_mask is not None:
                mask_flatten.append(padding_mask[idx].flatten(1))

        memory = torch.concat(memory, dim=1)
        memory_padding_mask = torch.concat(mask_flatten, dim=1) if mask_flatten else None

        # Get decoder input (query selection)
        (target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits) = self._get_decoder_input(
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
class RT_DETR_v1(DetectionBaseNet):
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

        hidden_dim = 256
        num_heads = 8
        dim_feedforward = 1024
        num_decoder_points = 4
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
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.decoder.class_embed.parameters():
                param.requires_grad = True
            for param in self.decoder.enc_score_head.parameters():
                param.requires_grad = True
            if self.num_denoising > 0:
                for param in self.denoising_class_embed.parameters():
                    param.requires_grad = True

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
            (loss_bbox, loss_giou) = self._box_loss(dn_out_bboxes[layer_idx], targets, indices, dn_num_boxes)

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
            (loss_bbox, loss_giou) = self._box_loss(out_bboxes[layer_idx], targets, indices, num_boxes)
            loss_ce_list.append(loss_ce)
            loss_bbox_list.append(loss_bbox)
            loss_giou_list.append(loss_giou)

        # Encoder auxiliary loss
        enc_indices = self.matcher(enc_topk_logits, enc_topk_bboxes, targets)
        loss_ce_enc = self._class_loss(enc_topk_logits, enc_topk_bboxes, targets, enc_indices, num_boxes)
        (loss_bbox_enc, loss_giou_enc) = self._box_loss(enc_topk_bboxes, targets, enc_indices, num_boxes)
        loss_ce_list.append(loss_ce_enc)
        loss_bbox_list.append(loss_bbox_enc)
        loss_giou_list.append(loss_giou_enc)

        loss_ce = torch.stack(loss_ce_list).sum()  # VFL weight is 1
        loss_bbox = torch.stack(loss_bbox_list).sum() * 5
        loss_giou = torch.stack(loss_giou_list).sum() * 2

        # Add denoising loss if available
        if dn_out_bboxes is not None and dn_out_logits is not None and dn_meta is not None:
            (loss_ce_dn, loss_bbox_dn, loss_giou_dn) = self._compute_denoising_loss(
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

        (denoising_class, denoising_bbox_unact, attn_mask, dn_meta) = self._prepare_cdn_queries(targets)

        (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits) = self.decoder(
            encoder_features,
            spatial_shapes,
            level_start_index,
            denoising_class,
            denoising_bbox_unact,
            attn_mask,
            masks,
        )

        if dn_meta is not None:
            (dn_num_split, _num_queries) = dn_meta["dn_num_split"]
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
        (topk_values, topk_indexes) = torch.topk(
            prob.view(class_logits.shape[0], -1), k=self.decoder.num_queries, dim=1
        )
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
                (B, _, H, W) = feat.size()
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
            (out_bboxes, out_logits, _, _) = self.decoder(
                encoder_features, spatial_shapes, level_start_index, padding_mask=mask_list
            )
            detections = self.postprocess_detections(out_logits[-1], out_bboxes[-1], images.image_sizes)

        return (detections, losses)

    def reparameterize_model(self) -> None:
        if self.reparameterized is True:
            return

        for module in self.modules():
            if hasattr(module, "reparameterize") is True:
                module.reparameterize()

        self.reparameterized = True


registry.register_model_config(
    "rt_detr_v1_s", RT_DETR_v1, config={"num_decoder_layers": 3, "expansion": 0.5, "depth_multiplier": 0.33}
)
registry.register_model_config("rt_detr_v1", RT_DETR_v1, config={"num_decoder_layers": 6})
