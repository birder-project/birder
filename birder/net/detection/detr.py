"""
DETR (DEtection TRansformer), adapted from
https://github.com/facebookresearch/detr/blob/main/models/detr.py

Paper "End-to-End Object Detection with Transformers", https://arxiv.org/abs/2005.12872

Changes from original:
* Move background index to first from last (to be inline with the rest of Birder detectors)
* Removed masking / padding and nested tensors (images are resized at the dataloader)
* Zero cost matrix elements on overflow (HungarianMatcher)
"""

# Reference license: Apache-2.0

import copy
import math
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import boxes as box_ops

from birder.common import training_utils
from birder.net.base import DetectorBackbone
from birder.net.detection.base import DetectionBaseNet


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
            out_prob = class_logits.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = box_regression.flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.concat([v["labels"] for v in targets], dim=0)
            tgt_bbox = torch.concat([v["boxes"] for v in targets], dim=0)

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - prob[target class].
            # The 1 is a constant that doesn't change the matching, it can be omitted.
            cost_class = -out_prob[:, tgt_ids]

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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        q = src + pos
        k = src + pos

        (src2, _) = self.self_attn(q, k, value=src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

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
        self, tgt: torch.Tensor, memory: torch.Tensor, pos: torch.Tensor, query_pos: torch.Tensor
    ) -> torch.Tensor:
        q = tgt + query_pos
        k = tgt + query_pos

        (tgt2, _) = self.self_attn(q, k, value=tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        (tgt2, _) = self.multihead_attn(query=tgt + query_pos, key=memory + pos, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out, pos=pos)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: nn.Module, return_intermediate: bool) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor, pos: torch.Tensor, query_pos: torch.Tensor
    ) -> torch.Tensor:
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
            if self.return_intermediate is True:
                intermediate.append(self.norm(output))

        if self.return_intermediate is True:
            return torch.stack(intermediate)

        return self.norm(output).unsqueeze(0)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        return_intermediate_dec: bool,
    ) -> None:
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec
        )

        # Weights initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, query_embed: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        # Flatten BxCxHxW to HWxBxC
        B = src.size(0)
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos=pos_embed)
        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1, 2)


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int, temperature: int = 10000, normalize: bool = False) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, _, H, W) = x.shape
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize is True:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.concat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class DETR(DetectionBaseNet):
    default_size = (640, 640)
    auto_register = True

    def __init__(
        self,
        num_classes: int,
        backbone: DetectorBackbone,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(num_classes, backbone, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is None, "config not supported"

        hidden_dim = 256
        nhead = 8
        num_encoder_layers = 6
        num_decoder_layers = 6
        dim_feedforward = 2048
        dropout = 0.1
        num_queries = 100

        self.hidden_dim = hidden_dim
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            return_intermediate_dec=True,
        )

        self.class_embed = nn.Linear(hidden_dim, self.num_classes)
        self.bbox_embed = MLP(hidden_dim, [hidden_dim, hidden_dim, 4], activation_layer=nn.ReLU)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(
            self.backbone.return_channels[-1], hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        self.matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = 0.1
        self.empty_weight = nn.Buffer(empty_weight)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes + 1
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes)

        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = 0.1
        self.empty_weight = nn.Buffer(empty_weight)

    def _get_src_permutation_idx(self, indices: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.concat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.concat([src for (src, _) in indices])
        return (batch_idx, src_idx)

    def _class_loss(
        self,
        cls_logits: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        indices: list[torch.Tensor],
    ) -> torch.Tensor:
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.concat([t["labels"][J] for t, (_, J) in zip(targets, indices)], dim=0)
        target_classes = torch.full(cls_logits.shape[:2], 0, dtype=torch.int64, device=cls_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(cls_logits.transpose(1, 2), target_classes, self.empty_weight)

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
            loss_ce_i = self._class_loss(cls_logits[idx], targets, indices)
            (loss_bbox_i, loss_giou_i) = self._box_loss(box_output[idx], targets, indices, num_boxes)
            loss_ce_list.append(loss_ce_i * 1)
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
        prob = F.softmax(class_logits, -1)
        (scores, labels) = prob[..., 1:].max(-1)
        labels = labels + 1
        target_sizes = torch.tensor(image_shapes, device=class_logits.device)

        # Convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_convert(box_regression, in_fmt="cxcywh", out_fmt="xyxy")

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
        x = features[self.backbone.return_stages[-1]]
        pos = self.pos_enc(x)
        hs = self.transformer(self.input_proj(x), self.query_embed.weight, pos_embed=pos)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        losses = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training:
            assert targets is not None, "targets should not be none when in training mode"

            # Convert target boxes
            for idx, target in enumerate(targets):
                boxes = target["boxes"]
                boxes = box_ops.box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
                boxes = boxes / torch.tensor(image_sizes[idx] * 2, dtype=torch.float32, device=x.device)
                targets[idx]["boxes"] = boxes

            losses = self.compute_loss(targets, outputs_class, outputs_coord)

        else:
            detections = self.postprocess_detections(outputs_class[-1], outputs_coord[-1], image_sizes_list)

        return (detections, losses)
