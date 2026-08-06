"""
Pooling-based Vision Transformer (PiT), adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/pit.py

Paper "Rethinking Spatial Dimensions of Vision Transformers", https://arxiv.org/abs/2103.16302

Changes from original:
* Always using distillation token (same as DeiT)
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import staged_stochastic_depth_rates
from birder.net.vit import Encoder


class SequentialTuple(nn.Sequential):
    def forward(  # pylint: disable=arguments-renamed
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for module in self:
            x = module(*x)

        return x


class PiTStage(nn.Module):
    def __init__(
        self,
        base_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        pool: Optional[Callable[..., nn.Module]],
        proj_drop: float,
        attn_drop: float,
        projection_dropout: float,
        drop_path_prob: list[float],
    ) -> None:
        super().__init__()
        embed_dim = base_dim * heads
        self.pool = pool
        self.encoder = Encoder(
            num_layers=depth,
            num_heads=heads,
            hidden_dim=embed_dim,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout=proj_drop,
            attention_dropout=attn_drop,
            projection_dropout=projection_dropout,
            dpr=drop_path_prob,
        )

    def forward(self, x: torch.Tensor, cls_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_length = cls_tokens.size(1)
        if self.pool is not None:
            x, cls_tokens = self.pool(x, cls_tokens)

        B, C, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = torch.concat((cls_tokens, x), dim=1)
        x = self.encoder(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = x.transpose(1, 2).reshape(B, C, H, W)

        return (x, cls_tokens)


class Pooling(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.conv = nn.Conv2d(
            in_features,
            out_features,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=in_features,
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, cls_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return (x, cls_token)


class PiT(DetectorBackbone):
    block_group_regex = r"body\.stage(\d+)\.encoder\.block\.(\d+)"

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, config=config, size=size)
        assert self.config is not None, "must set config"

        image_size = self.size
        patch_size: tuple[int, int] = self.config["patch_size"]
        patch_stride: tuple[int, int] = self.config["patch_stride"]
        depths: list[int] = self.config["depths"]
        base_dims: list[int] = self.config["base_dims"]
        heads: list[int] = self.config["heads"]
        dropout: float = self.config.get("dropout", 0.0)
        attention_dropout: float = self.config.get("attention_dropout", 0.0)
        projection_dropout: float = self.config.get("projection_dropout", 0.0)
        drop_path_rate: float = self.config["drop_path_rate"]

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        embed_dim = base_dims[0] * heads[0]
        height = (image_size[0] - patch_size[0]) // patch_stride[0] + 1
        width = (image_size[1] - patch_size[1]) // patch_stride[1] + 1

        self.stem = nn.Conv2d(
            self.input_channels, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=(0, 0)
        )

        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, height, width))
        self.cls_token = nn.Parameter(torch.randn(1, 2, embed_dim))

        dpr = staged_stochastic_depth_rates(drop_path_rate, depths)
        prev_dim = embed_dim

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i, depth in enumerate(depths):
            pool = None
            embed_dim = base_dims[i] * heads[i]
            if i > 0:
                pool = Pooling(prev_dim, embed_dim)

            stages[f"stage{i+1}"] = PiTStage(
                base_dims[i],
                depth,
                heads=heads[i],
                mlp_ratio=4.0,
                pool=pool,
                proj_drop=dropout,
                attn_drop=attention_dropout,
                projection_dropout=projection_dropout,
                drop_path_prob=dpr[i],
            )
            prev_dim = embed_dim
            return_channels.append(embed_dim)

        self.body = SequentialTuple(stages)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.return_stages = [f"stage{idx + 1}" for idx in range(len(depths))]
        self.return_channels = return_channels
        self.embedding_size = embed_dim
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()
        self.distillation_output = False

        self.max_stride = patch_stride[0] * 2 ** (len(depths) - 1)
        self.stem_stride = patch_stride[0]
        self.stem_width = base_dims[0] * heads[0]
        self.feature_dim = embed_dim

        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad_(True)

            for param in self.dist_classifier.parameters():
                param.requires_grad_(True)

        if unfreeze_features is True:
            for param in self.norm.parameters():
                param.requires_grad_(True)

    def strip_for_forward_features(self) -> None:
        super().strip_for_forward_features()
        self.norm = nn.Identity()
        self.dist_classifier = nn.Identity()

    def strip_for_detection_features(self) -> None:
        self.strip_for_forward_features()

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        out = {}
        for name, module in self.body.named_children():
            x, cls_tokens = module(x, cls_tokens)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        self.pos_embed.requires_grad_(False)
        for param in self.stem.parameters():
            param.requires_grad_(False)

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad_(False)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        for stage in self.body.children():
            x, cls_tokens = stage(x, cls_tokens)

        return (x, cls_tokens)

    def flatten_features(
        self, features: tuple[torch.Tensor, torch.Tensor], include_special_tokens: bool = True
    ) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__} does not support non-tensor features")

    def embedding_from_features(self, features: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        _, cls_tokens = features
        cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def set_distillation_output(self, enable: bool = True) -> None:
        self.distillation_output = enable

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        x_cls = x[:, 0]
        x_dist = x[:, 1]

        x_cls = self.classifier(x_cls)
        x_dist = self.dist_classifier(x_dist)

        if self.training is True and self.distillation_output is True:
            x = torch.stack([x_cls, x_dist], dim=1)
        else:
            # Classifier "token" as an average of both tokens (during normal training or inference)
            x = (x_cls + x_dist) / 2

        return x

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)

        height = (new_size[0] - self.patch_size[0]) // self.patch_stride[0] + 1
        width = (new_size[1] - self.patch_size[1]) // self.patch_stride[1] + 1

        with torch.no_grad():
            pos_embed = F.interpolate(self.pos_embed, (height, width), mode="bicubic", antialias=True)

        self.pos_embed = nn.Parameter(pos_embed)


registry.register_model_config(
    "pit_t",
    PiT,
    config={
        "patch_size": (16, 16),
        "patch_stride": (8, 8),
        "depths": [2, 6, 4],
        "base_dims": [32, 32, 32],
        "heads": [2, 4, 8],
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "pit_xs",
    PiT,
    config={
        "patch_size": (16, 16),
        "patch_stride": (8, 8),
        "depths": [2, 6, 4],
        "base_dims": [48, 48, 48],
        "heads": [2, 4, 8],
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "pit_s",
    PiT,
    config={
        "patch_size": (16, 16),
        "patch_stride": (8, 8),
        "depths": [2, 6, 4],
        "base_dims": [48, 48, 48],
        "heads": [3, 6, 12],
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "pit_b",
    PiT,
    config={
        "patch_size": (14, 14),
        "patch_stride": (7, 7),
        "depths": [3, 6, 4],
        "base_dims": [64, 64, 64],
        "heads": [4, 8, 16],
        "drop_path_rate": 0.1,
    },
)
