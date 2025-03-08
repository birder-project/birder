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
from birder.net.vit import Encoder


class SequentialTuple(nn.Sequential):
    def forward(  # pylint: disable=arguments-renamed
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for module in self:
            x = module(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        base_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        pool: Optional[Callable[..., nn.Module]],
        proj_drop: float,
        attn_drop: float,
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
            dpr=drop_path_prob,
        )

    def forward(self, xt: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        (x, cls_tokens) = xt
        token_length = cls_tokens.shape[1]
        if self.pool is not None:
            (x, cls_tokens) = self.pool(x, cls_tokens)

        (B, C, H, W) = x.size()
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
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is not None, "must set config"

        image_size = self.size
        patch_size: tuple[int, int] = self.config["patch_size"]
        patch_stride: tuple[int, int] = self.config["patch_stride"]
        depths: list[int] = self.config["depths"]
        base_dims: list[int] = self.config["base_dims"]
        heads: list[int] = self.config["heads"]
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

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = embed_dim

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i, depth in enumerate(depths):
            pool = None
            embed_dim = base_dims[i] * heads[i]
            if i > 0:
                pool = Pooling(prev_dim, embed_dim)

            stages[f"stage{i+1}"] = Transformer(
                base_dims[i],
                depth,
                heads=heads[i],
                mlp_ratio=4.0,
                pool=pool,
                proj_drop=0.0,
                attn_drop=0.0,
                drop_path_prob=dpr[i],
            )
            prev_dim = embed_dim
            return_channels.append(embed_dim)

        self.body = SequentialTuple(stages)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.return_stages = self.return_stages[: len(depths)]
        self.return_channels = return_channels
        self.embedding_size = embed_dim
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()
        self.distillation_output = False

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad = True

            for param in self.dist_classifier.parameters():
                param.requires_grad = True

        if unfreeze_features is True:
            for param in self.norm.parameters():
                param.requires_grad = True

    def transform_to_backbone(self) -> None:
        self.norm = nn.Identity()
        self.classifier = nn.Identity()
        self.dist_classifier = nn.Identity()

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        out = {}
        for name, module in self.body.named_children():
            (x, cls_tokens) = module((x, cls_tokens))
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.stem.parameters():
            param.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        (x, cls_tokens) = self.body((x, cls_tokens))
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

        self.pos_embed = nn.Parameter(
            F.interpolate(self.pos_embed.data, (height, width), mode="bicubic"), requires_grad=True
        )


registry.register_alias(
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
registry.register_alias(
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
registry.register_alias(
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
registry.register_alias(
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

registry.register_weights(
    "pit_t_il-common",
    {
        "description": "PiT tiny model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 18.4,
                "sha256": "8a1d98a2b7388e1efa14bbee89fc182b588a35d239544de521769c2d850ee5fe",
            }
        },
        "net": {"network": "pit_t", "tag": "il-common"},
    },
)
