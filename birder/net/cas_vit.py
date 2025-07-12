"""
CAS-ViT, adapted from
https://github.com/Tianfang-Zhang/CAS-ViT/blob/main/classification/model/rcvit.py

Paper "CAS-ViT: Convolutional Additive Self-attention Vision Transformers for Efficient Mobile Applications",
https://arxiv.org/abs/2408.03703

Changes from original:
* Removed biases before norms
"""

# Reference license: MIT

from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class ConvMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SpatialOperation(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class ChannelOperation(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class LocalIntegration(nn.Module):
    def __init__(self, dim: int, ratio: float, act_layer: Callable[..., nn.Module] = nn.GELU) -> None:
        super().__init__()
        mid_dim = round(ratio * dim)
        self.block = nn.Sequential(
            nn.Conv2d(dim, mid_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AdditiveTokenMixer(nn.Module):
    def __init__(self, dim: int, qkv_bias: bool, proj_drop: float) -> None:
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=qkv_bias)
        self.op_q = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.op_k = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.dwc = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim)

        self.proj = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (q, k, v) = self.qkv(x).chunk(3, dim=1)
        q = self.op_q(q)
        k = self.op_k(k)

        x = self.proj(self.dwc(q + k) * v)
        x = self.proj_drop(x)

        return x


class AdditiveBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_drop: float,
        drop_path: float,
        act_layer: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.local_perception = LocalIntegration(dim, ratio=1.0, act_layer=act_layer)
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = AdditiveTokenMixer(dim, qkv_bias=qkv_bias, proj_drop=proj_drop)

        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = ConvMLP(dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer)

        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local_perception(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# pylint: disable=invalid-name
class CAS_ViTStage(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_drop: float,
        drop_path: list[float],
        depth: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = Downsample(dim, out_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        layers = []
        for i in range(depth):
            layers.append(
                AdditiveBlock(
                    out_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_drop=proj_drop, drop_path=drop_path[i]
                )
            )

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


# pylint: disable=invalid-name
class CAS_ViT(DetectorBackbone):
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

        mlp_ratio = 4.0
        qkv_bias = False
        proj_drop = 0.0
        depths: list[int] = self.config["depths"]
        embed_dims: list[int] = self.config["embed_dims"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels, embed_dims[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            ),
            Conv2dNormActivation(embed_dims[0] // 2, embed_dims[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        num_stages = len(depths)

        prev_dim = embed_dims[0]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = CAS_ViTStage(
                prev_dim,
                embed_dims[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                drop_path=dpr[i],
                depth=depths[i],
                downsample=i > 0,
            )
            return_channels.append(embed_dims[i])
            prev_dim = embed_dims[i]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.BatchNorm2d(embed_dims[-1]),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embed_dims[-1]
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()
        self.distillation_output = False

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
            for param in self.features.parameters():
                param.requires_grad = True

    def transform_to_backbone(self) -> None:
        self.features = nn.Identity()
        self.classifier = nn.Identity()
        self.dist_classifier = nn.Identity()

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)

        out = {}
        for name, module in self.body.named_children():
            x = module(x)
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)

    def set_distillation_output(self, enable: bool = True) -> None:
        self.distillation_output = enable

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        x_cls = self.classifier(x)
        x_dist = self.dist_classifier(x)

        if self.training is True and self.distillation_output is True:
            x = torch.stack([x_cls, x_dist], dim=1)
        else:
            # Classifier "token" as an average of both tokens (during normal training or inference)
            x = (x_cls + x_dist) / 2

        return x


registry.register_model_config(
    "cas_vit_xs", CAS_ViT, config={"depths": [2, 2, 4, 2], "embed_dims": [48, 56, 112, 220], "drop_path_rate": 0.0}
)
registry.register_model_config(
    "cas_vit_s", CAS_ViT, config={"depths": [3, 3, 6, 3], "embed_dims": [48, 64, 128, 256], "drop_path_rate": 0.0}
)
registry.register_model_config(
    "cas_vit_m", CAS_ViT, config={"depths": [3, 3, 6, 3], "embed_dims": [64, 96, 192, 384], "drop_path_rate": 0.0}
)
registry.register_model_config(
    "cas_vit_t", CAS_ViT, config={"depths": [3, 3, 6, 3], "embed_dims": [96, 128, 256, 512], "drop_path_rate": 0.0}
)

registry.register_weights(
    "cas_vit_xs_il-common",
    {
        "description": "CAS-ViT extra small model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 11.4,
                "sha256": "e8d1c09f98a1a33334e7281d471684d260b312ac02d467bb6740e4a98d00daa8",
            }
        },
        "net": {"network": "cas_vit_xs", "tag": "il-common"},
    },
)
