"""
LeViT, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/levit.py

Paper "LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference", https://arxiv.org/abs/2104.01136

Changes from original:
* Removed attention bias cache
"""

# Reference license: BSD 3-Clause

from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.base import interpolate_attention_bias
from birder.net.vit import PatchEmbed


class LinearNorm(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.bn(x.flatten(0, 1)).reshape_as(x)


class Subsample(nn.Module):
    def __init__(self, stride: int, resolution: tuple[int, int]) -> None:
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, _, C) = x.shape
        x = x.view(B, self.resolution[0], self.resolution[1], C)
        x = x[:, :: self.stride, :: self.stride]
        return x.reshape(B, -1, C)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int,
        attn_ratio: float,
        resolution: tuple[int, int],
        act_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = int(attn_ratio * key_dim) * num_heads

        key_attn_dim = key_dim * num_heads
        self.qkv = LinearNorm(dim, self.val_attn_dim + key_attn_dim * 2)
        self.proj = nn.Sequential(
            act_layer(),
            LinearNorm(self.val_attn_dim, dim),
        )

        self.attention_biases = nn.Parameter(torch.zeros(self.num_heads, resolution[0] * resolution[1]))
        pos = torch.stack(
            torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]), indexing="ij")
        ).flatten(1)
        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
        self.attention_bias_idxs = nn.Buffer(rel_pos, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, _) = x.shape
        q, k, v = self.qkv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        attn = q @ k * self.scale + self.attention_biases[:, self.attention_bias_idxs]
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        x = self.proj(x)

        return x


class AttentionSubsample(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        key_dim: int,
        num_heads: int,
        attn_ratio: float,
        stride: int,
        resolution: tuple[int, int],
        act_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        self.stride = stride
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * self.num_heads
        self.scale = key_dim**-0.5

        key_attn_dim = key_dim * num_heads
        self.kv = LinearNorm(in_dim, self.val_attn_dim + key_attn_dim)
        self.q = nn.Sequential(
            Subsample(stride, resolution),
            LinearNorm(in_dim, key_attn_dim),
        )
        self.proj = nn.Sequential(
            act_layer(),
            LinearNorm(self.val_attn_dim, out_dim),
        )

        self.attention_biases = nn.Parameter(torch.zeros(self.num_heads, resolution[0] * resolution[1]))
        k_pos = torch.stack(
            torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]), indexing="ij")
        ).flatten(1)
        q_pos = torch.stack(
            torch.meshgrid(
                torch.arange(0, resolution[0], step=stride), torch.arange(0, resolution[1], step=stride), indexing="ij"
            )
        ).flatten(1)
        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
        self.attention_bias_idxs = nn.Buffer(rel_pos, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, _) = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
        k = k.permute(0, 2, 3, 1)  # BHCN
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)

        attn = q @ k * self.scale + self.attention_biases[:, self.attention_bias_idxs]
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
        x = self.proj(x)

        return x


class LeVitMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        self.ln1 = LinearNorm(in_features, hidden_features)
        self.act = act_layer()
        self.ln2 = LinearNorm(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = self.act(x)
        x = self.ln2(x)

        return x


class LeVitSubsample(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        key_dim: int,
        num_heads: int,
        attn_ratio: float,
        mlp_ratio: float,
        act_layer: Callable[..., nn.Module],
        resolution: tuple[int, int],
        drop_path: float,
    ) -> None:
        super().__init__()
        self.attn_downsample = AttentionSubsample(
            in_dim=in_dim,
            out_dim=out_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            stride=2,
            resolution=resolution,
            act_layer=act_layer,
        )

        self.mlp = LeVitMLP(out_dim, int(out_dim * mlp_ratio), act_layer=act_layer)
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_downsample(x)
        x = x + self.drop_path(self.mlp(x))
        return x


class LeVitBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int,
        attn_ratio: float,
        mlp_ratio: float,
        resolution: tuple[int, int],
        act_layer: Callable[..., nn.Module],
        drop_path: float,
    ) -> None:
        super().__init__()
        self.attn = Attention(
            dim=dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            resolution=resolution,
            act_layer=act_layer,
        )
        self.drop_path1 = StochasticDepth(drop_path, mode="row")

        self.mlp = LeVitMLP(dim, int(dim * mlp_ratio), act_layer=act_layer)
        self.drop_path2 = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(x))
        x = x + self.drop_path2(self.mlp(x))
        return x


class LevitStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        key_dim: int,
        depth: int,
        num_heads: int,
        attn_ratio: float,
        mlp_ratio: float,
        act_layer: Callable[..., nn.Module],
        resolution: tuple[int, int],
        downsample: bool,
        drop_path: float,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = LeVitSubsample(
                in_dim,
                out_dim,
                key_dim=key_dim,
                num_heads=in_dim // key_dim,
                attn_ratio=4.0,
                mlp_ratio=2.0,
                act_layer=act_layer,
                resolution=resolution,
                drop_path=drop_path,
            )
            resolution = ((resolution[0] - 1) // 2 + 1, (resolution[1] - 1) // 2 + 1)
        else:
            assert in_dim == out_dim
            self.downsample = nn.Identity()

        blocks = []
        for _ in range(depth):
            blocks.append(
                LeVitBlock(
                    out_dim,
                    key_dim,
                    num_heads=num_heads,
                    attn_ratio=attn_ratio,
                    mlp_ratio=mlp_ratio,
                    resolution=resolution,
                    act_layer=act_layer,
                    drop_path=drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class LeViT(BaseNet):
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

        embed_dim: list[int] = self.config["embed_dim"]
        key_dim: int = self.config["key_dim"]
        num_heads: list[int] = self.config["num_heads"]
        depths: list[int] = self.config["depths"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                embed_dim[0] // 8,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation_layer=nn.Hardswish,
            ),
            Conv2dNormActivation(
                embed_dim[0] // 8,
                embed_dim[0] // 4,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation_layer=nn.Hardswish,
            ),
            Conv2dNormActivation(
                embed_dim[0] // 4,
                embed_dim[0] // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation_layer=nn.Hardswish,
            ),
            Conv2dNormActivation(
                embed_dim[0] // 2,
                embed_dim[0],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation_layer=nn.Hardswish,
            ),
            PatchEmbed(),
        )

        resolution = (self.size[0] // 16, self.size[1] // 16)
        in_dim = embed_dim[0]
        num_stages = len(depths)
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_stages):
            if i == 0:
                stage_stride = 1
            else:
                stage_stride = 2

            stages[f"stage{i+1}"] = LevitStage(
                in_dim,
                embed_dim[i],
                key_dim=key_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                attn_ratio=2.0,
                mlp_ratio=2.0,
                act_layer=nn.Hardswish,
                resolution=resolution,
                downsample=stage_stride == 2,
                drop_path=drop_path_rate,  # No stochastic depth decay rule, simple float
            )
            resolution = ((resolution[0] - 1) // stage_stride + 1, (resolution[1] - 1) // stage_stride + 1)
            in_dim = embed_dim[i]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            Permute([0, 2, 1]),
            nn.BatchNorm1d(embed_dim[-1]),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(1),
        )
        self.embedding_size = embed_dim[-1]
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
            for param in self.features.parameters():
                param.requires_grad = True

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
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

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        old_size = self.size
        super().adjust_size(new_size)

        old_resolution = (old_size[0] // 16, old_size[1] // 16)
        resolution = (new_size[0] // 16, new_size[1] // 16)
        for stage in self.body:
            if isinstance(stage, LevitStage):
                for m in stage.modules():
                    if isinstance(m, AttentionSubsample):
                        # Update Subsample resolution
                        m.q[0].resolution = resolution

                        # Interpolate attention biases
                        m.attention_biases = nn.Parameter(
                            interpolate_attention_bias(m.attention_biases, old_resolution, resolution)
                        )

                        # Rebuild attention bias indices
                        k_pos = torch.stack(
                            torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]), indexing="ij")
                        ).flatten(1)
                        q_pos = torch.stack(
                            torch.meshgrid(
                                torch.arange(0, resolution[0], step=m.stride),
                                torch.arange(0, resolution[1], step=m.stride),
                                indexing="ij",
                            )
                        ).flatten(1)
                        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
                        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
                        m.attention_bias_idxs = nn.Buffer(rel_pos, persistent=False)

                        old_resolution = ((old_resolution[0] - 1) // 2 + 1, (old_resolution[1] - 1) // 2 + 1)
                        resolution = ((resolution[0] - 1) // 2 + 1, (resolution[1] - 1) // 2 + 1)

                    elif isinstance(m, Attention):
                        # Interpolate attention biases
                        m.attention_biases = nn.Parameter(
                            interpolate_attention_bias(m.attention_biases, old_resolution, resolution)
                        )

                        # Rebuild attention bias indices
                        pos = torch.stack(
                            torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]), indexing="ij")
                        ).flatten(1)
                        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
                        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
                        m.attention_bias_idxs = nn.Buffer(rel_pos, persistent=False)


registry.register_alias(
    "levit_128s",
    LeViT,
    config={
        "embed_dim": [128, 256, 384],
        "key_dim": 16,
        "num_heads": [4, 6, 8],
        "depths": [2, 3, 4],
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "levit_128",
    LeViT,
    config={
        "embed_dim": [128, 256, 384],
        "key_dim": 16,
        "num_heads": [4, 8, 12],
        "depths": [4, 4, 4],
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "levit_192",
    LeViT,
    config={
        "embed_dim": [192, 288, 384],
        "key_dim": 32,
        "num_heads": [3, 5, 6],
        "depths": [4, 4, 4],
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "levit_256",
    LeViT,
    config={
        "embed_dim": [256, 384, 512],
        "key_dim": 32,
        "num_heads": [4, 6, 8],
        "depths": [4, 4, 4],
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "levit_384",
    LeViT,
    config={
        "embed_dim": [384, 512, 768],
        "key_dim": 32,
        "num_heads": [6, 9, 12],
        "depths": [4, 4, 4],
        "drop_path_rate": 0.1,
    },
)

registry.register_weights(
    "levit_128s_il-common",
    {
        "description": "LeViT 128s model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 28.1,
                "sha256": "f70629adc76d029445f6eb70786075e71b9e7d99029186fefc4e1fb966aaf743",
            }
        },
        "net": {"network": "levit_128s", "tag": "il-common"},
    },
)
registry.register_weights(
    "levit_128_il-common",
    {
        "description": "LeViT 128 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 33.6,
                "sha256": "877006385c7bb9c9a48bb602803c10075a752f522941f085b241e5dfcdc77f9c",
            }
        },
        "net": {"network": "levit_128", "tag": "il-common"},
    },
)
