"""
EfficientFormer v1, adapted from
https://github.com/snap-research/EfficientFormer/blob/main/models/efficientformer.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientformer.py

Paper "EfficientFormer: Vision Transformers at MobileNet Speed",
https://arxiv.org/abs/2206.01191

Changes from original:
* Removed attention bias cache
"""

# Reference license: Apache-2.0 (both)

from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.base import interpolate_attention_bias


class Attention(nn.Module):
    def __init__(self, dim: int, key_dim: int, num_heads: int, attn_ratio: float, resolution: tuple[int, int]) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * num_heads
        key_attn_dim = key_dim * num_heads

        self.qkv = nn.Linear(dim, key_attn_dim * 2 + self.val_attn_dim)
        self.proj = nn.Linear(self.val_attn_dim, dim)

        pos = torch.stack(
            torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]), indexing="ij")
        ).flatten(1)
        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, resolution[0] * resolution[1]))
        self.attention_bias_idxs = nn.Buffer(rel_pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, _) = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.val_dim], dim=3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.attention_biases[:, self.attention_bias_idxs]

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
    ) -> None:
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


class Pooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x) - x


class ConvMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, in_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.drop = nn.Dropout(drop)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace is True:
            return x.mul_(self.gamma)

        return x * self.gamma


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_values: float, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        if self.inplace is True:
            return x.mul_(gamma)

        return x * gamma


class MetaBlock1d(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        resolution: tuple[int, int],
        proj_drop: float,
        drop_path: float,
        layer_scale_init_value: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = Attention(dim, key_dim=32, num_heads=8, attn_ratio=4.0, resolution=resolution)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=proj_drop)

        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.ls1 = LayerScale(dim, layer_scale_init_value)
        self.ls2 = LayerScale(dim, layer_scale_init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.ls1(self.token_mixer(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class MetaBlock2d(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        proj_drop: float,
        drop_path: float,
        layer_scale_init_value: float,
    ):
        super().__init__()
        self.token_mixer = Pooling()
        self.ls1 = LayerScale2d(dim, layer_scale_init_value)
        self.drop_path = StochasticDepth(drop_path, mode="row")

        self.mlp = ConvMLP(dim, hidden_features=int(dim * mlp_ratio), drop=proj_drop)
        self.ls2 = LayerScale2d(dim, layer_scale_init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.ls1(self.token_mixer(x)))
        x = x + self.drop_path(self.ls2(self.mlp(x)))
        return x


class Flat(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)


class EfficientFormerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        depth: int,
        downsample: bool,
        num_vit: int,
        mlp_ratio: float,
        resolution: tuple[int, int],
        proj_drop: float,
        drop_path: list[float],
        layer_scale_init_value: float,
    ):
        super().__init__()
        if downsample is True:
            self.downsample = Downsample(in_channels=dim, out_channels=dim_out, kernel_size=(3, 3), stride=(2, 2))
            dim = dim_out
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        blocks = []
        if num_vit >= depth:
            blocks.append(Flat())

        for block_idx in range(depth):
            remain_idx = depth - block_idx - 1
            if num_vit > remain_idx:
                blocks.append(
                    MetaBlock1d(
                        dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        proj_drop=proj_drop,
                        drop_path=drop_path[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            else:
                blocks.append(
                    MetaBlock2d(
                        dim,
                        mlp_ratio=mlp_ratio,
                        proj_drop=proj_drop,
                        drop_path=drop_path[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
                if num_vit > 0 and num_vit == remain_idx:
                    blocks.append(Flat())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


# pylint: disable=invalid-name
class EfficientFormer_v1(BaseNet):
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

        resolution = (int(self.size[0] / (2**5)), int(self.size[1] / (2**5)))
        layer_scale_init_value = 1e-5
        embed_dims: tuple[int, int, int, int] = self.config["embed_dims"]
        depths: tuple[int, int, int, int] = (3, 2, 6, 4)
        drop_path_rate: float = 0.0
        num_vit: int = 1

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels, embed_dims[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            ),
            Conv2dNormActivation(embed_dims[0] // 2, embed_dims[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        prev_dim = embed_dims[0]
        num_stages = len(depths)
        last_stage = num_stages - 1
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        downsample = (False,) + (True,) * (num_stages - 1)

        stages = []
        for i in range(num_stages):
            stages.append(
                EfficientFormerStage(
                    prev_dim,
                    embed_dims[i],
                    depths[i],
                    downsample=downsample[i],
                    num_vit=num_vit if i == last_stage else 0,
                    mlp_ratio=4.0,
                    resolution=resolution,
                    proj_drop=0.0,
                    drop_path=dpr[i],
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
            prev_dim = embed_dims[i]

        self.body = nn.Sequential(*stages)

        self.features = nn.Sequential(
            nn.LayerNorm(embed_dims[-1], eps=1e-6),
            Permute([0, 2, 1]),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(1),
        )
        self.embedding_size = embed_dims[-1]
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()
        self.distillation_output = False

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

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

        old_resolution = (int(old_size[0] / (2**5)), int(old_size[1] / (2**5)))
        resolution = (int(new_size[0] / (2**5)), int(new_size[1] / (2**5)))
        for m in self.body.modules():
            if isinstance(m, Attention):
                m.attention_biases = nn.Parameter(
                    interpolate_attention_bias(m.attention_biases, old_resolution, resolution)
                )

                pos = torch.stack(
                    torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]), indexing="ij")
                ).flatten(1)
                rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
                rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
                m.attention_bias_idxs = nn.Buffer(rel_pos)


registry.register_alias(
    "efficientformer_v1_l1",
    EfficientFormer_v1,
    config={"embed_dims": (48, 96, 224, 448), "depths": (3, 2, 6, 4), "drop_path_rate": 0.0, "num_vit": 1},
)
registry.register_alias(
    "efficientformer_v1_l3",
    EfficientFormer_v1,
    config={"embed_dims": (64, 128, 320, 512), "depths": (4, 4, 12, 6), "drop_path_rate": 0.1, "num_vit": 4},
)
registry.register_alias(
    "efficientformer_v1_l7",
    EfficientFormer_v1,
    config={"embed_dims": (96, 192, 384, 768), "depths": (6, 6, 18, 8), "drop_path_rate": 0.1, "num_vit": 8},
)

registry.register_weights(
    "efficientformer_v1_l1_il-common",
    {
        "description": "EfficientFormer v1 L1 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 45,
                "sha256": "e311c1d8e624de5c9571208674ec826ef16d47018d79dfdc649f8273f85cd51e",
            }
        },
        "net": {"network": "efficientformer_v1_l1", "tag": "il-common"},
    },
)
