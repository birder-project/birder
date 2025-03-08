"""
TinyViT, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/tiny_vit.py

Paper "TinyViT: Fast Pretraining Distillation for Small Vision Transformers", https://arxiv.org/abs/2207.10666

Changes from original:
* Window sizes based on image size
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import interpolate_attention_bias
from birder.net.convnext_v1 import LayerNorm2d


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = Conv2dNormActivation(
            in_channels,
            out_channels // 2,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            activation_layer=nn.GELU,
            inplace=None,
        )
        self.conv2 = Conv2dNormActivation(
            out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), activation_layer=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MBConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: float, drop_path: float) -> None:
        super().__init__()
        mid_channels = int(in_channels * expand_ratio)
        self.conv1 = Conv2dNormActivation(
            in_channels,
            mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=nn.GELU,
            inplace=None,
        )
        self.conv2 = Conv2dNormActivation(
            mid_channels,
            mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=mid_channels,
            activation_layer=nn.GELU,
            inplace=None,
        )
        self.conv3 = Conv2dNormActivation(
            mid_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=nn.GELU,
            inplace=None,
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut

        return x


class ConvLayer(nn.Module):
    def __init__(self, dim: int, depth: int, drop_path: list[float], conv_expand_ratio: float) -> None:
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(MBConv(dim, dim, conv_expand_ratio, drop_path[i]))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = Conv2dNormActivation(
            dim,
            out_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=nn.GELU,
            inplace=None,
        )
        self.conv2 = Conv2dNormActivation(
            out_dim,
            out_dim,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=out_dim,
            activation_layer=nn.GELU,
            inplace=None,
        )
        self.conv3 = Conv2dNormActivation(
            out_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class NormMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim: int, key_dim: int, num_heads: int, attn_ratio: float, resolution: tuple[int, int]) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.out_dim = self.val_dim * num_heads

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, num_heads * (self.val_dim + 2 * key_dim))
        self.proj = nn.Linear(self.out_dim, dim)

        self.define_bias_idxs(resolution)
        self.attention_biases = nn.Parameter(torch.zeros(self.num_heads, resolution[0] * resolution[1]))

    def define_bias_idxs(self, resolution: tuple[int, int]) -> None:
        k_pos = torch.stack(
            torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]), indexing="ij")
        ).flatten(1)
        q_pos = torch.stack(
            torch.meshgrid(torch.arange(0, resolution[0]), torch.arange(0, resolution[1]), indexing="ij")
        ).flatten(1)
        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
        self.attention_bias_idxs = nn.Buffer(torch.LongTensor(rel_pos), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_bias = self.attention_biases[:, self.attention_bias_idxs]
        (B, N, _) = x.shape

        # Normalization
        x = self.norm(x)
        qkv = self.qkv(x)
        (q, k, v) = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, attn_mask=attn_bias, scale=self.scale
        )
        x = x.transpose(1, 2).reshape(B, N, self.out_dim)
        x = self.proj(x)

        return x


class TinyVitBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int],
        mlp_ratio: float,
        drop: float,
        drop_path: list[float],
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        head_dim = dim // num_heads

        window_resolution = window_size
        self.attn = Attention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)
        self.drop_path1 = StochasticDepth(drop_path, mode="row")

        self.mlp = NormMLP(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = StochasticDepth(drop_path, mode="row")

        self.local_conv = Conv2dNormActivation(
            dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim, activation_layer=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, H, W, C) = x.shape
        L = H * W

        shortcut = x
        if H == self.window_size[0] and W == self.window_size[1]:
            x = x.reshape(B, L, C)
            x = self.attn(x)
            x = x.view(B, H, W, C)
        else:
            pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
            pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            # Window partition
            pH = H + pad_b  # pylint: disable=invalid-name
            pW = W + pad_r  # pylint: disable=invalid-name
            nH = pH // self.window_size[0]  # pylint: disable=invalid-name
            nW = pW // self.window_size[1]  # pylint: disable=invalid-name
            x = (
                x.view(B, nH, self.window_size[0], nW, self.window_size[1], C)
                .transpose(2, 3)
                .reshape(B * nH * nW, self.window_size[0] * self.window_size[1], C)
            )

            x = self.attn(x)

            # Window reverse
            x = x.view(B, nH, nW, self.window_size[0], self.window_size[1], C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

        x = shortcut + self.drop_path1(x)

        x = x.permute(0, 3, 1, 2)
        x = self.local_conv(x)
        x = x.reshape(B, C, L).transpose(1, 2)
        x = x + self.drop_path2(self.mlp(x))

        return x.view(B, H, W, C)


class TinyVitStage(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        window_size: tuple[int, int],
        mlp_ratio: float,
        drop: float,
        drop_path: list[list[float]],
        downsample: bool,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.out_dim = out_dim

        if downsample is True:
            self.downsample = PatchMerging(dim=dim, out_dim=out_dim)
        else:
            self.downsample = nn.Identity()
            assert dim == out_dim

        layers = []
        for i in range(depth):
            layers.append(
                TinyVitBlock(
                    dim=out_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i],
                )
            )

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.blocks(x)
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW

        return x


# pylint: disable=invalid-name
class Tiny_ViT(DetectorBackbone):
    block_group_regex = r"body\.stage\d+\.blocks\.(\d+)"

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

        self.window_scale_factors = [1, 1, 2, 1]
        embed_dims: list[int] = self.config["embed_dims"]
        depths: list[int] = self.config["depths"]
        num_heads: list[int] = self.config["num_heads"]
        drop_path_rate: float = self.config["drop_path_rate"]
        window_sizes = [
            (int(self.size[0] / (2**5) * scale), int(self.size[1] / (2**5) * scale))
            for scale in self.window_scale_factors
        ]

        self.stem = PatchEmbed(in_channels=self.input_channels, out_channels=embed_dims[0])

        num_stages = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        prev_dim = embed_dims[0]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for stage_idx in range(num_stages):
            if stage_idx == 0:
                stages[f"stage{stage_idx+1}"] = ConvLayer(
                    prev_dim,
                    depth=depths[stage_idx],
                    drop_path=dpr[: depths[stage_idx]],
                    conv_expand_ratio=4.0,
                )
            else:
                out_dim = embed_dims[stage_idx]
                stages[f"stage{stage_idx+1}"] = TinyVitStage(
                    dim=embed_dims[stage_idx - 1],
                    out_dim=out_dim,
                    depth=depths[stage_idx],
                    num_heads=num_heads[stage_idx],
                    window_size=window_sizes[stage_idx],
                    mlp_ratio=4.0,
                    drop=0.0,
                    drop_path=dpr[sum(depths[:stage_idx]) : sum(depths[: stage_idx + 1])],
                    downsample=True,
                )
                prev_dim = out_dim

            return_channels.append(prev_dim)

        num_features = embed_dims[-1]
        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            LayerNorm2d(num_features),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = num_features
        self.classifier = self.create_classifier()

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

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        old_size = self.size
        super().adjust_size(new_size)

        old_window_sizes = [
            (int(old_size[0] / (2**5) * scale), int(old_size[1] / (2**5) * scale))
            for scale in self.window_scale_factors
        ]
        window_sizes = [
            (int(new_size[0] / (2**5) * scale), int(new_size[1] / (2**5) * scale))
            for scale in self.window_scale_factors
        ]
        idx = 0
        for stage in self.body:
            if isinstance(stage, TinyVitStage):
                for m in stage.modules():
                    if isinstance(m, TinyVitBlock):
                        m.window_size = window_sizes[idx]

                        # This will update the index buffer
                        m.attn.define_bias_idxs(window_sizes[idx])

                        # Interpolate the actual table
                        m.attn.attention_biases = nn.Parameter(
                            interpolate_attention_bias(
                                m.attn.attention_biases, old_window_sizes[idx], window_sizes[idx], mode="bilinear"
                            )
                        )

            idx += 1


registry.register_alias(
    "tiny_vit_5m",
    Tiny_ViT,
    config={
        "embed_dims": [64, 128, 160, 320],
        "depths": [2, 2, 6, 2],
        "num_heads": [2, 4, 5, 10],
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "tiny_vit_11m",
    Tiny_ViT,
    config={
        "embed_dims": [64, 128, 256, 448],
        "depths": [2, 2, 6, 2],
        "num_heads": [2, 4, 8, 14],
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "tiny_vit_21m",
    Tiny_ViT,
    config={
        "embed_dims": [96, 192, 384, 576],
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 18],
        "drop_path_rate": 0.2,
    },
)

registry.register_weights(
    "tiny_vit_5m_il-common",
    {
        "description": "TinyViT 5M model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 20.0,
                "sha256": "57f84dc3144fc4e3ca39328d3a1446ca9e26ddb54e4c4d84301b7638bee2ec21",
            },
        },
        "net": {"network": "tiny_vit_5m", "tag": "il-common"},
    },
)
