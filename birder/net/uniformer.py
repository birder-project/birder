"""
UniFormer, adapted from
https://github.com/Sense-X/UniFormer/blob/main/image_classification/models/uniformer.py

Paper "UniFormer: Unifying Convolution and Self-attention for Visual Recognition", https://arxiv.org/abs/2201.09450
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class ConvMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, in_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, attn_drop: float, proj_drop: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)

        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float, drop_path: float) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.attn = nn.Conv2d(dim, dim, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=dim)
        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMLP(dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_values: float, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones([dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace is True:
            return x.mul_(self.gamma)

        return x * self.gamma


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        layer_scale_init_value: Optional[float],
        drop: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features=mlp_hidden_dim, drop=drop)

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale2d(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        (B, N, H, W) = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.layer_scale_1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))
        x = x.transpose(1, 2).reshape(B, N, H, W)

        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: tuple[int, int], in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        (B, _, H, W) = x.size()  # B, C, H, W
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class UniFormerStage(nn.Module):
    def __init__(
        self,
        block_type: Literal["conv", "attn"],
        patch_size: tuple[int, int],
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        layer_scale_init_value: Optional[float],
        drop: float,
        drop_path: list[float],
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)

        layers = []
        for i in range(depth):
            if block_type == "conv":
                layers.append(ConvBlock(embed_dim, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path[i]))
            elif block_type == "attn":
                layers.append(
                    AttentionBlock(
                        embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        layer_scale_init_value=layer_scale_init_value,
                        drop=0.0,
                        attn_drop=0.0,
                        drop_path=drop_path[i],
                    )
                )
            else:
                raise ValueError(f"Block type: {block_type} not supported")

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.blocks(x)

        return x


class UniFormer(DetectorBackbone):
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

        block_type = ["conv", "conv", "attn", "attn"]
        patch_size = [4, 2, 2, 2]
        depth: list[int] = self.config["depth"]
        embed_dim: list[int] = self.config["embed_dim"]
        mlp_ratio: list[float] = self.config["mlp_ratio"]
        head_dim: int = self.config["head_dim"]
        drop_path_rate: float = self.config["drop_path_rate"]
        layer_scale_init_value: Optional[float] = self.config["layer_scale_init_value"]

        num_stages = len(depth)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depth)).split(depth)]
        num_heads = [dim // head_dim for dim in embed_dim]

        prev_dim = self.input_channels
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = UniFormerStage(
                block_type=block_type[i],  # type: ignore[arg-type]
                patch_size=(patch_size[i], patch_size[i]),
                in_channels=prev_dim,
                embed_dim=embed_dim[i],
                depth=depth[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio[i],
                layer_scale_init_value=layer_scale_init_value,
                drop=0.0,
                drop_path=dpr[i],
            )
            return_channels.append(embed_dim[i])
            prev_dim = embed_dim[i]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.BatchNorm2d(embed_dim[-1]),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embed_dim[-1]
        self.classifier = self.create_classifier()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.features(x)


registry.register_alias(
    "uniformer_s",
    UniFormer,
    config={
        "depth": [3, 4, 8, 3],
        "embed_dim": [64, 128, 320, 512],
        "mlp_ratio": [4.0, 4.0, 4.0, 4.0],
        "head_dim": 64,
        "drop_path_rate": 0.1,
        "layer_scale_init_value": None,
    },
)
registry.register_alias(
    "uniformer_b",
    UniFormer,
    config={
        "depth": [5, 8, 20, 7],
        "embed_dim": [64, 128, 320, 512],
        "mlp_ratio": [4.0, 4.0, 4.0, 4.0],
        "head_dim": 64,
        "drop_path_rate": 0.3,
        "layer_scale_init_value": None,
    },
)
registry.register_alias(
    "uniformer_l",
    UniFormer,
    config={
        "depth": [5, 10, 24, 7],
        "embed_dim": [128, 192, 448, 640],
        "mlp_ratio": [4.0, 4.0, 4.0, 4.0],
        "head_dim": 64,
        "drop_path_rate": 0.4,
        "layer_scale_init_value": 1e-6,
    },
)
