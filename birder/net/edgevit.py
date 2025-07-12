"""
EdgeViT, adapted from
https://github.com/saic-fi/edgevit/blob/master/src/edgevit.py

Paper "EdgeViTs: Competing Light-weight CNNs on Mobile Devices with Vision Transformers",
https://arxiv.org/abs/2205.03436
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

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
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(
            hidden_features, out_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class GlobalSparseAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(kernel_size=(1, 1), stride=(self.sr, self.sr), padding=(0, 0))
            self.local_prop = nn.ConvTranspose2d(
                dim,
                dim,
                kernel_size=(self.sr, self.sr),
                stride=(self.sr, self.sr),
                padding=(0, 0),
                output_padding=(0, 0),
                groups=dim,
                bias=True,
            )
            self.norm = nn.LayerNorm(dim)

        else:
            self.local_prop = nn.Identity()
            self.sampler = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        (B, _, C) = x.size()  # B, N, C
        if self.sr > 1:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(H / self.sr), int(W / self.sr))
            x = self.local_prop(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)

        return x


class LocalAgg(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        act_layer: Callable[..., nn.Module],
        drop_path: float,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim, bias=True)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.attn = nn.Conv2d(dim, dim, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=dim, bias=True)
        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = ConvMLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SelfAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        drop_path: float,
        sr_ratio: int,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim, bias=True)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        (B, N, H, W) = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)

        return x


class LGLBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        drop_path: float,
        sr_ratio: int,
    ):
        super().__init__()

        if sr_ratio > 1:
            self.local_agg = LocalAgg(dim, mlp_ratio, act_layer, drop_path)

        else:
            self.local_agg = nn.Identity()

        self.self_attn = SelfAttn(dim, num_heads, mlp_ratio, act_layer, norm_layer, drop_path, sr_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_agg(x)
        x = self.self_attn(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
            bias=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        (B, _, H, W) = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class EdgeViTSage(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        drop_path: list[float],
        sr_ratio: int,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        layers = []
        for i in range(depth):
            layers.append(
                LGLBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop_path=drop_path[i],
                    sr_ratio=sr_ratio,
                )
            )
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.blocks(x)

        return x


class EdgeViT(DetectorBackbone):
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

        patch_size = [4, 2, 2, 2]
        mlp_ratio = [4.0, 4.0, 4.0, 4.0]
        sr_ratios = [4, 2, 2, 1]
        depth: list[int] = self.config["depth"]
        embed_dim: list[int] = self.config["embed_dim"]
        head_dim: int = self.config["head_dim"]
        drop_path_rate: float = self.config.get("drop_path_rate", 0.1)

        num_stages = len(depth)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depth)).split(depth)]
        num_heads = [dim // head_dim for dim in embed_dim]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        prev_dim = self.input_channels
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = EdgeViTSage(
                patch_size=patch_size[i],
                in_channels=prev_dim,
                embed_dim=embed_dim[i],
                depth=depth[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio[i],
                act_layer=nn.GELU,
                norm_layer=norm_layer,
                drop_path=dpr[i],
                sr_ratio=sr_ratios[i],
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
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)

    def create_classifier(self, embed_dim: Optional[int] = None) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size

        return nn.Linear(embed_dim, self.num_classes, bias=False)


registry.register_model_config(
    "edgevit_xxs", EdgeViT, config={"depth": [1, 1, 3, 2], "embed_dim": [36, 72, 144, 288], "head_dim": 36}
)
registry.register_model_config(
    "edgevit_xs", EdgeViT, config={"depth": [1, 1, 3, 1], "embed_dim": [48, 96, 240, 384], "head_dim": 48}
)
registry.register_model_config(
    "edgevit_s", EdgeViT, config={"depth": [1, 2, 5, 3], "embed_dim": [48, 96, 240, 384], "head_dim": 48}
)

registry.register_weights(
    "edgevit_xxs_il-common",
    {
        "description": "EdgeViT XXS model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 14.9,
                "sha256": "b4ae9c4f871bafc47c757b65ee8f7ce2dc427fff55171f4feaafdbb5ba02ff3f",
            }
        },
        "net": {"network": "edgevit_xxs", "tag": "il-common"},
    },
)
registry.register_weights(
    "edgevit_xs_il-common",
    {
        "description": "EdgeViT XS model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 25.0,
                "sha256": "38ab9c6f10586a84c2abaa55919cba56cbd2196ddf771befa09eb71c79d3634c",
            }
        },
        "net": {"network": "edgevit_xs", "tag": "il-common"},
    },
)
