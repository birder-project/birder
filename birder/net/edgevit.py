"""
EdgeViT, adapted from
https://github.com/saic-fi/edgevit/blob/master/src/edgevit.py

Paper "EdgeViTs: Competing Light-weight CNNs on Mobile Devices with Vision Transformers",
https://arxiv.org/abs/2205.03436
"""

# Reference license: Apache-2.0

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch
from torch import nn
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import BaseNet


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
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
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


class EdgeViT(BaseNet):
    default_size = 224

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        assert self.net_param is not None, "must set net-param"
        assert self.size is not None, "must set size"
        net_param = int(self.net_param)

        image_size = self.size
        mlp_ratio = [4.0, 4.0, 4.0, 4.0]
        sr_ratios = [4, 2, 2, 1]
        drop_path_rate = 0.1
        if net_param == 0:
            # XXS - extra-extra-small
            depth = [1, 1, 3, 2]
            embed_dim = [36, 72, 144, 288]
            head_dim = 36

        elif net_param == 1:
            # XS - extra-small
            depth = [1, 1, 3, 1]
            embed_dim = [48, 96, 240, 384]
            head_dim = 48

        elif net_param == 2:
            # S - small
            depth = [1, 2, 5, 3]
            embed_dim = [48, 96, 240, 384]
            head_dim = 48

        else:
            raise ValueError(f"net_param = {net_param} not supported")

        self.patch_embed1 = PatchEmbed(
            img_size=image_size, patch_size=4, in_channels=self.input_channels, embed_dim=embed_dim[0]
        )
        self.patch_embed2 = PatchEmbed(
            img_size=image_size // 4, patch_size=2, in_channels=embed_dim[0], embed_dim=embed_dim[1]
        )
        self.patch_embed3 = PatchEmbed(
            img_size=image_size // 8, patch_size=2, in_channels=embed_dim[1], embed_dim=embed_dim[2]
        )
        self.patch_embed4 = PatchEmbed(
            img_size=image_size // 16, patch_size=2, in_channels=embed_dim[2], embed_dim=embed_dim[3]
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # Stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        layers = []
        for i in range(depth[0]):
            layers.append(
                LGLBlock(
                    dim=embed_dim[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratio[0],
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                    sr_ratio=sr_ratios[0],
                )
            )
        self.blocks1 = nn.Sequential(*layers)

        layers = []
        for i in range(depth[1]):
            layers.append(
                LGLBlock(
                    dim=embed_dim[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratio[1],
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                    drop_path=dpr[i + depth[0]],
                    sr_ratio=sr_ratios[1],
                )
            )
        self.blocks2 = nn.Sequential(*layers)

        layers = []
        for i in range(depth[2]):
            layers.append(
                LGLBlock(
                    dim=embed_dim[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratio[2],
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                    drop_path=dpr[i + depth[0] + depth[1]],
                    sr_ratio=sr_ratios[2],
                )
            )
        self.blocks3 = nn.Sequential(*layers)

        layers = []
        for i in range(depth[3]):
            layers.append(
                LGLBlock(
                    dim=embed_dim[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratio[3],
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                    drop_path=dpr[i + depth[0] + depth[1] + depth[2]],
                    sr_ratio=sr_ratios[3],
                )
            )
        self.blocks4 = nn.Sequential(*layers)
        self.norm = nn.BatchNorm2d(embed_dim[-1])

        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = embed_dim[-1]
        self.classifier = self.create_classifier()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) is True:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) is True and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm) is True:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed1(x)
        x = self.blocks1(x)
        x = self.patch_embed2(x)
        x = self.blocks2(x)
        x = self.patch_embed3(x)
        x = self.blocks3(x)
        x = self.patch_embed4(x)
        x = self.blocks4(x)
        x = self.norm(x)

        return self.features(x)

    def create_classifier(self, embed_dim: Optional[int] = None) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size

        return nn.Linear(embed_dim, self.num_classes, bias=False)


registry.register_alias("edgevit_xxs", EdgeViT, 0)
registry.register_alias("edgevit_xs", EdgeViT, 1)
registry.register_alias("edgevit_s", EdgeViT, 2)

registry.register_weights(
    "edgevit_xxs_il-common",
    {
        "description": "EdgeViT XXS model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 14.9,
                "sha256": "295f034ccbaa6025ddd2b0edeea09a2987a5af40c01741e4f89d266e78dd5359",
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
                "file_size": 25,
                "sha256": "de9e8ef7a3df75222187a02d34d40af2d5a25e0f9b2ee011970752d92862f441",
            }
        },
        "net": {"network": "edgevit_xs", "tag": "il-common"},
    },
)
