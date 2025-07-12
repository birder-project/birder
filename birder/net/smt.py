"""
Scale-Aware Modulation Transformer (SMT), adapted from
https://github.com/AFeng-x/SMT/blob/main/models/smt.py

Paper "Scale-Aware Modulation Meet Transformer", https://arxiv.org/abs/2307.08579
"""

# Reference license: MIT

import math
from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.layers import LayerScale
from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class SequentialWithShape(nn.Sequential):
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:  # pylint: disable=arguments-differ
        for module in self:
            x = module(x, H, W)

        return x


class DWConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        (B, _, C) = x.size()
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.fc2(x)

        return x


class CAAttention(nn.Module):
    def __init__(self, dim: int, ca_num_heads: int, qkv_bias: bool, proj_drop: float, expand_ratio: int) -> None:
        super().__init__()
        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}"
        self.ca_num_heads = ca_num_heads

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups = dim // ca_num_heads

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)
        self.local_conv = nn.ModuleList()
        for i in range(self.ca_num_heads):
            self.local_conv.append(
                nn.Conv2d(
                    dim // self.ca_num_heads,
                    dim // self.ca_num_heads,
                    kernel_size=(3 + i * 2, 3 + i * 2),
                    stride=(1, 1),
                    padding=(1 + i, 1 + i),
                    groups=dim // self.ca_num_heads,
                )
            )

        self.proj0 = nn.Conv2d(
            dim, dim * expand_ratio, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=self.split_groups
        )
        self.bn = nn.BatchNorm2d(dim * expand_ratio)
        self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        (B, N, C) = x.size()

        v = self.v(x)
        s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1, 2)
        s_i_out = []
        for i, local_conv in enumerate(self.local_conv):
            s_i = s[i]
            s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
            s_i_out.append(s_i)

        s_out = torch.concat(s_i_out, dim=2)
        s_out = s_out.reshape(B, C, H, W)
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
        s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
        x = s_out * v

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SAAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        sa_num_heads: int,
        qkv_bias: bool,
        proj_drop: float,
        attn_drop: float,
    ) -> None:
        super().__init__()
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}"
        self.sa_num_heads = sa_num_heads

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        head_dim = dim // sa_num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.conv = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        (B, N, C) = x.size()

        q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
        (k, v) = kv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.conv(
            v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)
        ).view(B, C, N).transpose(1, 2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        ca_num_heads: int,
        sa_num_heads: int,
        qkv_bias: bool,
        proj_drop: float,
        attn_drop: float,
        ca_attention: bool,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        if ca_attention is True:
            self.attn = CAAttention(
                dim, ca_num_heads=ca_num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop, expand_ratio=expand_ratio
            )
        else:
            self.attn = SAAttention(
                dim, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop, attn_drop=attn_drop
            )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return self.attn(x, H, W)


class SMTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ca_num_heads: int,
        sa_num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        layer_scale_value: Optional[float],
        proj_drop: float,
        attn_drop: float,
        drop_path: float,
        ca_attention: bool,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            ca_num_heads=ca_num_heads,
            sa_num_heads=sa_num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ca_attention=ca_attention,
            expand_ratio=expand_ratio,
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(mlp_ratio * dim))

        if layer_scale_value is not None:
            self.layer_scale_1 = LayerScale(dim, layer_scale_value)
            self.layer_scale_2 = LayerScale(dim, layer_scale_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1(self.attn(self.norm1(x), H, W)))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x), H, W)))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size: tuple[int, int], stride: tuple[int, int], in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        (_, _, H, W) = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return (x, H, W)


class Stem(nn.Module):
    def __init__(self, kernel_size: tuple[int, int], stride: tuple[int, int], in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dNormActivation(
                in_channels,
                embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            ),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.conv(x)
        (_, _, H, W) = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return (x, H, W)


class SMTStage(nn.Module):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        dim: int,
        dim_out: int,
        depth: int,
        downsample_stem: bool,
        stem_conv: tuple[int, int],
        ca_num_heads: int,
        sa_num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        layer_scale_value: Optional[float],
        proj_drop: float,
        attn_drop: float,
        drop_path: list[float],
        ca_attention: Optional[bool],
        expand_ratio: int,
    ) -> None:
        super().__init__()
        if downsample_stem is True:
            self.downsample_block = Stem(kernel_size=stem_conv, stride=(2, 2), in_channels=dim, embed_dim=dim_out)
        else:
            self.downsample_block = OverlapPatchEmbed(
                patch_size=(3, 3), stride=(2, 2), in_channels=dim, embed_dim=dim_out
            )

        layers = []
        for i in range(depth):
            if ca_attention is not None:
                caa = ca_attention
            else:
                caa = i % 2 == 0

            layers.append(
                SMTBlock(
                    dim_out,
                    ca_num_heads=ca_num_heads,
                    sa_num_heads=sa_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    layer_scale_value=layer_scale_value,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    ca_attention=caa,
                    expand_ratio=expand_ratio,
                )
            )

        self.blocks = SequentialWithShape(*layers)
        self.norm = nn.LayerNorm(dim_out, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        (x, H, W) = self.downsample_block(x)
        x = self.blocks(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class SMT(DetectorBackbone):
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

        qkv_bias = True
        ca_num_heads = [4, 4, 4, -1]
        sa_num_heads = [-1, -1, 8, 16]
        ca_attentions = [True, True, None, False]  # Tri-state: None activated on even blocks only
        expand_ratio = 2
        embed_dims: list[int] = self.config["embed_dims"]
        mlp_ratios: list[float] = self.config["mlp_ratios"]
        depths: list[int] = self.config["depths"]
        stem_conv: tuple[int, int] = self.config["stem_conv"]
        drop_path_rate: float = self.config["drop_path_rate"]

        num_stages = len(depths)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = self.input_channels
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = SMTStage(
                prev_dim,
                embed_dims[i],
                depth=depths[i],
                downsample_stem=i == 0,
                stem_conv=stem_conv,
                ca_num_heads=ca_num_heads[i],
                sa_num_heads=sa_num_heads[i],
                mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                layer_scale_value=None,
                proj_drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[i],
                ca_attention=ca_attentions[i],
                expand_ratio=expand_ratio,
            )
            return_channels.append(embed_dims[i])
            prev_dim = embed_dims[i]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embed_dims[-1]
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)


registry.register_model_config(
    "smt_t",
    SMT,
    config={
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [4.0, 4.0, 4.0, 2.0],
        "depths": [2, 2, 8, 1],
        "stem_conv": (3, 3),
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "smt_s",
    SMT,
    config={
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [4.0, 4.0, 4.0, 2.0],
        "depths": [3, 4, 18, 2],
        "stem_conv": (3, 3),
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "smt_b",
    SMT,
    config={
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [8.0, 6.0, 4.0, 2.0],
        "depths": [4, 6, 28, 2],
        "stem_conv": (7, 7),
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "smt_l",
    SMT,
    config={
        "embed_dims": [96, 192, 384, 768],
        "mlp_ratios": [8.0, 6.0, 4.0, 2.0],
        "depths": [4, 6, 28, 4],
        "stem_conv": (7, 7),
        "drop_path_rate": 0.5,
    },
)

registry.register_weights(
    "smt_t_il-common",
    {
        "description": "SMT tiny model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 42.8,
                "sha256": "5b6d2541ed382d85bda88cf19ebf2d8fb0adb1b6475209f17eacf4a2f428bcc0",
            }
        },
        "net": {"network": "smt_t", "tag": "il-common"},
    },
)
