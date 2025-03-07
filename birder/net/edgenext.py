"""
EdgeNeXt, adapted from
https://github.com/mmaaz60/EdgeNeXt

Paper "EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications",
https://arxiv.org/abs/2206.10589
"""

# Reference license: MIT

import math
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
from birder.net.convnext_v1 import LayerNorm2d
from birder.net.xcit import PositionalEncodingFourier


class ConvBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        kernel_size: tuple[int, int],
        expand_ratio: int,
        layer_scale_init_value: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(
            dim,
            dim_out,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim_out, eps=1e-6)
        self.pw_conv1 = nn.Linear(dim_out, expand_ratio * dim_out)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(expand_ratio * dim_out, dim_out)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim_out))
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = shortcut + self.drop_path(x)

        return x


class CrossCovarianceAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, attn_drop: float, proj_drop: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 4, 1)
        (q, k, v) = qkv.unbind(0)

        q = F.normalize(q, dim=-1) * self.temperature
        k = F.normalize(k, dim=-1)
        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=1.0
        )

        x = x.permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SDTA(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float,
        layer_scale_init_value: float,
        expand_ratio: int,
        use_pos_emb: bool,
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        drop: float,
        num_scales: int,
    ) -> None:
        super().__init__()
        width = max(int(math.ceil(dim / num_scales)), int(math.floor(dim // num_scales)))
        self.width = width
        self.num_scales = max(1, num_scales - 1)

        conv_list = []
        for _ in range(self.num_scales):
            conv_list.append(nn.Conv2d(width, width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=width))

        self.conv_list = nn.ModuleList(conv_list)

        self.pos_embed = None
        if use_pos_emb is True:
            self.pos_embed = PositionalEncodingFourier(hidden_dim=32, dim=dim)

        self.norm_xca = nn.LayerNorm(dim, eps=1e-6)
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.xca = CrossCovarianceAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw_conv1 = nn.Linear(dim, expand_ratio * dim)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(expand_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)))
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # Replaced split with workaround from:
        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/edgenext.py
        spx = x.chunk(len(self.conv_list) + 1, dim=1)
        sp_out = []
        sp = spx[0]
        for i, conv in enumerate(self.conv_list):
            if i > 0:
                sp = sp + spx[i]

            sp = conv(sp)
            sp_out.append(sp)

        sp_out.append(spx[-1])
        x = torch.concat(sp_out, 1)

        # XCA
        (B, C, H, W) = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embed is not None:
            pos_encoding = self.pos_embed(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = x + self.drop_path(self.gamma_xca * self.xca(self.norm_xca(x)))
        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = shortcut + self.drop_path(x)

        return x


class EdgeNeXtStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        depth: int,
        num_global_blocks: int,
        num_heads: int,
        scales: int,
        kernel_size: tuple[int, int],
        expand_ratio: int,
        use_pos_emb: bool,
        layer_scale_init_value: float,
        drop_path_rates: list[float],
    ) -> None:
        super().__init__()
        self.grad_checkpointing = False

        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_channels, eps=1e-6),
                nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
            )
            in_channels = out_channels

        stage_blocks = []
        for i in range(depth):
            if i < depth - num_global_blocks:
                stage_blocks.append(
                    ConvBlock(
                        dim=in_channels,
                        dim_out=out_channels,
                        kernel_size=kernel_size,
                        expand_ratio=expand_ratio,
                        layer_scale_init_value=layer_scale_init_value,
                        drop_path=drop_path_rates[i],
                    )
                )
            else:
                stage_blocks.append(
                    SDTA(
                        dim=in_channels,
                        num_scales=scales,
                        num_heads=num_heads,
                        expand_ratio=expand_ratio,
                        use_pos_emb=use_pos_emb,
                        layer_scale_init_value=layer_scale_init_value,
                        drop_path=drop_path_rates[i],
                        qkv_bias=True,
                        attn_drop=0.0,
                        drop=0.0,
                    )
                )
            in_channels = out_channels

        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


class EdgeNeXt(DetectorBackbone):
    default_size = (256, 256)

    # pylint: disable=too-many-locals
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

        global_block_counts = [0, 1, 1, 1]
        d2_scales = [2, 2, 3, 4]
        expand_ratio = 4
        kernel_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]
        use_pos_emb = [False, True, False, False]
        layer_scale_init_value = 1e-6
        depths: list[int] = self.config["depths"]
        dims: list[int] = self.config["dims"]
        heads: list[int] = self.config["heads"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.stem = Conv2dNormActivation(
            self.input_channels,
            dims[0],
            kernel_size=(4, 4),
            stride=(4, 4),
            padding=(0, 0),
            bias=True,
            norm_layer=LayerNorm2d,
            activation_layer=None,
        )

        current_stride = 4
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        in_channels = dims[0]
        for i, (depth, dim, head) in enumerate(zip(depths, dims, heads)):
            stride = 2 if current_stride == 2 or i > 0 else 1
            current_stride *= stride
            stages[f"stage{i+1}"] = EdgeNeXtStage(
                in_channels=in_channels,
                out_channels=dim,
                stride=stride,
                depth=depth,
                num_global_blocks=global_block_counts[i],
                num_heads=head,
                drop_path_rates=dpr[i],
                scales=d2_scales[i],
                expand_ratio=expand_ratio,
                kernel_size=kernel_sizes[i],
                use_pos_emb=use_pos_emb[i],
                layer_scale_init_value=layer_scale_init_value,
            )
            return_channels.append(dim)
            in_channels = dim

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            LayerNorm2d(dims[-1], eps=1e-6),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = dims[-1]
        self.classifier = self.create_classifier()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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


registry.register_alias(
    "edgenext_xxs",
    EdgeNeXt,
    config={"depths": [2, 2, 6, 2], "dims": [24, 48, 88, 168], "heads": [4, 4, 4, 4], "drop_path_rate": 0.0},
)
registry.register_alias(
    "edgenext_xs",
    EdgeNeXt,
    config={"depths": [3, 3, 9, 3], "dims": [32, 64, 100, 192], "heads": [4, 4, 4, 4], "drop_path_rate": 0.0},
)
registry.register_alias(
    "edgenext_s",
    EdgeNeXt,
    config={"depths": [3, 3, 9, 3], "dims": [48, 96, 160, 304], "heads": [8, 8, 8, 8], "drop_path_rate": 0.1},
)
registry.register_alias(
    "edgenext_b",
    EdgeNeXt,
    config={"depths": [3, 3, 9, 3], "dims": [80, 160, 288, 584], "heads": [8, 8, 8, 8], "drop_path_rate": 0.1},
)

registry.register_weights(
    "edgenext_xxs_il-common",
    {
        "description": "EdgeNeXt extra extra small model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 4.7,
                "sha256": "05f567111bebe05f3e88cb73aa665407009decf275836ec3c1c823b3725c6725",
            }
        },
        "net": {"network": "edgenext_xxs", "tag": "il-common"},
    },
)
registry.register_weights(
    "edgenext_xs_il-common",
    {
        "description": "EdgeNeXt extra small model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 8.5,
                "sha256": "ceb0554f66dfaeb8cbba95137982756b1a5456f5359dda1657f9aab934ed7e0e",
            }
        },
        "net": {"network": "edgenext_xs", "tag": "il-common"},
    },
)
registry.register_weights(
    "edgenext_s_il-common",
    {
        "description": "EdgeNeXt small model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 20.7,
                "sha256": "2855125a641f143b97fff4ef27ac1386a3058d1659b4e8e5d723952e788ce0ef",
            }
        },
        "net": {"network": "edgenext_s", "tag": "il-common"},
    },
)
