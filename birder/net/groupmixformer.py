"""
GroupMixFormer, adapted from
https://github.com/AILab-CVC/GroupMixFormer/blob/main/models/groupmixformer.py

Paper "Advancing Vision Transformers with Group-Mix Attention", https://arxiv.org/abs/2311.15157
"""

# Reference license: MIT

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.conv1(x))


class Agg0(nn.Module):
    def __init__(self, seg_dim: int) -> None:
        super().__init__()
        self.conv = SeparableConv2d(seg_dim * 3, seg_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm = nn.LayerNorm(seg_dim)
        self.act = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        (B, C, _, _) = x.size()
        x = self.act(self.norm(x.reshape(B, C, -1).permute(0, 2, 1)))

        return x


class Aggregator(nn.Module):
    def __init__(self, dim: int, seg: int) -> None:
        super().__init__()
        self.dim = dim
        self.seg = seg
        self.seg_dim = self.dim // self.seg

        self.norm0 = nn.BatchNorm2d(self.seg_dim)
        self.act0 = nn.Hardswish()

        self.agg1 = SeparableConv2d(self.seg_dim, self.seg_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm1 = nn.BatchNorm2d(self.seg_dim)
        self.act1 = nn.Hardswish()

        self.agg2 = SeparableConv2d(self.seg_dim, self.seg_dim, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.norm2 = nn.BatchNorm2d(self.seg_dim)
        self.act2 = nn.Hardswish()

        self.agg3 = SeparableConv2d(self.seg_dim, self.seg_dim, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.norm3 = nn.BatchNorm2d(self.seg_dim)
        self.act3 = nn.Hardswish()

        self.agg0 = Agg0(self.seg_dim)

    def forward(self, x: torch.Tensor, size: tuple[int, int], num_head: int) -> tuple[torch.Tensor, torch.Tensor]:
        (B, _, C) = x.size()
        (H, W) = size
        # assert N == H * W

        x = x.transpose(1, 2).view(B, C, H, W)
        xs = x.split([self.seg_dim] * self.seg, dim=1)

        x_local = (
            xs[4].reshape(3, B // 3, self.seg_dim, H, W).permute(1, 0, 2, 3, 4).reshape(B // 3, 3 * self.seg_dim, H, W)
        )
        x_local = self.agg0(x_local)

        x0 = self.act0(self.norm0(xs[0]))
        x1 = self.act1(self.norm1(self.agg1(xs[1])))
        x2 = self.act2(self.norm2(self.agg2(xs[2])))
        x3 = self.act3(self.norm3(self.agg3(xs[3])))

        x = torch.concat([x0, x1, x2, x3], dim=1)

        C = C // 5 * 4
        x = x.reshape(3, B // 3, num_head, C // num_head, H * W).permute(0, 1, 2, 4, 3)

        return (x, x_local)


class ConvRelPosEnc(nn.Module):
    def __init__(self, channels: int, window: dict[int, int]) -> None:
        super().__init__()

        self.conv_list = nn.ModuleList()
        head_splits = []
        for cur_window, cur_head_split in window.items():
            padding_size = cur_window // 2
            cur_conv = nn.Conv2d(
                cur_head_split * channels,
                cur_head_split * channels,
                kernel_size=(cur_window, cur_window),
                stride=(1, 1),
                padding=(padding_size, padding_size),
                groups=cur_head_split * channels,
            )
            self.conv_list.append(cur_conv)
            head_splits.append(cur_head_split)

        self.channel_splits = [x * channels for x in head_splits]

    def forward(self, q: torch.Tensor, v: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        (B, L, _, C) = q.size()
        (H, W) = size
        # assert N == H * W

        v = v.reshape(B, L, H, W, C).permute(0, 1, 4, 2, 3).reshape(B, -1, H, W).contiguous()
        v_list = torch.split(v, self.channel_splits, dim=1)
        conv_v_list = []
        for i, conv in enumerate(self.conv_list):
            conv_v_list.append(conv(v_list[i]))

        conv_v = torch.concat(conv_v_list, dim=1)
        conv_v = conv_v.reshape(B, L, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, L, -1, C).contiguous()

        return q * conv_v


class ConvPosEnc(nn.Module):
    def __init__(self, dim: int, kernel_size: tuple[int, int]) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=dim,
        )

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        (B, _, C) = x.size()
        (H, W) = size
        # assert N == H * W

        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x) + x
        x = x.flatten(2).transpose(1, 2)

        return x


class EfficientAtt(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, proj_drop: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.aggregator = Aggregator(dim=dim, seg=5)

        t_dim = dim // 5 * 4
        self.crpe = ConvRelPosEnc(t_dim // num_heads, window={3: 2, 5: 3, 7: 3})

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        (B, N, C) = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3).reshape(3 * B, N, C)

        (qkv, x_agg0) = self.aggregator(qkv, size, self.num_heads)
        (q, k, v) = qkv.unbind(0)

        k_softmax = k.softmax(dim=2)
        k_softmax_t_dot_v = torch.einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        eff_att = torch.einsum("b h n k, b h k v -> b h n v", q, k_softmax_t_dot_v)
        crpe = self.crpe(q, v, size=size)

        x = self.scale * eff_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C // 5 * 4)
        x = torch.concat([x, x_agg0], dim=-1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PatchEmbed(nn.Module):
    def __init__(self, stride: int, in_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.stride = stride

        self.proj = SeparableConv2d(in_dim, embedding_dim, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1))
        self.norm = nn.BatchNorm2d(embedding_dim)
        self.act = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        (_, _, H, W) = x.size()
        x = self.act(self.norm(self.proj(x)))
        x = x.flatten(2).transpose(1, 2)

        return (x, (H // self.stride, W // self.stride))


class GroupMixAttentionBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool, drop: float, drop_path_rate: float
    ) -> None:
        super().__init__()
        self.cpe = ConvPosEnc(dim, kernel_size=(3, 3))
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EfficientAtt(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=drop)
        self.drop_path = StochasticDepth(drop_path_rate, mode="row")

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=drop)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = self.cpe(x, size)
        x = x + self.drop_path(self.attn(self.norm1(x), size))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class GroupMixFormerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        drop_path_rate: list[float],
        depth: int,
        patch_embed_stride: int,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
            stride=patch_embed_stride,
            in_dim=dim,
            embedding_dim=out_dim,
        )

        self.gma_blocks = nn.ModuleList()
        for i in range(depth):
            self.gma_blocks.append(
                GroupMixAttentionBlock(
                    dim=out_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    drop_path_rate=drop_path_rate[i],
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (x, size) = self.patch_embed(x)
        for block in self.gma_blocks:
            x = block(x, size)

        x = x.reshape(x.size(0), size[0], size[1], -1).permute(0, 3, 1, 2)

        return x


class GroupMixFormer(DetectorBackbone):
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

        num_heads = 8
        qkv_bias = True
        dropout = 0.0
        embed_dims: list[int] = self.config["embed_dims"]
        depths: list[int] = self.config["depths"]
        mlp_ratios: list[float] = self.config["mlp_ratios"]
        drop_path_rate: float = self.config["drop_path_rate"]

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        num_stages = len(depths)

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                embed_dims[0] // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation_layer=nn.Hardswish,
                bias=True,
            ),
            Conv2dNormActivation(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation_layer=nn.Hardswish,
                bias=True,
            ),
        )

        prev_dim = embed_dims[0]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = GroupMixFormerStage(
                prev_dim,
                embed_dims[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                drop=dropout,
                drop_path_rate=dpr[i],
                depth=depths[i],
                patch_embed_stride=1 if i == 0 else 2,
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
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
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
    "groupmixformer_mobile",
    GroupMixFormer,
    config={
        "embed_dims": [40, 80, 160, 160],
        "depths": [3, 3, 12, 4],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "groupmixformer_t",
    GroupMixFormer,
    config={
        "embed_dims": [80, 160, 200, 240],
        "depths": [4, 4, 12, 4],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "groupmixformer_s",
    GroupMixFormer,
    config={
        "embed_dims": [80, 160, 320, 320],
        "depths": [2, 4, 12, 4],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "drop_path_rate": 0.2,
    },
)
registry.register_alias(
    "groupmixformer_b",
    GroupMixFormer,
    config={
        "embed_dims": [200, 240, 320, 480],
        "depths": [8, 8, 12, 8],
        "mlp_ratios": [2.0, 2.0, 4.0, 4.0],
        "drop_path_rate": 0.4,
    },
)
registry.register_alias(
    "groupmixformer_l",
    GroupMixFormer,
    config={
        "embed_dims": [240, 320, 360, 480],
        "depths": [8, 10, 30, 10],
        "mlp_ratios": [4.0, 4.0, 2.0, 2.0],
        "drop_path_rate": 0.5,
    },
)

registry.register_weights(
    "groupmixformer_mobile_il-common",
    {
        "description": "GroupMixFormer mobile model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 22.0,
                "sha256": "c2f0607fe1691b428ed5f15b0a8ecc6333d78c653be736fcd5a6fb576642b417",
            }
        },
        "net": {"network": "groupmixformer_mobile", "tag": "il-common"},
    },
)
