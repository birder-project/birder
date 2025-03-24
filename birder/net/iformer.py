"""
iFormer (Inception Transformer), adapted from
https://github.com/sail-sg/iFormer/blob/main/models/inception_transformer.py

Paper "Inception Transformer", https://arxiv.org/abs/2205.12956

Changes from original:
* Removed biases before norms
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_values: float, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace is True:
            return x.mul_(self.gamma)

        return x * self.gamma


class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)

        return x


class HighMixer(nn.Module):
    def __init__(
        self, dim: int, kernel_size: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int]
    ) -> None:
        super().__init__()
        self.cnn_in = dim // 2
        cnn_dim = self.cnn_in * 2

        self.conv = nn.Conv2d(self.cnn_in, cnn_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.proj1 = nn.Conv2d(
            cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=cnn_dim, bias=False
        )
        self.act1 = nn.GELU()

        self.max_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(self.cnn_in, cnn_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cx = x[:, : self.cnn_in, :, :].contiguous()
        cx = self.conv(cx)
        cx = self.proj1(cx)
        cx = self.act1(cx)

        px = x[:, self.cnn_in :, :, :].contiguous()
        px = self.max_pool(px)
        px = self.proj2(px)
        px = self.act2(px)

        return torch.concat([cx, px], dim=1)


class LowMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        pool_size: tuple[int, int],
        attn_drop: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.dim = dim
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if pool_size[0] > 1 or pool_size[1] > 1:
            self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0, count_include_pad=False)
            self.upsample = nn.Upsample(scale_factor=pool_size)
        else:
            self.pool = nn.Identity()
            self.upsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        (B, _, H, W) = x.size()
        x = x.permute(0, 2, 3, 1).view(B, -1, self.dim)

        (B, N, C) = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)
        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0, scale=self.scale
        )
        x = x.transpose(2, 3).reshape(B, C, N)
        x = x.view(B, C, H, W)
        x = self.upsample(x)

        return x


class Mixer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        attention_head: int,
        pool_size: tuple[int, int],
        attn_drop: float,
        proj_drop: float,
    ) -> None:
        super().__init__()
        head_dim = dim // num_heads
        low_dim = attention_head * head_dim
        high_dim = dim - low_dim
        self.high_dim = high_dim

        self.high_mixer = HighMixer(high_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.low_mixer = LowMixer(
            low_dim,
            num_heads=attention_head,
            qkv_bias=qkv_bias,
            pool_size=pool_size,
            attn_drop=attn_drop,
        )

        self.conv_fuse = nn.Conv2d(
            low_dim + high_dim * 2,
            low_dim + high_dim * 2,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=low_dim + high_dim * 2,
            bias=False,
        )
        self.proj = nn.Conv2d(low_dim + high_dim * 2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)

        hx = x[:, : self.high_dim, :, :].contiguous()
        hx = self.high_mixer(hx)

        lx = x[:, self.high_dim :, :, :].contiguous()
        lx = self.low_mixer(lx)

        x = torch.concat([hx, lx], dim=1)
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x


class InceptionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        attention_head: int,
        pool_size: tuple[int, int],
        drop: float,
        attn_drop: float,
        drop_path: float,
        layer_scale_init_value: Optional[float],
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Mixer(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attention_head=attention_head,
            pool_size=pool_size,
            attn_drop=attn_drop,
            proj_drop=0.0,
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=drop)

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale2d(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))

        return x


class InceptionTransformerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        attention_heads: list[int],
        pool_size: tuple[int, int],
        drop: float,
        attn_drop: float,
        drop_path: list[float],
        layer_scale_init_value: Optional[float],
        depth: int,
        resolution: tuple[int, int],
        downsample: bool,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = PatchEmbed(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), in_channels=dim, embed_dim=out_dim
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        self.pos_embed = nn.Parameter(torch.zeros(1, resolution[0], resolution[1], out_dim))
        layers = []
        for i in range(depth):
            layers.append(
                InceptionTransformerBlock(
                    out_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attention_head=attention_heads[i],
                    pool_size=pool_size,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    layer_scale_init_value=layer_scale_init_value,
                )
            )

        self.blocks = nn.Sequential(*layers)

        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = x.permute(0, 3, 1, 2)

        return x


# pylint: disable=invalid-name
class iFormer(DetectorBackbone):
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

        img_size = self.size
        depths: list[int] = self.config["depths"]
        embed_dims: list[int] = self.config["embed_dims"]
        num_heads: list[int] = self.config["num_heads"]
        attention_heads: list[int] = self.config["attention_heads"]
        layer_scale_init_value: Optional[float] = self.config["layer_scale_init_value"]
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
                activation_layer=nn.GELU,
                inplace=None,
            ),
            Conv2dNormActivation(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation_layer=None,
            ),
            Permute([0, 2, 3, 1]),
        )

        head_index = 0
        resolution = (img_size[0] // 4, img_size[1] // 4)
        prev_dim = embed_dims[0]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = InceptionTransformerStage(
                prev_dim,
                embed_dims[i],
                num_heads=num_heads[i],
                mlp_ratio=4.0,
                qkv_bias=True,
                attention_heads=attention_heads[head_index : head_index + depths[i]],
                pool_size=(2, 2) if i < 2 else (1, 1),
                drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value if i >= 2 else None,
                depth=depths[i],
                resolution=resolution,
                downsample=i > 0,
            )
            return_channels.append(embed_dims[i])
            prev_dim = embed_dims[i]
            resolution = (resolution[0] // 2, resolution[1] // 2)
            head_index = head_index + depths[i]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            Permute([0, 2, 3, 1]),  # B C H W -> B H W C
            nn.LayerNorm(embed_dims[-1], eps=1e-6),
            Permute([0, 3, 1, 2]),  # B H W C -> B C H W
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embed_dims[-1]
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
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

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)

        resolution = (new_size[0] // 4, new_size[1] // 4)
        for stage in self.body.modules():
            if isinstance(stage, InceptionTransformerStage):
                stage.pos_embed = nn.Parameter(
                    F.interpolate(stage.pos_embed.permute(0, 3, 1, 2), size=resolution, mode="bilinear").permute(
                        0, 2, 3, 1
                    )
                )
                resolution = (resolution[0] // 2, resolution[1] // 2)


registry.register_alias(
    "iformer_s",
    iFormer,
    config={
        "depths": [3, 3, 9, 3],
        "embed_dims": [96, 192, 320, 384],
        "num_heads": [3, 6, 10, 12],
        "attention_heads": [1] * 3 + [3] * 3 + [7] * 4 + [9] * 5 + [11] * 3,
        "layer_scale_init_value": 1e-6,
        "drop_path_rate": 0.2,
    },
)
registry.register_alias(
    "iformer_b",
    iFormer,
    config={
        "depths": [4, 6, 14, 6],
        "embed_dims": [96, 192, 384, 512],
        "num_heads": [3, 6, 12, 16],
        "attention_heads": [1] * 4 + [3] * 6 + [8] * 7 + [10] * 7 + [15] * 6,
        "layer_scale_init_value": 1e-6,
        "drop_path_rate": 0.4,
    },
)
registry.register_alias(
    "iformer_l",
    iFormer,
    config={
        "depths": [4, 6, 18, 8],
        "embed_dims": [96, 192, 448, 640],
        "num_heads": [3, 6, 14, 20],
        "attention_heads": [1] * 4 + [3] * 6 + [10] * 9 + [12] * 9 + [19] * 8,
        "layer_scale_init_value": 1e-6,
        "drop_path_rate": 0.5,
    },
)

registry.register_weights(
    "iformer_s_arabian-peninsula",
    {
        "url": "https://huggingface.co/birder-project/iformer_s_arabian-peninsula/resolve/main",
        "description": "iFormer small model trained on the arabian-peninsula dataset",
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 79.5,
                "sha256": "28ccab8f908e2d8129bb968d2142b5f7d851f0f1e9d522c7b76512d223a944de",
            }
        },
        "net": {"network": "iformer_s", "tag": "arabian-peninsula"},
    },
)
