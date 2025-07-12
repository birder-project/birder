"""
Pyramid Vision Transformer v2, adapted from
https://github.com/whai362/PVT/blob/v2/classification/pvt_v2.py

Paper "PVT v2: Improved Baselines with Pyramid Vision Transformer", https://arxiv.org/abs/2106.13797

Changes from original:
* Add linear version to more models
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float, extra_relu: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU() if extra_relu else nn.Identity()
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=hidden_features,
            bias=True,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        (B, _, C) = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int,
        linear: bool,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if linear is False:
            self.pool = None
            self.act = None
            if sr_ratio > 1:
                self.sr = nn.Conv2d(
                    dim, dim, kernel_size=(sr_ratio, sr_ratio), stride=(sr_ratio, sr_ratio), padding=(0, 0)
                )
                self.norm = nn.LayerNorm(dim)
            else:
                self.sr = None
                self.norm = None
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.act = nn.GELU()
            self.sr = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            self.norm = nn.LayerNorm(dim)

        assert (self.pool is None and self.act is None) or (self.pool is not None and self.act is not None)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        (B, N, C) = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        if self.pool is not None and self.act is not None:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(self.pool(x)).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            x = self.act(x)

        else:
            if self.sr is not None:
                x = x.permute(0, 2, 1).reshape(B, C, H, W)
                x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
                x = self.norm(x)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        (k, v) = kv.unbind(0)

        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PyramidVisionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        sr_ratio: int,
        linear: bool,
        qkv_bias: bool,
        proj_drop: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            linear=linear,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=proj_drop,
            extra_relu=linear,
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size: tuple[int, int], stride: tuple[int, int], in_channels: int, embed_dim: int) -> None:
        super().__init__()
        assert patch_size[0] > stride[0], "Set larger patch_size than stride"
        assert patch_size[1] > stride[1], "Set larger patch_size than stride"
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        return x


class PyramidVisionTransformerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        depth: int,
        num_heads: int,
        sr_ratio: int,
        linear: bool,
        mlp_ratio: float,
        qkv_bias: bool,
        downsample: bool,
        proj_drop: float,
        attn_drop: float,
        drop_path: list[float],
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = OverlapPatchEmbed(
                patch_size=(3, 3),
                stride=(2, 2),
                in_channels=dim,
                embed_dim=dim_out,
            )
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                PyramidVisionTransformerBlock(
                    dim=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    sr_ratio=sr_ratio,
                    linear=linear,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)  # B, C, H, W -> B, H, W, C
        (B, H, W, C) = x.shape
        x = x.reshape(B, -1, C)
        for blk in self.blocks:
            x = blk(x, H, W)

        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


# pylint: disable=invalid-name
class PVT_v2(DetectorBackbone):
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

        num_heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]
        depths: list[int] = self.config["depths"]
        embed_dims: list[int] = self.config["embed_dims"]
        mlp_ratios: list[float] = self.config["mlp_ratios"]
        linear: bool = self.config["linear"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.patch_embed = OverlapPatchEmbed(
            patch_size=(7, 7),
            stride=(4, 4),
            in_channels=self.input_channels,
            embed_dim=embed_dims[0],
        )

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        num_stages = len(depths)
        prev_dim = embed_dims[0]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = PyramidVisionTransformerStage(
                dim=prev_dim,
                dim_out=embed_dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                sr_ratio=sr_ratios[i],
                linear=linear,
                mlp_ratio=mlp_ratios[i],
                qkv_bias=True,
                downsample=i > 0,
                proj_drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[i],
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
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.patch_embed(x)

        out = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)


registry.register_model_config(
    "pvt_v2_b0_li",
    PVT_v2,
    config={
        "depths": [2, 2, 2, 2],
        "embed_dims": [32, 64, 160, 256],
        "mlp_ratios": [8.0, 8.0, 4.0, 4.0],
        "linear": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "pvt_v2_b0",
    PVT_v2,
    config={
        "depths": [2, 2, 2, 2],
        "embed_dims": [32, 64, 160, 256],
        "mlp_ratios": [8.0, 8.0, 4.0, 4.0],
        "linear": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "pvt_v2_b1_li",
    PVT_v2,
    config={
        "depths": [2, 2, 2, 2],
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8.0, 8.0, 4.0, 4.0],
        "linear": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "pvt_v2_b1",
    PVT_v2,
    config={
        "depths": [2, 2, 2, 2],
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8.0, 8.0, 4.0, 4.0],
        "linear": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "pvt_v2_b2_li",
    PVT_v2,
    config={
        "depths": [3, 4, 6, 3],
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8.0, 8.0, 4.0, 4.0],
        "linear": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "pvt_v2_b2",
    PVT_v2,
    config={
        "depths": [3, 4, 6, 3],
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8.0, 8.0, 4.0, 4.0],
        "linear": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "pvt_v2_b3_li",
    PVT_v2,
    config={
        "depths": [3, 4, 18, 3],
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8.0, 8.0, 4.0, 4.0],
        "linear": True,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "pvt_v2_b3",
    PVT_v2,
    config={
        "depths": [3, 4, 18, 3],
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8.0, 8.0, 4.0, 4.0],
        "linear": False,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "pvt_v2_b4",
    PVT_v2,
    config={
        "depths": [3, 8, 27, 3],
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8.0, 8.0, 4.0, 4.0],
        "linear": False,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "pvt_v2_b5",
    PVT_v2,
    config={
        "depths": [3, 6, 40, 3],
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "linear": False,
        "drop_path_rate": 0.3,
    },
)

registry.register_weights(
    "pvt_v2_b0_li_il-common",
    {
        "description": "PVT v2 B0 linear model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 12.4,
                "sha256": "2c2e61844afc64b2efb3f0179f87029e4e418945ca9fae7b705b7d3287c36344",
            }
        },
        "net": {"network": "pvt_v2_b0_li", "tag": "il-common"},
    },
)
registry.register_weights(
    "pvt_v2_b0_il-common",
    {
        "description": "PVT v2 B0 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 13.4,
                "sha256": "0b93fed77fe67cc386b5f9ac852bf8e4e9c7a3230e11a7872bf868ee00da8e83",
            }
        },
        "net": {"network": "pvt_v2_b0", "tag": "il-common"},
    },
)
registry.register_weights(
    "pvt_v2_b2_mmcr-il-all256px",
    {
        "url": "https://huggingface.co/birder-project/pvt_v2_b2_mmcr-il-all/resolve/main",
        "description": "PVT v2 B2 model trained on the il-all dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 96.0,
                "sha256": "1b160a37d6fdd172ba21842de2d32cf2cc9524db155218a6949be81b88a70fe4",
            }
        },
        "net": {"network": "pvt_v2_b2", "tag": "il-all"},
    },
)
registry.register_weights(
    "pvt_v2_b2_mmcr-il-all",
    {
        "url": "https://huggingface.co/birder-project/pvt_v2_b2_mmcr-il-all/resolve/main",
        "description": "PVT v2 B2 model trained on the il-all dataset",
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 96.0,
                "sha256": "c0ffa0feb298e40c4335347ff648fe952fb0310e4c931d79be96a6af6bcbf1eb",
            }
        },
        "net": {"network": "pvt_v2_b2", "tag": "il-all"},
    },
)
