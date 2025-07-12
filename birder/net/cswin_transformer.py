"""
CSWin Transformer, adapted from
https://github.com/microsoft/CSWin-Transformer/blob/main/models/cswin.py

Paper "CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows",
https://arxiv.org/abs/2107.00652

Changes from original:
* Split size based on image size (image size // 32 where applicable)
"""

# Reference license: MIT

import math
from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.vit import PatchEmbed


def img2windows(img: torch.Tensor, h_sp: int, w_sp: int) -> torch.Tensor:
    (B, C, H, W) = img.size()
    img_reshape = img.view(B, C, H // h_sp, h_sp, W // w_sp, w_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, h_sp * w_sp, C)

    return img_perm


def windows2img(img_splits_hw: torch.Tensor, h_sp: int, w_sp: int, H: int, W: int) -> torch.Tensor:
    B = int(img_splits_hw.shape[0] / (H * W / h_sp / w_sp))
    img = img_splits_hw.view(B, H // h_sp, W // w_sp, h_sp, w_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return img


class LePEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        resolution: tuple[int, int],
        idx: int,
        split_size: tuple[int, int],
        num_heads: int,
        attn_drop: float,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.split_size = split_size
        self.idx = idx
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = head_dim**-0.5
        self.assign_sp_shape()

        self.get_v = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def assign_sp_shape(self) -> None:
        if self.idx == -1:
            self.h_sp = self.resolution[0]
            self.w_sp = self.resolution[1]
        elif self.idx == 0:
            self.h_sp = self.resolution[0]
            self.w_sp = self.split_size[1]
        elif self.idx == 1:
            self.h_sp = self.split_size[0]
            self.w_sp = self.resolution[1]
        else:
            raise ValueError("unsupported idx")

    def im2cswin(self, x: torch.Tensor) -> torch.Tensor:
        (B, _, C) = x.size()
        x = x.transpose(-2, -1).contiguous().view(B, C, self.resolution[0], self.resolution[1])
        x = img2windows(x, self.h_sp, self.w_sp)
        x = x.reshape(-1, self.h_sp * self.w_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        return x

    def get_lepe(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (B, _, C) = x.size()
        H = self.resolution[0]
        W = self.resolution[1]
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        h_sp = self.h_sp
        w_sp = self.w_sp
        x = x.view(B, C, H // h_sp, h_sp, W // w_sp, w_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, h_sp, w_sp)  # (B', C, H', W')

        lepe = self.get_v(x)  # (B', C, H', W')
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, h_sp * w_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, h_sp * w_sp).permute(0, 1, 3, 2).contiguous()

        return (x, lepe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (q, k, v) = x.unbind(0)

        (B, _, C) = q.shape

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        (v, lepe) = self.get_lepe(v)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B head N C @ B head C N --> B head N N
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.h_sp * self.w_sp, C)  # B head N N @ B head N C

        x = windows2img(x, self.h_sp, self.w_sp, self.resolution[0], self.resolution[1]).view(B, -1, C)

        return x


class MergeBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, resolution: tuple[int, int]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.norm = nn.LayerNorm(dim_out)
        self.resolution = resolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, _, C) = x.size()
        H = self.resolution[0]
        W = self.resolution[1]
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        (B, C) = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x


class CSWinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        resolution: tuple[int, int],
        num_heads: int,
        split_size: tuple[int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        proj_drop: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = nn.LayerNorm(dim)

        if resolution[0] == split_size[0] and resolution[1] == split_size[1]:
            self.branch_num = 1
        else:
            self.branch_num = 2

        self.proj = nn.Linear(dim, dim)

        if self.branch_num == 1:
            self.attentions = nn.ModuleList(
                [
                    LePEAttention(
                        dim,
                        resolution=resolution,
                        idx=-1,
                        split_size=split_size,
                        num_heads=num_heads,
                        attn_drop=attn_drop,
                    )
                    for _ in range(self.branch_num)
                ]
            )
        else:
            self.attentions = nn.ModuleList(
                [
                    LePEAttention(
                        dim // 2,
                        resolution=resolution,
                        idx=i,
                        split_size=split_size,
                        num_heads=num_heads // 2,
                        attn_drop=attn_drop,
                    )
                    for i in range(self.branch_num)
                ]
            )

        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, _, C) = x.shape

        qkv = self.qkv(self.norm1(x)).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        if self.branch_num == 2:
            # Some workaround for TorchScript
            outputs = []
            qkv_list = (qkv[:, :, :, : C // 2], qkv[:, :, :, C // 2 :])
            for attn, qkv_sp in zip(self.attentions, qkv_list):
                outputs.append(attn(qkv_sp))

            attn_x = torch.concat(outputs, dim=2)
        else:
            attn_x = self.attentions[0](qkv)

        attn_x = self.proj(attn_x)
        x = x + self.drop_path(attn_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class CSWinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        resolution: tuple[int, int],
        num_heads: int,
        split_size: tuple[int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        proj_drop: float,
        attn_drop: float,
        drop_path: list[float],
        depth: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = MergeBlock(dim, dim * 2, resolution=(resolution[0] * 2, resolution[1] * 2))
            dim = dim * 2
        else:
            self.downsample = nn.Identity()

        blocks = []
        for i in range(depth):
            blocks.append(
                CSWinBlock(
                    dim,
                    resolution=resolution,
                    num_heads=num_heads,
                    split_size=split_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


# pylint: disable=invalid-name
class CSWin_Transformer(DetectorBackbone):
    square_only = True

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

        embed_dim: int = self.config["embed_dim"]
        depths: list[int] = self.config["depths"]
        num_heads: list[int] = self.config["num_heads"]
        drop_path_rate: float = self.config["drop_path_rate"]
        mlp_ratio = 4.0
        self.split_size = [
            (1, 1),
            (2, 2),
            (int(self.size[0] / (2**5)), int(self.size[1] / (2**5))),
            (int(self.size[0] / (2**5)), int(self.size[1] / (2**5))),
        ]

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, embed_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2)),
            PatchEmbed(),
            nn.LayerNorm(embed_dim),
        )

        num_stages = len(depths)
        curr_dim = embed_dim
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = CSWinStage(
                curr_dim,
                resolution=(self.size[0] // (2 ** (i + 2)), self.size[1] // (2 ** (i + 2))),
                num_heads=num_heads[i],
                split_size=self.split_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[i],
                depth=depths[i],
                downsample=i > 0,
            )
            if i > 0:
                curr_dim = curr_dim * 2

            return_channels.append(curr_dim)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.LayerNorm(curr_dim),
            Permute([0, 2, 1]),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = curr_dim
        self.classifier = self.create_classifier()

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)

        out = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.return_stages:
                (B, L, C) = x.size()
                H = int(math.sqrt(L))
                W = H
                out[name] = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.stem.parameters():
            param.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)

        new_base = (new_size[0] // 4, new_size[1] // 4)
        idx = 0
        for stage in self.body.modules():
            if isinstance(stage, CSWinStage):
                for m in stage.modules():
                    if isinstance(m, MergeBlock):
                        m.resolution = (new_base[0] * 2, new_base[1] * 2)

                    elif isinstance(m, CSWinBlock):
                        for attn in m.attentions:
                            attn.resolution = new_base
                            attn.split_size = (new_size[0] // 32, new_size[1] // 32)
                            attn.assign_sp_shape()

                new_base = (new_base[0] // 2, new_base[1] // 2)
                self.split_size[idx] = (new_size[0] // 32, new_size[1] // 32)
                idx += 1


registry.register_model_config(
    "cswin_transformer_t",
    CSWin_Transformer,
    config={"embed_dim": 64, "depths": [1, 2, 21, 1], "num_heads": [2, 4, 8, 16], "drop_path_rate": 0.2},
)
registry.register_model_config(
    "cswin_transformer_s",
    CSWin_Transformer,
    config={"embed_dim": 64, "depths": [2, 4, 32, 2], "num_heads": [2, 4, 8, 16], "drop_path_rate": 0.4},
)
registry.register_model_config(
    "cswin_transformer_b",
    CSWin_Transformer,
    config={"embed_dim": 96, "depths": [2, 4, 32, 2], "num_heads": [4, 8, 16, 32], "drop_path_rate": 0.5},
)
registry.register_model_config(
    "cswin_transformer_l",
    CSWin_Transformer,
    config={"embed_dim": 144, "depths": [2, 4, 32, 2], "num_heads": [6, 12, 24, 24], "drop_path_rate": 0.5},
)
