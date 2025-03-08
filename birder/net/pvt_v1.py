"""
Pyramid Vision Transformer, adapted from
https://github.com/whai362/PVT/blob/v2/classification/pvt.py

Paper "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions",
https://arxiv.org/abs/2102.12122

Changes from original:
* No positional embedding on the CLS token
* Remove last norm
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.vit import adjust_position_embedding


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int,
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

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=(sr_ratio, sr_ratio), stride=(sr_ratio, sr_ratio), padding=(0, 0))
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        (B, N, C) = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

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
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=proj_drop)
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: tuple[int, int], in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=(0, 0))
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
        mlp_ratio: float,
        qkv_bias: bool,
        downsample: bool,
        cls_token: bool,
        img_size: tuple[int, int],
        proj_drop: float,
        attn_drop: float,
        drop_path: list[float],
    ) -> None:
        super().__init__()
        self.grad_checkpointing = False

        if downsample is True:
            self.downsample = PatchEmbed(patch_size=(2, 2), in_channels=dim, embed_dim=dim_out)
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        num_patches = img_size[0] * img_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim_out))
        self.blocks = nn.ModuleList(
            [
                PyramidVisionTransformerBlock(
                    dim=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    sr_ratio=sr_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim_out)
        if cls_token is True:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_out))
        else:
            self.cls_token = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)  # B, C, H, W -> B, H, W, C
        (B, H, W, C) = x.size()
        x = x.reshape(B, -1, C)
        x = x + self.pos_embed
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.concat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x, H, W)

        x = self.norm(x)
        if self.cls_token is not None:
            return x

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


# pylint: disable=invalid-name
class PVT_v1(DetectorBackbone):
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
        embed_dims = [64, 128, 320, 512]
        mlp_ratios = [8.0, 8.0, 4.0, 4.0]
        depths: list[int] = self.config["depths"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.patch_embed = PatchEmbed(
            patch_size=(4, 4),
            in_channels=self.input_channels,
            embed_dim=embed_dims[0],
        )
        img_size = (self.size[0] // 4, self.size[1] // 4)

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
                mlp_ratio=mlp_ratios[i],
                qkv_bias=True,
                downsample=i > 0,
                cls_token=i == num_stages - 1,
                img_size=(img_size[0] // (2**i), img_size[1] // (2**i)),
                proj_drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[i],
            )

            prev_dim = embed_dims[i]
            return_channels.append(embed_dims[i])

        self.body = nn.Sequential(stages)
        self.return_channels = return_channels
        self.embedding_size = embed_dims[-1]
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.patch_embed(x)

        out = {}
        for name, module in self.body.named_children():
            (B, _, H, W) = x.size()
            x = module(x)
            if name in self.return_stages:
                if name == "stage4":
                    # Remove class token and reshape
                    out[name] = x[:, 1:].reshape(B, H // 2, W // 2, -1).permute(0, 3, 1, 2).contiguous()
                else:
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

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.body(x)
        return x[:, 0]

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        old_size = self.size
        super().adjust_size(new_size)

        old_s = (old_size[0] // 4, old_size[1] // 4)
        s = (new_size[0] // 4, new_size[1] // 4)
        for m in self.body.modules():
            if isinstance(m, PyramidVisionTransformerStage):
                m.pos_embed = nn.Parameter(adjust_position_embedding(m.pos_embed, old_s, s, 0))
                old_s = (old_s[0] // 2, old_s[1] // 2)
                s = (s[0] // 2, s[1] // 2)


registry.register_alias("pvt_v1_t", PVT_v1, config={"depths": [2, 2, 2, 2], "drop_path_rate": 0.1})
registry.register_alias("pvt_v1_s", PVT_v1, config={"depths": [3, 4, 6, 3], "drop_path_rate": 0.1})
registry.register_alias("pvt_v1_m", PVT_v1, config={"depths": [3, 4, 18, 3], "drop_path_rate": 0.3})
registry.register_alias("pvt_v1_l", PVT_v1, config={"depths": [3, 8, 27, 3], "drop_path_rate": 0.3})
