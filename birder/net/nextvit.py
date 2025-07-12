"""
Next-ViT, adapted from
https://github.com/bytedance/Next-ViT/blob/main/classification/nextvit.py

"Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios",
https://arxiv.org/abs/2207.05501
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType
from birder.net.base import make_divisible


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        if stride[0] > 1 or stride[1] > 1:
            self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
            )
            self.norm = nn.BatchNorm2d(out_channels)

        elif in_channels != out_channels:
            self.avg_pool = nn.Identity()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
            )
            self.norm = nn.BatchNorm2d(out_channels)

        else:
            self.avg_pool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(self.avg_pool(x)))


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_channels: int, head_dim: int) -> None:
        super().__init__()
        self.group_conv3x3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=out_channels // head_dim,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.group_conv3x3(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.projection(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, mlp_ratio: float, bias: bool) -> None:
        super().__init__()
        hidden_dim = make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        return x


class NCB(nn.Module):
    """
    Next Convolution Block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
        drop_path: float,
        head_dim: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim)
        self.drop_path = StochasticDepth(drop_path, mode="row")

        self.norm = nn.BatchNorm2d(out_channels)
        self.mlp = MLP(out_channels, out_channels, mlp_ratio=mlp_ratio, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.drop_path(self.mhca(x))
        x = x + self.drop_path(self.mlp(self.norm(x)))

        return x


# pylint: disable=invalid-name
class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        head_dim: int,
        sr_ratio: int,
    ) -> None:
        super().__init__()
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, out_dim)

        n_ratio = sr_ratio**2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=n_ratio, stride=n_ratio)
            self.norm = nn.BatchNorm1d(dim)

        else:
            self.sr = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.size()
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        x = x.transpose(1, 2)
        x = self.sr(x)
        x = self.norm(x)
        x = x.transpose(1, 2)

        k = self.k(x)
        k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).transpose(1, 2)
        v = self.v(x)
        v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # pylint: disable=not-callable
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class NTB(nn.Module):
    """
    Next Transformer Block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_path: float,
        stride: tuple[int, int],
        sr_ratio: int,
        mlp_ratio: float,
        head_dim: int,
        mix_block_ratio: float,
    ) -> None:
        super().__init__()
        mhsa_out_channels = make_divisible(int(out_channels * mix_block_ratio), 32)
        mhca_out_channels = out_channels - mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels, mhsa_out_channels, stride)
        self.norm1 = nn.BatchNorm2d(mhsa_out_channels)
        self.e_mhsa = E_MHSA(
            mhsa_out_channels,
            mhsa_out_channels,
            head_dim=head_dim,
            sr_ratio=sr_ratio,
        )
        self.mhsa_drop_path = StochasticDepth(drop_path * mix_block_ratio, mode="row")

        self.projection = PatchEmbed(mhsa_out_channels, mhca_out_channels, stride=(1, 1))
        self.mhca = MHCA(mhca_out_channels, head_dim=head_dim)
        self.mhca_drop_path = StochasticDepth(drop_path * (1 - mix_block_ratio), mode="row")

        self.norm2 = nn.BatchNorm2d(out_channels)
        self.mlp = MLP(out_channels, out_channels, mlp_ratio=mlp_ratio, bias=True)
        self.mlp_drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        (B, C, H, W) = x.size()
        out = self.norm1(x)

        out = out.reshape(B, C, H * W).permute(0, 2, 1)
        out = self.mhsa_drop_path(self.e_mhsa(out))
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        x = x + out

        out = self.projection(x)
        out = out + self.mhca_drop_path(self.mhca(out))
        x = torch.concat([x, out], dim=1)

        out = self.norm2(x)
        x = x + self.mlp_drop_path(self.mlp(out))

        return x


class NextViT(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.stage\d+\.(\d+)"

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

        stem_chs = [64, 32, 64]
        strides = [1, 2, 2, 2]
        head_dim = 32
        mix_block_ratio = 0.75
        sr_ratios = [8, 4, 2, 1]
        depths: list[int] = self.config["depths"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.stage_out_channels = [
            [96] * (depths[0]),
            [192] * (depths[1] - 1) + [256],
            [384, 384, 384, 384, 512] * (depths[2] // 5),
            [768] * (depths[3] - 1) + [1024],
        ]

        self.stage_block_types = [
            [NCB] * depths[0],
            [NCB] * (depths[1] - 1) + [NTB],
            [NCB, NCB, NCB, NCB, NTB] * (depths[2] // 5),
            [NCB] * (depths[3] - 1) + [NTB],
        ]

        self.stem = nn.Sequential(
            Conv2dNormActivation(3, stem_chs[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            Conv2dNormActivation(stem_chs[0], stem_chs[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dNormActivation(stem_chs[1], stem_chs[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dNormActivation(stem_chs[2], stem_chs[2], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        input_channel = stem_chs[-1]
        idx = 0
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for stage_id, repeats in enumerate(depths):
            layers = []
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(repeats):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = (2, 2)
                else:
                    stride = (1, 1)

                output_channel = output_channels[block_id]
                block_type = block_types[block_id]  # type: ignore[index]
                if block_type is NCB:
                    layer = NCB(
                        input_channel,
                        output_channel,
                        stride=stride,
                        drop_path=dpr[idx + block_id],
                        head_dim=head_dim,
                        mlp_ratio=3,
                    )
                    layers.append(layer)

                elif block_type is NTB:
                    layer = NTB(
                        input_channel,
                        output_channel,
                        drop_path=dpr[idx + block_id],
                        stride=stride,
                        sr_ratio=sr_ratios[stage_id],
                        mlp_ratio=2,
                        head_dim=head_dim,
                        mix_block_ratio=mix_block_ratio,
                    )
                    layers.append(layer)

                else:
                    raise ValueError("Unsupported block type")

                input_channel = output_channel

            stages[f"stage{stage_id+1}"] = nn.Sequential(*layers)
            return_channels.append(output_channel)
            idx += repeats

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.BatchNorm2d(output_channel),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = output_channel
        self.classifier = self.create_classifier()

        self.stem_stride = 4
        self.stem_width = stem_chs[-1]
        self.encoding_size = output_channel

        # Weights initialization
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

    def masked_encoding_retention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: Optional[torch.Tensor] = None,
        return_keys: Literal["all", "features", "embedding"] = "features",
    ) -> TokenRetentionResultType:
        x = self.stem(x)
        x = mask_tensor(x, mask, patch_factor=self.max_stride // self.stem_stride, mask_token=mask_token)
        x = self.body(x)

        result: TokenRetentionResultType = {}
        if return_keys in ("all", "features"):
            result["features"] = x
        if return_keys in ("all", "embedding"):
            result["embedding"] = self.features(x)

        return result

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)


registry.register_model_config("nextvit_s", NextViT, config={"depths": [3, 4, 10, 3], "drop_path_rate": 0.1})
registry.register_model_config("nextvit_b", NextViT, config={"depths": [3, 4, 20, 3], "drop_path_rate": 0.2})
registry.register_model_config("nextvit_l", NextViT, config={"depths": [3, 4, 30, 3], "drop_path_rate": 0.2})
