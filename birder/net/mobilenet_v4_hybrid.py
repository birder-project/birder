"""
MobileNet v4, adapted from
https://github.com/jaiwei98/MobileNetV4-pytorch/blob/main/mobilenet/mobilenetv4.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention2d.py

Paper "MobileNetV4 -- Universal Models for the Mobile Ecosystem", https://arxiv.org/abs/2404.10518
"""

# Reference license: MIT and Apache-2.0

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.mobilenet_v4 import ConvNormActConfig
from birder.net.mobilenet_v4 import InvertedResidual
from birder.net.mobilenet_v4 import InvertedResidualConfig
from birder.net.mobilenet_v4 import UniversalInvertedBottleneck
from birder.net.mobilenet_v4 import UniversalInvertedBottleneckConfig


class MultiQueryAttentionBlockConfig:
    def __init__(
        self,
        in_channels: int,
        dw_kernel_size: tuple[int, int],
        num_heads: int,
        key_dim: int,
        value_dim: int,
        kv_strides: tuple[int, int],
        query_strides: tuple[int, int],
        dropout: float,
    ) -> None:
        self.in_channels = in_channels
        self.dw_kernel_size = dw_kernel_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.kv_strides = kv_strides
        self.query_strides = query_strides
        self.dropout = dropout

        # Compatibility
        self.stride = self.query_strides
        self.out_channels = self.in_channels


class MultiQueryAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        query_strides: tuple[int, int],
        kv_strides: tuple[int, int],
        dw_kernel_size: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.query_strides = query_strides

        query_layers = []
        if query_strides[0] > 1 or query_strides[0] > 1:
            query_layers.append(nn.AvgPool2d(kernel_size=query_strides, stride=query_strides, padding=(0, 0)))
            query_layers.append(nn.BatchNorm2d(in_channels))

        query_layers.append(
            nn.Conv2d(in_channels, self.num_heads * self.key_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )
        self.query = nn.Sequential(*query_layers)

        key_layers = []
        value_layers = []
        if kv_strides[0] > 1 or kv_strides[1] > 0:
            key_layers.append(
                Conv2dNormActivation(
                    in_channels,
                    in_channels,
                    kernel_size=dw_kernel_size,
                    stride=kv_strides,
                    padding=((dw_kernel_size[0] - 1) // 2, (dw_kernel_size[1] - 1) // 2),
                    groups=in_channels,
                    activation_layer=None,
                )
            )
            value_layers.append(
                Conv2dNormActivation(
                    in_channels,
                    in_channels,
                    kernel_size=dw_kernel_size,
                    stride=kv_strides,
                    padding=((dw_kernel_size[0] - 1) // 2, (dw_kernel_size[1] - 1) // 2),
                    groups=in_channels,
                    activation_layer=None,
                )
            )

        key_layers.append(nn.Conv2d(in_channels, self.key_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        value_layers.append(nn.Conv2d(in_channels, value_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.key = nn.Sequential(*key_layers)
        self.value = nn.Sequential(*value_layers)

        output_layers = []
        if query_strides[0] > 1 or query_strides[0] > 1:
            output_layers.append(nn.Upsample(scale_factor=self.query_strides, mode="bilinear", align_corners=False))

        output_layers.append(
            nn.Conv2d(self.num_heads * value_dim, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )
        output_layers.append(nn.Dropout(p=dropout))
        self.output = nn.Sequential(*output_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, C, H, W) = x.size()
        q = self.query(x)
        q = q.reshape(B, self.num_heads, self.key_dim, -1)
        q = q.transpose(-1, -2).contiguous()

        k = self.key(x)
        (B, C, _, _) = k.size()
        k = k.reshape(B, C, -1).transpose(1, 2)
        k = k.unsqueeze(1).contiguous()

        v = self.value(x)
        (B, C, _, _) = v.size()
        v = v.reshape(B, C, -1).transpose(1, 2)
        v = v.unsqueeze(1).contiguous()

        # Calculate attention score
        attn_score = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)  # pylint:disable=not-callable
        (B, _, _, C) = attn_score.size()
        feat_dim = C * self.num_heads
        attn_score = attn_score.transpose(1, 2)
        attn_score = (
            attn_score.reshape(B, H // self.query_strides[0], W // self.query_strides[1], feat_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        x = self.output(attn_score)

        return x


class MultiQueryAttentionBlock(nn.Module):
    def __init__(self, cnf: MultiQueryAttentionBlockConfig) -> None:
        super().__init__()

        self.input_norm = nn.BatchNorm2d(cnf.in_channels)
        self.multi_query_attention = MultiQueryAttention(
            cnf.in_channels,
            cnf.num_heads,
            cnf.key_dim,
            cnf.value_dim,
            cnf.query_strides,
            cnf.kv_strides,
            cnf.dw_kernel_size,
            cnf.dropout,
        )
        self.gamma = nn.Parameter(1e-5 * torch.ones(cnf.in_channels, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.input_norm(x)
        x = self.multi_query_attention(x)
        x = x * self.gamma
        x = x + shortcut

        return x


# pylint: disable=invalid-name
class MobileNet_v4_Hybrid(DetectorBackbone):
    default_size = 224

    # pylint: disable=too-many-branches
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        assert self.net_param is not None, "must set net-param"
        net_param = int(self.net_param)

        if net_param == 0:
            # Medium
            dropout = 0.2
            stochastic_depth_prob = 0.075
            stem_settings = ConvNormActConfig(self.input_channels, 32, (3, 3), (2, 2), (1, 1))
            net_settings = [
                # Stage 1
                InvertedResidualConfig(32, 48, (3, 3), (2, 2), (1, 1), 4.0, False),
                # Stage 2
                UniversalInvertedBottleneckConfig(48, 80, 4.0, (3, 3), (5, 5), (2, 2), True),
                UniversalInvertedBottleneckConfig(80, 80, 2.0, (3, 3), (3, 3), (1, 1), True),
                # Stage 3
                UniversalInvertedBottleneckConfig(80, 160, 6.0, (3, 3), (5, 5), (2, 2), True),
                UniversalInvertedBottleneckConfig(160, 160, 2.0, None, None, (1, 1), True),
                UniversalInvertedBottleneckConfig(160, 160, 4.0, (3, 3), (3, 3), (1, 1), True),
                UniversalInvertedBottleneckConfig(160, 160, 4.0, (3, 3), (5, 5), (1, 1), True),
                MultiQueryAttentionBlockConfig(160, (3, 3), 4, 64, 64, (2, 2), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(160, 160, 4.0, (3, 3), (3, 3), (1, 1), True),
                MultiQueryAttentionBlockConfig(160, (3, 3), 4, 64, 64, (2, 2), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(160, 160, 4.0, (3, 3), None, (1, 1), True),
                MultiQueryAttentionBlockConfig(160, (3, 3), 4, 64, 64, (2, 2), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(160, 160, 4.0, (3, 3), (3, 3), (1, 1), True),
                MultiQueryAttentionBlockConfig(160, (3, 3), 4, 64, 64, (2, 2), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(160, 160, 4.0, (3, 3), None, (1, 1), True),
                # Stage 4
                UniversalInvertedBottleneckConfig(160, 256, 6.0, (5, 5), (5, 5), (2, 2), True),
                UniversalInvertedBottleneckConfig(256, 256, 4.0, (5, 5), (5, 5), (1, 1), True),
                UniversalInvertedBottleneckConfig(256, 256, 4.0, (3, 3), (5, 5), (1, 1), True),
                UniversalInvertedBottleneckConfig(256, 256, 4.0, (3, 3), (5, 5), (1, 1), True),
                UniversalInvertedBottleneckConfig(256, 256, 2.0, None, None, (1, 1), True),
                UniversalInvertedBottleneckConfig(256, 256, 2.0, (3, 3), (5, 5), (1, 1), True),
                UniversalInvertedBottleneckConfig(256, 256, 2.0, None, None, (1, 1), True),
                UniversalInvertedBottleneckConfig(256, 256, 4.0, None, None, (1, 1), True),
                MultiQueryAttentionBlockConfig(256, (3, 3), 4, 64, 64, (1, 1), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(256, 256, 4.0, (3, 3), None, (1, 1), True),
                MultiQueryAttentionBlockConfig(256, (3, 3), 4, 64, 64, (1, 1), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(256, 256, 4.0, (5, 5), (5, 5), (1, 1), True),
                MultiQueryAttentionBlockConfig(256, (3, 3), 4, 64, 64, (1, 1), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(256, 256, 4.0, (5, 5), None, (1, 1), True),
                MultiQueryAttentionBlockConfig(256, (3, 3), 4, 64, 64, (1, 1), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(256, 256, 4.0, (5, 5), None, (1, 1), True),
            ]
            last_stage_settings = [
                ConvNormActConfig(256, 960, (1, 1), (1, 1), (0, 0)),
                ConvNormActConfig(960, 1280, (1, 1), (1, 1), (0, 0)),
            ]

        elif net_param == 1:
            # Large
            dropout = 0.2
            stochastic_depth_prob = 0.35
            stem_settings = ConvNormActConfig(self.input_channels, 24, (3, 3), (2, 2), (1, 1))
            net_settings = [
                # Stage 1
                InvertedResidualConfig(24, 48, (3, 3), (2, 2), (1, 1), 4.0, False),
                # Stage 2
                UniversalInvertedBottleneckConfig(48, 96, 4.0, (3, 3), (5, 5), (2, 2), True),
                UniversalInvertedBottleneckConfig(96, 96, 4.0, (3, 3), (3, 3), (1, 1), True),
                # Stage 3
                UniversalInvertedBottleneckConfig(96, 192, 4.0, (3, 3), (5, 5), (2, 2), True),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), (3, 3), (1, 1), True),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), (3, 3), (1, 1), True),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), (3, 3), (1, 1), True),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), (5, 5), (1, 1), True),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True),
                MultiQueryAttentionBlockConfig(192, (3, 3), 8, 48, 48, (2, 2), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True),
                MultiQueryAttentionBlockConfig(192, (3, 3), 8, 48, 48, (2, 2), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True),
                MultiQueryAttentionBlockConfig(192, (3, 3), 8, 48, 48, (2, 2), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True),
                MultiQueryAttentionBlockConfig(192, (3, 3), 8, 48, 48, (2, 2), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), None, (1, 1), True),
                # Stage 4
                UniversalInvertedBottleneckConfig(192, 512, 4.0, (5, 5), (5, 5), (2, 2), True),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (5, 5), (1, 1), True),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (5, 5), (1, 1), True),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (5, 5), (1, 1), True),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (3, 3), (1, 1), True),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (3, 3), (1, 1), True),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (5, 5), (1, 1), True),
                MultiQueryAttentionBlockConfig(512, (3, 3), 8, 64, 64, (1, 1), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
                MultiQueryAttentionBlockConfig(512, (3, 3), 8, 64, 64, (1, 1), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
                MultiQueryAttentionBlockConfig(512, (3, 3), 8, 64, 64, (1, 1), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
                MultiQueryAttentionBlockConfig(512, (3, 3), 8, 64, 64, (1, 1), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True),
            ]
            last_stage_settings = [
                ConvNormActConfig(512, 960, (1, 1), (1, 1), (0, 0)),
                ConvNormActConfig(960, 1280, (1, 1), (1, 1), (0, 0)),
            ]

        else:
            raise ValueError(f"net_param = {net_param} not supported")

        self.stem = Conv2dNormActivation(
            stem_settings.in_channels,
            stem_settings.out_channels,
            kernel_size=stem_settings.kernel,
            stride=stem_settings.stride,
            padding=stem_settings.padding,
        )

        layers: list[nn.Module] = []
        total_stage_blocks = len(net_settings) + len(last_stage_settings)
        i = 0
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for idx, block_settings in enumerate(net_settings):
            # Adjust stochastic depth probability based on the depth of the stage block
            sd_prob = stochastic_depth_prob * float(idx) / total_stage_blocks

            if idx > 0 and (block_settings.stride[0] > 1 or block_settings.stride[1] > 1):  # type: ignore
                stages[f"stage{i}"] = nn.Sequential(*layers)
                return_channels.append(net_settings[idx - 1].out_channels)  # type:ignore
                layers = []
                i += 1

            if isinstance(block_settings, ConvNormActConfig) is True:
                layers.append(
                    Conv2dNormActivation(
                        block_settings.in_channels,  # type: ignore
                        block_settings.out_channels,  # type: ignore
                        kernel_size=block_settings.kernel,  # type: ignore
                        stride=block_settings.stride,  # type: ignore
                        padding=block_settings.padding,  # type: ignore
                    )
                )

            elif isinstance(block_settings, InvertedResidualConfig) is True:
                layers.append(InvertedResidual(block_settings, sd_prob))  # type: ignore

            elif isinstance(block_settings, UniversalInvertedBottleneckConfig) is True:
                layers.append(UniversalInvertedBottleneck(block_settings, sd_prob))  # type: ignore

            elif isinstance(block_settings, MultiQueryAttentionBlockConfig) is True:
                layers.append(MultiQueryAttentionBlock(block_settings))  # type: ignore

            else:
                raise ValueError("Unknown config")

        stages[f"stage{i}"] = nn.Sequential(*layers)
        return_channels.append(net_settings[idx - 1].out_channels)  # type:ignore
        layers = []
        i += 1
        for block_settings in last_stage_settings:
            layers.append(
                Conv2dNormActivation(
                    block_settings.in_channels,
                    block_settings.out_channels,
                    kernel_size=block_settings.kernel,
                    stride=block_settings.stride,
                    padding=block_settings.padding,
                )
            )
        stages[f"stage{i}"] = nn.Sequential(*layers)
        return_channels.append(last_stage_settings[-1].out_channels)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Dropout(p=dropout),
        )
        self.return_channels = return_channels[1:5]
        self.embedding_size = last_stage_settings[-1].out_channels
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) is True:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) is True:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear) is True:
                nn.init.normal_(m.weight, 0, 0.01)
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


registry.register_alias("mobilenet_v4_hybrid_m", MobileNet_v4_Hybrid, 0)
registry.register_alias("mobilenet_v4_hybrid_l", MobileNet_v4_Hybrid, 1)
