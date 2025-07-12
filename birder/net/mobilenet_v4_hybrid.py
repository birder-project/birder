"""
MobileNet v4, adapted from
https://github.com/jaiwei98/MobileNetV4-pytorch/blob/main/mobilenet/mobilenetv4.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention2d.py

Paper "MobileNetV4 -- Universal Models for the Mobile Ecosystem", https://arxiv.org/abs/2404.10518
"""

# Reference license: MIT and Apache-2.0

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
from birder.layers import LayerScale2d
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType
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
            nn.Conv2d(
                in_channels,
                self.num_heads * self.key_dim,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            )
        )
        self.query = nn.Sequential(*query_layers)

        key_layers = []
        value_layers = []
        if kv_strides[0] > 1 or kv_strides[1] > 1:
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

        key_layers.append(
            nn.Conv2d(in_channels, self.key_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        )
        value_layers.append(
            nn.Conv2d(in_channels, value_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        )
        self.key = nn.Sequential(*key_layers)
        self.value = nn.Sequential(*value_layers)

        output_layers = []
        if query_strides[0] > 1 or query_strides[0] > 1:
            output_layers.append(nn.Upsample(scale_factor=self.query_strides, mode="bilinear", align_corners=False))

        output_layers.append(
            nn.Conv2d(
                self.num_heads * value_dim, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
            )
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
        attn_score = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)  # pylint: disable=not-callable
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
    def __init__(self, cnf: MultiQueryAttentionBlockConfig, stochastic_depth_prob: float) -> None:
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
        self.layer_scale = LayerScale2d(cnf.in_channels, 1e-5)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.input_norm(x)
        x = self.multi_query_attention(x)
        x = self.layer_scale(x)
        x = self.stochastic_depth(x) + shortcut

        return x


# pylint: disable=invalid-name
class MobileNet_v4_Hybrid(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.stage\d+\.(\d+)"

    # pylint: disable=too-many-branches
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

        dropout: float = self.config["dropout"]
        drop_path_rate: float = self.config["drop_path_rate"]
        net_size: str = self.config["net_size"]

        if net_size == "medium":
            stem_settings = ConvNormActConfig(0, 32, (3, 3), (2, 2), (1, 1))
            net_settings: list[Any] = [
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
            ]
            features_stage_settings = ConvNormActConfig(960, 1280, (1, 1), (1, 1), (0, 0))
        elif net_size == "large":
            stem_settings = ConvNormActConfig(0, 24, (3, 3), (2, 2), (1, 1), nn.GELU)
            net_settings = [
                # Stage 1
                InvertedResidualConfig(24, 48, (3, 3), (2, 2), (1, 1), 4.0, False, nn.GELU),
                # Stage 2
                UniversalInvertedBottleneckConfig(48, 96, 4.0, (3, 3), (5, 5), (2, 2), True, nn.GELU),
                UniversalInvertedBottleneckConfig(96, 96, 4.0, (3, 3), (3, 3), (1, 1), True, nn.GELU),
                # Stage 3
                UniversalInvertedBottleneckConfig(96, 192, 4.0, (3, 3), (5, 5), (2, 2), True, nn.GELU),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), (3, 3), (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), (3, 3), (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), (3, 3), (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), (5, 5), (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True, nn.GELU),
                MultiQueryAttentionBlockConfig(192, (3, 3), 8, 48, 48, (2, 2), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True, nn.GELU),
                MultiQueryAttentionBlockConfig(192, (3, 3), 8, 48, 48, (2, 2), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True, nn.GELU),
                MultiQueryAttentionBlockConfig(192, (3, 3), 8, 48, 48, (2, 2), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (5, 5), (3, 3), (1, 1), True, nn.GELU),
                MultiQueryAttentionBlockConfig(192, (3, 3), 8, 48, 48, (2, 2), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(192, 192, 4.0, (3, 3), None, (1, 1), True, nn.GELU),
                # Stage 4
                UniversalInvertedBottleneckConfig(192, 512, 4.0, (5, 5), (5, 5), (2, 2), True, nn.GELU),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (5, 5), (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (5, 5), (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (5, 5), (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (3, 3), (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (3, 3), (1, 1), True, nn.GELU),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), (5, 5), (1, 1), True, nn.GELU),
                MultiQueryAttentionBlockConfig(512, (3, 3), 8, 64, 64, (1, 1), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True, nn.GELU),
                MultiQueryAttentionBlockConfig(512, (3, 3), 8, 64, 64, (1, 1), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True, nn.GELU),
                MultiQueryAttentionBlockConfig(512, (3, 3), 8, 64, 64, (1, 1), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True, nn.GELU),
                MultiQueryAttentionBlockConfig(512, (3, 3), 8, 64, 64, (1, 1), (1, 1), 0.0),
                UniversalInvertedBottleneckConfig(512, 512, 4.0, (5, 5), None, (1, 1), True, nn.GELU),
            ]
            last_stage_settings = [
                ConvNormActConfig(512, 960, (1, 1), (1, 1), (0, 0), nn.GELU),
            ]
            features_stage_settings = ConvNormActConfig(960, 1280, (1, 1), (1, 1), (0, 0), nn.GELU)
        else:
            raise ValueError(f"Unknown net_size '{net_size}'")

        self.stem = Conv2dNormActivation(
            self.input_channels,
            stem_settings.out_channels,
            kernel_size=stem_settings.kernel,
            stride=stem_settings.stride,
            padding=stem_settings.padding,
            activation_layer=stem_settings.activation,
            inplace=None,
        )

        layers: list[nn.Module] = []
        total_stage_blocks = len(net_settings) + len(last_stage_settings)
        i = 0
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for idx, block_settings in enumerate(net_settings):
            # Adjust stochastic depth probability based on the depth of the stage block
            sd_prob = drop_path_rate * float(idx) / total_stage_blocks

            if idx > 0 and (block_settings.stride[0] > 1 or block_settings.stride[1] > 1):
                stages[f"stage{i+1}"] = nn.Sequential(*layers)
                return_channels.append(net_settings[idx - 1].out_channels)
                layers = []
                i += 1

            if isinstance(block_settings, ConvNormActConfig):
                layers.append(
                    Conv2dNormActivation(
                        block_settings.in_channels,
                        block_settings.out_channels,
                        kernel_size=block_settings.kernel,
                        stride=block_settings.stride,
                        padding=block_settings.padding,
                        activation_layer=block_settings.activation,
                        inplace=None,
                    )
                )

            elif isinstance(block_settings, InvertedResidualConfig):
                layers.append(InvertedResidual(block_settings, sd_prob))

            elif isinstance(block_settings, UniversalInvertedBottleneckConfig):
                layers.append(UniversalInvertedBottleneck(block_settings, 1e-5, sd_prob))

            elif isinstance(block_settings, MultiQueryAttentionBlockConfig):
                layers.append(MultiQueryAttentionBlock(block_settings, sd_prob))

            else:
                raise ValueError("Unknown config")

        stages[f"stage{i+1}"] = nn.Sequential(*layers)
        return_channels.append(net_settings[-1].out_channels)
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
                    activation_layer=block_settings.activation,
                    inplace=None,
                )
            )
        stages[f"stage{i+1}"] = nn.Sequential(*layers)
        return_channels.append(last_stage_settings[-1].out_channels)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Conv2dNormActivation(
                features_stage_settings.in_channels,
                features_stage_settings.out_channels,
                kernel_size=features_stage_settings.kernel,
                stride=features_stage_settings.stride,
                padding=features_stage_settings.padding,
                activation_layer=features_stage_settings.activation,
                inplace=None,
            ),
            nn.Flatten(1),
            nn.Dropout(p=dropout),
        )
        self.return_channels = return_channels[:4]
        self.embedding_size = features_stage_settings.out_channels
        self.classifier = self.create_classifier()

        self.stem_stride = stem_settings.stride[0]
        self.stem_width = stem_settings.out_channels
        self.encoding_size = features_stage_settings.out_channels

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def transform_to_backbone(self) -> None:
        self.body.stage5 = nn.Identity()
        self.features = nn.Identity()
        self.classifier = nn.Identity()

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


registry.register_model_config(
    "mobilenet_v4_hybrid_m", MobileNet_v4_Hybrid, config={"dropout": 0.2, "drop_path_rate": 0.075, "net_size": "medium"}
)
registry.register_model_config(
    "mobilenet_v4_hybrid_l", MobileNet_v4_Hybrid, config={"dropout": 0.2, "drop_path_rate": 0.35, "net_size": "large"}
)

registry.register_weights(
    "mobilenet_v4_hybrid_m_il-common",
    {
        "description": "MobileNet hybrid v4 medium model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 39.7,
                "sha256": "220df49e08ea49e24f30dcc777bf48c7aaea4aaa5909b56c931f41747381d390",
            }
        },
        "net": {"network": "mobilenet_v4_hybrid_m", "tag": "il-common"},
    },
)
