"""
MogaNet, adapted from
https://github.com/Westlake-AI/MogaNet/blob/main/models/moganet.py

Paper "MogaNet: Multi-order Gated Aggregation Network",
https://arxiv.org/abs/2211.03295
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


class ElementScale(nn.Module):
    def __init__(self, embed_dims: int, init_value: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    def __init__(self, embed_dims: int, feedforward_channels: int, kernel_size: int, ffn_drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=feedforward_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.dwconv = nn.Conv2d(
            feedforward_channels,
            feedforward_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=(kernel_size // 2, kernel_size // 2),
            groups=feedforward_channels,
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(feedforward_channels, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))  # C -> 1
        self.sigma = ElementScale(feedforward_channels, init_value=1e-5)
        self.decompose_act = nn.GELU()
        self.fc2 = nn.Conv2d(feedforward_channels, embed_dims, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)

        # Proj 2
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        x = self.fc2(x)
        x = self.drop(x)

        return x


class MultiOrderDWConv(nn.Module):
    def __init__(self, embed_dims: int, dw_dilation: list[int], channel_split: list[int]) -> None:
        super().__init__()

        split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # Padding: dilation * (kernel_size[0] - 1) // 2, dilation * (kernel_size[1] - 1) // 2
        self.dw_conv0 = nn.Conv2d(
            self.embed_dims,
            self.embed_dims,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=((1 + 4 * dw_dilation[0]) // 2, (1 + 4 * dw_dilation[0]) // 2),
            dilation=dw_dilation[0],
            groups=self.embed_dims,
        )
        self.dw_conv1 = nn.Conv2d(
            self.embed_dims_1,
            self.embed_dims_1,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=((1 + 4 * dw_dilation[1]) // 2, (1 + 4 * dw_dilation[1]) // 2),
            dilation=dw_dilation[1],
            groups=self.embed_dims_1,
        )
        self.dw_conv2 = nn.Conv2d(
            self.embed_dims_2,
            self.embed_dims_2,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=((1 + 6 * dw_dilation[2]) // 2, (1 + 6 * dw_dilation[2]) // 2),
            dilation=dw_dilation[2],
            groups=self.embed_dims_2,
        )
        self.pw_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = self.dw_conv0(x)
        x_1 = self.dw_conv1(x_0[:, self.embed_dims_0 : self.embed_dims_0 + self.embed_dims_1, ...])
        x_2 = self.dw_conv2(x_0[:, self.embed_dims - self.embed_dims_2 :, ...])

        x = torch.concat([x_0[:, : self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.pw_conv(x)

        return x


class MultiOrderGatedAggregation(nn.Module):
    def __init__(self, embed_dims: int, attn_dw_dilation: list[int], attn_channel_split: list[int]) -> None:
        super().__init__()

        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims, dw_dilation=attn_dw_dilation, channel_split=attn_channel_split
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()
        self.sigma = ElementScale(embed_dims, init_value=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.proj_1(x)
        x_d = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)

        g = self.gate(x)
        v = self.value(x)

        x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        x = x + shortcut

        return x


class MogaBlock(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        ffn_ratio: float,
        drop_rate: float,
        drop_path_rate: float,
        init_value: float,
        attn_dw_dilation: list[int],
        attn_channel_split: list[int],
    ) -> None:
        super().__init__()

        # Spatial attention
        self.norm1 = nn.BatchNorm2d(embed_dims, eps=1e-5)
        self.attn = MultiOrderGatedAggregation(
            embed_dims, attn_dw_dilation=attn_dw_dilation, attn_channel_split=attn_channel_split
        )
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

        # Channel MLP
        self.norm2 = nn.BatchNorm2d(embed_dims, eps=1e-5)
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            kernel_size=3,
            ffn_drop=drop_rate,
        )
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

        self.drop_path = StochasticDepth(drop_path_rate, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)

        # Channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)

        return x


class MogaNet(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.stage\d+\.(\d+)"

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

        init_value = 1e-5
        drop_rate = 0.0
        attn_dw_dilation = [1, 2, 3]
        attn_channel_split = [1, 3, 4]
        ffn_ratios = [8, 8, 4, 4]
        embed_dims: list[int] = self.config["embed_dims"]
        depths: list[int] = self.config["depths"]
        drop_path_rate: float = self.config["drop_path_rate"]

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
        )

        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        cur_block_idx = 0
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i, depth in enumerate(depths):
            blocks = []
            if i > 0:
                blocks.append(
                    Conv2dNormActivation(
                        embed_dims[i - 1],
                        embed_dims[i],
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=(1, 1),
                        activation_layer=None,
                    )
                )

            for j in range(depth):
                blocks.append(
                    MogaBlock(
                        embed_dims=embed_dims[i],
                        ffn_ratio=ffn_ratios[i],
                        drop_rate=drop_rate,
                        drop_path_rate=dpr[cur_block_idx + j],
                        init_value=init_value,
                        attn_dw_dilation=attn_dw_dilation,
                        attn_channel_split=attn_channel_split,
                    )
                )
            cur_block_idx += depth
            blocks.append(nn.BatchNorm2d(embed_dims[i]))
            stages[f"stage{i+1}"] = nn.Sequential(*blocks)
            return_channels.append(embed_dims[i])

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embed_dims[-1]
        self.classifier = self.create_classifier()

        self.stem_stride = 4
        self.stem_width = embed_dims[0]
        self.encoding_size = embed_dims[-1]

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
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
    "moganet_xt",
    MogaNet,
    config={"embed_dims": [32, 64, 96, 192], "depths": [3, 3, 10, 2], "drop_path_rate": 0.05},
)
registry.register_model_config(
    "moganet_t",
    MogaNet,
    config={"embed_dims": [32, 64, 128, 256], "depths": [3, 3, 12, 2], "drop_path_rate": 0.1},
)
registry.register_model_config(
    "moganet_s",
    MogaNet,
    config={"embed_dims": [64, 128, 320, 512], "depths": [2, 3, 12, 2], "drop_path_rate": 0.1},
)
registry.register_model_config(
    "moganet_b",
    MogaNet,
    config={"embed_dims": [64, 160, 320, 512], "depths": [4, 6, 22, 3], "drop_path_rate": 0.2},
)
registry.register_model_config(
    "moganet_l",
    MogaNet,
    config={"embed_dims": [64, 160, 320, 640], "depths": [4, 6, 44, 4], "drop_path_rate": 0.3},
)
registry.register_model_config(
    "moganet_xl",
    MogaNet,
    config={"embed_dims": [96, 192, 480, 960], "depths": [6, 6, 44, 4], "drop_path_rate": 0.4},
)

registry.register_weights(
    "moganet_xt_il-common",
    {
        "description": "MogaNet X-Tiny model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 11.1,
                "sha256": "2b353981631650be749bc5d51b4af3aafe36a562be3f5cb0a2d91cec383a3253",
            }
        },
        "net": {"network": "moganet_xt", "tag": "il-common"},
    },
)
