"""
Normalizer-Free Networks, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/nfnet.py

Paper "High-Performance Large-Scale Image Recognition Without Normalization", https://arxiv.org/abs/2102.06171
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import SqueezeExcitation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import make_divisible


class ScaledStdConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        groups: int = 1,
        bias: bool = True,
        gamma: float = 1.0,
        eps: float = 1e-5,
        gain_init: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-renamed
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1),
            None,
            None,
            weight=(self.gain * self.scale).view(-1),
            training=True,
            momentum=0.0,
            eps=self.eps,
        ).reshape_as(self.weight)
        z = F.conv2d(  # pylint: disable=not-callable
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return z


class GammaAct(nn.Module):
    # Activation Gamma
    # identity:    1.0
    # celu:        1.270926833152771
    # elu:         1.2716004848480225
    # gelu:        1.7015043497085571
    # leaky_relu:  1.70590341091156
    # log_sigmoid: 1.9193484783172607
    # log_softmax: 1.0002083778381348
    # relu:        1.7139588594436646
    # relu6:       1.7131484746932983
    # selu:        1.0008515119552612
    # sigmoid:     4.803835391998291
    # silu:        1.7881293296813965
    # softsign:    2.338853120803833
    # softplus:    1.9203323125839233
    # tanh:        1.5939117670059204
    def __init__(self, activation: Callable[..., nn.Module], gamma: float) -> None:
        super().__init__()
        self.act = activation()
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(x)
        x = x * self.gamma
        return x


class DownsampleAvg(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if stride > 1:
            self.pool = nn.AvgPool2d(
                kernel_size=(stride, stride),
                stride=(stride, stride),
                padding=(0, 0),
                ceil_mode=True,
                count_include_pad=False,
            )
        else:
            self.pool = nn.Identity()

        self.conv = ScaledStdConv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class NormFreeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        alpha: float,
        beta: float,
        bottle_ratio: float,
        group_size: int,
        skip_init: bool,
        attn_gain: float,
        act_layer: Callable[..., nn.Module],
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        mid_channels = make_divisible(out_channels * bottle_ratio, 8)
        groups = mid_channels // group_size
        if group_size % 8 == 0:
            mid_channels = group_size * groups

        self.alpha = alpha
        self.beta = beta
        self.attn_gain = attn_gain

        if in_channels == out_channels and stride == 1:
            self.downsample = None
        else:
            self.downsample = DownsampleAvg(in_channels, out_channels, stride=stride)

        self.act1 = act_layer()
        self.conv1 = ScaledStdConv2d(in_channels, mid_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.act2 = act_layer()
        self.conv2 = ScaledStdConv2d(
            mid_channels, mid_channels, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), groups=groups
        )
        self.act2b = act_layer()
        self.conv2b = ScaledStdConv2d(
            mid_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=groups
        )

        self.act3 = act_layer()
        self.conv3 = ScaledStdConv2d(
            mid_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            gain_init=1.0 if skip_init else 0.0,
        )
        self.attn = SqueezeExcitation(out_channels, out_channels // 2)
        self.drop_path = StochasticDepth(drop_path_rate, mode="row")
        self.skip_init_gain = nn.Parameter(torch.tensor(0.0)) if skip_init else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(x) * self.beta

        # Shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(out)

        # Residual branch
        out = self.conv1(out)
        out = self.conv2(self.act2(out))
        out = self.conv2b(self.act2b(out))
        out = self.conv3(self.act3(out))
        out = self.attn_gain * self.attn(out)
        out = self.drop_path(out)

        if self.skip_init_gain is not None:
            out = out.mul(self.skip_init_gain)

        out = out * self.alpha + shortcut

        return out


class NFNet(DetectorBackbone):
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

        alpha = 0.2
        stem_channels = 128
        channels = (256, 512, 1536, 1536)
        activation = nn.GELU
        activation_gamma = 1.7015043497085571
        depths: list[int] = self.config["depths"]
        drop_path_rate: float = self.config["drop_path_rate"]

        act_layer = partial(GammaAct, activation=activation, gamma=activation_gamma)
        self.stem = nn.Sequential(
            ScaledStdConv2d(self.input_channels, stem_channels // 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            act_layer(),
            ScaledStdConv2d(stem_channels // 8, stem_channels // 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            act_layer(),
            ScaledStdConv2d(stem_channels // 4, stem_channels // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            act_layer(),
            ScaledStdConv2d(stem_channels // 2, stem_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        drop_path_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_channels = stem_channels
        expected_var = 1.0
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for stage_idx, depth in enumerate(depths):
            stride = 1 if stage_idx == 0 else 2
            blocks = []
            out_channels = make_divisible(channels[stage_idx], 8)
            for block_idx in range(depth):
                blocks.append(
                    NormFreeBlock(
                        in_channels=prev_channels,
                        out_channels=out_channels,
                        stride=stride if block_idx == 0 else 1,
                        alpha=alpha,
                        beta=1.0 / expected_var**0.5,
                        bottle_ratio=0.5,
                        group_size=128,
                        skip_init=True,
                        attn_gain=2.0,
                        act_layer=act_layer,
                        drop_path_rate=drop_path_rates[stage_idx][block_idx],
                    )
                )
                if block_idx == 0:
                    expected_var = 1.0

                expected_var += alpha**2
                prev_channels = out_channels

            stages[f"stage{stage_idx+1}"] = nn.Sequential(*blocks)
            return_channels.append(out_channels)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            ScaledStdConv2d(prev_channels, prev_channels * 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = prev_channels * 2
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.bias, 0.0, 0.01)
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)


registry.register_model_config("nfnet_f0", NFNet, config={"depths": [1, 2, 6, 3], "drop_path_rate": 0.2})
registry.register_model_config("nfnet_f1", NFNet, config={"depths": [2, 4, 12, 6], "drop_path_rate": 0.3})
registry.register_model_config("nfnet_f2", NFNet, config={"depths": [3, 6, 18, 9], "drop_path_rate": 0.4})
registry.register_model_config("nfnet_f3", NFNet, config={"depths": [4, 8, 24, 12], "drop_path_rate": 0.4})
registry.register_model_config("nfnet_f4", NFNet, config={"depths": [5, 10, 30, 15], "drop_path_rate": 0.5})
registry.register_model_config("nfnet_f5", NFNet, config={"depths": [6, 12, 36, 18], "drop_path_rate": 0.5})
registry.register_model_config("nfnet_f6", NFNet, config={"depths": [7, 14, 42, 21], "drop_path_rate": 0.5})
