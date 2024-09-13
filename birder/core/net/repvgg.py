"""
RepVGG, adapted from
https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py

Paper "RepVGG: Making VGG-style ConvNets Great Again", https://arxiv.org/abs/2101.03697
"""

# Reference license: MIT

from typing import Optional

import torch
from torch import nn
from torchvision.ops import SqueezeExcitation

from birder.core.net.base import BaseNet
from birder.model_registry import registry


class RepVggBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
        use_se: bool,
        reparameterized: bool,
    ) -> None:
        super().__init__()
        self.reparameterized = reparameterized
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.groups = groups

        if use_se is True:
            self.se = SqueezeExcitation(out_channels, out_channels // 16)
        else:
            self.se = nn.Identity()

        self.activation = nn.ReLU()

        if reparameterized is True:
            self.reparam_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=padding,
                groups=groups,
                bias=True,
            )
        else:
            self.reparam_conv = None

            self.rbr_identity = None
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)

            self.conv_kxk = nn.Sequential()
            self.conv_kxk.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size, kernel_size),
                    stride=(stride, stride),
                    padding=(1, 1),
                    groups=self.groups,
                    bias=False,
                ),
            )
            self.conv_kxk.add_module("bn", nn.BatchNorm2d(num_features=out_channels))

            self.conv_1x1 = nn.Sequential()
            self.conv_1x1.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, stride),
                    padding=(0, 0),
                    groups=self.groups,
                    bias=False,
                ),
            )
            self.conv_1x1.add_module("bn", nn.BatchNorm2d(num_features=out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reparameterized forward pass
        if self.reparam_conv is not None:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass
        if self.rbr_identity is None:
            x = self.conv_1x1(x) + self.conv_kxk(x)
        else:
            identity = self.rbr_identity(x)
            x = self.conv_1x1(x) + self.conv_kxk(x) + identity

        return self.activation(self.se(x))

    def reparameterize(self) -> None:
        if self.reparameterized is True:
            return

        (kernel, bias) = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.conv_kxk.conv.in_channels,
            out_channels=self.conv_kxk.conv.out_channels,
            kernel_size=self.conv_kxk.conv.kernel_size,
            stride=self.conv_kxk.conv.stride,
            padding=self.conv_kxk.conv.padding,
            dilation=self.conv_kxk.conv.dilation,
            groups=self.conv_kxk.conv.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for param in self.parameters():
            param.detach_()

        del self.conv_kxk
        del self.conv_1x1
        if hasattr(self, "rbr_identity") is True:
            del self.rbr_identity

        self.reparameterized = True

    def _get_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Get weights and bias of scale branch
        kernel_1x1 = 0
        bias_1x1 = 0
        if self.conv_1x1 is not None:
            kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.conv_1x1)
            pad = self.kernel_size // 2
            kernel_1x1 = torch.nn.functional.pad(kernel_1x1, [pad, pad, pad, pad])

        # Get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_identity is not None:
            (kernel_identity, bias_identity) = self._fuse_bn_tensor(self.rbr_identity)

        # Get weights and bias of conv branches
        (kernel_conv, bias_conv) = self._fuse_bn_tensor(self.conv_kxk)

        kernel_final = kernel_conv + kernel_1x1 + kernel_identity
        bias_final = bias_conv + bias_1x1 + bias_identity

        return (kernel_final, bias_final)

    def _fuse_bn_tensor(self, branch: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential) is True:
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps

        else:
            assert isinstance(branch, nn.BatchNorm2d)
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros(
                (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                dtype=branch.weight.dtype,
                device=branch.weight.device,
            )
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1

            kernel = kernel_value
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        return kernel * t, beta - running_mean * gamma / std


class RepVggStage(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        num_blocks: int,
        groups: int,
        prev_blocks: int,
        use_se: bool,
        reparameterized: bool,
    ) -> None:
        super().__init__()
        strides = [2] + [1] * (num_blocks - 1)
        for idx, stride in enumerate(strides):
            if (prev_blocks + idx) < 27 and (prev_blocks + idx) % 2 == 0:
                current_groups = groups

            else:
                current_groups = 1

            self.append(
                RepVggBlock(
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=current_groups,
                    use_se=use_se,
                    reparameterized=reparameterized,
                )
            )
            in_planes = planes


class RepVgg(BaseNet):
    default_size = 224

    # pylint: disable=too-many-statements
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

        self.reparameterized = False
        widths = [64, 128, 256, 512]
        if net_param == 0:
            # A0
            width_multipliers = [0.75, 0.75, 0.75, 2.5]
            num_blocks_per_stage = [2, 4, 14, 1]
            groups = 1
            use_se = False

        elif net_param == 1:
            # A1
            width_multipliers = [1, 1, 1, 2.5]
            num_blocks_per_stage = [2, 4, 14, 1]
            groups = 1
            use_se = False

        elif net_param == 2:
            # A2
            width_multipliers = [1.5, 1.5, 1.5, 2.75]
            num_blocks_per_stage = [2, 4, 14, 1]
            groups = 1
            use_se = False

        elif net_param == 3:
            # B0
            width_multipliers = [1, 1, 1, 2.5]
            num_blocks_per_stage = [4, 6, 16, 1]
            groups = 1
            use_se = False

        elif net_param == 4:
            # B1
            width_multipliers = [2, 2, 2, 4]
            num_blocks_per_stage = [4, 6, 16, 1]
            groups = 1
            use_se = False

        elif net_param == 5:
            # B1 G4
            width_multipliers = [2, 2, 2, 4]
            num_blocks_per_stage = [4, 6, 16, 1]
            groups = 4
            use_se = False

        elif net_param == 6:
            # B2
            width_multipliers = [2.5, 2.5, 2.5, 5]
            num_blocks_per_stage = [4, 6, 16, 1]
            groups = 1
            use_se = False

        elif net_param == 7:
            # B2 G4
            width_multipliers = [2.5, 2.5, 2.5, 5]
            num_blocks_per_stage = [4, 6, 16, 1]
            groups = 4
            use_se = False

        elif net_param == 8:
            # B3
            width_multipliers = [3, 3, 3, 5]
            num_blocks_per_stage = [4, 6, 16, 1]
            groups = 1
            use_se = False

        elif net_param == 9:
            # B3 G4
            width_multipliers = [3, 3, 3, 5]
            num_blocks_per_stage = [4, 6, 16, 1]
            groups = 4
            use_se = False

        elif net_param == 10:
            # D2 SE
            width_multipliers = [2.5, 2.5, 2.5, 5]
            num_blocks_per_stage = [8, 14, 24, 1]
            groups = 1
            use_se = True

        else:
            raise ValueError(f"net_param = {net_param} not supported")

        in_planes = min(64, int(widths[0] * width_multipliers[0]))

        self.stem = RepVggBlock(
            in_channels=self.input_channels,
            out_channels=in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            use_se=False,
            reparameterized=self.reparameterized,
        )

        stages = []
        prev_blocks = 1  # Due to stem
        for idx in range(len(num_blocks_per_stage)):  # pylint: disable=consider-using-enumerate
            out_planes = int(widths[idx] * width_multipliers[idx])
            stages.append(
                RepVggStage(
                    in_planes,
                    out_planes,
                    num_blocks_per_stage[idx],
                    groups=groups,
                    prev_blocks=prev_blocks,
                    use_se=use_se,
                    reparameterized=self.reparameterized,
                )
            )
            in_planes = out_planes
            prev_blocks += num_blocks_per_stage[idx]

        self.body = nn.Sequential(*stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = int(widths[-1] * width_multipliers[3])
        self.classifier = self.create_classifier()

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)

    def reparameterize_model(self) -> None:
        for module in self.modules():
            if hasattr(module, "reparameterize") is True:
                module.reparameterize()

        self.reparameterized = True


registry.register_alias("repvgg_a0", RepVgg, 0)
registry.register_alias("repvgg_a1", RepVgg, 1)
registry.register_alias("repvgg_a2", RepVgg, 2)
registry.register_alias("repvgg_b0", RepVgg, 3)
registry.register_alias("repvgg_b1", RepVgg, 4)
registry.register_alias("repvgg_b1g4", RepVgg, 5)
registry.register_alias("repvgg_b2", RepVgg, 6)
registry.register_alias("repvgg_b2g4", RepVgg, 7)
registry.register_alias("repvgg_b3", RepVgg, 8)
registry.register_alias("repvgg_b3g4", RepVgg, 9)
registry.register_alias("repvgg_d2se", RepVgg, 10)