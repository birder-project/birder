"""
MobileOne, adapted from
https://github.com/apple/ml-mobileone/blob/main/mobileone.py

Paper "MobileOne: An Improved One millisecond Mobile Backbone",
https://arxiv.org/abs/2206.04040
"""

# Reference license: Apple MIT License

from typing import Optional

import torch
from torch import nn
from torchvision.ops import SqueezeExcitation

from birder.model_registry import registry
from birder.net.base import BaseNet


class MobileOneBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
        use_se: bool,
        num_conv_branches: int,
        reparameterized: bool,
    ) -> None:
        super().__init__()
        self.reparameterized = reparameterized
        self.groups = groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.num_conv_branches = num_conv_branches

        if use_se is True:
            self.se = SqueezeExcitation(out_channels, out_channels // 16)
        else:
            self.se = nn.Identity()

        self.activation = nn.ReLU()

        if reparameterized is True:
            self.reparam_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=True,
            )
        else:
            self.reparam_conv = None

            # Re-parameterizable skip connection
            self.rbr_skip = None
            if out_channels == in_channels and stride == 1:
                self.rbr_skip = nn.BatchNorm2d(num_features=in_channels)

            # Re-parameterizable conv branches
            self.rbr_conv = nn.ModuleList()
            for _ in range(self.num_conv_branches):
                mod_list = nn.Sequential()
                mod_list.add_module(
                    "conv",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=self.groups,
                        bias=False,
                    ),
                )
                mod_list.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
                self.rbr_conv.append(mod_list)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = nn.Sequential()
                self.rbr_scale.add_module(
                    "conv",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(1, 1),
                        stride=stride,
                        padding=(0, 0),
                        groups=self.groups,
                        bias=False,
                    ),
                )
                self.rbr_scale.add_module("bn", nn.BatchNorm2d(num_features=out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reparameterized forward pass
        if self.reparam_conv is not None:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        out = scale_out + identity_out
        for conv in self.rbr_conv:
            out += conv(x)

        return self.activation(self.se(out))

    def reparameterize(self) -> None:
        """
        Following works like "RepVGG: Making VGG-style ConvNets Great Again" - https://arxiv.org/pdf/2101.03697.pdf.
        We re-parameterize multi-branched architecture used at training time
        to obtain a plain CNN-like structure for inference.
        """

        if self.reparameterized is True:
            return

        (kernel, bias) = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.rbr_conv[0].conv.in_channels,
            out_channels=self.rbr_conv[0].conv.out_channels,
            kernel_size=self.rbr_conv[0].conv.kernel_size,
            stride=self.rbr_conv[0].conv.stride,
            padding=self.rbr_conv[0].conv.padding,
            dilation=self.rbr_conv[0].conv.dilation,
            groups=self.rbr_conv[0].conv.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for param in self.parameters():
            param.detach_()

        del self.rbr_conv
        del self.rbr_scale
        if hasattr(self, "rbr_skip") is True:
            del self.rbr_skip

        self.reparameterized = True

    def _get_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83
        """

        # Get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # Get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            (kernel_identity, bias_identity) = self._fuse_bn_tensor(self.rbr_skip)

        # Get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            (_kernel, _bias) = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity

        return (kernel_final, bias_final)

    def _fuse_bn_tensor(self, branch: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Method to fuse batchnorm layer with preceding convolution layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95
        """

        if isinstance(branch, nn.Sequential) is True:
            kernel_value = branch.conv.weight
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

            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        return (kernel_value * t, beta - running_mean * gamma / std)


class MobileOneStage(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        num_blocks: int,
        num_se_blocks: int,
        num_conv_branches: int,
        reparameterized: bool,
    ) -> None:
        super().__init__()
        strides = [2] + [1] * (num_blocks - 1)
        for idx, stride in enumerate(strides):
            use_se = False
            if idx >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            self.append(
                MobileOneBlock(
                    in_channels=in_planes,
                    out_channels=in_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_planes,
                    use_se=use_se,
                    num_conv_branches=num_conv_branches,
                    reparameterized=reparameterized,
                )
            )
            # Pointwise conv
            self.append(
                MobileOneBlock(
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    use_se=use_se,
                    num_conv_branches=num_conv_branches,
                    reparameterized=reparameterized,
                )
            )
            in_planes = planes


class MobileOne(BaseNet):
    default_size = 224

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
        num_blocks_per_stage = [2, 8, 10, 1]
        widths = [64, 128, 256, 512]
        if net_param == 0:
            # s0
            width_multipliers = [0.75, 1.0, 1.0, 2.0]
            num_conv_branches = 4
            num_se_blocks = [0, 0, 0, 0]

        elif net_param == 1:
            # s1
            width_multipliers = [1.5, 1.5, 2.0, 2.5]
            num_conv_branches = 1
            num_se_blocks = [0, 0, 0, 0]

        elif net_param == 2:
            # s2
            width_multipliers = [1.5, 2.0, 2.5, 4.0]
            num_conv_branches = 1
            num_se_blocks = [0, 0, 0, 0]

        elif net_param == 3:
            # s3
            width_multipliers = [2.0, 2.5, 3.0, 4.0]
            num_conv_branches = 1
            num_se_blocks = [0, 0, 0, 0]

        elif net_param == 4:
            # s4
            width_multipliers = [3.0, 3.5, 3.5, 4.0]
            num_conv_branches = 1
            num_se_blocks = [0, 0, 5, 1]

        else:
            raise ValueError(f"net_param = {net_param} not supported")

        in_planes = min(64, int(widths[0] * width_multipliers[0]))
        self.stem = MobileOneBlock(
            in_channels=self.input_channels,
            out_channels=in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            use_se=False,
            num_conv_branches=1,
            reparameterized=self.reparameterized,
        )

        stages = []
        for idx in range(len(num_blocks_per_stage)):  # pylint: disable=consider-using-enumerate
            out_planes = int(widths[idx] * width_multipliers[idx])
            stages.append(
                MobileOneStage(
                    in_planes,
                    out_planes,
                    num_blocks_per_stage[idx],
                    num_se_blocks=num_se_blocks[idx],
                    num_conv_branches=num_conv_branches,
                    reparameterized=self.reparameterized,
                )
            )
            in_planes = out_planes

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


registry.register_alias("mobileone_s0", MobileOne, 0)
registry.register_alias("mobileone_s1", MobileOne, 1)
registry.register_alias("mobileone_s2", MobileOne, 2)
registry.register_alias("mobileone_s3", MobileOne, 3)
registry.register_alias("mobileone_s4", MobileOne, 4)

registry.register_weights(
    "mobileone_s0_il-common",
    {
        "description": "MobileOne S0 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 18.5,
                "sha256": "57576873c88bb9dbdfeb4f89c9250c589e8ffbfe5c24f9c119bf5f4df40198db",
            }
        },
        "net": {"network": "mobileone_s0", "tag": "il-common"},
    },
)
registry.register_weights(
    "mobileone_s0_il-common_reparameterized",
    {
        "description": "MobileOne S0 (reparameterized) model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 5.5,
                "sha256": "43995722d2a7fcfacb837cf2fb595e16a6b55b91b996c5bd59b3105be3470d2b",
            }
        },
        "net": {"network": "mobileone_s0", "tag": "il-common_reparameterized", "reparameterized": True},
    },
)
