"""
UniRepLKNet, adapted from
https://github.com/AILab-CVC/UniRepLKNet/blob/main/unireplknet.py

Paper "UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud,
Time-Series and Image Recognition", https://arxiv.org/abs/2311.15599
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from functools import partial
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_weights
from torchvision.ops import SqueezeExcitation
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.layers import LayerNorm2d
from birder.layers import LayerScale2d
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType
from birder.net.base import staged_stochastic_depth_rates

# large_kernel_size: ((branch_kernel_size, dilation), ...)
_DILATED_BRANCH_SPECS: dict[int, tuple[tuple[int, int], ...]] = {
    5: ((3, 1), (3, 2)),
    7: ((5, 1), (3, 2), (3, 3)),
    9: ((5, 1), (5, 2), (3, 3), (3, 4)),
    11: ((5, 1), (5, 2), (3, 3), (3, 4), (3, 5)),
    13: ((5, 1), (7, 2), (3, 3), (3, 4), (3, 5)),
    15: ((5, 1), (7, 2), (3, 3), (3, 5), (3, 7)),
    17: ((5, 1), (9, 2), (3, 4), (3, 5), (3, 7)),
}


def _build_kernel_sizes(depths: list[int], stage3_kernel_pattern: list[int]) -> list[list[int]]:
    if len(depths) != 4:
        raise ValueError("depths must describe four stages")
    if len(stage3_kernel_pattern) == 0:
        raise ValueError("stage3_kernel_pattern must not be empty")

    num_repeats, remainder = divmod(depths[2], len(stage3_kernel_pattern))
    if remainder != 0:
        raise ValueError("stage3_kernel_pattern length must divide the stage 3 depth")

    return [
        [3] * depths[0],
        [13] * depths[1],
        stage3_kernel_pattern * num_repeats,
        [13] * depths[3],
    ]


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    return fuse_conv_bn_weights(  # type: ignore[no-any-return]
        conv.weight,
        conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )


def _new_fused_conv(conv: nn.Conv2d, kernel: torch.Tensor, bias: torch.Tensor) -> nn.Conv2d:
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        padding_mode=conv.padding_mode,
        device=kernel.device,
        dtype=kernel.dtype,
    )
    fused_conv.weight.copy_(kernel)
    fused_conv.bias.copy_(bias)

    return fused_conv


class GRN(nn.Module):
    """
    Global Response Normalization with a removable bias for reparameterization
    """

    def __init__(self, dim: int, use_bias: bool) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if use_bias is True:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        else:
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)  # pylint: disable=not-callable
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        x = self.gamma * (x * nx) + x
        if self.beta is not None:
            x = x + self.beta

        return x

    def remove_bias(self) -> None:
        self.register_parameter("beta", None)


class DilatedReparamBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, reparameterized: bool) -> None:
        super().__init__()
        if kernel_size not in _DILATED_BRANCH_SPECS:
            raise ValueError(f"Unsupported dilated reparameterization kernel size: {kernel_size}")

        self.channels = channels
        self.kernel_size = kernel_size
        self.reparameterized = reparameterized

        if reparameterized is True:
            self.reparam_conv = nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(1, 1),
                padding=(kernel_size // 2, kernel_size // 2),
                groups=channels,
            )
        else:
            self.reparam_conv = None
            self.origin = nn.Sequential()
            self.origin.add_module(
                "conv",
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(kernel_size, kernel_size),
                    stride=(1, 1),
                    padding=(kernel_size // 2, kernel_size // 2),
                    groups=channels,
                    bias=False,
                ),
            )
            self.origin.add_module("bn", nn.BatchNorm2d(channels))

            branches = []
            for branch_kernel_size, dilation in _DILATED_BRANCH_SPECS[kernel_size]:
                effective_kernel_size = dilation * (branch_kernel_size - 1) + 1
                branches.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        channels,
                                        channels,
                                        kernel_size=(branch_kernel_size, branch_kernel_size),
                                        stride=(1, 1),
                                        padding=(effective_kernel_size // 2, effective_kernel_size // 2),
                                        dilation=(dilation, dilation),
                                        groups=channels,
                                        bias=False,
                                    ),
                                ),
                                ("bn", nn.BatchNorm2d(channels)),
                            ]
                        )
                    )
                )

            self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            return self.reparam_conv(x)

        x_out = self.origin(x)
        for branch in self.branches:
            x_out = x_out + branch(x)

        return x_out

    @staticmethod
    def _convert_dilated_kernel(kernel: torch.Tensor, dilation: int) -> torch.Tensor:
        if dilation == 1:
            return kernel

        identity_kernel = torch.ones((1, 1, 1, 1), dtype=kernel.dtype, device=kernel.device)
        return F.conv_transpose2d(kernel, identity_kernel, stride=dilation)  # pylint: disable=not-callable

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def reparameterize(self) -> None:
        if self.reparameterized is True:
            return

        origin_conv = self.origin.conv
        origin_bn = self.origin.bn
        kernel, bias = _fuse_conv_bn(origin_conv, origin_bn)
        for branch, (_, dilation) in zip(self.branches, _DILATED_BRANCH_SPECS[self.kernel_size]):
            branch_kernel, branch_bias = _fuse_conv_bn(branch.conv, branch.bn)
            branch_kernel = self._convert_dilated_kernel(branch_kernel, dilation)
            pad = (self.kernel_size - branch_kernel.size(-1)) // 2
            kernel = kernel + F.pad(branch_kernel, [pad, pad, pad, pad])
            bias = bias + branch_bias

        self.reparam_conv = _new_fused_conv(origin_conv, kernel, bias)
        for param in self.parameters():
            param.detach_()

        del self.origin
        del self.branches
        self.reparameterized = True


class UniRepLKNetBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        ffn_ratio: float,
        layer_scale_init_value: Optional[float],
        stochastic_depth_prob: float,
        reparameterized: bool,
    ) -> None:
        super().__init__()
        self.reparameterized = reparameterized

        if kernel_size == 0:
            self.spatial_mixer: nn.Module = nn.Identity()
            self.spatial_norm: nn.Module = nn.Identity()
        elif kernel_size >= 7:
            self.spatial_mixer = DilatedReparamBlock(channels, kernel_size, reparameterized)
            if reparameterized is True:
                self.spatial_norm = nn.Identity()
            else:
                self.spatial_norm = nn.BatchNorm2d(channels)
        elif kernel_size in (3, 5):
            self.spatial_mixer = nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(1, 1),
                padding=(kernel_size // 2, kernel_size // 2),
                groups=channels,
                bias=reparameterized,
            )
            if reparameterized is True:
                self.spatial_norm = nn.Identity()
            else:
                self.spatial_norm = nn.BatchNorm2d(channels)
        else:
            raise ValueError("kernel_size must be 0, 3, 5, or a supported dilated reparameterization size")

        self.se = SqueezeExcitation(channels, channels // 4)

        hidden_channels = int(ffn_ratio * channels)
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.activation = nn.GELU()
        self.grn = GRN(hidden_channels, use_bias=not reparameterized)
        self.fc2 = nn.Linear(hidden_channels, channels, bias=reparameterized)
        if reparameterized is True:
            self.output_norm = nn.Identity()
        else:
            self.output_norm = nn.BatchNorm2d(channels)

        if reparameterized is False and layer_scale_init_value is not None and layer_scale_init_value > 0:
            self.layer_scale = LayerScale2d(channels, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.spatial_mixer(x)
        x = self.spatial_norm(x)
        x = self.se(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.output_norm(x)
        x = self.layer_scale(x)
        x = self.stochastic_depth(x)

        return identity + x

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def reparameterize(self) -> None:
        if self.reparameterized is True:
            return

        if isinstance(self.spatial_mixer, DilatedReparamBlock):
            self.spatial_mixer.reparameterize()
            spatial_conv = self.spatial_mixer.reparam_conv
        elif isinstance(self.spatial_mixer, nn.Conv2d):
            spatial_conv = self.spatial_mixer
        else:
            spatial_conv = None

        if spatial_conv is not None:
            kernel, bias = _fuse_conv_bn(spatial_conv, self.spatial_norm)
            spatial_conv = _new_fused_conv(spatial_conv, kernel, bias)
            if isinstance(self.spatial_mixer, DilatedReparamBlock):
                self.spatial_mixer.reparam_conv = spatial_conv
            else:
                self.spatial_mixer = spatial_conv

            self.spatial_norm = nn.Identity()

        if isinstance(self.layer_scale, LayerScale2d):
            final_scale = self.layer_scale.gamma.flatten()
        else:
            final_scale = torch.ones(  # type: ignore[unreachable]
                self.fc2.out_features,
                dtype=self.fc2.weight.dtype,
                device=self.fc2.weight.device,
            )

        if self.grn.beta is not None:
            projected_grn_bias = self.fc2.weight @ self.grn.beta.flatten()
        else:
            projected_grn_bias = torch.zeros_like(self.output_norm.running_mean)

        if self.fc2.bias is not None:
            projected_grn_bias = projected_grn_bias + self.fc2.bias

        std = torch.sqrt(self.output_norm.running_var + self.output_norm.eps)
        bn_scale = self.output_norm.weight / std
        new_fc2 = nn.Linear(
            self.fc2.in_features,
            self.fc2.out_features,
            device=self.fc2.weight.device,
            dtype=self.fc2.weight.dtype,
        )
        new_fc2.weight.copy_(self.fc2.weight * (bn_scale * final_scale).unsqueeze(1))
        new_fc2.bias.copy_(
            (self.output_norm.bias + (projected_grn_bias - self.output_norm.running_mean) * bn_scale) * final_scale
        )

        self.fc2 = new_fc2
        self.grn.remove_bias()
        self.output_norm = nn.Identity()
        self.layer_scale = nn.Identity()
        for param in self.parameters():
            param.detach_()

        self.reparameterized = True


class UniRepLKNetStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        ffn_ratio: float,
        layer_scale_init_value: Optional[float],
        stochastic_depth_probs: list[float],
        downsample: bool,
        reparameterized: bool,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                ),
                LayerNorm2d(out_channels, eps=1e-6),
            )
        else:
            assert in_channels == out_channels
            self.downsample = nn.Identity()

        self.blocks = nn.Sequential(
            *[
                UniRepLKNetBlock(
                    out_channels,
                    kernel_size=kernel_size,
                    ffn_ratio=ffn_ratio,
                    layer_scale_init_value=layer_scale_init_value,
                    stochastic_depth_prob=stochastic_depth_probs[idx],
                    reparameterized=reparameterized,
                )
                for idx, kernel_size in enumerate(kernel_sizes)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


class UniRepLKNet(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.stage(\d+)\.blocks\.(\d+)"

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, config=config, size=size)
        assert self.config is not None, "must set config"

        channels: list[int] = self.config["channels"]
        depths: list[int] = self.config["depths"]
        stage3_kernel_pattern: list[int] = self.config["stage3_kernel_pattern"]
        ffn_ratio: float = self.config.get("ffn_ratio", 4.0)
        layer_scale_init_value: Optional[float] = self.config.get("layer_scale_init_value", 1e-6)
        drop_path_rate: float = self.config["drop_path_rate"]

        if not len(channels) == len(depths) == 4:
            raise ValueError("channels and depths must describe four stages")

        kernel_sizes = _build_kernel_sizes(depths, stage3_kernel_pattern)
        self.reparameterized = False

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, channels[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            LayerNorm2d(channels[0] // 2, eps=1e-6),
            nn.GELU(),
            nn.Conv2d(channels[0] // 2, channels[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            LayerNorm2d(channels[0], eps=1e-6),
        )

        dpr = staged_stochastic_depth_rates(drop_path_rate, depths)
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        prev_channels = channels[0]
        for idx in range(4):
            stages[f"stage{idx+1}"] = UniRepLKNetStage(
                prev_channels,
                channels[idx],
                kernel_sizes=kernel_sizes[idx],
                ffn_ratio=ffn_ratio,
                layer_scale_init_value=layer_scale_init_value,
                stochastic_depth_probs=dpr[idx],
                downsample=idx > 0,
                reparameterized=self.reparameterized,
            )
            prev_channels = channels[idx]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(channels[-1], eps=1e-6),
        )
        self.return_channels = channels
        self.embedding_size = channels[-1]
        self.classifier = self.create_classifier()

        self.max_stride = 32
        self.stem_stride = 4
        self.stem_width = channels[0]
        self.feature_dim = channels[-1]
        self.decoder_block = partial(
            UniRepLKNetBlock,
            kernel_size=3,
            ffn_ratio=ffn_ratio,
            layer_scale_init_value=layer_scale_init_value,
            stochastic_depth_prob=0.0,
            reparameterized=False,
        )

        # Weight initialization
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
            param.requires_grad_(False)

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad_(False)

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

    def embedding_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.features(features)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def reparameterize_model(self) -> None:
        if self.reparameterized is True:
            return

        for module in self.modules():
            if isinstance(module, UniRepLKNetBlock):
                module.reparameterize()

        self.reparameterized = True


registry.register_model_config(
    "unireplknet_a",
    UniRepLKNet,
    config={
        "channels": [40, 80, 160, 320],
        "depths": [2, 2, 6, 2],
        "stage3_kernel_pattern": [13],
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "unireplknet_f",
    UniRepLKNet,
    config={
        "channels": [48, 96, 192, 384],
        "depths": [2, 2, 6, 2],
        "stage3_kernel_pattern": [13],
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "unireplknet_p",
    UniRepLKNet,
    config={
        "channels": [64, 128, 256, 512],
        "depths": [2, 2, 6, 2],
        "stage3_kernel_pattern": [13],
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "unireplknet_n",
    UniRepLKNet,
    config={
        "channels": [80, 160, 320, 640],
        "depths": [2, 2, 8, 2],
        "stage3_kernel_pattern": [13],
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "unireplknet_t",
    UniRepLKNet,
    config={
        "channels": [80, 160, 320, 640],
        "depths": [3, 3, 18, 3],
        "stage3_kernel_pattern": [13, 3],
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "unireplknet_s",
    UniRepLKNet,
    config={
        "channels": [96, 192, 384, 768],
        "depths": [3, 3, 27, 3],
        "stage3_kernel_pattern": [13, 3, 3],
        "drop_path_rate": 0.4,
    },
)
registry.register_model_config(
    "unireplknet_b",
    UniRepLKNet,
    config={
        "channels": [128, 256, 512, 1024],
        "depths": [3, 3, 27, 3],
        "stage3_kernel_pattern": [13, 3, 3],
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "unireplknet_l",
    UniRepLKNet,
    config={
        "channels": [192, 384, 768, 1536],
        "depths": [3, 3, 27, 3],
        "stage3_kernel_pattern": [13, 3, 3],
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "unireplknet_xl",
    UniRepLKNet,
    config={
        "channels": [256, 512, 1024, 2048],
        "depths": [3, 3, 27, 3],
        "stage3_kernel_pattern": [13, 3, 3],
        "drop_path_rate": 0.2,
    },
)
