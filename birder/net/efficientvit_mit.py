"""
EfficientViT (MIT), adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientvit_mit.py

Paper "EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction",
https://arxiv.org/abs/2205.14756
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import get_activation_module


class ResidualBlock(nn.Module):
    def __init__(self, main: nn.Module, shortcut: nn.Module, drop_path: float) -> None:
        super().__init__()
        self.main = main
        self.shortcut = shortcut
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.drop_path(self.main(x))
        res = res + self.shortcut(x)

        return res


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        norm_layer: tuple[Optional[Callable[..., nn.Module]], Optional[Callable[..., nn.Module]]],
        act_layer: tuple[Optional[Callable[..., nn.Module]], Optional[Callable[..., nn.Module]]],
    ) -> None:
        super().__init__()
        self.depth_conv = Conv2dNormActivation(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=in_channels,
            norm_layer=norm_layer[0],
            activation_layer=act_layer[0],
            inplace=None,
        )
        self.point_conv = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            norm_layer=norm_layer[1],
            activation_layer=act_layer[1],
            inplace=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        norm_layer: tuple[Optional[Callable[..., nn.Module]], Optional[Callable[..., nn.Module]]],
        act_layer: tuple[Optional[Callable[..., nn.Module]], Optional[Callable[..., nn.Module]]],
    ) -> None:
        super().__init__()
        mid_channels = in_channels

        self.conv1 = Conv2dNormActivation(
            in_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            norm_layer=norm_layer[0],
            activation_layer=act_layer[0],
            inplace=None,
        )
        self.conv2 = Conv2dNormActivation(
            mid_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            norm_layer=norm_layer[1],
            activation_layer=act_layer[1],
            inplace=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        expand_ratio: float,
        norm_layer: tuple[Optional[Callable[..., nn.Module]], ...],
        act_layer: tuple[Optional[Callable[..., nn.Module]], ...],
    ) -> None:
        super().__init__()
        mid_channels = round(in_channels * expand_ratio)

        self.inverted_conv = Conv2dNormActivation(
            in_channels,
            mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            norm_layer=norm_layer[0],
            activation_layer=act_layer[0],
            inplace=None,
        )
        self.depth_conv = Conv2dNormActivation(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=mid_channels,
            norm_layer=norm_layer[1],
            activation_layer=act_layer[1],
            inplace=None,
        )
        self.point_conv = Conv2dNormActivation(
            mid_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            norm_layer=norm_layer[2],
            activation_layer=act_layer[2],
            inplace=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)

        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        expand_ratio: float,
        norm_layer: tuple[Optional[Callable[..., nn.Module]], Optional[Callable[..., nn.Module]]],
        act_layer: tuple[Optional[Callable[..., nn.Module]], Optional[Callable[..., nn.Module]]],
    ) -> None:
        super().__init__()
        mid_channels = round(in_channels * expand_ratio)

        self.spatial_conv = Conv2dNormActivation(
            in_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            norm_layer=norm_layer[0],
            activation_layer=act_layer[0],
            inplace=None,
        )
        self.point_conv = Conv2dNormActivation(
            mid_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            norm_layer=norm_layer[1],
            activation_layer=act_layer[1],
            inplace=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)

        return x


class LiteMLA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads_ratio: float,
        dim: int,
        norm_layer: tuple[Optional[Callable[..., nn.Module]], Optional[Callable[..., nn.Module]]],
        scales: list[int],
        eps: float,
    ) -> None:
        super().__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio)
        total_dim = heads * dim

        self.dim = dim
        self.qkv = Conv2dNormActivation(
            in_channels,
            3 * total_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            norm_layer=norm_layer[0],
            activation_layer=None,
            inplace=None,
            bias=False,
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        kernel_size=(scale, scale),
                        stride=(1, 1),
                        padding=(scale // 2, scale // 2),
                        groups=3 * total_dim,
                        bias=False,
                    ),
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0),
                        groups=3 * heads,
                        bias=False,
                    ),
                )
                for scale in scales
            ]
        )
        self.kernel_func = nn.ReLU(inplace=False)

        self.proj = Conv2dNormActivation(
            total_dim * (1 + len(scales)),
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            norm_layer=norm_layer[1],
            activation_layer=None,
            inplace=None,
            bias=False,
        )

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dtype = v.dtype
        q = q.float()
        k = k.float()
        v = v.float()
        kv = k.transpose(-1, -2) @ v
        out = q @ kv
        out = out[..., :-1] / (out[..., -1:] + self.eps)
        return out.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, _, H, W) = x.size()

        # Generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv_list = [qkv]
        for op in self.aggreg:
            multi_scale_qkv_list.append(op(qkv))

        multi_scale_qkv = torch.concat(multi_scale_qkv_list, dim=1)
        multi_scale_qkv = multi_scale_qkv.reshape(B, -1, 3 * self.dim, H * W).transpose(-1, -2)
        (q, k, v) = multi_scale_qkv.chunk(3, dim=-1)

        # Lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        v = F.pad(v, (0, 1), mode="constant", value=1.0)

        with torch.autocast(device_type=v.device.type, enabled=False):
            out = self._attn(q, k, v)

        # Final projection
        out = out.transpose(-1, -2).reshape(B, -1, H, W)
        out = self.proj(out)

        return out


class EfficientVitBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float,
        head_dim: int,
        expand_ratio: float,
        norm_layer: Callable[..., nn.Module],
        act_layer: Callable[..., nn.Module],
        drop_path: float,
    ) -> None:
        super().__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=head_dim,
                norm_layer=(None, norm_layer),
                scales=[5],
                eps=1e-5,
            ),
            nn.Identity(),
            drop_path=drop_path,
        )
        self.local_module = ResidualBlock(
            MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                expand_ratio=expand_ratio,
                norm_layer=(None, None, norm_layer),
                act_layer=(act_layer, act_layer, None),
            ),
            nn.Identity(),
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)

        return x


class EfficientVitStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        head_dim: int,
        expand_ratio: float,
        expand_divisor: int,
        norm_layer: Callable[..., nn.Module],
        act_layer: Callable[..., nn.Module],
        drop_path: list[float],
        depth: int,
        fused_block: bool,
        fewer_norms: bool,
        vit_stage: bool,
    ):
        super().__init__()
        blocks = [
            self.get_block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                expand_ratio=expand_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                fewer_norms=fewer_norms,
                fused_block=fused_block,
            )
        ]
        in_channels = out_channels

        if vit_stage is True:
            for i in range(depth):
                blocks.append(
                    EfficientVitBlock(
                        in_channels=in_channels,
                        heads_ratio=1.0,
                        head_dim=head_dim,
                        expand_ratio=expand_ratio // expand_divisor,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        drop_path=drop_path[i],
                    )
                )
        else:
            for i in range(1, depth):
                blocks.append(
                    ResidualBlock(
                        self.get_block(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            expand_ratio=expand_ratio // expand_divisor,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            fewer_norms=fewer_norms,
                            fused_block=fused_block,
                        ),
                        nn.Identity(),
                        drop_path=drop_path[i],
                    )
                )

        self.blocks = nn.Sequential(*blocks)

    @staticmethod
    def get_block(
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        expand_ratio: float,
        norm_layer: Callable[..., nn.Module],
        act_layer: Callable[..., nn.Module],
        fewer_norms: bool,
        fused_block: bool,
    ) -> nn.Module:
        norm_layers: tuple[Optional[Callable[..., nn.Module]], ...]

        if fused_block is False:
            if fewer_norms is True:
                norm_layers = (None, None, norm_layer)
            else:
                norm_layers = (norm_layer, norm_layer, norm_layer)

            return MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                expand_ratio=expand_ratio,
                norm_layer=norm_layers,
                act_layer=(act_layer, act_layer, None),
            )

        if fewer_norms is True:
            norm_layers = (None, norm_layer)
        else:
            norm_layers = (norm_layer, norm_layer)

        return FusedMBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expand_ratio=expand_ratio,
            norm_layer=norm_layers,
            act_layer=(act_layer, None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Stem(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        norm_layer: Callable[..., nn.Module],
        act_layer: Callable[..., nn.Module],
        stem_block: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        self.append(
            Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                norm_layer=norm_layer,
                activation_layer=act_layer,
                inplace=None,
            ),
        )
        for _ in range(depth):
            self.append(
                ResidualBlock(
                    stem_block(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        norm_layer=(norm_layer, norm_layer),
                        act_layer=(act_layer, None),
                    ),
                    nn.Identity(),
                    drop_path=0.0,
                ),
            )


# pylint: disable=invalid-name
class EfficientViT_MIT(DetectorBackbone):
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

        norm_layer = nn.BatchNorm2d
        widths: list[int] = self.config["widths"]
        depths: list[int] = self.config["depths"]
        head_dim: int = self.config["head_dim"]
        conv_stages: int = self.config["conv_stages"]
        fused_blocks: int = self.config["fused_blocks"]
        head_widths: tuple[int, int] = self.config["head_widths"]
        expand_ratio: float = self.config["expand_ratio"]
        vit_expand_ratio: float = self.config["vit_expand_ratio"]
        expand_divisor: int = self.config["expand_divisor"]
        act_layer_name: str = self.config["act_layer_name"]
        stem_block_name: str = self.config["stem_block_name"]
        drop_path_rate: float = self.config["drop_path_rate"]

        act_layer = get_activation_module(act_layer_name)
        if stem_block_name == "DSConv":
            stem_block = DSConv
        elif stem_block_name == "ConvBlock":
            stem_block = ConvBlock
        else:
            raise ValueError(f"Unknown stem_block_name '{stem_block_name}'")

        self.stem = Stem(
            self.input_channels, widths[0], depths[0], norm_layer=norm_layer, act_layer=act_layer, stem_block=stem_block
        )

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths[1:])).split(depths[1:])]
        in_channels = widths[0]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i, (w, d) in enumerate(zip(widths[1:], depths[1:])):
            stages[f"stage{i+1}"] = EfficientVitStage(
                in_channels,
                w,
                head_dim=head_dim,
                expand_ratio=vit_expand_ratio if i >= conv_stages else expand_ratio,
                expand_divisor=expand_divisor,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop_path=dpr[i],
                depth=d,
                fused_block=i < fused_blocks,
                fewer_norms=i >= 2,
                vit_stage=i >= conv_stages,
            )
            return_channels.append(w)
            in_channels = w

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            Conv2dNormActivation(
                widths[-1],
                head_widths[0],
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                norm_layer=norm_layer,
                activation_layer=act_layer,
                inplace=None,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Linear(head_widths[0], head_widths[1], bias=False),
            nn.LayerNorm(head_widths[1]),
            act_layer(),
        )
        self.return_channels = return_channels
        self.embedding_size = head_widths[1]
        self.classifier = self.create_classifier()

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


registry.register_alias(
    "efficientvit_mit_b0",
    EfficientViT_MIT,
    config={
        "widths": [8, 16, 32, 64, 128],
        "depths": [1, 2, 2, 2, 2],
        "head_dim": 16,
        "conv_stages": 2,
        "fused_blocks": 0,
        "head_widths": (1024, 1280),
        "expand_ratio": 4.0,
        "vit_expand_ratio": 4.0,
        "expand_divisor": 1,
        "act_layer_name": "hard_swish",
        "stem_block_name": "DSConv",
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "efficientvit_mit_b1",
    EfficientViT_MIT,
    config={
        "widths": [16, 32, 64, 128, 256],
        "depths": [1, 2, 3, 3, 4],
        "head_dim": 16,
        "conv_stages": 2,
        "fused_blocks": 0,
        "head_widths": (1536, 1600),
        "expand_ratio": 4.0,
        "vit_expand_ratio": 4.0,
        "expand_divisor": 1,
        "act_layer_name": "hard_swish",
        "stem_block_name": "DSConv",
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "efficientvit_mit_b2",
    EfficientViT_MIT,
    config={
        "widths": [24, 48, 96, 192, 384],
        "depths": [1, 3, 4, 4, 6],
        "head_dim": 32,
        "conv_stages": 2,
        "fused_blocks": 0,
        "head_widths": (2304, 2560),
        "expand_ratio": 4.0,
        "vit_expand_ratio": 4.0,
        "expand_divisor": 1,
        "act_layer_name": "hard_swish",
        "stem_block_name": "DSConv",
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "efficientvit_mit_b3",
    EfficientViT_MIT,
    config={
        "widths": [32, 64, 128, 256, 512],
        "depths": [1, 4, 6, 6, 9],
        "head_dim": 32,
        "conv_stages": 2,
        "fused_blocks": 0,
        "head_widths": (2304, 2560),
        "expand_ratio": 4.0,
        "vit_expand_ratio": 4.0,
        "expand_divisor": 1,
        "act_layer_name": "hard_swish",
        "stem_block_name": "DSConv",
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "efficientvit_mit_l1",
    EfficientViT_MIT,
    config={
        "widths": [32, 64, 128, 256, 512],
        "depths": [1, 2, 2, 7, 6],
        "head_dim": 32,
        "conv_stages": 3,
        "fused_blocks": 2,
        "head_widths": (3072, 3200),
        "expand_ratio": 16.0,
        "vit_expand_ratio": 24.0,
        "expand_divisor": 4,
        "act_layer_name": "gelu_tanh",
        "stem_block_name": "ConvBlock",
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "efficientvit_mit_l2",
    EfficientViT_MIT,
    config={
        "widths": [32, 64, 128, 256, 512],
        "depths": [1, 3, 3, 9, 8],
        "head_dim": 32,
        "conv_stages": 3,
        "fused_blocks": 2,
        "head_widths": (3072, 3200),
        "expand_ratio": 16.0,
        "vit_expand_ratio": 24.0,
        "expand_divisor": 4,
        "act_layer_name": "gelu_tanh",
        "stem_block_name": "ConvBlock",
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "efficientvit_mit_l3",
    EfficientViT_MIT,
    config={
        "widths": [64, 128, 256, 512, 1024],
        "depths": [1, 3, 3, 9, 8],
        "head_dim": 32,
        "conv_stages": 3,
        "fused_blocks": 2,
        "head_widths": (6144, 6400),
        "expand_ratio": 16.0,
        "vit_expand_ratio": 24.0,
        "expand_divisor": 4,
        "act_layer_name": "gelu_tanh",
        "stem_block_name": "ConvBlock",
        "drop_path_rate": 0.1,
    },
)

registry.register_weights(
    "efficientvit_mit_b0_il-common",
    {
        "description": "EfficientViT (MIT) B0 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 10.0,
                "sha256": "421197ed3bef179bb52db8675a5de5aa962ae2999e1f8d5459c87db5d956d138",
            }
        },
        "net": {"network": "efficientvit_mit_b0", "tag": "il-common"},
    },
)
