"""
HorNet, adapted from
https://github.com/raoyongming/HorNet/blob/master/hornet.py

Paper "HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions",
https://arxiv.org/abs/2207.14284
"""

# Reference license: MIT

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class ChannelsFirstLayerNorm(nn.Module):
    """
    Channels first corresponds to inputs with shape (batch_size, channels, height, width)
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = nn.Buffer(torch.tensor(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]

        return x


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_value: float, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_value * torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace is True:
            return x.mul_(self.gamma)

        return x * self.gamma


class GlobalLocalFilter(nn.Module):
    def __init__(self, dim: int, h: int, w: int) -> None:
        super().__init__()
        self.pre_norm = ChannelsFirstLayerNorm(dim, eps=1e-6)
        self.dw = nn.Conv2d(
            dim // 2, dim // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim // 2, bias=False
        )
        self.complex_weight = nn.Parameter(torch.empty(dim // 2, h, w, 2, dtype=torch.float32))
        self.post_norm = ChannelsFirstLayerNorm(dim, eps=1e-6)

        # Weight initialization
        nn.init.trunc_normal_(self.complex_weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        (x1, x2) = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        (B, C, a, b) = x2.size()
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm="ortho")  # pylint: disable=not-callable

        weight = torch.view_as_complex(self.complex_weight.contiguous())
        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm="ortho")  # pylint: disable=not-callable

        x = torch.concat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)

        return x


# pylint: disable=invalid-name
class gnConv(nn.Module):
    def __init__(self, dim: int, order: int, gf_layer: bool, h: int, w: int, scale: float) -> None:
        super().__init__()
        self.order = order
        self.dims = [dim // 2**i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        if gf_layer is False:
            self.dwconv = nn.Conv2d(
                sum(self.dims), sum(self.dims), kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=sum(self.dims)
            )
        else:
            self.dwconv = GlobalLocalFilter(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.pws = nn.ModuleList(
            [
                nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
                for i in range(order - 1)
            ]
        )

        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused_x = self.proj_in(x)
        (pwa, abc) = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i, mod in enumerate(self.pws):
            x = mod(x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class HorBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_scale_init_value: Optional[float],
        order: int,
        gf_layer: bool,
        h: int,
        w: int,
        scale: float,
        drop_path: float,
    ) -> None:
        super().__init__()

        self.norm1 = ChannelsFirstLayerNorm(dim, eps=1e-6)
        self.gn_conv = gnConv(dim, order=order, gf_layer=gf_layer, h=h, w=w, scale=scale)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.pw_conv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * dim, dim)

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale2d(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm1(x)
        x = self.gn_conv(x)
        x = self.layer_scale_1(x)
        x = res + self.drop_path(x)

        res = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = self.layer_scale_2(x)
        x = res + self.drop_path(x)

        return x


class HorStage(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        layer_scale_init_value: Optional[float],
        order: int,
        gf_layer: bool,
        h: int,
        w: int,
        scale: float,
        drop_path: list[float],
        depth: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = nn.Sequential(
                ChannelsFirstLayerNorm(dim, eps=1e-6),
                nn.Conv2d(dim, out_dim, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        layers = []
        for i in range(depth):
            layers.append(
                HorBlock(
                    out_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    order=order,
                    gf_layer=gf_layer,
                    h=h,
                    w=w,
                    scale=scale,
                    drop_path=drop_path[i],
                )
            )

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


class HorNet(DetectorBackbone):
    square_only = True

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

        s = 1.0 / 3.0
        layer_scale_init_value = 1e-6
        gn_conv_order = [2, 3, 4, 5]
        depths: list[int] = self.config["depths"]
        base_dim: int = self.config["base_dim"]
        gf_layer: list[bool] = self.config["gf_layer"]
        drop_path_rate: float = self.config["drop_path_rate"]

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        num_stages = len(depths)
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        gn_conv_h = [self.size[0] // 4, self.size[0] // 8, self.size[0] // 16, self.size[0] // 32]
        gn_conv_w = [(h // 2) + 1 for h in gn_conv_h]

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, dims[0], kernel_size=(4, 4), stride=(4, 4), padding=(0, 0)),
            ChannelsFirstLayerNorm(dims[0], eps=1e-6),
        )

        prev_dim = dims[0]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = HorStage(
                prev_dim,
                dims[i],
                layer_scale_init_value=layer_scale_init_value,
                order=gn_conv_order[i],
                gf_layer=gf_layer[i],
                h=gn_conv_h[i],
                w=gn_conv_w[i],
                scale=s,
                drop_path=dpr[i],
                depth=depths[i],
                downsample=i > 0,
            )
            return_channels.append(dims[i])
            prev_dim = dims[i]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(dims[-1], eps=1e-6),
        )
        self.return_channels = return_channels
        self.embedding_size = dims[-1]
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
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

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)

        gn_conv_h = [new_size[0] // 4, new_size[0] // 8, new_size[0] // 16, new_size[0] // 32]
        gn_conv_w = [(h // 2) + 1 for h in gn_conv_h]
        for i, module in enumerate(self.body.children()):
            if isinstance(module, HorStage):
                for m in module.modules():
                    if isinstance(m, HorBlock):
                        if isinstance(m.gn_conv.dwconv, GlobalLocalFilter):
                            weight = m.gn_conv.dwconv.complex_weight
                            weight = F.interpolate(
                                weight.permute(3, 0, 1, 2),
                                size=(gn_conv_h[i], gn_conv_w[i]),
                                mode="bilinear",
                                align_corners=True,
                            ).permute(1, 2, 3, 0)
                            m.gn_conv.dwconv.complex_weight = nn.Parameter(weight)


registry.register_alias(
    "hornet_tiny_7x7",
    HorNet,
    config={
        "depths": [2, 3, 18, 2],
        "base_dim": 64,
        "gf_layer": [False, False, False, False],
        "drop_path_rate": 0.2,
    },
)
registry.register_alias(
    "hornet_tiny_gf",
    HorNet,
    config={
        "depths": [2, 3, 18, 2],
        "base_dim": 64,
        "gf_layer": [False, False, True, True],
        "drop_path_rate": 0.2,
    },
)
registry.register_alias(
    "hornet_small_7x7",
    HorNet,
    config={
        "depths": [2, 3, 18, 2],
        "base_dim": 96,
        "gf_layer": [False, False, False, False],
        "drop_path_rate": 0.4,
    },
)
registry.register_alias(
    "hornet_small_gf",
    HorNet,
    config={
        "depths": [2, 3, 18, 2],
        "base_dim": 96,
        "gf_layer": [False, False, True, True],
        "drop_path_rate": 0.4,
    },
)
registry.register_alias(
    "hornet_base_7x7",
    HorNet,
    config={
        "depths": [2, 3, 18, 2],
        "base_dim": 128,
        "gf_layer": [False, False, False, False],
        "drop_path_rate": 0.5,
    },
)
registry.register_alias(
    "hornet_base_gf",
    HorNet,
    config={
        "depths": [2, 3, 18, 2],
        "base_dim": 128,
        "gf_layer": [False, False, True, True],
        "drop_path_rate": 0.5,
    },
)
registry.register_alias(
    "hornet_large_7x7",
    HorNet,
    config={
        "depths": [2, 3, 18, 2],
        "base_dim": 192,
        "gf_layer": [False, False, False, False],
        "drop_path_rate": 0.4,
    },
)
registry.register_alias(
    "hornet_large_gf",
    HorNet,
    config={
        "depths": [2, 3, 18, 2],
        "base_dim": 192,
        "gf_layer": [False, False, True, True],
        "drop_path_rate": 0.4,
    },
)
