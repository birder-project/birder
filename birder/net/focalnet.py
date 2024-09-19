"""
FocalNet, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/focalnet.py

Paper "Focal Modulation Networks", https://arxiv.org/abs/2203.11926
"""

# Reference license: Apache-2.0

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import BaseNet


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-renamed
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps=self.eps)
        x = x.permute(0, 3, 1, 2)

        return x


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        if self.inplace is True:
            return x.mul_(gamma)

        return x * gamma


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: Callable[..., nn.Module],
        drop: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(
            hidden_features, out_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True
        )
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 4,
        overlap: bool = False,
        norm_layer: Callable[..., torch.Tensor] = LayerNorm2d,
    ) -> None:
        super().__init__()
        padding = (0, 0)
        kernel_size = (stride, stride)
        if overlap is True:
            assert stride in (2, 4)
            if stride == 4:
                kernel_size = (7, 7)
                padding = (2, 2)

            elif stride == 2:
                kernel_size = (3, 3)
                padding = (1, 1)

        self.proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True
        )
        self.norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        return x


class FocalModulation(nn.Module):
    def __init__(
        self,
        dim: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int,
        bias: bool,
        use_post_norm: bool,
        normalize_modulator: bool,
        proj_drop: float,
        norm_layer: Callable[..., torch.Tensor] = LayerNorm2d,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_post_norm = use_post_norm
        self.normalize_modulator = normalize_modulator
        self.input_split = [dim, dim, self.focal_level + 1]

        self.f = nn.Conv2d(
            dim,
            2 * dim + (self.focal_level + 1),
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias,
        )
        self.h = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=kernel_size,
                        stride=(1, 1),
                        padding=kernel_size // 2,
                        groups=dim,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)

        self.norm = norm_layer(dim) if self.use_post_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre linear projection
        x = self.f(x)
        (q, ctx, gates) = torch.split(x, self.input_split, 1)

        # Context aggregation
        ctx_all = 0.0
        for idx, focal_layer in enumerate(self.focal_layers):
            ctx = focal_layer(ctx)
            ctx_all = ctx_all + ctx * gates[:, idx : idx + 1]

        ctx_global = self.act(ctx.mean((2, 3), keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level :]

        # Normalize context
        if self.normalize_modulator is True:
            ctx_all = ctx_all / (self.focal_level + 1)

        # Focal modulation
        x_out = q * self.h(ctx_all)
        x_out = self.norm(x_out)

        # Post linear projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out


class FocalNetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        focal_level: int,
        focal_window: int,
        use_post_norm: bool,
        use_post_norm_in_modulation: bool,
        normalize_modulator: bool,
        layer_scale_value: Optional[float],
        proj_drop: float,
        drop_path: float,
        norm_layer: Callable[..., torch.Tensor] = LayerNorm2d,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_post_norm = use_post_norm

        self.norm1 = norm_layer(dim) if not use_post_norm else nn.Identity()
        self.modulation = FocalModulation(
            dim,
            focal_window=focal_window,
            focal_level=self.focal_level,
            focal_factor=2,
            bias=True,
            use_post_norm=use_post_norm_in_modulation,
            normalize_modulator=normalize_modulator,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm1_post = norm_layer(dim) if use_post_norm else nn.Identity()
        self.ls1 = LayerScale2d(dim, layer_scale_value) if layer_scale_value is not None else nn.Identity()
        self.drop_path1 = StochasticDepth(drop_path, mode="row") if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim) if not use_post_norm else nn.Identity()
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=nn.GELU,
            drop=proj_drop,
        )
        self.norm2_post = norm_layer(dim) if use_post_norm else nn.Identity()
        self.ls2 = LayerScale2d(dim, layer_scale_value) if layer_scale_value is not None else nn.Identity()
        self.drop_path2 = StochasticDepth(drop_path, mode="row") if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # Focal modulation
        x = self.norm1(x)
        x = self.modulation(x)
        x = self.norm1_post(x)
        x = shortcut + self.drop_path1(self.ls1(x))

        # FFN
        x = x + self.drop_path2(self.ls2(self.norm2_post(self.mlp(self.norm2(x)))))

        return x


class FocalNetStage(nn.Module):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        dim: int,
        out_dim: int,
        depth: int,
        mlp_ratio: float,
        downsample: bool,
        focal_level: int,
        focal_window: int,
        use_overlap_down: bool,
        use_post_norm: bool,
        use_post_norm_in_modulation: bool,
        normalize_modulator: bool,
        layer_scale_value: Optional[float],
        proj_drop: float,
        drop_path: list[float],
        norm_layer: Callable[..., torch.Tensor] = LayerNorm2d,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        if downsample is True:
            self.downsample = Downsample(
                in_channels=dim,
                out_channels=out_dim,
                stride=2,
                overlap=use_overlap_down,
                norm_layer=norm_layer,
            )

        else:
            self.downsample = nn.Identity()

        blocks = []
        for i in range(depth):
            blocks.append(
                FocalNetBlock(
                    dim=out_dim,
                    mlp_ratio=mlp_ratio,
                    focal_level=focal_level,
                    focal_window=focal_window,
                    use_post_norm=use_post_norm,
                    use_post_norm_in_modulation=use_post_norm_in_modulation,
                    normalize_modulator=normalize_modulator,
                    layer_scale_value=layer_scale_value,
                    proj_drop=proj_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


class FocalNet(BaseNet):
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

        if net_param == 0:
            # Tiny SRF (small receptive field)
            depths = [2, 2, 6, 2]
            embed_dim = 96
            focal_levels = (2, 2, 2, 2)
            focal_windows = (3, 3, 3, 3)
            layer_scale_value = None
            use_post_norm = False
            use_overlap_down = False
            use_post_norm_in_modulation = False

        elif net_param == 1:
            # Tiny LRF (large receptive field)
            depths = [2, 2, 6, 2]
            embed_dim = 96
            focal_levels = (3, 3, 3, 3)
            focal_windows = (3, 3, 3, 3)
            layer_scale_value = None
            use_post_norm = False
            use_overlap_down = False
            use_post_norm_in_modulation = False

        elif net_param == 2:
            # Small SRF (small receptive field)
            depths = [2, 2, 18, 2]
            embed_dim = 96
            focal_levels = (2, 2, 2, 2)
            focal_windows = (3, 3, 3, 3)
            layer_scale_value = None
            use_post_norm = False
            use_overlap_down = False
            use_post_norm_in_modulation = False

        elif net_param == 3:
            # Small LRF (large receptive field)
            depths = [2, 2, 18, 2]
            embed_dim = 96
            focal_levels = (3, 3, 3, 3)
            focal_windows = (3, 3, 3, 3)
            layer_scale_value = None
            use_post_norm = False
            use_overlap_down = False
            use_post_norm_in_modulation = False

        elif net_param == 4:
            # Base SRF (small receptive field)
            depths = [2, 2, 18, 2]
            embed_dim = 128
            focal_levels = (2, 2, 2, 2)
            focal_windows = (3, 3, 3, 3)
            layer_scale_value = None
            use_post_norm = False
            use_overlap_down = False
            use_post_norm_in_modulation = False

        elif net_param == 5:
            # Base LRF (large receptive field)
            depths = [2, 2, 18, 2]
            embed_dim = 128
            focal_levels = (3, 3, 3, 3)
            focal_windows = (3, 3, 3, 3)
            layer_scale_value = None
            use_post_norm = False
            use_overlap_down = False
            use_post_norm_in_modulation = False

        elif net_param == 6:
            # Large 3
            depths = [2, 2, 18, 2]
            embed_dim = 192
            focal_levels = (3, 3, 3, 3)
            focal_windows = (5, 5, 5, 5)
            layer_scale_value = 1e-4
            use_post_norm = True
            use_overlap_down = True
            use_post_norm_in_modulation = False

        elif net_param == 7:
            # Large 4
            depths = [2, 2, 18, 2]
            embed_dim = 192
            focal_levels = (4, 4, 4, 4)
            focal_windows = (3, 3, 3, 3)
            layer_scale_value = 1e-4
            use_post_norm = True
            use_overlap_down = True
            use_post_norm_in_modulation = False

        elif net_param == 8:
            # X-Large 3
            depths = [2, 2, 18, 2]
            embed_dim = 256
            focal_levels = (3, 3, 3, 3)
            focal_windows = (5, 5, 5, 5)
            layer_scale_value = 1e-4
            use_post_norm = True
            use_overlap_down = True
            use_post_norm_in_modulation = False

        elif net_param == 9:
            # X-Large 4
            depths = [2, 2, 18, 2]
            embed_dim = 256
            focal_levels = (4, 4, 4, 4)
            focal_windows = (3, 3, 3, 3)
            layer_scale_value = 1e-4
            use_post_norm = True
            use_overlap_down = True
            use_post_norm_in_modulation = False

        elif net_param == 10:
            # Huge 3
            depths = [2, 2, 18, 2]
            embed_dim = 352
            focal_levels = (3, 3, 3, 3)
            focal_windows = (5, 5, 5, 5)
            layer_scale_value = 1e-4
            use_post_norm = True
            use_overlap_down = True
            use_post_norm_in_modulation = True

        elif net_param == 11:
            # Huge 4
            depths = [2, 2, 18, 2]
            embed_dim = 352
            focal_levels = (4, 4, 4, 4)
            focal_windows = (3, 3, 3, 3)
            layer_scale_value = 1e-4
            use_post_norm = True
            use_overlap_down = True
            use_post_norm_in_modulation = True

        else:
            raise ValueError(f"net_param = {net_param} not supported")

        drop_path_rate = 0.1
        proj_drop_rate = 0.0
        num_layers = len(depths)
        embed_dims = [embed_dim * (2**i) for i in range(num_layers)]
        num_features = embed_dims[-1]
        norm_layer = partial(LayerNorm2d, eps=1e-5)

        self.stem = Downsample(
            in_channels=3,
            out_channels=embed_dims[0],
            overlap=use_overlap_down,
            norm_layer=norm_layer,
        )

        in_dim = embed_dims[0]
        dpr: list[float] = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        layers = []
        for i_layer in range(num_layers):
            out_dim = embed_dims[i_layer]
            layers.append(
                FocalNetStage(
                    dim=in_dim,
                    out_dim=out_dim,
                    depth=depths[i_layer],
                    mlp_ratio=4.0,
                    downsample=i_layer > 0,
                    focal_level=focal_levels[i_layer],
                    focal_window=focal_windows[i_layer],
                    use_overlap_down=use_overlap_down,
                    use_post_norm=use_post_norm,
                    use_post_norm_in_modulation=use_post_norm_in_modulation,
                    normalize_modulator=False,
                    layer_scale_value=layer_scale_value,
                    proj_drop=proj_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    norm_layer=norm_layer,
                )
            )
            in_dim = out_dim

        layers.append(norm_layer(num_features))

        self.body = nn.Sequential(*layers)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = num_features
        self.classifier = self.create_classifier()

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)


registry.register_alias("focalnet_t_srf", FocalNet, 0)
registry.register_alias("focalnet_t_lrf", FocalNet, 1)
registry.register_alias("focalnet_s_srf", FocalNet, 2)
registry.register_alias("focalnet_s_lrf", FocalNet, 3)
registry.register_alias("focalnet_b_srf", FocalNet, 4)
registry.register_alias("focalnet_b_lrf", FocalNet, 5)
registry.register_alias("focalnet_l3", FocalNet, 6)
registry.register_alias("focalnet_l4", FocalNet, 7)
registry.register_alias("focalnet_xl3", FocalNet, 8)
registry.register_alias("focalnet_xl4", FocalNet, 9)
registry.register_alias("focalnet_h3", FocalNet, 10)
registry.register_alias("focalnet_h4", FocalNet, 11)
