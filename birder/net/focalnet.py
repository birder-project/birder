"""
FocalNet, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/focalnet.py

Paper "Focal Modulation Networks", https://arxiv.org/abs/2203.11926
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torch import nn
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.layers import LayerNorm2d
from birder.layers import LayerScale2d
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType


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
        self.drop_path1 = StochasticDepth(drop_path, mode="row")

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
        self.drop_path2 = StochasticDepth(drop_path, mode="row")

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
    # pylint: disable=too-many-arguments,too-many-positional-arguments
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


class FocalNet(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.stage\d+\.blocks\.(\d+)"

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

        proj_drop_rate = 0.0
        depths: list[int] = self.config["depths"]
        embed_dim: int = self.config["embed_dim"]
        focal_levels: tuple[int, int, int, int] = self.config["focal_levels"]
        focal_windows: tuple[int, int, int, int] = self.config["focal_windows"]
        layer_scale_value: Optional[float] = self.config["layer_scale_value"]
        use_post_norm: bool = self.config["use_post_norm"]
        use_overlap_down: bool = self.config["use_overlap_down"]
        use_post_norm_in_modulation: bool = self.config["use_post_norm_in_modulation"]
        drop_path_rate: float = self.config["drop_path_rate"]

        num_stages = len(depths)
        embed_dims = [embed_dim * (2**i) for i in range(num_stages)]
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

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for idx in range(num_stages):
            out_dim = embed_dims[idx]
            stages[f"stage{idx+1}"] = FocalNetStage(
                dim=in_dim,
                out_dim=out_dim,
                depth=depths[idx],
                mlp_ratio=4.0,
                downsample=idx > 0,
                focal_level=focal_levels[idx],
                focal_window=focal_windows[idx],
                use_overlap_down=use_overlap_down,
                use_post_norm=use_post_norm,
                use_post_norm_in_modulation=use_post_norm_in_modulation,
                normalize_modulator=False,
                layer_scale_value=layer_scale_value,
                proj_drop=proj_drop_rate,
                drop_path=dpr[sum(depths[:idx]) : sum(depths[: idx + 1])],
                norm_layer=norm_layer,
            )

            return_channels.append(out_dim)
            in_dim = out_dim

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            norm_layer(num_features),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = num_features
        self.classifier = self.create_classifier()

        self.stem_stride = 4
        self.stem_width = embed_dims[0]
        self.encoding_size = num_features

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
    "focalnet_t_srf",
    FocalNet,
    config={
        "depths": [2, 2, 6, 2],
        "embed_dim": 96,
        "focal_levels": (2, 2, 2, 2),
        "focal_windows": (3, 3, 3, 3),
        "layer_scale_value": None,
        "use_post_norm": False,
        "use_overlap_down": False,
        "use_post_norm_in_modulation": False,
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "focalnet_t_lrf",
    FocalNet,
    config={
        "depths": [2, 2, 6, 2],
        "embed_dim": 96,
        "focal_levels": (3, 3, 3, 3),
        "focal_windows": (3, 3, 3, 3),
        "layer_scale_value": None,
        "use_post_norm": False,
        "use_overlap_down": False,
        "use_post_norm_in_modulation": False,
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "focalnet_s_srf",
    FocalNet,
    config={
        "depths": [2, 2, 18, 2],
        "embed_dim": 96,
        "focal_levels": (2, 2, 2, 2),
        "focal_windows": (3, 3, 3, 3),
        "layer_scale_value": None,
        "use_post_norm": False,
        "use_overlap_down": False,
        "use_post_norm_in_modulation": False,
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "focalnet_s_lrf",
    FocalNet,
    config={
        "depths": [2, 2, 18, 2],
        "embed_dim": 96,
        "focal_levels": (3, 3, 3, 3),
        "focal_windows": (3, 3, 3, 3),
        "layer_scale_value": None,
        "use_post_norm": False,
        "use_overlap_down": False,
        "use_post_norm_in_modulation": False,
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "focalnet_b_srf",
    FocalNet,
    config={
        "depths": [2, 2, 18, 2],
        "embed_dim": 128,
        "focal_levels": (2, 2, 2, 2),
        "focal_windows": (3, 3, 3, 3),
        "layer_scale_value": None,
        "use_post_norm": False,
        "use_overlap_down": False,
        "use_post_norm_in_modulation": False,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "focalnet_b_lrf",
    FocalNet,
    config={
        "depths": [2, 2, 18, 2],
        "embed_dim": 128,
        "focal_levels": (3, 3, 3, 3),
        "focal_windows": (3, 3, 3, 3),
        "layer_scale_value": None,
        "use_post_norm": False,
        "use_overlap_down": False,
        "use_post_norm_in_modulation": False,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "focalnet_l3",
    FocalNet,
    config={
        "depths": [2, 2, 18, 2],
        "embed_dim": 192,
        "focal_levels": (3, 3, 3, 3),
        "focal_windows": (5, 5, 5, 5),
        "layer_scale_value": 1e-4,
        "use_post_norm": True,
        "use_overlap_down": True,
        "use_post_norm_in_modulation": False,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "focalnet_l4",
    FocalNet,
    config={
        "depths": [2, 2, 18, 2],
        "embed_dim": 192,
        "focal_levels": (4, 4, 4, 4),
        "focal_windows": (3, 3, 3, 3),
        "layer_scale_value": 1e-4,
        "use_post_norm": True,
        "use_overlap_down": True,
        "use_post_norm_in_modulation": False,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "focalnet_xl3",
    FocalNet,
    config={
        "depths": [2, 2, 18, 2],
        "embed_dim": 256,
        "focal_levels": (3, 3, 3, 3),
        "focal_windows": (5, 5, 5, 5),
        "layer_scale_value": 1e-4,
        "use_post_norm": True,
        "use_overlap_down": True,
        "use_post_norm_in_modulation": False,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "focalnet_xl4",
    FocalNet,
    config={
        "depths": [2, 2, 18, 2],
        "embed_dim": 256,
        "focal_levels": (4, 4, 4, 4),
        "focal_windows": (3, 3, 3, 3),
        "layer_scale_value": 1e-4,
        "use_post_norm": True,
        "use_overlap_down": True,
        "use_post_norm_in_modulation": False,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "focalnet_h3",
    FocalNet,
    config={
        "depths": [2, 2, 18, 2],
        "embed_dim": 352,
        "focal_levels": (3, 3, 3, 3),
        "focal_windows": (5, 5, 5, 5),
        "layer_scale_value": 1e-4,
        "use_post_norm": True,
        "use_overlap_down": True,
        "use_post_norm_in_modulation": True,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "focalnet_h4",
    FocalNet,
    config={
        "depths": [2, 2, 18, 2],
        "embed_dim": 352,
        "focal_levels": (4, 4, 4, 4),
        "focal_windows": (3, 3, 3, 3),
        "layer_scale_value": 1e-4,
        "use_post_norm": True,
        "use_overlap_down": True,
        "use_post_norm_in_modulation": True,
        "drop_path_rate": 0.3,
    },
)

registry.register_weights(
    "focalnet_b_lrf_intermediate-eu-common",
    {
        "url": "https://huggingface.co/birder-project/focalnet_b_lrf_intermediate-eu-common/resolve/main",
        "description": (
            "FocalNet Base (LRF) model with intermediate training, then fine-tuned on the eu-common dataset"
        ),
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 337.6,
                "sha256": "9318e35c033606bb58d4992867d19fe4718e416d33955319a2841ed4ab7dcb5d",
            },
        },
        "net": {"network": "focalnet_b_lrf", "tag": "intermediate-eu-common"},
    },
)
registry.register_weights(
    "focalnet_b_lrf_intermediate-arabian-peninsula",
    {
        "url": "https://huggingface.co/birder-project/focalnet_b_lrf_intermediate-arabian-peninsula/resolve/main",
        "description": (
            "FocalNet Base (LRF) model with intermediate training, then fine-tuned on the arabian-peninsula dataset"
        ),
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 337.7,
                "sha256": "9a225c0a58cb4114ada7ea20c9410a4dea116670335cc0eabaa8448f8e9247ab",
            },
        },
        "net": {"network": "focalnet_b_lrf", "tag": "intermediate-arabian-peninsula"},
    },
)
