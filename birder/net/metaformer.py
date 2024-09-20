"""
MetaFormer, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/metaformer.py
and
https://github.com/sail-sg/metaformer

Paper "MetaFormer Baselines for Vision", https://arxiv.org/abs/2210.13452
"""

# Reference license: Apache-2.0 (both)

from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.convnext_v1 import LayerNorm2d


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        norm_layer: Optional[Callable[..., nn.Module]],
    ) -> None:
        super().__init__()
        self.norm = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x


class ConvMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: Callable[..., nn.Module],
        bias: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Conv2d(
            hidden_features, out_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class Scale(nn.Module):
    def __init__(self, dim: int, init_value: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class SquaredReLU(nn.Module):
    """
    Squared ReLU: https://arxiv.org/abs/2109.08668
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0, inplace: bool = False) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=True)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.relu(x) ** 2 + self.bias


class Attention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 32, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.num_heads = dim // head_dim
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        (B, H, W, _) = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)

        x = F.scaled_dot_product_attention(  # pylint:disable=not-callable
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
        )

        x = x.transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        return x


class GroupNorm1(nn.GroupNorm):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_groups=1, num_channels=num_channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-renamed
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class GroupNorm1NoBias(nn.GroupNorm):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_groups=1, num_channels=num_channels, eps=1e-6)
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-renamed
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class LayerNorm2dNoBias(LayerNorm2d):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels, eps=1e-6)
        self.bias = None


class SepConv(nn.Module):
    """
    Same as MobileNet v2 inverted separable convolution without the normalization
    """

    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 2,
        act1_layer: Callable[..., nn.Module] = StarReLU,
        kernel_size: tuple[int, int] = (7, 7),
        padding: tuple[int, int] = (3, 3),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        mid_channels = int(expansion_ratio * dim)
        self.pw_conv1 = nn.Conv2d(dim, mid_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.act = act1_layer()
        self.dwconv = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
            groups=mid_channels,
            bias=False,
        )
        self.pw_conv2 = nn.Conv2d(mid_channels, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.pw_conv2(x)

        return x


class Pooling(nn.Module):
    """
    PoolFormer: https://arxiv.org/abs/2111.11418
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x) - x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(
        self,
        dim: int,
        token_mixer: Callable[..., nn.Module],
        mlp_act: Callable[..., nn.Module],
        mlp_bias: bool,
        norm_layer: Callable[..., nn.Module],
        proj_drop: float,
        drop_path: float,
        layer_scale_init_value: Optional[float],
        res_scale_init_value: Optional[float],
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, proj_drop=proj_drop)
        self.drop_path1 = StochasticDepth(drop_path, mode="row")
        if layer_scale_init_value is None:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()
        else:
            self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value)
            self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value)

        if res_scale_init_value is None:
            self.res_scale1 = nn.Identity()
            self.res_scale2 = nn.Identity()
        else:
            self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value)
            self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value)

        self.norm2 = norm_layer(dim)
        self.mlp = ConvMLP(dim, 4 * dim, dim, act_layer=mlp_act, bias=mlp_bias, dropout=proj_drop)
        self.drop_path2 = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_scale1(x) + self.layer_scale1(self.drop_path1(self.token_mixer(self.norm1(x))))
        x = self.res_scale2(x) + self.layer_scale2(self.drop_path2(self.mlp(self.norm2(x))))

        return x


class MetaFormerStage(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        depth: int,
        token_mixer: Callable[..., nn.Module],
        mlp_act: Callable[..., nn.Module],
        mlp_bias: bool,
        downsample_norm: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        proj_drop: float,
        dp_rates: list[float],
        layer_scale_init_value: Optional[float],
        res_scale_init_value: Optional[float],
    ) -> None:
        super().__init__()

        if in_chs == out_chs:
            self.downsample = nn.Identity()
        else:
            self.downsample = Downsample(
                in_chs,
                out_chs,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                norm_layer=downsample_norm,
            )

        layers = []
        for i in range(depth):
            layers.append(
                MetaFormerBlock(
                    dim=out_chs,
                    token_mixer=token_mixer,
                    mlp_act=mlp_act,
                    mlp_bias=mlp_bias,
                    norm_layer=norm_layer,
                    proj_drop=proj_drop,
                    drop_path=dp_rates[i],
                    layer_scale_init_value=layer_scale_init_value,
                    res_scale_init_value=res_scale_init_value,
                )
            )

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


class MetaFormer(BaseNet):
    default_size = 224

    # pylint: disable=too-many-statements,too-many-branches
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

        # PoolFormer v1
        if net_param == 0:
            # s 12
            depths = [2, 2, 6, 2]
            dims = [64, 128, 320, 512]
            token_mixers: list[type[nn.Module]] = [Pooling, Pooling, Pooling, Pooling]
            mlp_act = nn.GELU
            mlp_bias = True
            layer_scale_init_values: list[Optional[float]] = [1e-5, 1e-5, 1e-5, 1e-5]
            res_scale_init_values: list[Optional[float]] = [None, None, None, None]
            norm_layers = [GroupNorm1, GroupNorm1, GroupNorm1, GroupNorm1]
            downsample_norm = nn.Identity
            drop_path_rate = 0.1
            use_mlp_head = False
            mlp_head_dropout = 0.0

        elif net_param == 1:
            # s 24
            depths = [4, 4, 12, 4]
            dims = [64, 128, 320, 512]
            token_mixers = [Pooling, Pooling, Pooling, Pooling]
            mlp_act = nn.GELU
            mlp_bias = True
            layer_scale_init_values = [1e-5, 1e-5, 1e-5, 1e-5]
            res_scale_init_values = [None, None, None, None]
            norm_layers = [GroupNorm1, GroupNorm1, GroupNorm1, GroupNorm1]
            downsample_norm = nn.Identity
            drop_path_rate = 0.1
            use_mlp_head = False
            mlp_head_dropout = 0.0

        elif net_param == 2:
            # s 36
            depths = [6, 6, 18, 6]
            dims = [64, 128, 320, 512]
            token_mixers = [Pooling, Pooling, Pooling, Pooling]
            mlp_act = nn.GELU
            mlp_bias = True
            layer_scale_init_values = [1e-6, 1e-6, 1e-6, 1e-6]
            res_scale_init_values = [None, None, None, None]
            norm_layers = [GroupNorm1, GroupNorm1, GroupNorm1, GroupNorm1]
            downsample_norm = nn.Identity
            drop_path_rate = 0.2
            use_mlp_head = False
            mlp_head_dropout = 0.0

        elif net_param == 3:
            # m 36
            depths = [6, 6, 18, 6]
            dims = [96, 192, 384, 768]
            token_mixers = [Pooling, Pooling, Pooling, Pooling]
            mlp_act = nn.GELU
            mlp_bias = True
            layer_scale_init_values = [1e-6, 1e-6, 1e-6, 1e-6]
            res_scale_init_values = [None, None, None, None]
            norm_layers = [GroupNorm1, GroupNorm1, GroupNorm1, GroupNorm1]
            downsample_norm = nn.Identity
            drop_path_rate = 0.3
            use_mlp_head = False
            mlp_head_dropout = 0.0

        elif net_param == 4:
            # m 48
            depths = [8, 8, 24, 8]
            dims = [96, 192, 384, 768]
            token_mixers = [Pooling, Pooling, Pooling, Pooling]
            mlp_act = nn.GELU
            mlp_bias = True
            layer_scale_init_values = [1e-6, 1e-6, 1e-6, 1e-6]
            res_scale_init_values = [None, None, None, None]
            norm_layers = [GroupNorm1, GroupNorm1, GroupNorm1, GroupNorm1]
            downsample_norm = nn.Identity
            drop_path_rate = 0.4
            use_mlp_head = False
            mlp_head_dropout = 0.0

        # PoolFormer v2
        elif net_param == 10:
            # s 12
            depths = [2, 2, 6, 2]
            dims = [64, 128, 320, 512]
            token_mixers = [Pooling, Pooling, Pooling, Pooling]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [GroupNorm1NoBias, GroupNorm1NoBias, GroupNorm1NoBias, GroupNorm1NoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.1
            use_mlp_head = False
            mlp_head_dropout = 0.0

        elif net_param == 11:
            # s 24
            depths = [4, 4, 12, 4]
            dims = [64, 128, 320, 512]
            token_mixers = [Pooling, Pooling, Pooling, Pooling]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [GroupNorm1NoBias, GroupNorm1NoBias, GroupNorm1NoBias, GroupNorm1NoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.1
            use_mlp_head = False
            mlp_head_dropout = 0.0

        elif net_param == 12:
            # s 36
            depths = [6, 6, 18, 6]
            dims = [64, 128, 320, 512]
            token_mixers = [Pooling, Pooling, Pooling, Pooling]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [GroupNorm1NoBias, GroupNorm1NoBias, GroupNorm1NoBias, GroupNorm1NoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.2
            use_mlp_head = False
            mlp_head_dropout = 0.0

        elif net_param == 13:
            # m 36
            depths = [6, 6, 18, 6]
            dims = [96, 192, 384, 768]
            token_mixers = [Pooling, Pooling, Pooling, Pooling]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [GroupNorm1NoBias, GroupNorm1NoBias, GroupNorm1NoBias, GroupNorm1NoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.3
            use_mlp_head = False
            mlp_head_dropout = 0.0

        elif net_param == 14:
            # m 48
            depths = [8, 8, 24, 8]
            dims = [96, 192, 384, 768]
            token_mixers = [Pooling, Pooling, Pooling, Pooling]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [GroupNorm1NoBias, GroupNorm1NoBias, GroupNorm1NoBias, GroupNorm1NoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.4
            use_mlp_head = False
            mlp_head_dropout = 0.0

        # ConvFormer
        elif net_param == 20:
            # s 18
            depths = [3, 3, 9, 3]
            dims = [64, 128, 320, 512]
            token_mixers = [SepConv, SepConv, SepConv, SepConv]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.2
            use_mlp_head = True
            mlp_head_dropout = 0.0

        elif net_param == 21:
            # s 36
            depths = [3, 12, 18, 3]
            dims = [64, 128, 320, 512]
            token_mixers = [SepConv, SepConv, SepConv, SepConv]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.3
            use_mlp_head = True
            mlp_head_dropout = 0.0

        elif net_param == 22:
            # m 36
            depths = [3, 12, 18, 3]
            dims = [96, 192, 384, 576]
            token_mixers = [SepConv, SepConv, SepConv, SepConv]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.4
            use_mlp_head = True
            mlp_head_dropout = 0.0

        elif net_param == 23:
            # b 36
            depths = [3, 12, 18, 3]
            dims = [128, 256, 512, 768]
            token_mixers = [SepConv, SepConv, SepConv, SepConv]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.6
            use_mlp_head = True
            mlp_head_dropout = 0.0

        # CAFormer
        elif net_param == 30:
            # s 18
            depths = [3, 3, 9, 3]
            dims = [64, 128, 320, 512]
            token_mixers = [SepConv, SepConv, Attention, Attention]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.15
            use_mlp_head = True
            mlp_head_dropout = 0.0

        elif net_param == 31:
            # s 36
            depths = [3, 12, 18, 3]
            dims = [64, 128, 320, 512]
            token_mixers = [SepConv, SepConv, Attention, Attention]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.3
            use_mlp_head = True
            mlp_head_dropout = 0.4

        elif net_param == 32:
            # m 36
            depths = [3, 12, 18, 3]
            dims = [96, 192, 384, 576]
            token_mixers = [SepConv, SepConv, Attention, Attention]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.4
            use_mlp_head = True
            mlp_head_dropout = 0.4

        elif net_param == 33:
            # b 36
            depths = [3, 12, 18, 3]
            dims = [128, 256, 512, 768]
            token_mixers = [SepConv, SepConv, Attention, Attention]
            mlp_act = StarReLU
            mlp_bias = False
            layer_scale_init_values = [None, None, None, None]
            res_scale_init_values = [None, None, 1.0, 1.0]
            norm_layers = [LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias, LayerNorm2dNoBias]
            downsample_norm = LayerNorm2dNoBias
            drop_path_rate = 0.6
            use_mlp_head = True
            mlp_head_dropout = 0.5

        else:
            raise ValueError(f"net_param = {net_param} not supported")

        self.use_mlp_head = use_mlp_head
        self.mlp_head_dropout = mlp_head_dropout
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, dims[0], kernel_size=(7, 7), stride=(4, 4), padding=(2, 2)),
            downsample_norm(dims[0]),
        )

        num_stages = len(depths)
        stages = []
        prev_dim = dims[0]
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]

        for i in range(num_stages):
            stages += [
                MetaFormerStage(
                    prev_dim,
                    dims[i],
                    depth=depths[i],
                    token_mixer=token_mixers[i],
                    mlp_act=mlp_act,
                    mlp_bias=mlp_bias,
                    proj_drop=0.0,
                    dp_rates=dp_rates[i],
                    layer_scale_init_value=layer_scale_init_values[i],
                    res_scale_init_value=res_scale_init_values[i],
                    downsample_norm=downsample_norm,
                    norm_layer=norm_layers[i],
                )
            ]
            prev_dim = dims[i]

        self.body = nn.Sequential(*stages)

        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            LayerNorm2d(dims[-1], eps=1e-6),
            nn.Flatten(1),
        )
        self.embedding_size = dims[-1]
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) is True:
                nn.init.trunc_normal_(m.weight, std=0.02)

            elif isinstance(m, nn.Linear) is True:
                nn.init.trunc_normal_(m.weight, std=0.02)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)

    def create_classifier(self, embed_dim: Optional[int] = None) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size

        if self.use_mlp_head is False:
            return nn.Linear(embed_dim, self.num_classes)

        return nn.Sequential(
            nn.Dropout(self.mlp_head_dropout),
            nn.Linear(embed_dim, 4 * embed_dim),
            SquaredReLU(),
            nn.LayerNorm(4 * embed_dim),
            nn.Linear(4 * embed_dim, self.num_classes),
        )


registry.register_alias("poolformer_v1_s12", MetaFormer, 0)
registry.register_alias("poolformer_v1_s24", MetaFormer, 1)
registry.register_alias("poolformer_v1_s36", MetaFormer, 2)
registry.register_alias("poolformer_v1_m36", MetaFormer, 3)
registry.register_alias("poolformer_v1_m48", MetaFormer, 4)

registry.register_alias("poolformer_v2_s12", MetaFormer, 10)
registry.register_alias("poolformer_v2_s24", MetaFormer, 11)
registry.register_alias("poolformer_v2_s36", MetaFormer, 12)
registry.register_alias("poolformer_v2_m36", MetaFormer, 13)
registry.register_alias("poolformer_v2_m48", MetaFormer, 14)

registry.register_alias("convformer_s18", MetaFormer, 20)
registry.register_alias("convformer_s36", MetaFormer, 21)
registry.register_alias("convformer_m36", MetaFormer, 22)
registry.register_alias("convformer_b36", MetaFormer, 23)

registry.register_alias("caformer_s18", MetaFormer, 30)
registry.register_alias("caformer_s36", MetaFormer, 31)
registry.register_alias("caformer_m36", MetaFormer, 32)
registry.register_alias("caformer_b36", MetaFormer, 33)