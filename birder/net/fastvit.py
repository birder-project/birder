"""
FastViT, adapted from
https://github.com/apple/ml-fastvit/blob/main/models/fastvit.py
and
https://github.com/apple/ml-mobileclip/blob/main/mobileclip/models/mci.py

Paper "FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization",
https://arxiv.org/abs/2303.14189
and
Paper "MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training",
https://arxiv.org/abs/2311.17049

Changes from original:
* Fixed ReparamLargeKernelConv activation (at forward)
"""

# Reference license: Apple MIT License (both)

from collections import OrderedDict
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import SqueezeExcitation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.mobileone import MobileOneBlock


class ReparamLargeKernelConv(nn.Module):
    """
    Building Block of RepLKNet

    This class defines overparameterized large kernel conv block
    introduced in RepLKNet https://arxiv.org/abs/2203.06717

    Reference: https://github.com/DingXiaoH/RepLKNet-pytorch (MIT License)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int,
        use_se: bool,
        reparameterized: bool,
    ) -> None:
        super().__init__()
        assert small_kernel <= kernel_size

        self.reparameterized = reparameterized
        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = kernel_size // 2
        if reparameterized is True:
            self.lkb_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                groups=self.groups,
                bias=True,
            )
        else:
            self.lkb_reparam = None

            self.lkb_origin = nn.Sequential()
            self.lkb_origin.add_module(
                "conv",
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    groups=self.groups,
                    bias=False,
                ),
            )
            self.lkb_origin.add_module("bn", nn.BatchNorm2d(num_features=out_channels))

            self.small_conv = nn.Sequential()
            self.small_conv.add_module(
                "conv",
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=small_kernel,
                    stride=self.stride,
                    padding=small_kernel // 2,
                    groups=self.groups,
                    bias=False,
                ),
            )
            self.small_conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))

        if use_se is True:
            self.se = SqueezeExcitation(out_channels, out_channels // 4)
        else:
            self.se = nn.Identity()

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lkb_reparam is not None:
            # Reparameterized forward pass
            x = self.lkb_reparam(x)
        else:
            # Multi-branched train-time forward pass
            x = self.lkb_origin(x) + self.small_conv(x)

        x = self.se(x)
        x = self.activation(x)

        return x

    def reparameterize(self) -> None:
        if self.reparameterized is True:
            return

        (eq_k, eq_b) = self._get_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=True,
        )

        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b

        # Delete un-used branches
        for param in self.parameters():
            param.detach_()

        del self.lkb_origin
        del self.small_conv

        self.reparameterized = True

    def _get_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        (eq_k, eq_b) = self._fuse_bn_tensor(self.lkb_origin.conv, self.lkb_origin.bn)

        (small_k, small_b) = self._fuse_bn_tensor(self.small_conv.conv, self.small_conv.bn)
        eq_b += small_b
        eq_k += F.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)

        return (eq_k, eq_b)

    @staticmethod
    def _fuse_bn_tensor(conv: torch.Tensor, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_value = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        return (kernel_value * t, beta - running_mean * gamma / std)


class MHSA(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, "dim must be divisible by head_dim"

        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, C, H, W) = x.shape
        N = H * W
        x = x.flatten(2).transpose(-2, -1)  # (B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)
        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        use_se: bool,
        reparameterized: bool,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            ReparamLargeKernelConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel=3,
                use_se=use_se,
                reparameterized=reparameterized,
            ),
            MobileOneBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                use_se=False,
                use_act=True,
                use_scale_branch=True,
                num_conv_branches=1,
                reparameterized=reparameterized,
                activation_layer=nn.GELU,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_value: float, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_value * torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace is True:
            return x.mul_(self.gamma)

        return x * self.gamma


class RepMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        layer_scale_init_value: Optional[float],
        reparameterized: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.reparameterized = reparameterized

        if reparameterized is True:
            self.reparam_conv = nn.Conv2d(
                self.dim,
                self.dim,
                kernel_size=self.kernel_size,
                stride=(1, 1),
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.reparam_conv = None

            self.norm = MobileOneBlock(
                in_channels=dim,
                out_channels=dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=dim,
                use_se=False,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
                reparameterized=False,
                activation_layer=nn.GELU,
            )
            self.mixer = MobileOneBlock(
                in_channels=dim,
                out_channels=dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=dim,
                use_se=False,
                use_act=False,
                use_scale_branch=True,
                num_conv_branches=1,
                reparameterized=False,
                activation_layer=nn.GELU,
            )
            if layer_scale_init_value is not None:
                self.layer_scale = LayerScale2d(dim, layer_scale_init_value)
            else:
                self.layer_scale = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reparameterized forward pass
        if self.reparam_conv is not None:
            return self.reparam_conv(x)

        # Multi-branched train-time forward pass
        return x + self.layer_scale(self.mixer(x) - self.norm(x))

    def reparameterize(self) -> None:
        if self.reparameterized is True:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if isinstance(self.layer_scale, LayerScale2d):
            w = self.mixer.id_tensor + self.layer_scale.gamma.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale.gamma) * (self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias)
        else:
            w = (  # type: ignore[unreachable]
                self.mixer.id_tensor + self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            self.dim,
            self.dim,
            kernel_size=self.kernel_size,
            stride=(1, 1),
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        # Delete un-used branches
        for param in self.parameters():
            param.detach_()

        del self.mixer
        del self.norm
        del self.layer_scale

        self.reparameterized = True


class ConvMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        drop: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(7, 7),
                stride=(1, 1),
                padding=(3, 3),
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=in_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.drop = nn.Dropout(drop)

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepCPE(nn.Module):
    """
    Implementation of conditional positional encoding

    For more details refer to paper:
    Conditional Positional Encodings for Vision Transformers https://arxiv.org/pdf/2102.10882.pdf

    In this implementation, we can reparameterize this module to eliminate a skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        spatial_shape: tuple[int, int],
        reparameterized: bool,
    ) -> None:
        super().__init__()
        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim
        self.reparameterized = reparameterized

        if reparameterized is True:
            self.reparam_conv = nn.Conv2d(
                self.in_channels,
                self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=(1, 1),
                padding=self.spatial_shape[0] // 2,
                groups=self.groups,
                bias=True,
            )
        else:
            self.reparam_conv = None

            self.pe = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=self.spatial_shape,
                stride=(1, 1),
                padding=self.spatial_shape[0] // 2,
                groups=self.groups,
                bias=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reparameterized forward pass
        if self.reparam_conv is not None:
            return self.reparam_conv(x)

        # Skip connection train-time forward pass
        return self.pe(x) + x

    def reparameterize(self) -> None:
        if self.reparameterized is True:
            return

        # Build equivalent Id tensor
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pe.weight.dtype,
            device=self.pe.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1

        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=(1, 1),
            padding=self.spatial_shape[0] // 2,
            groups=self.embed_dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        # Delete un-used branches
        for param in self.parameters():
            param.detach_()

        del self.pe
        self.reparameterized = True


class RepMixerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        mlp_ratio: float,
        proj_drop: float,
        drop_path: float,
        layer_scale_init_value: Optional[float],
        reparameterized: bool,
    ) -> None:
        super().__init__()

        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            layer_scale_init_value=layer_scale_init_value,
            reparameterized=reparameterized,
        )
        self.mlp = ConvMLP(
            in_channels=dim,
            hidden_channels=int(dim * mlp_ratio),
            drop=proj_drop,
        )

        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()

        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mixer(x)
        x = x + self.drop_path(self.layer_scale(self.mlp(x)))

        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        proj_drop: float,
        drop_path: float,
        layer_scale_init_value: Optional[float],
    ):
        super().__init__()

        self.norm = nn.BatchNorm2d(dim)
        self.token_mixer = MHSA(dim, head_dim=32, qkv_bias=False, attn_drop=0.0, proj_drop=0.0)
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()

        self.mlp = ConvMLP(
            in_channels=dim,
            hidden_channels=int(dim * mlp_ratio),
            drop=proj_drop,
        )
        if layer_scale_init_value is not None:
            self.layer_scale_2 = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale_2 = nn.Identity()

        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1(self.token_mixer(self.norm(x))))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(x)))
        return x


class FastVitStage(nn.Module):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        dim: int,
        dim_out: int,
        depth: int,
        token_mixer_type: Literal["repmixer", "attention"],
        downsample: bool,
        se_downsample: bool,
        down_patch_size: int,
        down_stride: int,
        use_cpe: bool,
        kernel_size: int,
        mlp_ratio: float,
        proj_drop_rate: float,
        drop_path_rate: list[float],
        layer_scale_init_value: Optional[float],
        reparameterized: bool,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = PatchEmbed(
                patch_size=down_patch_size,
                stride=down_stride,
                in_channels=dim,
                embed_dim=dim_out,
                use_se=se_downsample,
                reparameterized=reparameterized,
            )
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        if use_cpe is True:
            self.pos_emb = RepCPE(dim_out, dim_out, spatial_shape=(7, 7), reparameterized=reparameterized)
        else:
            self.pos_emb = nn.Identity()

        blocks = []
        for block_idx in range(depth):
            if token_mixer_type == "repmixer":  # nosec
                blocks.append(
                    RepMixerBlock(
                        dim_out,
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        proj_drop=proj_drop_rate,
                        drop_path=drop_path_rate[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                        reparameterized=reparameterized,
                    )
                )
            elif token_mixer_type == "attention":  # nosec
                blocks.append(
                    AttentionBlock(
                        dim_out,
                        mlp_ratio=mlp_ratio,
                        proj_drop=proj_drop_rate,
                        drop_path=drop_path_rate[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            else:
                raise ValueError(f"Token mixer type: {token_mixer_type} not supported")

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.pos_emb(x)
        x = self.blocks(x)

        return x


class FastViT(DetectorBackbone):
    default_size = (256, 256)

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

        self.reparameterized = False
        cls_ratio = 2.0
        layers: tuple[int, int, int, int] = self.config["layers"]
        embed_dims: tuple[int, int, int, int] = self.config["embed_dims"]
        mlp_ratios: tuple[int, int, int, int] = self.config["mlp_ratios"]
        se_downsamples: tuple[bool, bool, bool, bool] = self.config["se_downsamples"]
        use_cpe: tuple[bool, bool, bool, bool] = self.config["use_cpe"]
        token_mixers: tuple[str, str, str, str] = self.config["token_mixers"]
        layer_scale_init_value: float = self.config["layer_scale_init_value"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.stem = nn.Sequential(
            MobileOneBlock(
                in_channels=self.input_channels,
                out_channels=embed_dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                use_se=False,
                use_act=True,
                use_scale_branch=True,
                num_conv_branches=1,
                reparameterized=self.reparameterized,
                activation_layer=nn.GELU,
            ),
            MobileOneBlock(
                in_channels=embed_dims[0],
                out_channels=embed_dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
                groups=embed_dims[0],
                use_se=False,
                use_act=True,
                use_scale_branch=True,
                num_conv_branches=1,
                reparameterized=self.reparameterized,
                activation_layer=nn.GELU,
            ),
            MobileOneBlock(
                in_channels=embed_dims[0],
                out_channels=embed_dims[0],
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                use_se=False,
                use_act=True,
                use_scale_branch=True,
                num_conv_branches=1,
                reparameterized=self.reparameterized,
                activation_layer=nn.GELU,
            ),
        )

        num_stages = len(layers)
        prev_dim = embed_dims[0]
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        downsamples = (False,) + (True,) * (num_stages - 1)
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            downsample = downsamples[i] or prev_dim != embed_dims[i]
            stages[f"stage{i+1}"] = FastVitStage(
                dim=prev_dim,
                dim_out=embed_dims[i],
                depth=layers[i],
                token_mixer_type=token_mixers[i],  # type: ignore[arg-type]
                downsample=downsample,
                se_downsample=se_downsamples[i],
                down_patch_size=7,
                down_stride=2,
                use_cpe=use_cpe[i],
                kernel_size=3,
                mlp_ratio=mlp_ratios[i],
                proj_drop_rate=0.0,
                drop_path_rate=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                reparameterized=self.reparameterized,
            )
            return_channels.append(embed_dims[i])
            prev_dim = embed_dims[i]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            MobileOneBlock(
                in_channels=embed_dims[-1],
                out_channels=int(embed_dims[-1] * cls_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
                groups=embed_dims[-1],
                use_se=True,
                use_act=True,
                use_scale_branch=True,
                num_conv_branches=1,
                reparameterized=self.reparameterized,
                activation_layer=nn.GELU,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = int(embed_dims[-1] * cls_ratio)
        self.classifier = self.create_classifier()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
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

    def reparameterize_model(self) -> None:
        for module in self.modules():
            if hasattr(module, "reparameterize") is True:
                module.reparameterize()

        self.reparameterized = True


registry.register_alias(
    "fastvit_t8",
    FastViT,
    config={
        "layers": (2, 2, 4, 2),
        "embed_dims": (48, 96, 192, 384),
        "mlp_ratios": (3, 3, 3, 3),
        "se_downsamples": (False, False, False, False),
        "use_cpe": (False, False, False, False),
        "token_mixers": ("repmixer", "repmixer", "repmixer", "repmixer"),
        "layer_scale_init_value": 1e-5,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "fastvit_t12",
    FastViT,
    config={
        "layers": (2, 2, 6, 2),
        "embed_dims": (64, 128, 256, 512),
        "mlp_ratios": (3, 3, 3, 3),
        "se_downsamples": (False, False, False, False),
        "use_cpe": (False, False, False, False),
        "token_mixers": ("repmixer", "repmixer", "repmixer", "repmixer"),
        "layer_scale_init_value": 1e-5,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "fastvit_s12",
    FastViT,
    config={
        "layers": (2, 2, 6, 2),
        "embed_dims": (64, 128, 256, 512),
        "mlp_ratios": (4, 4, 4, 4),
        "se_downsamples": (False, False, False, False),
        "use_cpe": (False, False, False, False),
        "token_mixers": ("repmixer", "repmixer", "repmixer", "repmixer"),
        "layer_scale_init_value": 1e-5,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "fastvit_sa12",
    FastViT,
    config={
        "layers": (2, 2, 6, 2),
        "embed_dims": (64, 128, 256, 512),
        "mlp_ratios": (4, 4, 4, 4),
        "se_downsamples": (False, False, False, False),
        "use_cpe": (False, False, False, True),
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
        "layer_scale_init_value": 1e-5,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "fastvit_sa24",
    FastViT,
    config={
        "layers": (4, 4, 12, 4),
        "embed_dims": (64, 128, 256, 512),
        "mlp_ratios": (4, 4, 4, 4),
        "se_downsamples": (False, False, False, False),
        "use_cpe": (False, False, False, True),
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
        "layer_scale_init_value": 1e-5,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "fastvit_sa36",
    FastViT,
    config={
        "layers": (6, 6, 18, 6),
        "embed_dims": (64, 128, 256, 512),
        "mlp_ratios": (4, 4, 4, 4),
        "se_downsamples": (False, False, False, False),
        "use_cpe": (False, False, False, True),
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
        "layer_scale_init_value": 1e-6,
        "drop_path_rate": 0.2,
    },
)
registry.register_alias(
    "fastvit_ma36",
    FastViT,
    config={
        "layers": (6, 6, 18, 6),
        "embed_dims": (76, 152, 304, 608),
        "mlp_ratios": (4, 4, 4, 4),
        "se_downsamples": (False, False, False, False),
        "use_cpe": (False, False, False, True),
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
        "layer_scale_init_value": 1e-6,
        "drop_path_rate": 0.35,
    },
)

registry.register_alias(
    "mobileclip_i0",
    FastViT,
    config={
        "layers": (2, 6, 10, 2),
        "embed_dims": (64, 128, 256, 512),
        "mlp_ratios": (3, 3, 3, 3),
        "se_downsamples": (False, False, True, True),
        "use_cpe": (False, False, False, True),
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
        "layer_scale_init_value": 1e-5,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "mobileclip_i1",
    FastViT,
    config={
        "layers": (4, 12, 20, 4),
        "embed_dims": (64, 128, 256, 512),
        "mlp_ratios": (3, 3, 3, 3),
        "se_downsamples": (False, False, True, True),
        "use_cpe": (False, False, False, True),
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
        "layer_scale_init_value": 1e-5,
        "drop_path_rate": 0.05,
    },
)
registry.register_alias(
    "mobileclip_i2",
    FastViT,
    config={
        "layers": (4, 12, 24, 4),
        "embed_dims": (80, 160, 320, 640),
        "mlp_ratios": (3, 3, 3, 3),
        "se_downsamples": (False, False, True, True),
        "use_cpe": (False, False, False, True),
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
        "layer_scale_init_value": 1e-5,
        "drop_path_rate": 0.15,
    },
)

registry.register_weights(
    "fastvit_t8_il-common",
    {
        "description": "FastViT T-8 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 13.8,
                "sha256": "ec976d4edebf2215807de264e67d8bf65aa6da893295d4132cd418ea42b2171f",
            }
        },
        "net": {"network": "fastvit_t8", "tag": "il-common"},
    },
)
registry.register_weights(
    "fastvit_t8_il-common_reparameterized",
    {
        "description": "FastViT T-8 (reparameterized) model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 13.5,
                "sha256": "062c98f4acd50f726346a572a6f505c1b804c8ca6618da8c3719ff6764e9ac3b",
            }
        },
        "net": {"network": "fastvit_t8", "tag": "il-common_reparameterized", "reparameterized": True},
    },
)
registry.register_weights(
    "fastvit_t12_il-common",
    {
        "description": "FastViT T-12 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 26.7,
                "sha256": "66a61c8ba504357869af29349b3554b28567f020b539ad7c89d7b4d90de9b490",
            }
        },
        "net": {"network": "fastvit_t12", "tag": "il-common"},
    },
)
registry.register_weights(
    "fastvit_t12_il-common_reparameterized",
    {
        "description": "FastViT T-12 (reparameterized) model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 26.3,
                "sha256": "debb0dc5a96c9bb843e57bdea995c0b1535aceb0ad26aa8fb5d23ecdb8427f9b",
            }
        },
        "net": {"network": "fastvit_t12", "tag": "il-common_reparameterized", "reparameterized": True},
    },
)
registry.register_weights(
    "fastvit_sa12_il-common",
    {
        "description": "FastViT SA-12 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 42.0,
                "sha256": "5478786787fe8f23cff31216de941e6d4b0aab307ea0b9051115f7691906707f",
            }
        },
        "net": {"network": "fastvit_sa12", "tag": "il-common"},
    },
)
registry.register_weights(
    "fastvit_sa12_il-common_reparameterized",
    {
        "description": "FastViT SA-12 (reparameterized) model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 41.7,
                "sha256": "6e83955ac63b323760be62383c21ef72417c7bca461347401c9cd0d1d424ff62",
            }
        },
        "net": {"network": "fastvit_sa12", "tag": "il-common_reparameterized", "reparameterized": True},
    },
)
registry.register_weights(
    "mobileclip_i0_il-common",
    {
        "description": "MobileClip i0 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 43.5,
                "sha256": "f6599c5c5373d0928add5747c9e6524d5273b24c3c580a1c7b80acc7c10fd655",
            }
        },
        "net": {"network": "mobileclip_i0", "tag": "il-common"},
    },
)
registry.register_weights(
    "mobileclip_i0_il-common_reparameterized",
    {
        "description": "MobileClip i0 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 42.9,
                "sha256": "aec9bf62c0eaa12943158b056e7cef2fa93ebd78519080ad18bfacc5ef006670",
            }
        },
        "net": {"network": "mobileclip_i0", "tag": "il-common_reparameterized", "reparameterized": True},
    },
)
