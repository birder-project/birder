"""
EfficientFormer v2, adapted from
https://github.com/snap-research/EfficientFormer/blob/main/models/efficientformer_v2.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientformer_v2.py

Paper "Rethinking Vision Transformers for MobileNet Size and Speed",
https://arxiv.org/abs/2212.08059

Changes from original:
* Removed attention bias cache
"""

# Reference license: Apache-2.0 (both)

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import BaseNet


# pylint: disable=too-many-instance-attributes
class Attention2d(nn.Module):
    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int,
        attn_ratio: float,
        resolution: tuple[int, int],
        stride: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5

        if stride is not None:
            resolution = (math.ceil(resolution[0] / stride), math.ceil(resolution[1] / stride))
            self.stride_conv = Conv2dNormActivation(
                dim, dim, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), groups=dim, activation_layer=None
            )
            self.upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
        else:
            self.stride_conv = nn.Identity()
            self.upsample = nn.Identity()

        self.resolution = resolution
        self.N = self.resolution[0] * self.resolution[1]
        self.dh = int(attn_ratio * key_dim) * num_heads
        kh = key_dim * self.num_heads

        self.q = Conv2dNormActivation(dim, kh, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=None)
        self.k = Conv2dNormActivation(dim, kh, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=None)
        self.v = Conv2dNormActivation(
            dim, self.dh, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=None
        )
        self.v_local = Conv2dNormActivation(
            self.dh, self.dh, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=self.dh, activation_layer=None
        )
        self.talking_head1 = nn.Conv2d(
            self.num_heads, self.num_heads, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.talking_head2 = nn.Conv2d(
            self.num_heads, self.num_heads, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

        self.act = nn.GELU()
        self.proj = Conv2dNormActivation(
            self.dh, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=None
        )

        pos = torch.stack(
            torch.meshgrid(torch.arange(self.resolution[0]), torch.arange(self.resolution[1]), indexing="ij")
        ).flatten(1)
        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * self.resolution[1]) + rel_pos[1]
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, self.N))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(rel_pos), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)  # (B, C, H, W)
        x = self.stride_conv(x)

        q = self.q(x).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        k = self.k(x).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        attn = attn + self.attention_biases[:, self.attention_bias_idxs]
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v).transpose(2, 3)
        x = x.reshape(B, self.dh, self.resolution[0], self.resolution[1]) + v_local
        x = self.upsample(x)

        x = self.act(x)
        x = self.proj(x)

        return x


class LocalGlobalQuery(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))
        self.local = nn.Conv2d(in_dim, in_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=in_dim)
        self.proj = Conv2dNormActivation(
            in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)

        return q


class Attention2dDownsample(nn.Module):
    def __init__(
        self,
        dim: int,
        key_dim: int,
        out_dim: int,
        num_heads: int,
        attn_ratio: float,
        resolution: tuple[int, int],
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.resolution = resolution
        self.resolution2 = (math.ceil(self.resolution[0] / 2), math.ceil(self.resolution[1] / 2))
        self.N = self.resolution[0] * self.resolution[1]
        self.N2 = self.resolution2[0] * self.resolution2[1]  # pylint:disable=invalid-name

        self.dh = int(attn_ratio * key_dim) * num_heads
        kh = key_dim * self.num_heads

        self.q = LocalGlobalQuery(dim, kh)
        self.k = Conv2dNormActivation(dim, kh, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=None)
        self.v = Conv2dNormActivation(
            dim, self.dh, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=None
        )
        self.v_local = Conv2dNormActivation(
            self.dh, self.dh, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=self.dh, activation_layer=None
        )

        self.act = nn.GELU()
        self.proj = Conv2dNormActivation(
            self.dh, out_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=None
        )

        self.attention_biases = nn.Parameter(torch.zeros(num_heads, self.N))
        k_pos = torch.stack(
            torch.meshgrid(torch.arange(self.resolution[0]), torch.arange(self.resolution[1]), indexing="ij")
        ).flatten(1)
        q_pos = torch.stack(
            torch.meshgrid(
                torch.arange(0, self.resolution[0], step=2), torch.arange(0, self.resolution[1], step=2), indexing="ij"
            )
        ).flatten(1)
        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * self.resolution[1]) + rel_pos[1]
        self.register_buffer("attention_bias_idxs", rel_pos, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)  # (B, C, H, W)

        q = self.q(x).reshape(B, self.num_heads, -1, self.N2).permute(0, 1, 3, 2)
        k = self.k(x).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        attn = attn + self.attention_biases[:, self.attention_bias_idxs]
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(2, 3)
        x = x.reshape(B, self.dh, self.resolution2[0], self.resolution2[1]) + v_local
        x = self.act(x)
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        resolution: tuple[int, int],
        use_attn: bool,
    ) -> None:
        super().__init__()
        self.conv = Conv2dNormActivation(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation_layer=None
        )

        if use_attn is True:
            self.attn = Attention2dDownsample(
                dim=in_channels,
                key_dim=16,
                out_dim=out_channels,
                num_heads=8,
                attn_ratio=4,
                resolution=resolution,
            )
        else:
            self.attn = None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.attn is not None:
            return self.attn(x) + out

        return out  # type: ignore[unreachable]


class ConvMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        drop: float,
    ) -> None:
        super().__init__()
        self.fc1 = Conv2dNormActivation(in_features, hidden_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.mid = Conv2dNormActivation(
            hidden_features, hidden_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=hidden_features
        )
        self.drop1 = nn.Dropout(drop)
        self.fc2 = Conv2dNormActivation(
            hidden_features, in_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=None
        )
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.mid(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_values: float, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        if self.inplace is True:
            return x.mul_(gamma)

        return x * gamma


class EfficientFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        proj_drop: float,
        drop_path: float,
        layer_scale_init_value: float,
        resolution: tuple[int, int],
        stride: Optional[int],
        use_attn: bool,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.stride = stride
        self.use_attn = use_attn

        if use_attn is True:
            self.token_mixer = Attention2d(
                dim,
                key_dim=32,
                num_heads=8,
                attn_ratio=4.0,
                resolution=resolution,
                stride=stride,
            )
            self.ls1 = LayerScale2d(dim, layer_scale_init_value)
            self.drop_path1 = StochasticDepth(drop_path, mode="row")
        else:
            self.token_mixer = None  # type: ignore[assignment]
            self.ls1 = None  # type: ignore[assignment]
            self.drop_path1 = None

        self.mlp = ConvMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=proj_drop,
        )
        self.ls2 = LayerScale2d(dim, layer_scale_init_value)
        self.drop_path2 = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.token_mixer is not None:
            x = x + self.drop_path1(self.ls1(self.token_mixer(x)))

        x = x + self.drop_path2(self.ls2(self.mlp(x)))

        return x


class EfficientFormerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        depth: int,
        resolution: tuple[int, int],
        downsample: bool,
        block_stride: Optional[int],
        downsample_use_attn: bool,
        block_use_attn: bool,
        num_vit: int,
        mlp_ratios: list[int],
        proj_drop: float,
        drop_path: list[float],
        layer_scale_init_value: float,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        if downsample is True:
            self.downsample_block = Downsample(
                dim,
                dim_out,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                resolution=resolution,
                use_attn=downsample_use_attn,
            )
            dim = dim_out
            resolution = (math.ceil(resolution[0] / 2), math.ceil(resolution[1] / 2))
        else:
            assert dim == dim_out
            self.downsample_block = nn.Identity()

        blocks = []
        for block_idx in range(depth):
            remain_idx = depth - num_vit - 1
            blocks.append(
                EfficientFormerBlock(
                    dim,
                    mlp_ratio=mlp_ratios[block_idx],
                    proj_drop=proj_drop,
                    drop_path=drop_path[block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                    resolution=resolution,
                    stride=block_stride,
                    use_attn=block_use_attn and block_idx > remain_idx,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_block(x)
        x = self.blocks(x)

        return x


# pylint: disable=invalid-name
class EfficientFormer_v2(BaseNet):
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

        layer_scale_init_value = 1e-5
        if net_param == 0:
            # S0
            embed_dims = (32, 48, 96, 176)
            depths = (2, 2, 6, 4)
            drop_path_rate = 0.0
            num_vit = 2
            mlp_ratios = [
                [4, 4],
                [4, 4],
                [4, 3, 3, 3, 4, 4],
                [4, 3, 3, 4],
            ]

        elif net_param == 1:
            # S1
            embed_dims = (32, 48, 120, 224)
            depths = (3, 3, 9, 6)
            drop_path_rate = 0.0
            num_vit = 2
            mlp_ratios = [
                [4, 4, 4],
                [4, 4, 4],
                [4, 4, 3, 3, 3, 3, 4, 4, 4],
                [4, 4, 3, 3, 4, 4],
            ]

        elif net_param == 2:
            # S2
            embed_dims = (32, 64, 144, 288)
            depths = (4, 4, 12, 8)
            drop_path_rate = 0.02
            num_vit = 4
            mlp_ratios = [
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
                [4, 4, 3, 3, 3, 3, 4, 4],
            ]

        elif net_param == 3:
            # L
            embed_dims = (40, 80, 192, 384)
            depths = (5, 5, 15, 10)
            drop_path_rate = 0.1
            num_vit = 6
            mlp_ratios = [
                [4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4],
                [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
                [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
            ]

        else:
            raise ValueError(f"net_param = {net_param} not supported")

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                embed_dims[0] // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation_layer=nn.GELU,
                inplace=None,
            ),
            Conv2dNormActivation(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation_layer=nn.GELU,
                inplace=None,
            ),
        )

        prev_dim = embed_dims[0]
        stride = 4
        num_stages = len(depths)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        downsample = (False,) + (True,) * (num_stages - 1)

        stages = []
        for i in range(num_stages):
            curr_resolution = (math.ceil(self.size / stride), math.ceil(self.size / stride))
            stages.append(
                EfficientFormerStage(
                    prev_dim,
                    embed_dims[i],
                    depth=depths[i],
                    resolution=curr_resolution,
                    downsample=downsample[i],
                    block_stride=2 if i == 2 else None,
                    downsample_use_attn=i >= 3,
                    block_use_attn=i >= 2,
                    num_vit=num_vit,
                    mlp_ratios=mlp_ratios[i],
                    proj_drop=0.0,
                    drop_path=dpr[i],
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
            prev_dim = embed_dims[i]
            if downsample[i] is True:
                stride *= 2

        stages.append(nn.BatchNorm2d(embed_dims[-1]))
        self.body = nn.Sequential(*stages)

        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = embed_dims[-1]
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()
        self.distillation_output = False

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) is True:
                nn.init.trunc_normal_(m.weight, std=0.02)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()

    def freeze(self, freeze_classifier: bool = True) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad = True

            for param in self.dist_classifier.parameters():
                param.requires_grad = True

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)

    def set_distillation_output(self, enable: bool = True) -> None:
        self.distillation_output = enable

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        x_cls = self.classifier(x)
        x_dist = self.dist_classifier(x)

        if self.training is True and self.distillation_output is True:
            x = torch.stack([x_cls, x_dist], dim=1)
        else:
            # Classifier "token" as an average of both tokens (during normal training or inference)
            x = (x_cls + x_dist) / 2

        return x

    def adjust_size(self, new_size: int) -> None:
        old_size = self.size
        super().adjust_size(new_size)

        old_base = old_size // 4
        new_base = new_size // 4
        for stage in self.body.modules():
            if isinstance(stage, EfficientFormerStage) is True:
                if stage.downsample is True:
                    if stage.downsample_block.attn is not None:
                        attn = stage.downsample_block.attn
                        attn.resolution = (new_base, new_base)
                        attn.resolution2 = (math.ceil(new_base / 2), math.ceil(new_base / 2))
                        attn.N = attn.resolution[0] * attn.resolution[1]
                        attn.N2 = attn.resolution2[0] * attn.resolution2[1]

                        # Interpolate attention_biases
                        orig_dtype = attn.attention_biases.dtype
                        attention_biases = attn.attention_biases.float()  # Interpolate needs float32
                        attention_biases = attention_biases.reshape(1, old_base, old_base, -1).permute(0, 3, 1, 2)
                        attention_biases = F.interpolate(
                            attention_biases,
                            size=(new_base, new_base),
                            mode="bicubic",
                            antialias=True,
                        )
                        attention_biases = attention_biases.permute(0, 2, 3, 1).reshape(attn.num_heads, -1)
                        attention_biases = attention_biases.to(orig_dtype)
                        attn.attention_biases = torch.nn.Parameter(attention_biases)

                        k_pos = torch.stack(
                            torch.meshgrid(
                                torch.arange(attn.resolution[0]), torch.arange(attn.resolution[1]), indexing="ij"
                            )
                        ).flatten(1)
                        q_pos = torch.stack(
                            torch.meshgrid(
                                torch.arange(0, attn.resolution[0], step=2),
                                torch.arange(0, attn.resolution[1], step=2),
                                indexing="ij",
                            )
                        ).flatten(1)
                        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
                        rel_pos = (rel_pos[0] * attn.resolution[1]) + rel_pos[1]
                        attn.attention_bias_idxs = torch.LongTensor(rel_pos)

                    old_base = old_base // 2
                    new_base = new_base // 2

                for m in stage.modules():
                    if isinstance(m, EfficientFormerBlock) is True:
                        if m.resolution[0] == new_base and m.resolution[1] == new_base:
                            return

                        c_old_base = old_base
                        c_new_base = new_base
                        if m.token_mixer is not None and m.use_attn is True and m.stride is not None:
                            c_old_base = math.ceil(old_base / m.stride)
                            c_new_base = math.ceil(new_base / m.stride)

                        if m.token_mixer is not None:
                            m.token_mixer.resolution = (c_new_base, c_new_base)
                            m.token_mixer.N = m.token_mixer.resolution[0] * m.token_mixer.resolution[1]

                            pos = torch.stack(
                                torch.meshgrid(torch.arange(c_new_base), torch.arange(c_new_base), indexing="ij")
                            ).flatten(1)
                            rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
                            rel_pos = (rel_pos[0] * c_new_base) + rel_pos[1]
                            orig_dtype = m.token_mixer.attention_biases.dtype
                            attention_biases = m.token_mixer.attention_biases.float()  # Interpolate needs float32
                            attention_biases = attention_biases.reshape(1, c_old_base, c_old_base, -1).permute(
                                0, 3, 1, 2
                            )
                            attention_biases = F.interpolate(
                                attention_biases,
                                size=(c_new_base, c_new_base),
                                mode="bicubic",
                                antialias=True,
                            )
                            attention_biases = attention_biases.permute(0, 2, 3, 1).reshape(m.token_mixer.num_heads, -1)
                            attention_biases = attention_biases.to(orig_dtype)
                            m.token_mixer.attention_biases = torch.nn.Parameter(attention_biases)
                            m.token_mixer.attention_bias_idxs = torch.LongTensor(rel_pos)

        logging.info(f"Resized attention base resolution: {old_base} to {new_base}")


registry.register_alias("efficientformer_v2_s0", EfficientFormer_v2, 0)
registry.register_alias("efficientformer_v2_s1", EfficientFormer_v2, 1)
registry.register_alias("efficientformer_v2_s2", EfficientFormer_v2, 2)
registry.register_alias("efficientformer_v2_l", EfficientFormer_v2, 3)