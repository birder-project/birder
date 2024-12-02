"""
MaxViT, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/maxvit.py

Paper "MaxViT: Multi-Axis Vision Transformer", https://arxiv.org/abs/2204.01697
"""

# Reference license: BSD 3-Clause

import logging
import math
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import PreTrainEncoder


def _get_conv_output_shape(
    input_size: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[int, int]:
    return (
        (input_size[0] - kernel_size[0] + 2 * padding[0]) // stride[0] + 1,
        (input_size[1] - kernel_size[1] + 2 * padding[1]) // stride[1] + 1,
    )


def _make_block_input_shapes(input_size: tuple[int, int], n_blocks: int) -> list[tuple[int, int]]:
    shapes = []
    block_input_shape = _get_conv_output_shape(input_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    for _ in range(n_blocks):
        block_input_shape = _get_conv_output_shape(block_input_shape, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        shapes.append(block_input_shape)

    return shapes


def _get_relative_position_index(height: int, width: int) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid([torch.arange(height), torch.arange(width)], indexing="ij"))
    coords_flat = torch.flatten(coords, 1)
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += height - 1
    relative_coords[:, :, 1] += width - 1
    relative_coords[:, :, 0] *= 2 * width - 1

    return relative_coords.sum(-1)


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        squeeze_ratio: float,
        stride: tuple[int, int],
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        p_stochastic_dropout: float,
    ) -> None:
        super().__init__()

        if stride[0] != 1 or stride[1] != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.AvgPool2d(kernel_size=(2, 2), stride=stride, padding=(0, 0)),
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            )
        else:
            self.proj = nn.Identity()

        mid_channels = int(out_channels * expansion_ratio)
        sqz_channels = int(out_channels * squeeze_ratio)
        self.stochastic_depth = StochasticDepth(p_stochastic_dropout, mode="row")

        self.layers = nn.Sequential(
            norm_layer(in_channels),
            Conv2dNormActivation(
                in_channels,
                mid_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                inplace=None,
            ),
            Conv2dNormActivation(
                mid_channels,
                mid_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=mid_channels,
                bias=False,
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                inplace=None,
            ),
            SqueezeExcitation(mid_channels, sqz_channels, activation=nn.SiLU),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x)
        x = self.stochastic_depth(self.layers(x))

        return res + x


class RelativePositionalMultiHeadAttention(nn.Module):
    def __init__(self, feat_dim: int, head_dim: int, max_seq_len: int) -> None:
        super().__init__()

        if feat_dim % head_dim != 0:
            raise ValueError(f"feat_dim: {feat_dim} must be divisible by head_dim: {head_dim}")

        self.n_heads = feat_dim // head_dim
        self.head_dim = head_dim
        self.size = int(math.sqrt(max_seq_len))
        self.max_seq_len = max_seq_len

        self.to_qkv = nn.Linear(feat_dim, self.n_heads * self.head_dim * 3)
        self.scale_factor = feat_dim**-0.5

        self.merge = nn.Linear(self.head_dim * self.n_heads, feat_dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.empty(((2 * self.size - 1) * (2 * self.size - 1), self.n_heads), dtype=torch.float32),
        )
        self.relative_position_index = nn.Buffer(_get_relative_position_index(self.size, self.size))

        # Initialize with truncated normal the bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_relative_positional_bias(self) -> torch.Tensor:
        bias_index = self.relative_position_index.view(-1)
        relative_bias = self.relative_position_bias_table[bias_index].view(self.max_seq_len, self.max_seq_len, -1)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        return relative_bias.unsqueeze(dim=0)

    # pylint: disable=invalid-name
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, G, P, D) = x.size()
        H = self.n_heads
        DH = self.head_dim

        qkv = self.to_qkv(x)
        (q, k, v) = torch.chunk(qkv, 3, dim=-1)

        q = q.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)

        k = k * self.scale_factor
        dot_prod = torch.einsum("B G H I D, B G H J D -> B G H I J", q, k)
        pos_bias = self.get_relative_positional_bias()

        dot_prod = F.softmax(dot_prod + pos_bias, dim=-1)

        out = torch.einsum("B G H I J, B G H J D -> B G H I D", dot_prod, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, G, P, D)

        out = self.merge(out)
        return out


class SwapAxes(nn.Module):
    def __init__(self, axis0: int, axis1: int) -> None:
        super().__init__()
        self.axis0 = axis0
        self.axis1 = axis1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.swapaxes(x, self.axis0, self.axis1)
        return res


class WindowPartition(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, p: int) -> torch.Tensor:
        (B, C, H, W) = x.size()
        P = p

        # Chunk up H and W dimensions
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1)

        # Collapse P * P dimension
        x = x.reshape(B, (H // P) * (W // P), P * P, C)

        return x


class WindowDepartition(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    # pylint: disable=invalid-name
    def forward(self, x: torch.Tensor, p: int, h_partitions: int, w_partitions: int) -> torch.Tensor:
        (B, _G, _PP, C) = x.size()
        P = p
        (HP, WP) = h_partitions, w_partitions

        # Split P * P dimension into 2 P tile dimension
        x = x.reshape(B, HP, WP, P, P, C)

        # Permute into B, C, HP, P, WP, P
        x = x.permute(0, 5, 1, 3, 2, 4)

        # Reshape into B, C, H, W
        x = x.reshape(B, C, HP * P, WP * P)

        return x


class PartitionAttentionLayer(nn.Module):
    """
    Layer for partitioning the input tensor into non-overlapping windows and applying attention to each window
    """

    def __init__(
        self,
        in_channels: int,
        head_dim: int,
        partition_size: int,
        partition_type: str,
        grid_size: tuple[int, int],
        mlp_ratio: int,
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        attention_dropout: float,
        mlp_dropout: float,
        p_stochastic_dropout: float,
    ) -> None:
        super().__init__()

        self.n_heads = in_channels // head_dim
        self.head_dim = head_dim
        self.n_partitions = grid_size[0] // partition_size
        self.partition_type = partition_type
        self.grid_size = grid_size

        if partition_type not in ["grid", "window"]:
            raise ValueError("partition_type must be either 'grid' or 'window'")

        if partition_type == "window":
            self.p = partition_size
            self.g = self.n_partitions
        else:
            self.p = self.n_partitions
            self.g = partition_size

        self.gh = self.grid_size[0] // self.p
        self.gw = self.grid_size[1] // self.p

        # Undefined behavior if H or W are not divisible by p
        # https://github.com/google-research/maxvit/blob/da76cf0d8a6ec668cc31b399c4126186da7da944/maxvit/models/maxvit.py#L766
        torch._assert(
            self.grid_size[0] % self.p == 0 and self.grid_size[1] % self.p == 0,
            f"Grid size must be divisible by partition size. Got grid size of "
            f"{self.grid_size} and partition size of {self.p}",
        )

        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()
        self.partition_swap = SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()
        self.departition_swap = SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()

        self.attn_layer = nn.Sequential(
            norm_layer(in_channels),
            RelativePositionalMultiHeadAttention(in_channels, head_dim, partition_size**2),
            nn.Dropout(attention_dropout),
        )

        # Pre-normalization similar to transformer layers
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * mlp_ratio),
            activation_layer(),
            nn.Linear(in_channels * mlp_ratio, in_channels),
            nn.Dropout(mlp_dropout),
        )

        # layer scale factors
        self.stochastic_dropout = StochasticDepth(p_stochastic_dropout, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.partition_op(x, self.p)
        x = self.partition_swap(x)
        x = x + self.stochastic_dropout(self.attn_layer(x))
        x = x + self.stochastic_dropout(self.mlp_layer(x))
        x = self.departition_swap(x)
        x = self.departition_op(x, self.p, self.gh, self.gw)

        return x


class MaxVitLayer(nn.Module):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        stride: tuple[int, int],
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        head_dim: int,
        mlp_ratio: int,
        mlp_dropout: float,
        attention_dropout: float,
        p_stochastic_dropout: float,
        partition_size: int,
        grid_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion_ratio=expansion_ratio,
                squeeze_ratio=squeeze_ratio,
                stride=stride,
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                p_stochastic_dropout=p_stochastic_dropout,
            ),
            PartitionAttentionLayer(
                in_channels=out_channels,
                head_dim=head_dim,
                partition_size=partition_size,
                partition_type="window",
                grid_size=grid_size,
                mlp_ratio=mlp_ratio,
                activation_layer=activation_layer,
                norm_layer=nn.LayerNorm,
                attention_dropout=attention_dropout,
                mlp_dropout=mlp_dropout,
                p_stochastic_dropout=p_stochastic_dropout,
            ),
            PartitionAttentionLayer(
                in_channels=out_channels,
                head_dim=head_dim,
                partition_size=partition_size,
                partition_type="grid",
                grid_size=grid_size,
                mlp_ratio=mlp_ratio,
                activation_layer=activation_layer,
                norm_layer=nn.LayerNorm,
                attention_dropout=attention_dropout,
                mlp_dropout=mlp_dropout,
                p_stochastic_dropout=p_stochastic_dropout,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


class MaxVitBlock(nn.Module):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        # Conv parameters
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        # Conv + transformer parameters
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        # Transformer parameters
        head_dim: int,
        mlp_ratio: int,
        mlp_dropout: float,
        attention_dropout: float,
        # Partitioning parameters
        partition_size: int,
        input_grid_size: tuple[int, int],
        # Number of layers
        n_layers: int,
        p_stochastic: list[float],
    ) -> None:
        super().__init__()
        if len(p_stochastic) != n_layers:
            raise ValueError(f"p_stochastic must have length n_layers={n_layers}, got p_stochastic={p_stochastic}.")

        # Account for the first stride of the first layer
        self.grid_size = _get_conv_output_shape(input_grid_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        layers = []
        for idx, p in enumerate(p_stochastic):
            if idx == 0:
                stride = (2, 2)
            else:
                stride = (1, 1)

            layers.append(
                MaxVitLayer(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    grid_size=self.grid_size,
                    p_stochastic_dropout=p,
                ),
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class MaxViT(DetectorBackbone, PreTrainEncoder):
    default_size = 224
    block_group_regex = r"body\.stage\d+\.block\.(\d+)"

    # pylint: disable=too-many-locals
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is not None, "must set config"
        assert self.size is not None, "must set size"

        image_size = (self.size, self.size)
        squeeze_ratio = 0.25
        expansion_ratio = 4
        mlp_ratio = 4
        mlp_dropout = 0.0
        attention_dropout = 0.0
        partition_size = int(self.size / (2**5))
        block_channels: list[int] = self.config["block_channels"]
        block_layers: list[int] = self.config["block_layers"]
        stem_channels: int = self.config["stem_channels"]
        head_dim: int = self.config["head_dim"]
        drop_path_rate: float = self.config["drop_path_rate"]

        # Make sure input size will be divisible by the partition size in all blocks
        # Undefined behavior if H or W are not divisible by p
        block_input_sizes = _make_block_input_shapes(image_size, len(block_channels))
        for idx, block_input_size in enumerate(block_input_sizes):
            if block_input_size[0] % partition_size != 0 or block_input_size[1] % partition_size != 0:
                raise ValueError(
                    f"Input size {block_input_size} of block {idx} is not divisible by partition size "
                    f"{partition_size}. Consider changing the partition size or the input size.\n"
                    f"Current configuration yields the following block input sizes: {block_input_sizes}"
                )

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=1 - 0.99)
        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                stem_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
                norm_layer=norm_layer,
                activation_layer=nn.GELU,
                inplace=None,
            ),
            nn.Conv2d(
                stem_channels,
                stem_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=True,
            ),
        )

        # Account for stem stride
        image_size = _get_conv_output_shape(image_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.partition_size = partition_size

        # Blocks
        in_channels = [stem_channels] + block_channels[:-1]
        out_channels = block_channels

        # Pre-compute the stochastic depth probabilities from 0 to stochastic_depth_prob
        # since we have N blocks with L layers, we will have N * L probabilities uniformly distributed
        # over the range [0, stochastic_depth_prob]
        p_stochastic = np.linspace(0, drop_path_rate, sum(block_layers)).tolist()

        p_idx = 0
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i, (in_channel, out_channel, num_layers) in enumerate(zip(in_channels, out_channels, block_layers)):
            stages[f"stage{i+1}"] = MaxVitBlock(
                in_channels=in_channel,
                out_channels=out_channel,
                squeeze_ratio=squeeze_ratio,
                expansion_ratio=expansion_ratio,
                norm_layer=norm_layer,
                activation_layer=nn.GELU,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                mlp_dropout=mlp_dropout,
                attention_dropout=attention_dropout,
                partition_size=partition_size,
                input_grid_size=image_size,
                n_layers=num_layers,
                p_stochastic=p_stochastic[p_idx : p_idx + num_layers],
            )
            return_channels.append(out_channel)
            image_size = stages[f"stage{i+1}"].grid_size
            p_idx += num_layers

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = block_channels[-1]
        self.classifier = self.create_classifier()

        self.encoding_size = stem_channels

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
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

    def masked_encoding(
        self, x: torch.Tensor, mask_ratio: float, mask_token: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, ...]:
        assert mask_token is not None

        (B, _, H, W) = x.shape
        L = (H // 32) * (W // 32)  # Patch size = 32
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(B, L, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0

        # Un-shuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Upsample mask
        scale = 2**4
        assert len(mask.shape) == 2

        upscale_mask = (
            mask.reshape(-1, (H // 32), (W // 32)).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
        )
        upscale_mask = upscale_mask.unsqueeze(1).type_as(x)

        x = self.stem(x)
        mask_tokens = mask_token.permute(0, 3, 1, 2)
        mask_tokens = mask_tokens.expand(B, -1, (H // 2), (W // 2))  # Patch stride = 2
        x = x * (1.0 - upscale_mask) + (mask_tokens * upscale_mask)
        x = self.body(x)

        return (x, mask)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)

    def create_classifier(self, embed_dim: Optional[int] = None) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size

        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, self.num_classes, bias=False),
        )

    def adjust_size(self, new_size: int) -> None:
        if new_size == self.size:
            return

        logging.info(f"Adjusting model input resolution from {self.size} to {new_size}")
        super().adjust_size(new_size)

        new_grid_size = _get_conv_output_shape((new_size, new_size), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        new_grid_size = _get_conv_output_shape(new_grid_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.partition_size = int(new_size / (2**5))
        for m in self.body.modules():
            if isinstance(m, MaxVitBlock):
                m.grid_size = _get_conv_output_shape(new_grid_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                for layer in m.block:
                    for i in range(1, 3):
                        mod = layer.layers[i]  # PartitionAttentionLayer
                        mod.n_partitions = new_grid_size[0] // self.partition_size
                        mod.grid_size = new_grid_size

                        if mod.partition_type == "window":
                            mod.p = self.partition_size
                            mod.g = mod.n_partitions

                        else:
                            mod.p = mod.n_partitions
                            mod.g = self.partition_size

                        mod.gh = mod.grid_size[0] // mod.p
                        mod.gw = mod.grid_size[1] // mod.p

                        # Undefined behavior if H or W are not divisible by p
                        # https://github.com/google-research/maxvit/blob/da76cf0d8a6ec668cc31b399c4126186da7da944/maxvit/models/maxvit.py#L766
                        torch._assert(  # pylint: disable=protected-access
                            mod.grid_size[0] % mod.p == 0 and mod.grid_size[1] % mod.p == 0,
                            f"Grid size must be divisible by partition size. Got grid size of "
                            f"{mod.grid_size} and partition size of {mod.p}",
                        )

                        attn = mod.attn_layer[1]  # RelativePositionalMultiHeadAttention
                        attn.size = self.partition_size
                        attn.max_seq_len = self.partition_size**2
                        attn.relative_position_index = nn.Buffer(_get_relative_position_index(attn.size, attn.size))

                        # Interpolate relative_position_bias_table, adapted from
                        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed_rel.py
                        dst_size = (2 * attn.size - 1, 2 * attn.size - 1)
                        rel_pos_bias = attn.relative_position_bias_table
                        rel_pos_bias = rel_pos_bias.detach()

                        (src_num_pos, num_attn_heads) = rel_pos_bias.shape
                        src_size = (int(src_num_pos**0.5), int(src_num_pos**0.5))

                        def _calc(src: int, dst: int) -> list[float]:
                            (left, right) = 1.01, 1.5
                            while right - left > 1e-6:
                                q = (left + right) / 2.0
                                gp = (1.0 - q ** (src // 2)) / (1.0 - q)  # Geometric progression
                                if gp > dst // 2:
                                    right = q

                                else:
                                    left = q

                            dis = []
                            cur = 1.0
                            for i in range(src // 2):
                                dis.append(cur)
                                cur += q ** (i + 1)

                            r_ids = [-_ for _ in reversed(dis)]
                            return r_ids + [0] + dis

                        y = _calc(src_size[0], dst_size[0])
                        x = _calc(src_size[1], dst_size[1])

                        ty = dst_size[0] // 2.0
                        tx = dst_size[1] // 2.0
                        dy = torch.arange(-ty, ty + 0.1, 1.0)
                        dx = torch.arange(-tx, tx + 0.1, 1.0)
                        dxy = torch.meshgrid(dx, dy, indexing="ij")

                        all_rel_pos_bias = []
                        for i in range(num_attn_heads):
                            z = rel_pos_bias[:, i].view(src_size[0], src_size[1]).float()
                            rgi = interpolate.RegularGridInterpolator(
                                (x, y), z.numpy().T, method="cubic", bounds_error=False, fill_value=None
                            )
                            r = torch.Tensor(rgi(dxy)).T.contiguous().to(rel_pos_bias.device)

                            r = r.view(-1, 1)
                            all_rel_pos_bias.append(r)

                        rel_pos_bias = torch.concat(all_rel_pos_bias, dim=-1)
                        attn.relative_position_bias_table = nn.Parameter(rel_pos_bias)

                new_grid_size = m.grid_size


registry.register_alias(
    "maxvit_t",
    MaxViT,
    config={
        "block_channels": [64, 128, 256, 512],
        "block_layers": [2, 2, 5, 2],
        "stem_channels": 64,
        "head_dim": 32,
        "drop_path_rate": 0.2,
    },
)
registry.register_alias(
    "maxvit_s",
    MaxViT,
    config={
        "block_channels": [96, 128, 256, 512],
        "block_layers": [2, 2, 5, 2],
        "stem_channels": 64,
        "head_dim": 32,
        "drop_path_rate": 0.3,
    },
)
registry.register_alias(
    "maxvit_b",
    MaxViT,
    config={
        "block_channels": [96, 192, 384, 768],
        "block_layers": [2, 6, 14, 2],
        "stem_channels": 64,
        "head_dim": 32,
        "drop_path_rate": 0.4,
    },
)
registry.register_alias(
    "maxvit_l",
    MaxViT,
    config={
        "block_channels": [128, 256, 512, 1024],
        "block_layers": [2, 6, 14, 2],
        "stem_channels": 128,
        "head_dim": 32,
        "drop_path_rate": 0.5,
    },
)
