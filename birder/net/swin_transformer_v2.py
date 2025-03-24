"""
Swin Transformer v2, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py
https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py

Paper "Swin Transformer V2: Scaling Up Capacity and Resolution", https://arxiv.org/abs/2111.09883

Changes from original:
* Window size based on image size (image size // 32)
"""

# Reference license: BSD 3-Clause and MIT

from collections import OrderedDict
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType
from birder.net.swin_transformer_v1 import get_relative_position_bias
from birder.net.swin_transformer_v1 import patch_merging_pad
from birder.net.swin_transformer_v1 import shifted_window_attention


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(2 * dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = patch_merging_pad(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        x = self.norm(x)

        return x


class ShiftedWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool,
        proj_bias: bool,
    ) -> None:
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")

        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # MLP to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )
        if qkv_bias is True:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length : 2 * length].data.zero_()

    def define_relative_position_bias_table(self) -> None:
        # Get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        relative_coords_table *= 8  # Normalize to (-8, 8)
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        )
        self.relative_coords_table = nn.Buffer(relative_coords_table)

    def define_relative_position_index(self) -> None:
        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.relative_position_index = nn.Buffer(relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relative_position_bias = get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,
            self.window_size,
        )
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
        )


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        mlp_ratio: float,
        stochastic_depth_prob: float,
    ) -> None:
        super().__init__()

        self.input_resolution = input_resolution
        window_size_h = window_size[0]
        window_size_w = window_size[1]
        shift_size_h = shift_size[0]
        shift_size_w = shift_size[1]
        if self.input_resolution[0] <= window_size_h:
            shift_size_h = 0
            window_size_h = self.input_resolution[0]
        if self.input_resolution[1] <= window_size_w:
            shift_size_w = 0
            window_size_w = self.input_resolution[1]

        window_size = (window_size_h, window_size_w)
        shift_size = (shift_size_h, shift_size_w)

        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = ShiftedWindowAttention(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=True,
            proj_bias=True,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))

        return x


# pylint: disable=invalid-name
class Swin_Transformer_v2(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    default_size = (256, 256)
    block_group_regex = r"body\.stage\d+\.(\d+)"

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

        patch_size = 4
        embed_dim: int = self.config["embed_dim"]
        depths: list[int] = self.config["depths"]
        num_heads: list[int] = self.config["num_heads"]
        drop_path_rate: float = self.config["drop_path_rate"]
        self.window_scale_factor: int = self.config["window_scale_factor"]
        mlp_ratio = 4.0
        window_size_h = int(self.size[0] / (2**5)) * self.window_scale_factor
        window_size_w = int(self.size[1] / (2**5)) * self.window_scale_factor
        window_size = (window_size_h, window_size_w)

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                embed_dim,
                kernel_size=(patch_size, patch_size),
                stride=patch_size,
                padding=(0, 0),
                bias=True,
            ),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(embed_dim, eps=1e-5),
        )

        resolution = (self.size[0] // patch_size, self.size[1] // patch_size)
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        layers = []
        for i_stage, depth in enumerate(depths):
            dim = embed_dim * 2**i_stage
            for i_layer in range(depth):
                # Adjust stochastic depth probability based on the depth of the stage block
                sd_prob = drop_path_rate * float(stage_block_id) / (total_stage_blocks - 1)
                if i_layer % 2 == 0:
                    shift_size = (0, 0)
                else:
                    shift_size = (window_size[0] // 2, window_size[1] // 2)

                layers.append(
                    SwinTransformerBlock(
                        dim,
                        resolution,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        stochastic_depth_prob=sd_prob,
                    )
                )
                stage_block_id += 1

            stages[f"stage{i_stage+1}"] = nn.Sequential(*layers)
            return_channels.append(dim)
            layers = []

            # Add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(PatchMerging(dim))
                resolution = (resolution[0] // 2, resolution[1] // 2)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.LayerNorm(num_features, eps=1e-5),
            Permute([0, 3, 1, 2]),  # B H W C -> B C H W
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = num_features
        self.classifier = self.create_classifier()

        self.stem_stride = patch_size
        self.stem_width = embed_dim
        self.encoding_size = num_features

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)

        out = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.return_stages:
                out[name] = x.permute(0, 3, 1, 2).contiguous()

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
        x = mask_tensor(
            x, mask, channels_last=True, patch_factor=self.max_stride // self.stem_stride, mask_token=mask_token
        )
        x = self.body(x)

        result: TokenRetentionResultType = {}
        if return_keys in ("all", "features"):
            result["features"] = x.permute(0, 3, 1, 2).contiguous()
        if return_keys in ("all", "embedding"):
            result["embedding"] = self.features(x)

        return result

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        old_size = self.size
        super().adjust_size(new_size)

        for m in self.body.modules():
            if isinstance(m, SwinTransformerBlock):
                new_window_size_h = int(new_size[0] / (2**5)) * self.window_scale_factor
                new_window_size_w = int(new_size[1] / (2**5)) * self.window_scale_factor
                new_window_size = (new_window_size_h, new_window_size_w)

                shift_size_h = m.attn.shift_size[0]
                shift_size_w = m.attn.shift_size[1]
                window_size_h = new_window_size[0]
                window_size_w = new_window_size[1]

                # Adjust resolution
                scale_h = old_size[0] // m.input_resolution[0]
                scale_w = old_size[1] // m.input_resolution[1]
                m.input_resolution = (new_size[0] // scale_h, new_size[1] // scale_w)

                if m.input_resolution[0] <= window_size_h:
                    shift_size_h = 0
                    window_size_h = m.input_resolution[0]

                if m.input_resolution[1] <= window_size_w:
                    shift_size_w = 0
                    window_size_w = m.input_resolution[1]

                m.attn.window_size = (window_size_h, window_size_w)

                if m.attn.shift_size[0] != 0:
                    shift_size_h = m.attn.window_size[0] // 2

                if m.attn.shift_size[1] != 0:
                    shift_size_w = m.attn.window_size[1] // 2

                m.attn.shift_size = (shift_size_h, shift_size_w)

                m.attn.define_relative_position_bias_table()
                m.attn.define_relative_position_index()


# Window factor = 1
registry.register_alias(
    "swin_transformer_v2_t",
    Swin_Transformer_v2,
    config={
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "drop_path_rate": 0.2,
        "window_scale_factor": 1,
    },
)
registry.register_alias(
    "swin_transformer_v2_s",
    Swin_Transformer_v2,
    config={
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
        "drop_path_rate": 0.3,
        "window_scale_factor": 1,
    },
)
registry.register_alias(
    "swin_transformer_v2_b",
    Swin_Transformer_v2,
    config={
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "drop_path_rate": 0.5,
        "window_scale_factor": 1,
    },
)
registry.register_alias(
    "swin_transformer_v2_l",
    Swin_Transformer_v2,
    config={
        "embed_dim": 192,
        "depths": [2, 2, 18, 2],
        "num_heads": [6, 12, 24, 48],
        "drop_path_rate": 0.5,
        "window_scale_factor": 1,
    },
)

# Window factor = 2
registry.register_alias(
    "swin_transformer_v2_w2_t",
    Swin_Transformer_v2,
    config={
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "drop_path_rate": 0.2,
        "window_scale_factor": 2,
    },
)
registry.register_alias(
    "swin_transformer_v2_w2_s",
    Swin_Transformer_v2,
    config={
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
        "drop_path_rate": 0.3,
        "window_scale_factor": 2,
    },
)
registry.register_alias(
    "swin_transformer_v2_w2_b",
    Swin_Transformer_v2,
    config={
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "drop_path_rate": 0.5,
        "window_scale_factor": 2,
    },
)
registry.register_alias(
    "swin_transformer_v2_w2_l",
    Swin_Transformer_v2,
    config={
        "embed_dim": 192,
        "depths": [2, 2, 18, 2],
        "num_heads": [6, 12, 24, 48],
        "drop_path_rate": 0.5,
        "window_scale_factor": 2,
    },
)

registry.register_weights(
    "swin_transformer_v2_s_intermediate-arabian-peninsula",
    {
        "url": (
            "https://huggingface.co/birder-project/swin_transformer_v2_s_intermediate-arabian-peninsula/resolve/main"
        ),
        "description": (
            "Swin Transformer v2 small model with intermediate training, "
            "then fine-tuned on the arabian-peninsula dataset"
        ),
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 193.0,
                "sha256": "766706779aef6b55860cabd5712bcb758f463896100645648f917495214dd478",
            },
        },
        "net": {"network": "swin_transformer_v2_s", "tag": "intermediate-arabian-peninsula"},
    },
)
