"""
Swin Transformer, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py

Paper "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
https://arxiv.org/abs/2103.14030

Changes from original:
* Window size based on image size (image size // 32)
"""

# Reference license: BSD 3-Clause

import math
from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.fx
import torch.nn.functional as F
from scipy import interpolate
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


def patch_merging_pad(x: torch.Tensor) -> torch.Tensor:
    (H, W, _) = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
    x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
    x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
    x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
    x = torch.concat([x0, x1, x2, x3], dim=-1)  # ... H/2 W/2 4*C

    return x


torch.fx.wrap("patch_merging_pad")


def get_relative_position_bias(
    relative_position_bias_table: torch.Tensor,
    relative_position_index: torch.Tensor,
    window_size: tuple[int, int],
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

    return relative_position_bias


torch.fx.wrap("get_relative_position_bias")


# pylint: disable=too-many-locals
def shifted_window_attention(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    relative_position_bias: torch.Tensor,
    window_size: tuple[int, int],
    num_heads: int,
    shift_size: tuple[int, int],
    qkv_bias: Optional[torch.Tensor] = None,
    proj_bias: Optional[torch.Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    (B, H, W, C) = x.size()

    # Pad feature maps to multiples of window size
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
    (_, pad_h, pad_w, _) = x.size()

    # If window size is larger than feature size, there is no need to shift window
    shift_size_w = shift_size[0]
    shift_size_h = shift_size[1]
    if window_size[0] >= pad_h:
        shift_size_h = 0
    if window_size[1] >= pad_w:
        shift_size_w = 0

    shift_size = (shift_size_w, shift_size_h)

    # Cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # Partition windows
    num_windows = (pad_h // window_size[0]) * (pad_w // window_size[1])
    x = x.view(B, pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # Multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()

    qkv = F.linear(x, qkv_weight, qkv_bias)  # pylint: disable=not-callable
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]
    if logit_scale is not None:
        # Cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale

    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))

    # Add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # Generate attention mask
        attn_mask = x.new_zeros((pad_h, pad_w))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1

        attn_mask = attn_mask.view(pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)  # pylint: disable=not-callable

    # Reverse windows
    x = x.view(B, pad_h // window_size[0], pad_w // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_h, pad_w, C)

    # Reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # Unpad features
    x = x[:, :H, :W, :].contiguous()

    return x


torch.fx.wrap("shifted_window_attention")


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C

        return x


class ShiftedWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
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

    def define_relative_position_bias_table(self) -> None:
        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

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
            self.relative_position_bias_table, self.relative_position_index, self.window_size
        )
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
        )


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        mlp_ratio: float,
        stochastic_depth_prob: float,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttention(
            dim,
            window_size,
            shift_size,
            num_heads,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))

        return x


# pylint: disable=invalid-name
class Swin_Transformer_v1(DetectorBackbone):
    block_group_regex = r"body\.stage\d+\.(\d+)"

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

        mlp_ratio = 4.0
        patch_size = 4
        window_size = (int(self.size[0] / (2**5)), int(self.size[1] / (2**5)))
        embed_dim: int = self.config["embed_dim"]
        depths: list[int] = self.config["depths"]
        num_heads: list[int] = self.config["num_heads"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                embed_dim,
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                padding=(0, 0),
                bias=True,
            ),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(embed_dim, eps=1e-5),
        )

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

        # Weight initialization
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        old_size = self.size
        super().adjust_size(new_size)

        for m in self.body.modules():
            if isinstance(m, SwinTransformerBlock):
                old_window_size = (old_size[0] // (2**5), old_size[1] // (2**5))
                new_window_size = (new_size[0] // (2**5), new_size[1] // (2**5))
                m.attn.window_size = new_window_size
                shift_size_w = m.attn.shift_size[0]
                shift_size_h = m.attn.shift_size[1]
                if m.attn.shift_size[0] != 0:
                    shift_size_w = m.attn.window_size[0] // 2

                if m.attn.shift_size[1] != 0:
                    shift_size_h = m.attn.window_size[1] // 2

                m.attn.shift_size = (shift_size_w, shift_size_h)

                m.attn.define_relative_position_index()

                # Interpolate relative_position_bias_table, adapted from
                # https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed_rel.py
                src_size = (2 * old_window_size[0] - 1, 2 * old_window_size[1] - 1)
                dst_size = (2 * new_window_size[0] - 1, 2 * new_window_size[1] - 1)
                rel_pos_bias = m.attn.relative_position_bias_table
                rel_pos_bias = rel_pos_bias.detach()
                num_attn_heads = rel_pos_bias.size(1)

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
                m.attn.relative_position_bias_table = nn.Parameter(rel_pos_bias)


registry.register_model_config(
    "swin_transformer_v1_t",
    Swin_Transformer_v1,
    config={
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "swin_transformer_v1_s",
    Swin_Transformer_v1,
    config={
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "swin_transformer_v1_b",
    Swin_Transformer_v1,
    config={
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "drop_path_rate": 0.5,
    },
)
registry.register_model_config(
    "swin_transformer_v1_l",
    Swin_Transformer_v1,
    config={
        "embed_dim": 192,
        "depths": [2, 2, 18, 2],
        "num_heads": [6, 12, 24, 48],
        "drop_path_rate": 0.5,
    },
)
