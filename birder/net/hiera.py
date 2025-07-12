"""
Hiera, adapted from
https://github.com/facebookresearch/hiera/blob/main/hiera/hiera.py

Paper "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles", https://arxiv.org/abs/2306.00989

AbsWin variant from the paper "Window Attention is Bugged: How not to Interpolate Position Embeddings",
https://arxiv.org/abs/2311.05613

Changes from original:
* Support only 2d
"""

# Reference license: Apache-2.0

import math
from collections import OrderedDict
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_from_indices
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenOmissionResultType
from birder.net.vit import adjust_position_embedding


def get_resized_mask(target_size: list[int], mask: torch.Tensor) -> torch.Tensor:
    # target_size: [(T), (H), W]
    # (spatial) mask: [B, C, (t), (h), w]
    if mask is None:
        return mask

    assert len(mask.shape[2:]) == len(target_size)
    if mask.shape[2:] != target_size:
        return F.interpolate(mask.float(), size=target_size)

    return mask


def undo_windowing(x: torch.Tensor, shape: tuple[int, int], mu_shape: list[int]) -> torch.Tensor:
    """
    Restore spatial organization by undoing windowed organization of mask units.

    Parameters
    ----------
    x
        Organized by mask units windows, e.g. in 2d [B, #MUy*#MUx, MUy, MUx, C]
    shape
        Current spatial shape, if it were not organized into mask unit windows, e.g. in 2d [B, #MUy*MUy, #MUx*MUx, C]
    mu_shape
        current mask unit shape, e.g. in 2d [MUy, MUx]

    Returns
    -------
    Tensor with shape of [B, #MUy*MUy, #MUx*MUx, C]
    """

    D = len(shape)
    (B, C) = x.shape[0], x.shape[-1]
    # [B, #MUy*#MUx, MUy, MUx, C] -> [B, #MUy, #MUx, MUy, MUx, C]
    num_mu = [s // mu for s, mu in zip(shape, mu_shape)]
    x = x.view(B, *num_mu, *mu_shape, C)

    # [B, #MUy, #MUx, MUy, MUx, C] -> [B, #MUy*MUy, #MUx*MUx, C]
    permute = [0] + sum([list(p) for p in zip(range(1, 1 + D), range(1 + D, 1 + 2 * D))], []) + [len(x.shape) - 1]
    x = x.permute(permute).reshape(B, *shape, C)

    return x


class PatchEmbed(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, kernel: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int]
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            mask = get_resized_mask(target_size=x.shape[2:], mask=mask)
            x = self.proj(x * mask.to(torch.bool))
        else:
            x = self.proj(x)

        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)

        return x


class Unroll(nn.Module):
    """
    Reorders the tokens such that patches are contiguous in memory.
    E.g., given [B, (H, W), C] and stride of (Sy, Sx), this will re-order the tokens as
                           [B, (Sy, Sx, H // Sy, W // Sx), C]

    This allows operations like Max2d to be computed as x.view(B, Sx*Sy, -1, C).max(dim=1).
    Not only is this faster, but it also makes it easy to support inputs of arbitrary
    dimensions in addition to patch-wise sparsity.

    Performing this operation multiple times in sequence puts entire windows as contiguous
    in memory. For instance, if you applied the stride (2, 2) 3 times, entire windows of
    size 8x8 would be contiguous in memory, allowing operations like mask unit attention
    computed easily and efficiently, while also allowing max to be applied sequentially.

    Note: This means that intermediate values of the model are not in HxW order, so they
    need to be re-rolled if you want to use the intermediate values as a HxW feature map.
    The last block of the network is fine though, since by then the strides are all consumed.

    Modified to only fit 2d due to TorchScript constraints.
    """

    def __init__(
        self, input_size: tuple[int, int], patch_stride: tuple[int, int], unroll_schedule: list[tuple[int, int]]
    ) -> None:
        super().__init__()
        self.size = (input_size[0] // patch_stride[0], input_size[1] // patch_stride[1])
        self.schedule = unroll_schedule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: flattened patch embeddings [B, N, C]
        Output: patch embeddings [B, N, C] permuted such that [B, 4, N//4, C].max(1) etc. performs MaxPoolNd
        """

        (B, _, C) = x.shape
        cur_size = self.size
        x = x.view(B, *cur_size, C)

        for strides in self.schedule:
            # Move patches with the given strides to the batch dimension

            # Create a view of the tensor with the patch stride as separate dims
            # For example in 2d: [B, H // Sy, Sy, W // Sx, Sx, C]
            cur_size = (cur_size[0] // strides[0], cur_size[1] // strides[1])
            new_shape = (B, cur_size[0], strides[0], cur_size[1], strides[1], C)
            x = x.view(new_shape)

            # Move the patch stride into the batch dimension
            # For example in 2d: [B, Sy, Sx, H // Sy, W // Sx, C]
            L = len(new_shape)
            permute = [0] + list(range(2, L - 1, 2)) + list(range(1, L - 1, 2)) + [L - 1]
            x = x.permute(permute)

            # Now finally flatten the relevant dims into the batch dimension
            x = x.flatten(0, len(strides))
            B *= strides[0] * strides[1]

        x = x.reshape(-1, self.size[0] * self.size[1], C)

        return x


class Reroll(nn.Module):
    """
    Undo the "unroll" operation so that intermediate features are usable
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        patch_stride: tuple[int, int],
        unroll_schedule: list[tuple[int, int]],
        stage_ends: list[int],
        q_pool: int,
    ):
        super().__init__()
        self.size = (input_size[0] // patch_stride[0], input_size[1] // patch_stride[1])

        # The first stage has to reverse everything
        # The next stage has to reverse all but the first unroll, etc.
        self.schedule = {}
        size = self.size
        for i in range(stage_ends[-1] + 1):
            self.schedule[i] = (unroll_schedule, size)
            # Schedule unchanged if no pooling at a stage end
            if i in stage_ends[:q_pool]:
                if len(unroll_schedule) > 0:
                    size = (size[0] // unroll_schedule[0][0], size[1] // unroll_schedule[0][1])

                unroll_schedule = unroll_schedule[1:]

    def forward(self, x: torch.Tensor, block_idx: int, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Roll the given tensor back up to spatial order assuming it's from the given block.

        If no mask is provided:
            - Returns [B, H, W, C] for 2d, [B, T, H, W, C] for 3d, etc.
        If a mask is provided:
            - Returns [B, #MUs, MUy, MUx, C] for 2d, etc.
        """

        (schedule, size) = self.schedule[block_idx]
        (B, N, C) = x.size()

        D = len(size)
        cur_mu_shape = [1] * D

        for strides in schedule:
            # Extract the current patch from N
            x = x.view(B, *strides, N // math.prod(strides), *cur_mu_shape, C)

            # Move that patch into the current MU
            # Example in 2d: [B, Sy, Sx, N//(Sy*Sx), MUy, MUx, C] -> [B, N//(Sy*Sx), Sy, MUy, Sx, MUx, C]
            L = len(x.shape)
            permute = [0, 1 + D] + sum([list(p) for p in zip(range(1, 1 + D), range(1 + D + 1, L - 1))], []) + [L - 1]
            x = x.permute(permute)

            # Reshape to [B, N//(Sy*Sx), *MU, C]
            for i in range(D):
                cur_mu_shape[i] *= strides[i]

            x = x.reshape(B, -1, *cur_mu_shape, C)
            N = x.shape[1]

        # Current shape (e.g., 2d: [B, #MUy*#MUx, MUy, MUx, C])
        x = x.view(B, N, *cur_mu_shape, C)

        # If masked, return [B, #MUs, MUy, MUx, C]
        if mask is not None:
            return x

        # If not masked, we can return [B, H, W, C]
        x = undo_windowing(x, size, cur_mu_shape)

        return x


class MaskUnitAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int,
        window_size: int,
        use_mask_unit_attn: bool,
    ) -> None:
        super().__init__()
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride
        self.head_dim = dim_out // heads
        self.scale = self.head_dim**-0.5
        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, _) = x.shape
        if self.use_mask_unit_attn is True:
            num_windows = N // (self.q_stride * self.window_size)
        else:
            num_windows = 1

        qkv = self.qkv(x).reshape(B, -1, num_windows, 3, self.heads, self.head_dim).permute(3, 0, 4, 2, 1, 5)
        (q, k, v) = qkv.unbind(0)

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim).amax(dim=3)

        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # pylint: disable=not-callable

        x = x.transpose(1, 3).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        return x


class HieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float,
        drop_path: float,
        q_stride: int,
        window_size: int,
        use_mask_unit_attn: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out

        self.norm1 = nn.LayerNorm(dim)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
        else:
            self.proj = None

        self.attn = MaskUnitAttention(dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn)
        self.norm2 = nn.LayerNorm(dim_out)
        self.mlp = MLP(dim_out, [int(dim_out * mlp_ratio), dim_out], activation_layer=nn.GELU)
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        if self.proj is not None:
            x = self.proj(x_norm)
            x = x.view(x.shape[0], self.attn.q_stride, -1, x.shape[-1]).amax(dim=1)  # Max-pool

        x = x + self.drop_path(self.attn(x_norm))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# pylint: disable=too-many-instance-attributes
class Hiera(DetectorBackbone, PreTrainEncoder, MaskedTokenOmissionMixin):
    scriptable = False
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

        image_size = self.size
        mask_unit_size = (8, 8)
        q_stride = (2, 2)
        q_pool = 3
        patch_kernel = (7, 7)
        patch_stride = (4, 4)
        patch_padding = (3, 3)
        mask_unit_attn = (True, True, False, False)
        dim_mul = 2.0
        head_mul = 2.0
        mlp_ratio = 4.0
        depths: list[int] = self.config["depths"]
        embed_dim: int = self.config["embed_dim"]
        num_heads: int = self.config["num_heads"]
        abs_win_pos_embed: bool = self.config["abs_win_pos_embed"]
        drop_path_rate: float = self.config["drop_path_rate"]

        tokens_spatial_shape = [i // s for i, s in zip(image_size, patch_stride)]
        num_tokens = math.prod(tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)
        assert q_pool < len(depths)
        assert patch_stride[0] == patch_stride[1]

        self.tokens_spatial_shape = tokens_spatial_shape
        self.mask_unit_size = mask_unit_size
        self.patch_stride = patch_stride
        self.q_stride = q_stride
        self.q_pool = q_pool
        self.mu_size = flat_mu_size
        self.mask_spatial_shape = [i // s for i, s in zip(tokens_spatial_shape, mask_unit_size)]
        self.stage_ends = [sum(depths[:i]) - 1 for i in range(1, len(depths) + 1)]
        self.num_special_tokens = 0
        self.num_layers = sum(depths)

        stem_dim = embed_dim
        self.stem = PatchEmbed(self.input_channels, stem_dim, patch_kernel, patch_stride, patch_padding)

        global_pos_size = (image_size[0] // 2**4, image_size[1] // 2**4)
        if abs_win_pos_embed is True:
            # AbsWin variant
            self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *global_pos_size))
            self.pos_embed_win = nn.Parameter(torch.zeros(1, embed_dim, *mask_unit_size))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
            self.pos_embed_win = None

        self.unroll = Unroll(image_size, patch_stride, [q_stride] * len(self.stage_ends[:-1]))
        self.reroll = Reroll(image_size, patch_stride, [q_stride] * len(self.stage_ends[:-1]), self.stage_ends, q_pool)

        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # Stochastic depth decay rule

        cur_stage = 0
        layers = []
        self.block_count = []
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(self.num_layers):
            dim_out = embed_dim

            # Mask unit or global attention.
            # Lag by 1 block, so that global attention, applied post pooling on lower resolution
            use_mask_unit_attn = mask_unit_attn[cur_stage]

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride

            layers.append(
                HieraBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                    window_size=flat_mu_size,
                    use_mask_unit_attn=use_mask_unit_attn,
                )
            )
            embed_dim = dim_out

            if i in self.stage_ends:
                stages[f"stage{cur_stage+1}"] = nn.Sequential(*layers)
                return_channels.append(dim_out)
                self.block_count.append(i)
                layers = []

        assert len(layers) == 0

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embed_dim
        self.classifier = self.create_classifier()

        self.max_stride = patch_stride[0] * 8
        self.stem_stride = patch_stride[0]
        self.stem_width = stem_dim
        self.encoding_size = embed_dim
        # self.decoder_block

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.02)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.02)
                nn.init.ones_(m.weight)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.pos_embed_win is not None:
            nn.init.trunc_normal_(self.pos_embed_win, std=0.02)
        if isinstance(self.classifier, nn.Linear):
            self.classifier.weight.data.mul_(0.001)
            self.classifier.bias.data.mul_(0.001)

    def _get_pos_embed(self) -> torch.Tensor:
        if self.pos_embed_win is not None:
            # Reference interpolation adapted from:
            # https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/backbones/hieradet.py#L273
            pos_embed_win = self.pos_embed_win.tile(self.mask_spatial_shape)
            pos_embed = F.interpolate(
                self.pos_embed,
                size=pos_embed_win.shape[-2:],
                mode="bicubic",
                antialias=True,
            )
            pos_embed = pos_embed + pos_embed_win
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
        else:
            pos_embed = self.pos_embed

        return pos_embed

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = x + self._get_pos_embed()
        x = self.unroll(x)

        out = {}
        for idx, (name, module) in enumerate(self.body.named_children()):
            x = module(x)
            if name in self.return_stages:
                z = self.reroll(x, self.block_count[idx])
                out[name] = z.permute(0, 3, 1, 2).contiguous()

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.stem.parameters():
            param.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def masked_encoding_omission(
        self,
        x: torch.Tensor,
        ids_keep: Optional[torch.Tensor] = None,
        return_all_features: bool = False,
        return_keys: Literal["all", "tokens", "embedding"] = "tokens",
    ) -> TokenOmissionResultType:
        torch._assert(return_all_features is False, "not supported")  # pylint: disable=protected-access
        if ids_keep is not None:
            num_windows = math.prod(self.mask_spatial_shape)

            # Mask already in opposite form
            mask = mask_from_indices(ids_keep, num_windows)
            mask = mask.bool()
            patch_mask = mask.view(x.shape[0], 1, *self.mask_spatial_shape)  # B, C, *mask_spatial_shape
        else:
            patch_mask = None

        x = self.stem(x, patch_mask)
        x = x + self._get_pos_embed()
        x = self.unroll(x)

        # Discard masked tokens
        if ids_keep is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(x.shape[0], -1, x.shape[-1])

        x = self.body(x)

        result: TokenOmissionResultType = {}
        if return_keys in ("all", "tokens"):
            result["tokens"] = x

        if return_keys in ("all", "embedding"):
            x = x.mean(dim=1)
            x = self.features(x)
            result["embedding"] = x

        return result

    def masked_encoding(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        # Binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = 1 - mask

        # Un-shuffle to get the binary mask
        mask = mask.bool()
        patch_mask = mask.view(x.shape[0], 1, *self.mask_spatial_shape)  # B, C, *mask_spatial_shape

        x = self.stem(x, patch_mask)
        x = x + self._get_pos_embed()
        x = self.unroll(x)

        # Discard masked tokens
        x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(x.shape[0], -1, x.shape[-1])

        out = []
        for idx, (name, module) in enumerate(self.body.named_children()):
            x = module(x)
            if name in self.return_stages:
                out.append(self.reroll(x, self.block_count[idx], mask=mask))

        return (out, mask)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = x + self._get_pos_embed()
        x = self.unroll(x)
        x = self.body(x)

        return x

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.features(x)

        return x

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        old_size = self.size
        super().adjust_size(new_size)

        if self.pos_embed_win is not None:
            global_pos_size = (new_size[0] // 2**4, new_size[1] // 2**4)
            pos_embed = F.interpolate(
                self.pos_embed,
                size=global_pos_size,
                mode="bicubic",
                antialias=True,
            )
            self.pos_embed = nn.Parameter(pos_embed)

        else:
            self.pos_embed = nn.Parameter(
                adjust_position_embedding(
                    self.pos_embed,
                    (old_size[0] // self.patch_stride[0], old_size[1] // self.patch_stride[1]),
                    (new_size[0] // self.patch_stride[0], new_size[1] // self.patch_stride[1]),
                    0,
                )
            )

        # Re-init vars
        self.tokens_spatial_shape = [i // s for i, s in zip(new_size, self.patch_stride)]
        self.mask_spatial_shape = [i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)]

        # Re-init rolls
        self.unroll = Unroll(new_size, self.patch_stride, [self.q_stride] * len(self.stage_ends[:-1]))
        self.reroll = Reroll(
            new_size,
            self.patch_stride,
            [self.q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            self.q_pool,
        )


registry.register_model_config(
    "hiera_tiny",
    Hiera,
    config={
        "depths": [1, 2, 7, 2],
        "embed_dim": 96,
        "num_heads": 1,
        "abs_win_pos_embed": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "hiera_small",
    Hiera,
    config={
        "depths": [1, 2, 11, 2],
        "embed_dim": 96,
        "num_heads": 1,
        "abs_win_pos_embed": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "hiera_base",
    Hiera,
    config={
        "depths": [2, 3, 16, 3],
        "embed_dim": 96,
        "num_heads": 1,
        "abs_win_pos_embed": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "hiera_base_plus",
    Hiera,
    config={
        "depths": [2, 3, 16, 3],
        "embed_dim": 112,
        "num_heads": 2,
        "abs_win_pos_embed": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "hiera_large",
    Hiera,
    config={
        "depths": [2, 6, 36, 4],
        "embed_dim": 144,
        "num_heads": 2,
        "abs_win_pos_embed": False,
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "hiera_huge",
    Hiera,
    config={
        "depths": [2, 6, 36, 4],
        "embed_dim": 256,
        "num_heads": 4,
        "abs_win_pos_embed": False,
        "drop_path_rate": 0.3,
    },
)

# AbsWin variant
registry.register_model_config(
    "hiera_abswin_tiny",
    Hiera,
    config={
        "depths": [1, 2, 7, 2],
        "embed_dim": 96,
        "num_heads": 1,
        "abs_win_pos_embed": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "hiera_abswin_small",
    Hiera,
    config={
        "depths": [1, 2, 11, 2],
        "embed_dim": 96,
        "num_heads": 1,
        "abs_win_pos_embed": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "hiera_abswin_base",
    Hiera,
    config={
        "depths": [2, 3, 16, 3],
        "embed_dim": 96,
        "num_heads": 1,
        "abs_win_pos_embed": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "hiera_abswin_base_plus",
    Hiera,
    config={
        "depths": [2, 3, 16, 3],
        "embed_dim": 112,
        "num_heads": 2,
        "abs_win_pos_embed": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "hiera_abswin_large",
    Hiera,
    config={
        "depths": [2, 6, 36, 4],
        "embed_dim": 144,
        "num_heads": 2,
        "abs_win_pos_embed": True,
        "drop_path_rate": 0.2,
    },
)
registry.register_model_config(
    "hiera_abswin_huge",
    Hiera,
    config={
        "depths": [2, 6, 36, 4],
        "embed_dim": 256,
        "num_heads": 4,
        "abs_win_pos_embed": True,
        "drop_path_rate": 0.3,
    },
)

registry.register_weights(
    "hiera_abswin_base_mim",
    {
        "url": "https://huggingface.co/birder-project/hiera_abswin_base_mim/resolve/main",
        "description": (
            "Hiera base with abswin image encoder pre-trained using Masked Image Modeling (MIM) for 400 epochs. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 192.7,
                "sha256": "d8ce30c2d9d46af8bdfe7c591610d12c4e592ad4bc4b00e2ba41457e683702da",
            },
            "safetensors": {
                "file_size": 192.6,
                "sha256": "b03f9ec88f7b391fe6b2f5ab7707c156351ab4151c95163f631db0712c48b843",
            },
        },
        "net": {"network": "hiera_abswin_base", "tag": "mim"},
    },
)
registry.register_weights(
    "hiera_abswin_base_mim-intermediate-eu-common",
    {
        "url": "https://huggingface.co/birder-project/hiera_abswin_base_mim-intermediate-eu-common/resolve/main",
        "description": (
            "Hiera base with abswin model with MIM pretraining and intermediate training, "
            "then fine-tuned on the eu-common dataset"
        ),
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 194.9,
                "sha256": "e6e9d28e0cd98a5c72c1a4a67301c62d2ade97e6d162f34ad630e5ffb2f9ec2a",
            }
        },
        "net": {"network": "hiera_abswin_base", "tag": "mim-intermediate-eu-common"},
    },
)
