import math
from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from birder.layers import FFN
from birder.layers import LayerScale
from birder.model_registry import registry
from birder.net._vit_configs import BASE
from birder.net._vit_configs import LARGE
from birder.net._vit_configs import SMALL
from birder.net.base import DetectorBackbone
from birder.net.base import normalize_out_indices
from birder.net.vit import PatchEmbed
from birder.net.vit import adjust_position_embedding


def window_partition(x: torch.Tensor, window_size: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_w = (window_size[1] - W % window_size[1]) % window_size[1]
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp = H + pad_h
    Wp = W + pad_w
    x = x.view(B, Hp // window_size[0], window_size[0], Wp // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)

    return (windows, (Hp, Wp))


def window_unpartition(
    windows: torch.Tensor, window_size: tuple[int, int], pad_hw: tuple[int, int], hw: tuple[int, int]
) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.size(0) // (Hp * Wp // window_size[0] // window_size[1])
    x = windows.view(B, Hp // window_size[0], Wp // window_size[1], window_size[0], window_size[1], windows.size(-1))
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, windows.size(-1))

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()

    return x


def windowed_to_global(
    x: torch.Tensor,
    num_special_tokens: int,
    window_size: tuple[int, int],
    window_grid: tuple[int, int],
    grid_size: tuple[int, int],
) -> torch.Tensor:
    num_windows = window_grid[0] * window_grid[1]
    B = x.size(0) // num_windows
    if num_special_tokens > 0:
        special_tokens = x[:, :num_special_tokens, :].view(B, num_windows, num_special_tokens, x.size(-1)).mean(dim=1)
        patch_tokens = x[:, num_special_tokens:, :]
    else:
        special_tokens = None
        patch_tokens = x

    pad_hw = (window_grid[0] * window_size[0], window_grid[1] * window_size[1])
    patch_tokens = patch_tokens.view(patch_tokens.size(0), window_size[0], window_size[1], patch_tokens.size(-1))
    patch_tokens = window_unpartition(patch_tokens, window_size, pad_hw, grid_size)
    B, H, W, C = patch_tokens.shape
    patch_tokens = patch_tokens.view(B, H * W, C)

    if special_tokens is None:
        return patch_tokens

    return torch.concat([special_tokens, patch_tokens], dim=1)


def global_to_windowed(
    x: torch.Tensor,
    num_special_tokens: int,
    window_size: tuple[int, int],
    window_grid: tuple[int, int],
    grid_size: tuple[int, int],
) -> torch.Tensor:
    if num_special_tokens > 0:
        special_tokens = x[:, :num_special_tokens, :]
        patch_tokens = x[:, num_special_tokens:, :]
    else:
        special_tokens = None
        patch_tokens = x

    B, _, C = patch_tokens.shape
    grid_h, grid_w = grid_size
    patch_tokens = patch_tokens.view(B, grid_h, grid_w, C)
    patch_tokens, _ = window_partition(patch_tokens, window_size)
    patch_tokens = patch_tokens.view(patch_tokens.size(0), window_size[0] * window_size[1], patch_tokens.size(-1))

    if special_tokens is None:
        return patch_tokens

    num_windows = window_grid[0] * window_grid[1]
    special_tokens = special_tokens.unsqueeze(1).expand(-1, num_windows, -1, -1)
    special_tokens = special_tokens.reshape(B * num_windows, num_special_tokens, C)
    return torch.concat([special_tokens, patch_tokens], dim=1)


def make_window_attention_mask(
    batch_size: int,
    num_special_tokens: int,
    window_size: tuple[int, int],
    window_grid: tuple[int, int],
    grid_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if grid_size[0] % window_size[0] == 0 and grid_size[1] % window_size[1] == 0:
        return None

    patch_mask = torch.ones((1, grid_size[0], grid_size[1], 1), device=device)
    patch_mask, _ = window_partition(patch_mask, window_size)
    patch_mask = patch_mask.view(1, window_grid[0] * window_grid[1], window_size[0] * window_size[1]) > 0
    patch_mask = patch_mask.expand(batch_size, -1, -1).reshape(batch_size * window_grid[0] * window_grid[1], -1)

    if num_special_tokens == 0:
        empty_windows = ~patch_mask.any(dim=1, keepdim=True)
        patch_mask = patch_mask | empty_windows
    else:
        special_mask = torch.ones((patch_mask.size(0), num_special_tokens), device=device, dtype=torch.bool)
        patch_mask = torch.concat([special_mask, patch_mask], dim=1)

    attn_mask = torch.zeros((patch_mask.size(0), 1, 1, patch_mask.size(1)), device=device, dtype=dtype)
    attn_mask.masked_fill_(~patch_mask[:, None, None, :], float("-inf"))

    return attn_mask


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float, proj_drop: float) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return self.proj_drop(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: Optional[int],
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        global_attn: bool,
        num_special_tokens: int,
        activation_layer: Callable[..., nn.Module] = nn.GELU,
        layer_scale_init_value: Optional[float] = None,
        norm_layer_eps: float = 1e-6,
        mlp_layer: Callable[..., nn.Module] = FFN,
    ) -> None:
        super().__init__()
        if mlp_dim is None:
            mlp_dim = hidden_dim * 4

        self.global_attn = global_attn
        self.num_special_tokens = num_special_tokens

        self.norm1 = nn.LayerNorm(hidden_dim, eps=norm_layer_eps)
        self.attn = Attention(hidden_dim, num_heads=num_heads, attn_drop=attention_dropout, proj_drop=0.0)
        self.drop_path = StochasticDepth(drop_path, mode="row")
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()

        self.norm2 = nn.LayerNorm(hidden_dim, eps=norm_layer_eps)
        self.mlp = mlp_layer(hidden_dim, mlp_dim, act_layer=activation_layer, dropout=dropout)
        if layer_scale_init_value is not None:
            self.layer_scale_2 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_2 = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        window_size: tuple[int, int],
        window_grid: tuple[int, int],
        grid_size: tuple[int, int],
        local_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        if self.global_attn is True:
            x = windowed_to_global(x, self.num_special_tokens, window_size, window_grid, grid_size)
            x = self.attn(x)
            x = global_to_windowed(x, self.num_special_tokens, window_size, window_grid, grid_size)
        else:
            x = self.attn(x, attn_mask=local_attn_mask)

        x = shortcut + self.drop_path(self.layer_scale_1(x))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        dpr: list[float],
        global_attn_indexes: list[int],
        num_special_tokens: int,
        activation_layer: Callable[..., nn.Module] = nn.GELU,
        layer_scale_init_value: Optional[float] = None,
        norm_layer_eps: float = 1e-6,
        mlp_layer: Callable[..., nn.Module] = FFN,
    ) -> None:
        super().__init__()
        pre_layers = []
        if dropout > 0.0:
            pre_layers.append(nn.Dropout(dropout))

        self.pre_block = nn.Sequential(*pre_layers)
        global_attn_index_set = set(global_attn_indexes)

        layers = []
        for i in range(num_layers):
            layers.append(
                EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    dpr[i],
                    global_attn=i in global_attn_index_set,
                    num_special_tokens=num_special_tokens,
                    activation_layer=activation_layer,
                    layer_scale_init_value=layer_scale_init_value,
                    norm_layer_eps=norm_layer_eps,
                    mlp_layer=mlp_layer,
                )
            )

        self.block = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        window_size: tuple[int, int],
        window_grid: tuple[int, int],
        grid_size: tuple[int, int],
        local_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pre_block(x)
        for blk in self.block:
            x = blk(x, window_size, window_grid, grid_size, local_attn_mask=local_attn_mask)

        return x

    def forward_features(
        self,
        x: torch.Tensor,
        window_size: tuple[int, int],
        window_grid: tuple[int, int],
        grid_size: tuple[int, int],
        local_attn_mask: Optional[torch.Tensor] = None,
        out_indices: Optional[list[int]] = None,
    ) -> list[torch.Tensor]:
        x = self.pre_block(x)

        out_indices_set = set(out_indices) if out_indices is not None else None
        xs = []
        for idx, blk in enumerate(self.block):
            x = blk(x, window_size, window_grid, grid_size, local_attn_mask=local_attn_mask)
            if out_indices_set is None or idx in out_indices_set:
                xs.append(x)

        return xs


class ViT_Windowed(DetectorBackbone):
    block_group_regex = r"encoder\.block\.(\d+)"

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, config=config, size=size)
        assert self.config is not None, "must set config"

        image_size = self.size
        attention_dropout = 0.0
        dropout = 0.0
        pos_embed_special_tokens: bool = self.config.get("pos_embed_special_tokens", True)
        patch_size: int = self.config["patch_size"]
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_dim: int = self.config["mlp_dim"]
        window_size: Optional[tuple[int, int]] = self.config.get("window_size", None)
        num_windows: Optional[tuple[int, int]] = self.config.get("num_windows", None)
        global_attn_indexes: list[int] = self.config["global_attn_indexes"]
        layer_scale_init_value: Optional[float] = self.config.get("layer_scale_init_value", None)
        norm_layer_eps: float = self.config.get("norm_layer_eps", 1e-6)
        num_reg_tokens: int = self.config.get("num_reg_tokens", 0)
        class_token: bool = self.config.get("class_token", True)
        mask_padded_attn: bool = self.config.get("mask_padded_attn", True)
        out_indices: Optional[list[int]] = self.config.get("out_indices", None)
        drop_path_rate: float = self.config["drop_path_rate"]

        torch._assert(image_size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(image_size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")
        self.pos_embed_special_tokens = pos_embed_special_tokens
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.num_windows = num_windows
        self.global_attn_indexes = global_attn_indexes
        self.num_reg_tokens = num_reg_tokens
        self.mask_padded_attn = mask_padded_attn
        self.out_indices = normalize_out_indices(out_indices, num_layers)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.conv_proj = nn.Conv2d(
            self.input_channels,
            hidden_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
        )
        self.patch_embed = PatchEmbed()

        seq_length = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.num_special_tokens = 0

        if class_token is True:
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.num_special_tokens += 1
            if pos_embed_special_tokens is True:
                seq_length += 1
        else:
            self.class_token = None

        if self.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, self.num_reg_tokens, hidden_dim))
            self.num_special_tokens += self.num_reg_tokens
            if pos_embed_special_tokens is True:
                seq_length += self.num_reg_tokens
        else:
            self.reg_tokens = None

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))

        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            dpr,
            global_attn_indexes=global_attn_indexes,
            num_special_tokens=self.num_special_tokens,
            activation_layer=nn.GELU,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer_eps=norm_layer_eps,
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=norm_layer_eps)

        num_return_stages = len(self.out_indices) if self.out_indices is not None else 1
        self.return_stages = [f"stage{stage_idx + 1}" for stage_idx in range(num_return_stages)]
        self.return_channels = [hidden_dim] * num_return_stages
        self.embedding_size = hidden_dim
        self.classifier = self.create_classifier()

        self.max_stride = patch_size
        self.stem_stride = patch_size
        self.stem_width = hidden_dim

        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

        if isinstance(self.classifier, nn.Linear):
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    def _get_window_size(self, grid_h: int, grid_w: int) -> tuple[int, int]:
        if self.num_windows is None:
            return self.window_size  # type: ignore[return-value]

        return (
            max(1, math.ceil(grid_h / self.num_windows[0])),
            max(1, math.ceil(grid_w / self.num_windows[1])),
        )

    def _get_pos_embed(self, grid_h: int, grid_w: int) -> torch.Tensor:
        if self.dynamic_size is False:
            return self.pos_embedding

        orig_grid_h = self.size[0] // self.patch_size
        orig_grid_w = self.size[1] // self.patch_size
        if grid_h == orig_grid_h and grid_w == orig_grid_w:
            return self.pos_embedding

        return adjust_position_embedding(
            self.pos_embedding,
            (orig_grid_h, orig_grid_w),
            (grid_h, grid_w),
            self.num_special_tokens if self.pos_embed_special_tokens is True else 0,
            antialias=False,
        )

    def _get_special_tokens(
        self, batch_size: int, num_windows: int, pos_embedding: torch.Tensor
    ) -> Optional[torch.Tensor]:
        special_tokens = []
        if self.reg_tokens is not None:
            reg_tokens = self.reg_tokens
            if self.pos_embed_special_tokens is True:
                reg_tokens = reg_tokens + pos_embedding[:, : self.num_reg_tokens, :]

            special_tokens.append(reg_tokens.expand(batch_size, -1, -1))

        if self.class_token is not None:
            cls_token = self.class_token
            if self.pos_embed_special_tokens is True:
                cls_token = cls_token + pos_embedding[:, self.num_reg_tokens : self.num_reg_tokens + 1, :]

            special_tokens.append(cls_token.expand(batch_size, -1, -1))

        if len(special_tokens) == 0:
            return None

        special_tokens = torch.concat(special_tokens, dim=1)
        special_tokens = special_tokens.unsqueeze(1).expand(-1, num_windows, -1, -1)
        return special_tokens.reshape(batch_size * num_windows, self.num_special_tokens, self.hidden_dim)

    def _prepare_windowed_tokens(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int], tuple[int, int], Optional[torch.Tensor]]:
        batch_size = x.size(0)
        x = self.conv_proj(x)
        grid_h, grid_w = x.shape[-2:]
        pos_embedding = self._get_pos_embed(grid_h, grid_w)
        window_size = self._get_window_size(grid_h, grid_w)

        x = self.patch_embed(x)
        if self.pos_embed_special_tokens is True:
            x = x + pos_embedding[:, self.num_special_tokens :, :]
        else:
            x = x + pos_embedding

        x = x.view(x.size(0), grid_h, grid_w, self.hidden_dim)
        x, pad_hw = window_partition(x, window_size)
        window_grid = (pad_hw[0] // window_size[0], pad_hw[1] // window_size[1])
        num_windows = window_grid[0] * window_grid[1]
        x = x.view(x.size(0), window_size[0] * window_size[1], x.size(-1))

        special_tokens = self._get_special_tokens(batch_size, num_windows, pos_embedding)
        if special_tokens is not None:
            x = torch.concat([special_tokens, x], dim=1)

        if self.mask_padded_attn is True:
            local_attn_mask = make_window_attention_mask(
                batch_size, self.num_special_tokens, window_size, window_grid, (grid_h, grid_w), x.device, x.dtype
            )
        else:
            local_attn_mask = None

        return (x, window_size, window_grid, (grid_h, grid_w), local_attn_mask)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x, window_size, window_grid, grid_size, local_attn_mask = self._prepare_windowed_tokens(x)

        if self.out_indices is None:
            xs = [self.encoder(x, window_size, window_grid, grid_size, local_attn_mask=local_attn_mask)]
        else:
            xs = self.encoder.forward_features(
                x, window_size, window_grid, grid_size, local_attn_mask=local_attn_mask, out_indices=self.out_indices
            )

        out: dict[str, torch.Tensor] = {}
        for stage_name, stage_x in zip(self.return_stages, xs, strict=True):
            if self.num_special_tokens > 0:
                stage_x = stage_x[:, self.num_special_tokens :, :]

            pad_hw = (window_grid[0] * window_size[0], window_grid[1] * window_size[1])
            stage_x = stage_x.view(stage_x.size(0), window_size[0], window_size[1], stage_x.size(-1))
            out[stage_name] = window_unpartition(stage_x, window_size, pad_hw, grid_size).permute(0, 3, 1, 2)

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.conv_proj.parameters():
            param.requires_grad_(False)

        self.pos_embedding.requires_grad_(False)

        for idx, module in enumerate(self.encoder.block.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad_(False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x, window_size, window_grid, grid_size, local_attn_mask = self._prepare_windowed_tokens(x)
        x = self.encoder(x, window_size, window_grid, grid_size, local_attn_mask=local_attn_mask)
        x = windowed_to_global(x, self.num_special_tokens, window_size, window_grid, grid_size)

        return self.norm(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if self.class_token is None:
            return x[:, self.num_special_tokens :].mean(dim=1)

        return x[:, self.num_reg_tokens]

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        assert new_size[0] % self.patch_size == 0, "Input shape indivisible by patch size!"
        assert new_size[1] % self.patch_size == 0, "Input shape indivisible by patch size!"

        old_size = self.size
        super().adjust_size(new_size)

        with torch.no_grad():
            pos_embedding = adjust_position_embedding(
                self.pos_embedding,
                (old_size[0] // self.patch_size, old_size[1] // self.patch_size),
                (new_size[0] // self.patch_size, new_size[1] // self.patch_size),
                self.num_special_tokens if self.pos_embed_special_tokens is True else 0,
            )

        self.pos_embedding = nn.Parameter(pos_embedding)


registry.register_model_config(
    "vit_windowed_s16",
    ViT_Windowed,
    config={"patch_size": 16, **SMALL, "num_windows": (2, 2), "global_attn_indexes": [5, 11]},
)
registry.register_model_config(
    "vit_windowed_b16",
    ViT_Windowed,
    config={"patch_size": 16, **BASE, "num_windows": (2, 2), "global_attn_indexes": [5, 11]},
)
registry.register_model_config(
    "vit_windowed_l16",
    ViT_Windowed,
    config={"patch_size": 16, **LARGE, "num_windows": (2, 2), "global_attn_indexes": [5, 11, 17, 23]},
)

# With registers
####################

registry.register_model_config(
    "vit_windowed_reg4_s14_nps_ls_avg",
    ViT_Windowed,
    config={
        "pos_embed_special_tokens": False,
        "patch_size": 14,
        **SMALL,
        "class_token": False,
        "num_windows": (2, 2),
        "global_attn_indexes": [5, 11],
        "layer_scale_init_value": 1e-5,
        "num_reg_tokens": 4,
    },
)
