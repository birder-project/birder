"""
ViT SAM, adapted from
https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py

Paper "Exploring Plain Vision Transformer Backbones for Object Detection",
https://arxiv.org/abs/2203.16527

and as used as an image encoder at the paper "Segment Anything", https://arxiv.org/abs/2304.02643
"""

# Reference license: Apache-2.0

from functools import partial
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.convnext_v1 import LayerNorm2d
from birder.net.vit import EncoderBlock as MAEDecoderBlock


# pylint: disable=invalid-name
def window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    (B, H, W, C) = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp = H + pad_h
    Wp = W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return (windows, (Hp, Wp))


# pylint: disable=invalid-name
def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]
) -> torch.Tensor:
    (Hp, Wp) = pad_hw
    (H, W) = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()

    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)

    # Interpolate rel pos if needed
    if rel_pos.shape[0] != max_rel_dist:
        # Adjust size is a one off interpolation, should prevent us from getting here
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode="linear"
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def get_decomposed_rel_pos_bias(
    q: torch.Tensor, rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor, q_size: tuple[int, int], k_size: tuple[int, int]
) -> torch.Tensor:
    (q_h, q_w) = q_size
    (k_h, k_w) = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    (B, _, dim) = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn_bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    attn_bias = attn_bias.reshape(-1, q_h * q_w, k_h * k_w)

    return attn_bias


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, kernel_size: tuple[int, int], stride: tuple[int, int]) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)

        # B C H W -> B H W C
        return x.permute(0, 2, 3, 1)


class Attention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, qkv_bias: bool, use_rel_pos: bool, input_size: Optional[tuple[int, int]] = None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos is True:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, H, W, _) = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        if self.use_rel_pos is True:
            attn_bias = get_decomposed_rel_pos_bias(q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        else:
            attn_bias = None

        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, attn_mask=attn_bias, scale=self.scale
        )

        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_path: float,
        use_rel_pos: bool,
        window_size: int,
        input_size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.window_size = window_size

        # Attention block
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.drop_path1 = StochasticDepth(drop_path, mode="row")

        # MLP block
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU)
        self.drop_path2 = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (_, H, W, _) = x.shape
        shortcut = x

        x = self.norm1(x)
        pad_hw = (0, 0)
        if self.window_size > 0:
            (x, pad_hw) = window_partition(x, self.window_size)

        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = self.drop_path1(x) + shortcut
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


# pylint: disable=invalid-name
class ViT_SAM(DetectorBackbone):
    block_group_regex = r"body\.(\d+)"

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
        use_rel_pos = True
        out_channels = 256
        patch_size: int = self.config["patch_size"]
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_ratio: float = self.config["mlp_ratio"]
        window_size: int = self.config["window_size"]
        global_attn_indexes: list[int] = self.config["global_attn_indexes"]
        drop_path_rate: float = self.config["drop_path_rate"]

        torch._assert(image_size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(image_size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.global_attn_indexes = global_attn_indexes
        self.num_special_tokens = 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # Stochastic depth decay rule

        self.patch_embed = PatchEmbed(
            in_channels=self.input_channels,
            embed_dim=hidden_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

        # Absolute position embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, image_size[0] // patch_size, image_size[1] // patch_size, hidden_dim)
        )

        layers = []
        for i in range(num_layers):
            layers.append(
                EncoderBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    use_rel_pos=use_rel_pos,
                    window_size=window_size if i not in global_attn_indexes else 0,
                    input_size=(image_size[0] // patch_size, image_size[1] // patch_size),
                )
            )

        self.body = nn.Sequential(*layers)
        self.neck = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            LayerNorm2d(out_channels),
        )
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )

        self.return_stages = ["neck"]  # Actually meaningless, but for completeness
        self.return_channels = [out_channels]
        self.embedding_size = out_channels
        self.classifier = self.create_classifier()

        self.encoding_size = hidden_dim * (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.decoder_block = partial(
            MAEDecoderBlock,
            16,
            mlp_dim=None,
            dropout=0,
            attention_dropout=0,
            drop_path=0,
            activation_layer=nn.GELU,
        )

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_embedding

        x = self.body(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return {self.return_stages[0]: x}

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        self.pos_embedding.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embedding

        x = self.body(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        x = self.features(x)

        return x

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)

        new_base_size_h = new_size[0] // self.patch_size
        new_base_size_w = new_size[1] // self.patch_size
        for idx, m in enumerate(self.body):
            if idx not in self.global_attn_indexes:
                continue

            if m.attn.use_rel_pos is True:
                rel_pos_h = m.attn.rel_pos_h
                rel_pos_w = m.attn.rel_pos_w

                orig_dtype = rel_pos_h.dtype
                rel_pos_h = rel_pos_h.float()
                rel_pos_w = rel_pos_w.float()

                rel_pos_h = rel_pos_h.permute(1, 0)
                rel_pos_w = rel_pos_w.permute(1, 0)
                rel_pos_h = rel_pos_h.unsqueeze(0)
                rel_pos_w = rel_pos_w.unsqueeze(0)

                rel_pos_h = F.interpolate(rel_pos_h, size=2 * new_base_size_h - 1, mode="linear")
                rel_pos_w = F.interpolate(rel_pos_w, size=2 * new_base_size_w - 1, mode="linear")
                rel_pos_h = rel_pos_h.squeeze(0)
                rel_pos_w = rel_pos_w.squeeze(0)
                rel_pos_h = rel_pos_h.permute(1, 0)
                rel_pos_w = rel_pos_w.permute(1, 0)
                rel_pos_h = rel_pos_h.to(orig_dtype)
                rel_pos_w = rel_pos_w.to(orig_dtype)

                m.attn.rel_pos_h = nn.Parameter(rel_pos_h)
                m.attn.rel_pos_w = nn.Parameter(rel_pos_w)

        orig_dtype = self.pos_embedding.dtype
        pos_embedding = self.pos_embedding.float()
        pos_embedding = pos_embedding.permute(0, 3, 1, 2)
        pos_embedding = F.interpolate(
            pos_embedding, size=(new_base_size_h, new_base_size_w), mode="bicubic", antialias=True
        )
        pos_embedding = pos_embedding.permute(0, 2, 3, 1)
        pos_embedding = pos_embedding.to(orig_dtype)

        self.pos_embedding = nn.Parameter(pos_embedding)

    def load_vit_weights(self, state_dict: dict[str, Any]) -> None:
        """
        As suggested at "Exploring Plain Vision Transformer Backbones for Object Detection"
        (https://arxiv.org/abs/2203.16527), weights can be transfer from vanilla ViT for faster start-up.
        The relative position embedding and "neck" convolutions are left intact.
        """

        # NOTE: SoVit variant not supported

        # Remove all special token
        num_special_tokens = 0
        if "class_token" in state_dict:
            num_special_tokens += 1
            del state_dict["class_token"]

        if "reg_tokens" in state_dict:
            num_special_tokens += state_dict["reg_tokens"].size(1)
            del state_dict["reg_tokens"]

        # Remove final norm
        del state_dict["norm.weight"]
        del state_dict["norm.bias"]

        # Remove classifier weights
        if "classifier.weight" in state_dict:
            del state_dict["classifier.weight"]
            del state_dict["classifier.bias"]

        # Adjust pos_embedding
        if state_dict["pos_embedding"].ndim == 2:
            state_dict["pos_embedding"] = state_dict["pos_embedding"][num_special_tokens:, :]
        else:
            state_dict["pos_embedding"] = state_dict["pos_embedding"][:, num_special_tokens:, :]

        state_dict["pos_embedding"] = state_dict["pos_embedding"].reshape(
            1, self.size[0] // self.patch_size, self.size[1] // self.patch_size, -1
        )

        # Modify encoder weight names
        modified_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("conv_proj.") is True:
                k = k.replace("conv_proj.", "patch_embed.proj.")
            elif k.startswith("encoder.block.") is True:
                k = k.replace("encoder.block.", "body.")
                k = k.replace("ln1", "norm1")
                k = k.replace("ln2", "norm2")
                if "self_attention.in_proj" in k:
                    k = k.replace("self_attention.in_proj_", "attn.qkv.")
                elif "self_attention.out_proj" in k:
                    k = k.replace("self_attention.out_proj.", "attn.proj.")

            modified_state_dict[k] = v

        # Load the modified state dict
        incompatible_keys = self.load_state_dict(modified_state_dict, strict=False)
        assert len(incompatible_keys.unexpected_keys) == 0


registry.register_alias(
    "vit_sam_b16",
    ViT_SAM,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_ratio": 4.0,
        "window_size": 14,
        "global_attn_indexes": [2, 5, 8, 11],
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "vit_sam_l16",
    ViT_SAM,
    config={
        "patch_size": 16,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_ratio": 4.0,
        "window_size": 14,
        "global_attn_indexes": [5, 11, 17, 23],
        "drop_path_rate": 0.4,
    },
)
registry.register_alias(
    "vit_sam_h16",
    ViT_SAM,
    config={
        "patch_size": 16,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_ratio": 4.0,
        "window_size": 14,
        "global_attn_indexes": [7, 15, 23, 31],
        "drop_path_rate": 0.5,
    },
)
