"""
ViT, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

Paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
https://arxiv.org/abs/2010.11929
and
Paper "Vision Transformers Need Registers", https://arxiv.org/abs/2309.16588
and
Paper "Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design", https://arxiv.org/abs/2305.13035
and
Paper "Scaling Vision Transformers", https://arxiv.org/abs/2106.04560
"""

# Reference license: BSD 3-Clause and Apache-2.0

import math
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.layers import FFN
from birder.layers import LayerScale
from birder.layers import SwiGLU_FFN
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenOmissionResultType
from birder.net.base import TokenRetentionResultType


def adjust_position_embedding(
    pos_embedding: torch.Tensor,
    old_base_size: tuple[int, int],
    new_base_size: tuple[int, int],
    num_prefix_tokens: int,
    antialias: bool = True,
) -> torch.Tensor:
    """
    Adapted from
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py
    """

    pos_embedding_prefix = pos_embedding[:, :num_prefix_tokens]
    pos_embedding = pos_embedding[:, num_prefix_tokens:]

    # Interpolation
    embed_dim = pos_embedding.shape[-1]
    orig_dtype = pos_embedding.dtype
    pos_embedding = pos_embedding.float()  # Interpolate needs float32
    pos_embedding = pos_embedding.reshape(1, old_base_size[0], old_base_size[1], -1).permute(0, 3, 1, 2)
    pos_embedding = F.interpolate(pos_embedding, size=new_base_size, mode="bicubic", antialias=antialias)
    pos_embedding = pos_embedding.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    pos_embedding = pos_embedding.to(orig_dtype)

    # Add back special tokens
    return torch.concat([pos_embedding_prefix, pos_embedding], dim=1)


class MultiHeadAttentionPool(nn.Module):
    """
    Adapted from:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention_pool.py#L12
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        qkv_bias: bool,
        latent_len: int = 1,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, dim))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.mlp = FFN(dim, mlp_dim, act_layer=nn.GELU)

        # Weight initialization
        nn.init.trunc_normal_(self.latent, std=dim**-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.size()

        q_latent = self.latent.expand(B, self.latent_len, -1)
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        (k, v) = kv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # pylint: disable=not-callable
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = x + self.mlp(self.norm(x))

        return x


class PatchEmbed(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The entire forward is equivalent to x.flatten(2).transpose(1, 2)
        """

        (n, hidden_dim, h, w) = x.size()

        # (n, hidden_dim, h, w) -> (n, hidden_dim, (h * w))
        x = x.reshape(n, hidden_dim, h * w)

        # (n, hidden_dim, (h * w)) -> (n, (h * w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: Optional[int],
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        activation_layer: Callable[..., nn.Module],
        layer_scale_init_value: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        mlp_layer: Callable[..., nn.Module] = FFN,
    ) -> None:
        super().__init__()
        self.need_attn = False

        if mlp_dim is None:
            mlp_dim = hidden_dim * 4

        # Attention block
        self.ln1 = norm_layer(hidden_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.drop_path1 = StochasticDepth(drop_path, mode="row")
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()

        # MLP block
        self.ln2 = norm_layer(hidden_dim, eps=1e-6)
        self.mlp = mlp_layer(hidden_dim, mlp_dim, act_layer=activation_layer, dropout=dropout)
        self.drop_path2 = StochasticDepth(drop_path, mode="row")
        if layer_scale_init_value is not None:
            self.layer_scale_2 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.size()}")
        branch1 = self.ln1(x)
        (branch1, _) = self.self_attention(
            branch1, branch1, branch1, need_weights=self.need_attn, average_attn_weights=False
        )
        branch1 = self.layer_scale_1(branch1)
        branch1 = self.drop_path1(branch1) + x

        branch2 = self.ln2(branch1)
        branch2 = self.mlp(branch2)
        branch2 = self.layer_scale_2(branch2)

        x = self.drop_path2(branch2) + branch1

        return x

    def set_need_attn(self) -> None:
        self.need_attn = True


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
        activation_layer: Callable[..., nn.Module] = nn.GELU,
        layer_scale_init_value: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        mlp_layer: Callable[..., nn.Module] = FFN,
    ) -> None:
        super().__init__()
        layers = []
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        for i in range(num_layers):
            layers.append(
                EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    dpr[i],
                    activation_layer=activation_layer,
                    layer_scale_init_value=layer_scale_init_value,
                    norm_layer=norm_layer,
                    mlp_layer=mlp_layer,
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.size()}")
        x = self.block(x)

        return x

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        xs = []
        for blk in self.block:
            x = blk(x)
            xs.append(x)

        return xs

    def set_need_attn(self) -> None:
        for b in self.block:
            b.set_need_attn()


# pylint: disable=too-many-instance-attributes
class ViT(DetectorBackbone, PreTrainEncoder, MaskedTokenOmissionMixin, MaskedTokenRetentionMixin):
    block_group_regex = r"encoder\.block\.(\d+)"

    # pylint: disable=too-many-locals,too-many-branches
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
        attention_dropout = 0.0
        dropout = 0.0
        pos_embed_special_tokens: bool = self.config.get("pos_embed_special_tokens", True)
        patch_size: int = self.config["patch_size"]
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_dim: int = self.config["mlp_dim"]
        layer_scale_init_value: Optional[float] = self.config.get("layer_scale_init_value", None)
        num_reg_tokens: int = self.config.get("num_reg_tokens", 0)
        class_token: bool = self.config.get("class_token", True)
        attn_pool_head: bool = self.config.get("attn_pool_head", False)
        norm_layer_type: str = self.config.get("norm_layer_type", "LayerNorm")
        mlp_layer_type: str = self.config.get("mlp_layer_type", "FFN")
        drop_path_rate: float = self.config["drop_path_rate"]

        if norm_layer_type == "LayerNorm":
            norm_layer = nn.LayerNorm
        elif norm_layer_type == "RMSNorm":
            norm_layer = nn.RMSNorm
        else:
            raise ValueError(f"Unknown norm_layer_type '{norm_layer_type}'")

        if mlp_layer_type == "FFN":
            mlp_layer = FFN
            act_layer = nn.GELU
        elif mlp_layer_type == "SwiGLU_FFN":
            mlp_layer = SwiGLU_FFN
            act_layer = nn.SiLU
        else:
            raise ValueError(f"Unknown mlp_layer_type '{mlp_layer_type}'")

        torch._assert(image_size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(image_size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")
        self.pos_embed_special_tokens = pos_embed_special_tokens
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_reg_tokens = num_reg_tokens
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # Stochastic depth decay rule

        self.conv_proj = nn.Conv2d(
            self.input_channels,
            hidden_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
            bias=True,
        )
        self.patch_embed = PatchEmbed()

        seq_length = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.num_special_tokens = 0

        # Add a class token
        if class_token is True:
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.num_special_tokens += 1
            if pos_embed_special_tokens is True:
                seq_length += 1
        else:
            self.class_token = None

        # Add optional register tokens
        if self.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, self.num_reg_tokens, hidden_dim))
            self.num_special_tokens += self.num_reg_tokens
            if pos_embed_special_tokens is True:
                seq_length += self.num_reg_tokens
        else:
            self.reg_tokens = None

        # Add positional embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT

        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            dpr,
            activation_layer=act_layer,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )
        self.norm = norm_layer(hidden_dim, eps=1e-6)

        if attn_pool_head is False:
            self.attn_pool = None
        else:
            self.attn_pool = MultiHeadAttentionPool(hidden_dim, num_heads, mlp_dim, qkv_bias=True, latent_len=1)

        self.return_stages = ["neck"]  # Actually meaningless, just for completeness
        self.return_channels = [hidden_dim]
        self.embedding_size = hidden_dim
        self.classifier = self.create_classifier()

        self.max_stride = patch_size
        self.stem_stride = patch_size
        self.stem_width = hidden_dim
        self.encoding_size = hidden_dim
        self.decoder_block = partial(
            EncoderBlock,
            16,
            mlp_dim=None,
            dropout=0,
            attention_dropout=0,
            drop_path=0,
            activation_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )

        # Weight initialization
        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        if isinstance(self.classifier, nn.Linear):
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    def _get_pos_embed(self, H: int, W: int) -> torch.Tensor:
        if self.dynamic_size is False:
            return self.pos_embedding

        if H == self.size[0] and W == self.size[1]:
            return self.pos_embedding

        return adjust_position_embedding(
            self.pos_embedding,
            (self.size[0] // self.patch_size, self.size[1] // self.patch_size),
            (H // self.patch_size, W // self.patch_size),
            self.num_special_tokens if self.pos_embed_special_tokens is True else 0,
            antialias=False,
        )

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad = True

        if unfreeze_features is True:
            if self.attn_pool is not None:
                for param in self.attn_pool.parameters():
                    param.requires_grad = True

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        (H, W) = x.shape[-2:]
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        if self.pos_embed_special_tokens is False:
            x = x + self._get_pos_embed(H, W)

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        if self.pos_embed_special_tokens is True:
            x = x + self._get_pos_embed(H, W)

        x = self.encoder(x)
        x = self.norm(x)

        x = x[:, self.num_special_tokens :]
        x = x.permute(0, 2, 1)
        (B, C, _) = x.size()
        x = x.reshape(B, C, H // self.patch_size, W // self.patch_size)

        return {self.return_stages[0]: x}

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.conv_proj.parameters():
            param.requires_grad = False

        self.pos_embedding.requires_grad = False

        for idx, module in enumerate(self.encoder.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    # pylint: disable=too-many-branches
    def masked_encoding_omission(
        self,
        x: torch.Tensor,
        ids_keep: Optional[torch.Tensor] = None,
        return_all_features: bool = False,
        return_keys: Literal["all", "tokens", "embedding"] = "tokens",
    ) -> TokenOmissionResultType:
        (H, W) = x.shape[-2:]

        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        # Add pos embedding without special tokens
        pos_embedding = self._get_pos_embed(H, W)
        if self.pos_embed_special_tokens is True:
            x = x + pos_embedding[:, self.num_special_tokens :, :]
        else:
            x = x + pos_embedding

        # Mask tokens
        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(2)))

        # Append class and register tokens
        if self.class_token is not None:
            if self.pos_embed_special_tokens is True:
                cls_token = self.class_token + pos_embedding[:, self.num_reg_tokens : self.num_reg_tokens + 1, :]
            else:
                cls_token = self.class_token

            batch_class_token = cls_token.expand(x.shape[0], -1, -1)
            x = torch.concat((batch_class_token, x), dim=1)

        if self.reg_tokens is not None:
            if self.pos_embed_special_tokens is True:
                reg_tokens = self.reg_tokens + pos_embedding[:, 0 : self.num_reg_tokens, :]
            else:
                reg_tokens = self.reg_tokens

            batch_reg_tokens = reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        # Apply transformer
        if return_all_features is True:
            xs = self.encoder.forward_features(x)
            xs[-1] = self.norm(xs[-1])
            x = torch.stack(xs, dim=-1)
        else:
            x = self.encoder(x)
            x = self.norm(x)

        result: TokenOmissionResultType = {}
        if return_keys in ("all", "tokens"):
            result["tokens"] = x

        if return_keys in ("all", "embedding"):
            if return_all_features is True:
                x = x[..., -1]

            if self.attn_pool is not None:
                x = self.attn_pool(x)
                result["embedding"] = x[:, 0]
            elif self.class_token is None:
                x = x[:, self.num_special_tokens :]
                result["embedding"] = x.mean(dim=1)
            else:
                result["embedding"] = x[:, self.num_reg_tokens]

        return result

    def masked_encoding_retention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: Optional[torch.Tensor] = None,
        return_keys: Literal["all", "features", "embedding"] = "features",
    ) -> TokenRetentionResultType:
        (H, W) = x.shape[-2:]

        x = self.conv_proj(x)
        x = mask_tensor(x, mask, mask_token=mask_token, patch_factor=self.max_stride // self.stem_stride)

        # Reshape and permute the input tensor
        x = self.patch_embed(x)

        if self.pos_embed_special_tokens is False:
            x = x + self._get_pos_embed(H, W)

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        if self.pos_embed_special_tokens is True:
            x = x + self._get_pos_embed(H, W)

        x = self.encoder(x)
        x = self.norm(x)

        result: TokenRetentionResultType = {}
        if return_keys in ("all", "features"):
            features = x[:, self.num_special_tokens :]
            features = features.permute(0, 2, 1)
            (B, C, _) = features.size()
            features = features.reshape(B, C, H // self.patch_size, W // self.patch_size)
            result["features"] = features

        if return_keys in ("all", "embedding"):
            if self.attn_pool is not None:
                x = self.attn_pool(x)
                result["embedding"] = x[:, 0]
            elif self.class_token is None:
                x = x[:, self.num_special_tokens :]
                result["embedding"] = x.mean(dim=1)
            else:
                result["embedding"] = x[:, self.num_reg_tokens]

        return result

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        (H, W) = x.shape[-2:]

        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        if self.pos_embed_special_tokens is False:
            x = x + self._get_pos_embed(H, W)

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        if self.pos_embed_special_tokens is True:
            x = x + self._get_pos_embed(H, W)

        x = self.encoder(x)
        x = self.norm(x)

        return x

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            return x[:, 0]

        if self.class_token is None:
            x = x[:, self.num_special_tokens :]
            return x.mean(dim=1)

        # Classifier "token" as used by standard language architectures
        return x[:, self.num_reg_tokens]

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        assert new_size[0] % self.patch_size == 0, "Input shape indivisible by patch size!"
        assert new_size[1] % self.patch_size == 0, "Input shape indivisible by patch size!"

        old_size = self.size
        super().adjust_size(new_size)

        if self.pos_embed_special_tokens is True:
            num_prefix_tokens = self.num_special_tokens
        else:
            num_prefix_tokens = 0

        # Add back class tokens
        self.pos_embedding = nn.Parameter(
            # On rounding error see: https://github.com/facebookresearch/dino/issues/8
            adjust_position_embedding(
                self.pos_embedding,
                (old_size[0] // self.patch_size, old_size[1] // self.patch_size),
                (new_size[0] // self.patch_size, new_size[1] // self.patch_size),
                num_prefix_tokens,
            )
        )


registry.register_model_config(
    "vit_t32",
    ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 3,
        "hidden_dim": 192,
        "mlp_dim": 768,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_t16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 3,
        "hidden_dim": 192,
        "mlp_dim": 768,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_s32",
    ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_s16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_s14",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_m32",
    ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_m16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_m14",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_b32",
    ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_b16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_b14",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_l32",
    ViT,
    config={
        "patch_size": 32,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_l16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_l14",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_h16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_h14",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(  # From "Scaling Vision Transformers"
    "vit_g14",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 40,
        "num_heads": 16,
        "hidden_dim": 1408,
        "mlp_dim": 6144,
        "drop_path_rate": 0.1,
    },
)

# With registers
registry.register_model_config(
    "vit_reg1_t16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 3,
        "hidden_dim": 192,
        "mlp_dim": 768,
        "num_reg_tokens": 1,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_reg1_s32",
    ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "num_reg_tokens": 1,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_reg1_s16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "num_reg_tokens": 1,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_reg1_s16_ls",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "layer_scale_init_value": 1e-5,
        "num_reg_tokens": 1,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_reg1_s16_rms_ls",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "layer_scale_init_value": 1e-5,
        "num_reg_tokens": 1,
        "norm_layer_type": "RMSNorm",
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_reg1_s14",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "num_reg_tokens": 1,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_reg4_m32",
    ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_reg4_m16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_reg4_m16_rms_avg",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "num_reg_tokens": 4,
        "class_token": False,
        "norm_layer_type": "RMSNorm",
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_reg4_m14",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_reg4_b32",
    ViT,
    config={
        "patch_size": 32,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_model_config(
    "vit_reg4_b16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg4_b16_avg",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_reg_tokens": 4,
        "class_token": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg4_b14",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg8_b14_ap",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_reg_tokens": 8,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg4_l32",
    ViT,
    config={
        "patch_size": 32,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg4_l16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg4_l14",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg8_l14_ap",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_reg_tokens": 8,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg8_l14_rms_ap",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_reg_tokens": 8,
        "class_token": False,
        "attn_pool_head": True,
        "norm_layer_type": "RMSNorm",
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg4_h16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg4_h14",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(  # From "Scaling Vision Transformers"
    "vit_reg4_g14",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 40,
        "num_heads": 16,
        "hidden_dim": 1408,
        "mlp_dim": 6144,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)

# Shape-optimized vision transformer (SoViT)
registry.register_model_config(
    "vit_so150m_p14_ap",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 18,
        "num_heads": 16,
        "hidden_dim": 896,  # Changed from 880 for RoPE divisibility
        "mlp_dim": 2320,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_so400m_p14_ap",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 27,
        "num_heads": 16,
        "hidden_dim": 1152,
        "mlp_dim": 4304,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg4_so150m_p14_ap",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 18,
        "num_heads": 16,
        "hidden_dim": 896,  # Changed from 880 for RoPE divisibility
        "mlp_dim": 2320,
        "num_reg_tokens": 4,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg8_so150m_p14_avg",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 18,
        "num_heads": 16,
        "hidden_dim": 896,  # Changed from 880 for RoPE divisibility
        "mlp_dim": 2320,
        "num_reg_tokens": 8,
        "class_token": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg8_so150m_p14_swiglu_avg",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 18,
        "num_heads": 16,
        "hidden_dim": 896,  # Changed from 880 for RoPE divisibility
        "mlp_dim": 2320,
        "num_reg_tokens": 8,
        "class_token": False,
        "mlp_layer_type": "SwiGLU_FFN",
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg8_so150m_p14_ap",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 18,
        "num_heads": 16,
        "hidden_dim": 896,  # Changed from 880 for RoPE divisibility
        "mlp_dim": 2320,
        "num_reg_tokens": 8,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg4_so400m_p14_ap",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 27,
        "num_heads": 16,
        "hidden_dim": 1152,
        "mlp_dim": 4304,
        "num_reg_tokens": 4,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "vit_reg8_so400m_p14_ap",
    ViT,
    config={
        "patch_size": 14,
        "num_layers": 27,
        "num_heads": 16,
        "hidden_dim": 1152,
        "mlp_dim": 4304,
        "num_reg_tokens": 8,
        "class_token": False,
        "attn_pool_head": True,
        "drop_path_rate": 0.1,
    },
)

registry.register_weights(
    "vit_l16_mim_200",
    {
        "url": "https://huggingface.co/birder-project/vit_l16_mim/resolve/main",
        "description": (
            "ViT l16 image encoder pre-trained using Masked Image Modeling (MIM) for 200 epochs. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 1157.1,
                "sha256": "003b15a79cd528339de1b19304bbd04fd5885df36b80e19202cd6ef6f8ffbed1",
            },
        },
        "net": {"network": "vit_l16", "tag": "mim"},
    },
)
registry.register_weights(
    "vit_l16_mim_400",
    {
        "url": "https://huggingface.co/birder-project/vit_l16_mim/resolve/main",
        "description": (
            "ViT l16 image encoder pre-trained using Masked Image Modeling (MIM) for 400 epochs. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 1157.1,
                "sha256": "c6083c6532996addaf4efe29276aa55f9a3c77984f862f720c6131f86b847994",
            },
        },
        "net": {"network": "vit_l16", "tag": "mim"},
    },
)
registry.register_weights(
    "vit_l16_mim-eu-common",
    {
        "url": "https://huggingface.co/birder-project/vit_l16_mim-eu-common/resolve/main",
        "description": "ViT l16 model with MIM pretraining, then fine-tuned on the eu-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 1160.1,
                "sha256": "3b7235b90f76fb1e0e36d4c4111777a4cc4e4500552fe840c51170b208310d16",
            },
        },
        "net": {"network": "vit_l16", "tag": "mim-eu-common"},
    },
)

# With registers
registry.register_weights(
    "vit_reg4_m16_rms_avg_i-jepa",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_m16_rms_avg_i-jepa/resolve/main",
        "description": (
            "ViT reg4 m16 RMS norm image encoder pre-trained using I-JEPA"
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 146.2,
                "sha256": "bc4c9e600e93322440fb68c1001216d49c54c7587cdf61544f363f9537152f4a",
            },
        },
        "net": {"network": "vit_reg4_m16_rms_avg", "tag": "i-jepa"},
    },
)
registry.register_weights(
    "vit_reg4_m16_rms_avg_i-jepa-inat21-256px",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_m16_rms_avg_i-jepa-inat21/resolve/main",
        "description": (
            "ViT reg4 m16 RMS norm image encoder pre-trained using I-JEPA"
            "then fine-tuned on the iNaturalist 2021 dataset"
        ),
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 166.8,
                "sha256": "1cfa7ebea3db95363bf9e35fc24be94e419debe5db58746fe3320fbcb8bda6dd",
            },
        },
        "net": {"network": "vit_reg4_m16_rms_avg", "tag": "i-jepa-inat21-256px"},
    },
)
registry.register_weights(
    "vit_reg4_m16_rms_avg_i-jepa-inat21",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_m16_rms_avg_i-jepa-inat21/resolve/main",
        "description": (
            "ViT reg4 m16 RMS norm image encoder pre-trained using I-JEPA"
            "then fine-tuned on the iNaturalist 2021 dataset"
        ),
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 167.4,
                "sha256": "1cfa7ebea3db95363bf9e35fc24be94e419debe5db58746fe3320fbcb8bda6dd",
            },
        },
        "net": {"network": "vit_reg4_m16_rms_avg", "tag": "i-jepa-inat21"},
    },
)
registry.register_weights(
    "vit_reg4_m16_rms_avg_i-jepa-imagenet21k",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_m16_rms_avg_i-jepa-imagenet21k/resolve/main",
        "description": (
            "ViT reg4 m16 RMS norm image encoder pre-trained using I-JEPA then fine-tuned on the ImageNet-21K dataset"
        ),
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 184.2,
                "sha256": "d6d9fc47ecbad04a83b178bcd2eeecbd77569cc2a17fbdf52e02feda54523c3f",
            },
        },
        "net": {"network": "vit_reg4_m16_rms_avg", "tag": "i-jepa-imagenet21k"},
    },
)
registry.register_weights(
    "vit_reg4_b16_mim_200",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_b16_mim/resolve/main",
        "description": (
            "ViT reg4 b16 image encoder pre-trained using Masked Image Modeling (MIM) for 200 epochs. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 327.4,
                "sha256": "6b044cd7834293e344309f809070db3fe9ede489478e7549ad96255f9d76b329",
            },
        },
        "net": {"network": "vit_reg4_b16", "tag": "mim"},
    },
)
registry.register_weights(
    "vit_reg4_b16_mim_300",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_b16_mim/resolve/main",
        "description": (
            "ViT reg4 b16 image encoder pre-trained using Masked Image Modeling (MIM) for 300 epochs. "
            "This model has not been fine-tuned for a specific classification task"
        ),
        "resolution": (224, 224),
        "formats": {
            "pt": {
                "file_size": 327.4,
                "sha256": "e0df2e79f8ed0612d12c736cc6317be1b9b354e468715a5077366f7676fdd2ce",
            },
        },
        "net": {"network": "vit_reg4_b16", "tag": "mim"},
    },
)
registry.register_weights(
    "vit_reg4_b16_mim-intermediate-il-common",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_b16_mim-intermediate-il-common/resolve/main",
        "description": (
            "ViT reg4 b16 model with MIM pretraining and intermediate training, "
            "then fine-tuned on the il-common dataset"
        ),
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 328.7,
                "sha256": "3d1564be46b23081c76aa87c7e90324214b6ced899d4b38d59d1a4154b13f01c",
            },
        },
        "net": {"network": "vit_reg4_b16", "tag": "mim-intermediate-il-common"},
    },
)
registry.register_weights(
    "vit_reg4_b16_mim-intermediate-arabian-peninsula",
    {
        "url": "https://huggingface.co/birder-project/vit_reg4_b16_mim-intermediate-arabian-peninsula/resolve/main",
        "description": (
            "ViT reg4 b16 model with MIM pretraining and intermediate training, "
            "then fine-tuned on the arabian-peninsula dataset"
        ),
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 330.7,
                "sha256": "e011f931a5a4d96ef21283d70911a55ea649eadfefa9c163a48b996797f0d9da",
            },
        },
        "net": {"network": "vit_reg4_b16", "tag": "mim-intermediate-arabian-peninsula"},
    },
)
