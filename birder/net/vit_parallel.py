"""
ViT Parallel, adapted from
https://github.com/facebookresearch/deit/blob/main/models_v2.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

Paper "Three things everyone should know about Vision Transformers", https://arxiv.org/abs/2203.09795
"""

# Reference license: Apache-2.0 (both)

import math
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import StochasticDepth

from birder.common.masking import mask_tensor
from birder.layers import LayerScale
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenOmissionResultType
from birder.net.base import TokenRetentionResultType
from birder.net.vit import PatchEmbed
from birder.net.vit import adjust_position_embedding


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float, proj_drop: float) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)

        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class EncoderParallelBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: Optional[int],
        num_parallel: int,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        activation_layer: Callable[..., nn.Module],
        layer_scale_init_value: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        if mlp_dim is None:
            mlp_dim = hidden_dim * 4

        # Attention blocks
        self.attn_blocks = nn.ModuleList()
        for _ in range(num_parallel):
            self.attn_blocks.append(
                nn.Sequential(
                    norm_layer(hidden_dim, eps=1e-6),
                    Attention(hidden_dim, num_heads, attn_drop=attention_dropout, proj_drop=dropout),
                    LayerScale(hidden_dim, layer_scale_init_value) if layer_scale_init_value else nn.Identity(),
                    StochasticDepth(drop_path, mode="row"),
                )
            )

        # MLP blocks
        self.mlp_blocks = nn.ModuleList()
        for _ in range(num_parallel):
            self.mlp_blocks.append(
                nn.Sequential(
                    norm_layer(hidden_dim, eps=1e-6),
                    MLP(
                        hidden_dim,
                        [mlp_dim, hidden_dim],
                        activation_layer=activation_layer,
                        inplace=None,
                        dropout=dropout,
                    ),
                    LayerScale(hidden_dim, layer_scale_init_value) if layer_scale_init_value else nn.Identity(),
                    StochasticDepth(drop_path, mode="row"),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.stack([attn(x) for attn in self.attn_blocks]).sum(dim=0)
        x = x + torch.stack([mlp(x) for mlp in self.mlp_blocks]).sum(dim=0)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        num_parallel: int,
        dropout: float,
        attention_dropout: float,
        dpr: list[float],
        layer_scale_init_value: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        layers = []
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        for i in range(num_layers):
            layers.append(
                EncoderParallelBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    num_parallel,
                    dropout,
                    attention_dropout,
                    dpr[i],
                    activation_layer=nn.GELU,
                    layer_scale_init_value=layer_scale_init_value,
                    norm_layer=norm_layer,
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


# pylint: disable=invalid-name,too-many-instance-attributes
class ViT_Parallel(DetectorBackbone, PreTrainEncoder, MaskedTokenOmissionMixin, MaskedTokenRetentionMixin):
    block_group_regex = r"encoder\.block\.(\d+)"

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
        patch_size: int = self.config["patch_size"]
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_dim: int = self.config["mlp_dim"]
        num_parallel: int = self.config["num_parallel"]
        layer_scale_init_value: Optional[float] = self.config.get("layer_scale_init_value", None)
        num_reg_tokens: int = self.config.get("num_reg_tokens", 0)
        class_token: bool = self.config.get("class_token", True)
        norm_layer_type: str = self.config.get("norm_layer_type", "LayerNorm")
        drop_path_rate: float = self.config["drop_path_rate"]

        if norm_layer_type == "LayerNorm":
            norm_layer = nn.LayerNorm
        elif norm_layer_type == "RMSNorm":
            norm_layer = nn.RMSNorm
        else:
            raise ValueError(f"Unknown norm_layer_type '{norm_layer_type}'")

        torch._assert(image_size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(image_size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")
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
            seq_length += 1
            self.num_special_tokens += 1
        else:
            self.class_token = None

        # Add optional register tokens
        if self.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, self.num_reg_tokens, hidden_dim))
            seq_length += self.num_reg_tokens
            self.num_special_tokens += self.num_reg_tokens
        else:
            self.reg_tokens = None

        # Add positional embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT

        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            num_parallel,
            dropout,
            attention_dropout,
            dpr,
            layer_scale_init_value,
            norm_layer=norm_layer,
        )
        self.norm = norm_layer(hidden_dim, eps=1e-6)

        self.return_stages = ["neck"]  # Actually meaningless, but for completeness
        self.return_channels = [hidden_dim]
        self.embedding_size = hidden_dim
        self.classifier = self.create_classifier()

        self.max_stride = patch_size
        self.stem_stride = patch_size
        self.stem_width = hidden_dim
        self.encoding_size = hidden_dim
        self.decoder_block = partial(
            EncoderParallelBlock,
            16,
            mlp_dim=None,
            num_parallel=num_parallel,
            dropout=0,
            attention_dropout=0,
            drop_path=0,
            activation_layer=nn.GELU,
            norm_layer=norm_layer,
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
            self.num_special_tokens,
            antialias=False,
        )

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad = True

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        (H, W) = x.shape[-2:]
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

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
        x = x + pos_embedding[:, self.num_special_tokens :, :]

        # Mask tokens
        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(2)))

        # Append class and register tokens
        if self.class_token is not None:
            cls_token = self.class_token + pos_embedding[:, self.num_reg_tokens : self.num_reg_tokens + 1, :]
            batch_class_token = cls_token.expand(x.shape[0], -1, -1)
            x = torch.concat((batch_class_token, x), dim=1)

        if self.reg_tokens is not None:
            reg_tokens = self.reg_tokens + pos_embedding[:, 0 : self.num_reg_tokens, :]
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

            if self.class_token is None:
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

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

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
            if self.class_token is None:
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

        # Expand the class token to the full batch
        if self.class_token is not None:
            batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        x = x + self._get_pos_embed(H, W)
        x = self.encoder(x)
        x = self.norm(x)

        return x

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)

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

        # Add back class tokens
        self.pos_embedding = nn.Parameter(
            # On rounding error see: https://github.com/facebookresearch/dino/issues/8
            adjust_position_embedding(
                self.pos_embedding,
                (old_size[0] // self.patch_size, old_size[1] // self.patch_size),
                (new_size[0] // self.patch_size, new_size[1] // self.patch_size),
                self.num_special_tokens,
            )
        )


registry.register_model_config(
    "vit_parallel_s16_18x2_ls",
    ViT_Parallel,
    config={
        "patch_size": 16,
        "num_layers": 18,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "num_parallel": 2,
        "layer_scale_init_value": 1e-5,
        "drop_path_rate": 0.05,
    },
)
registry.register_model_config(
    "vit_parallel_s16_18x2_ls_avg",
    ViT_Parallel,
    config={
        "patch_size": 16,
        "num_layers": 18,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "num_parallel": 2,
        "layer_scale_init_value": 1e-5,
        "class_token": False,
        "drop_path_rate": 0.05,
    },
)
registry.register_model_config(
    "vit_parallel_b16_18x2_ls",
    ViT_Parallel,
    config={
        "patch_size": 16,
        "num_layers": 18,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_parallel": 2,
        "layer_scale_init_value": 1e-5,
        "drop_path_rate": 0.2,
    },
)
