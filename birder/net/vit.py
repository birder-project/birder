"""
ViT, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

Paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
https://arxiv.org/abs/2010.11929
"""

# Reference license: BSD 3-Clause

import logging
import math
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import PreTrainEncoder


def adjust_position_embedding(
    num_pos_tokens: int, pos_embedding: torch.Tensor, new_base_size: int, num_prefix_tokens: int
) -> torch.Tensor:
    """
    Adapted from
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py
    """

    old_size = int(math.sqrt(num_pos_tokens - num_prefix_tokens))

    pos_embedding_prefix = pos_embedding[:, :num_prefix_tokens]
    pos_embedding = pos_embedding[:, num_prefix_tokens:]

    # Interpolation
    embed_dim = pos_embedding.shape[-1]
    orig_dtype = pos_embedding.dtype
    pos_embedding = pos_embedding.float()  # Interpolate needs float32
    pos_embedding = pos_embedding.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    pos_embedding = F.interpolate(pos_embedding, size=(new_base_size, new_base_size), mode="bicubic", antialias=True)
    pos_embedding = pos_embedding.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    pos_embedding = pos_embedding.to(orig_dtype)

    # Add back class tokens
    return nn.Parameter(torch.concat([pos_embedding_prefix, pos_embedding], dim=1))


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
    ) -> None:
        super().__init__()
        self.need_attn = False

        if mlp_dim is None:
            mlp_dim = hidden_dim * 4

        # Attention block
        self.ln1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.drop_path1 = StochasticDepth(drop_path, mode="row")

        # MLP block
        self.ln2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLP(
            hidden_dim, [mlp_dim, hidden_dim], activation_layer=activation_layer, inplace=None, dropout=dropout
        )
        self.drop_path2 = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.size()}")
        branch1 = self.ln1(x)
        (branch1, _) = self.self_attention(
            branch1, branch1, branch1, need_weights=self.need_attn, average_attn_weights=False
        )
        branch1 = self.drop_path1(branch1) + x

        branch2 = self.ln2(branch1)
        branch2 = self.mlp(branch2)

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
                    activation_layer=nn.GELU,
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.size()}")
        x = self.block(x)

        return x

    def set_need_attn(self) -> None:
        for b in self.block:
            b.set_need_attn()


class ViT(PreTrainEncoder):
    default_size = 224
    block_group_regex = r"encoder\.block\.(\d+)"

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[int] = None,
        num_reg_tokens: int = 0,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is not None, "must set config"

        self.num_reg_tokens = num_reg_tokens
        image_size = self.size
        attention_dropout = 0.0
        dropout = 0.0
        patch_size: int = self.config["patch_size"]
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_dim: int = self.config["mlp_dim"]
        drop_path_rate: float = self.config["drop_path_rate"]

        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_special_tokens = 1 + self.num_reg_tokens
        self.attention_dropout = attention_dropout
        self.dropout = dropout
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

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        # Add optional register tokens
        if self.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, self.num_reg_tokens, hidden_dim))
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
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.embedding_size = hidden_dim
        self.classifier = self.create_classifier()

        self.encoding_size = hidden_dim * seq_length
        self.decoder_block = partial(
            EncoderBlock,
            16,
            mlp_dim=None,
            dropout=0,
            attention_dropout=0,
            drop_path=0,
            activation_layer=nn.GELU,
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

    def masked_encoding(
        self, x: torch.Tensor, mask_ratio: float, _mask_token: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        # Add pos embedding without special tokens
        x = x + self.pos_embedding[:, self.num_reg_tokens + 1 :, :]

        # Masking: length -> length * mask_ratio
        # Perform per-sample random masking by per-sample shuffling.
        # Per-sample shuffling is done by argsort random noise.
        (N, L, D) = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # Noise in [0, 1]

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # Un-shuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x = x_masked

        # Append class and register tokens
        cls_token = self.class_token + self.pos_embedding[:, self.num_reg_tokens : self.num_reg_tokens + 1, :]
        batch_class_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.concat((batch_class_token, x), dim=1)

        if self.reg_tokens is not None:
            reg_tokens = self.reg_tokens + self.pos_embedding[:, 0 : self.num_reg_tokens, :]
            batch_reg_tokens = reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        # Apply transformer
        x = self.encoder(x)
        x = self.norm(x)

        return (x, mask, ids_restore)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.concat([batch_class_token, x], dim=1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.concat([batch_reg_tokens, x], dim=1)

        x = x + self.pos_embedding
        x = self.encoder(x)
        x = self.norm(x)

        # Classifier "token" as used by standard language architectures
        return x[:, self.num_reg_tokens]

    def adjust_size(self, new_size: int) -> None:
        if new_size == self.size:
            return

        logging.info(f"Adjusting model input resolution from {self.size} to {new_size}")
        super().adjust_size(new_size)

        # Sort out sizes
        num_pos_tokens = self.pos_embedding.shape[1]

        # Add back class tokens
        self.pos_embedding = nn.Parameter(
            adjust_position_embedding(
                num_pos_tokens, self.pos_embedding, new_size // self.patch_size, 1 + self.num_reg_tokens
            )
        )

        # Update encoding size
        self.encoding_size = self.pos_embedding.numel()


registry.register_alias(
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
registry.register_alias(
    "vit_b16",
    ViT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
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
registry.register_alias(
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
registry.register_alias(
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
registry.register_alias(
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

registry.register_weights(
    "vit_l16_mim_200",
    {
        "url": "https://huggingface.co/birder-project/vit_l16_mim/resolve/main/vit_l16_mim_200.pt",
        "description": (
            "ViT l16 image encoder pre-trained using Masked Image Modeling (MIM). "
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
