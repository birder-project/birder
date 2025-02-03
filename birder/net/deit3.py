"""
Paper "DeiT III: Revenge of the ViT", https://arxiv.org/abs/2204.07118
"""

import logging
import math
from typing import Any
from typing import Optional

import torch
from torch import nn

from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.vit import Encoder
from birder.net.vit import PatchEmbed
from birder.net.vit import adjust_position_embedding


class DeiT3(BaseNet):
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
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is not None, "must set config"

        pos_embed_class = False
        layer_scale_init_value = 1e-5
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
        self.num_special_tokens = 1
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.pos_embed_class = pos_embed_class
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # Stochastic depth decay rule

        self.conv_proj = nn.Conv2d(
            3,
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
        if pos_embed_class is True:
            seq_length += 1

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
            layer_scale_init_value=layer_scale_init_value,
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.embedding_size = hidden_dim
        self.classifier = self.create_classifier()

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

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad = True

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(x.shape[0], -1, -1)

        if self.pos_embed_class is True:
            x = torch.concat([batch_class_token, x], dim=1)
            x = x + self.pos_embedding
        else:
            x = x + self.pos_embedding
            x = torch.concat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        x = self.norm(x)
        x = x[:, 0]

        return x

    def adjust_size(self, new_size: int) -> None:
        if new_size == self.size:
            return

        logging.info(f"Adjusting model input resolution from {self.size} to {new_size}")
        super().adjust_size(new_size)

        # Sort out sizes
        num_pos_tokens = self.pos_embedding.shape[1]
        if self.pos_embed_class is True:
            num_prefix_tokens = 1
        else:
            num_prefix_tokens = 0

        # Add back class tokens
        self.pos_embedding = nn.Parameter(
            adjust_position_embedding(
                num_pos_tokens, self.pos_embedding, new_size // self.patch_size, num_prefix_tokens
            )
        )


registry.register_alias(
    "deit3_t16",
    DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 3,
        "hidden_dim": 192,
        "mlp_dim": 768,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "deit3_s16",
    DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "drop_path_rate": 0.05,
    },
)
registry.register_alias(
    "deit3_m16",
    DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "deit3_b16",
    DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.2,
    },
)
registry.register_alias(
    "deit3_l16",
    DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "drop_path_rate": 0.45,
    },
)
registry.register_alias(
    "deit3_h14",
    DeiT3,
    config={
        "patch_size": 14,
        "num_layers": 32,
        "num_heads": 16,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "drop_path_rate": 0.55,
    },
)

# registry.register_weights(
#     "deit3_t16_il-common",
#     {
#         "description": "DeiT3 tiny model trained on the il-common dataset",
#         "resolution": (256, 256),
#         "formats": {
#             "pt": {
#                 "file_size": 21.7,
#                 "sha256": "a883c5240cd3f8b6b003218e6016e18d01f8a5243481acbeeb3cbc3bf68a76af",
#             }
#         },
#         "net": {"network": "deit3_t16", "tag": "il-common"},
#     },
# )
