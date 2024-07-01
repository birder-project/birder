"""
DeiT, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/deit.py

Paper "Training data-efficient image transformers & distillation through attention",
https://arxiv.org/abs/2012.12877
"""

# Reference license: Apache-2.0

import logging
import math
from typing import Optional

import torch
from torch import nn

from birder.core.net.base import BaseNet
from birder.core.net.vit import Encoder
from birder.core.net.vit import PatchEmbed
from birder.core.net.vit import adjust_position_embedding


class DeiT(BaseNet):
    default_size = 224

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
        pos_embed_class: bool = True,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        assert self.net_param is not None, "must set net-param"
        net_param = int(self.net_param)

        image_size = self.size
        attention_dropout = 0.0
        dropout = 0.0
        drop_path_rate = 0.1
        if net_param == 0:
            # Tiny 16 (t16)
            patch_size = 16
            num_layers = 12
            num_heads = 3
            hidden_dim = 192
            mlp_dim = 768

        elif net_param == 1:
            # Small 16 (s16)
            patch_size = 16
            num_layers = 12
            num_heads = 6
            hidden_dim = 384
            mlp_dim = 1536

        elif net_param == 2:
            # Base 16 (b16)
            patch_size = 16
            num_layers = 12
            num_heads = 12
            hidden_dim = 768
            mlp_dim = 3072

        else:
            raise ValueError(f"net_param = {net_param} not supported")

        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
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
        )

        self.embedding_size = hidden_dim
        self.classifier = self.create_classifier()

        # Weight initialization
        if isinstance(self.conv_proj, nn.Conv2d) is True:
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        if isinstance(self.classifier, nn.Linear) is True:
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)

        if self.pos_embed_class is True:
            x = torch.concat([batch_class_token, x], dim=1)
            x = x + self.pos_embedding

        else:
            x = x + self.pos_embedding
            x = torch.concat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x

    def create_classifier(self) -> nn.Module:
        return nn.Linear(self.embedding_size, self.num_classes)

    def adjust_size(self, new_size: int) -> None:
        super().adjust_size(new_size)

        # Sort out sizes
        num_pos_tokens = self.pos_embedding.shape[1]
        num_new_tokens = (new_size // self.patch_size) ** 2
        if self.pos_embed_class is True:
            num_prefix_tokens = 1
            num_new_tokens += 1  # Adding the class token

        else:
            num_prefix_tokens = 0

        if num_new_tokens == num_pos_tokens:
            return

        # Add back class tokens
        self.pos_embedding = nn.Parameter(
            adjust_position_embedding(
                num_pos_tokens, self.pos_embedding, new_size // self.patch_size, num_prefix_tokens
            )
        )

        logging.info(f"Resized position embedding: {num_pos_tokens} to {num_new_tokens}")
