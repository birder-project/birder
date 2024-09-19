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

from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.vit import Encoder
from birder.net.vit import PatchEmbed
from birder.net.vit import adjust_position_embedding


# pylint: disable=too-many-instance-attributes
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
        if net_param == 0:
            # Tiny 16 (t16)
            patch_size = 16
            num_layers = 12
            num_heads = 3
            hidden_dim = 192
            mlp_dim = 768
            drop_path_rate = 0.0

        elif net_param == 1:
            # Small 16 (s16)
            patch_size = 16
            num_layers = 12
            num_heads = 6
            hidden_dim = 384
            mlp_dim = 1536
            drop_path_rate = 0.1

        elif net_param == 2:
            # Base 16 (b16)
            patch_size = 16
            num_layers = 12
            num_heads = 12
            hidden_dim = 768
            mlp_dim = 3072
            drop_path_rate = 0.1

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

        # Add distillation token
        self.dist_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
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
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()
        self.distillation_output = False

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

        if isinstance(self.dist_classifier, nn.Linear) is True:
            nn.init.zeros_(self.dist_classifier.weight)
            nn.init.zeros_(self.dist_classifier.bias)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()

    def freeze(self, freeze_classifier: bool = True) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad = True

            for param in self.dist_classifier.parameters():
                param.requires_grad = True

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
        batch_dist_token = self.dist_token.expand(x.shape[0], -1, -1)

        if self.pos_embed_class is True:
            x = torch.concat([batch_class_token, batch_dist_token, x], dim=1)
            x = x + self.pos_embedding
        else:
            x = x + self.pos_embedding
            x = torch.concat([batch_class_token, batch_dist_token, x], dim=1)

        x = self.encoder(x)
        x = x[:, 0:2]

        return x

    def set_distillation_output(self, enable: bool = True) -> None:
        self.distillation_output = enable

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        x_cls = x[:, 0]
        x_dist = x[:, 1]

        x_cls = self.classifier(x_cls)
        x_dist = self.dist_classifier(x_dist)

        if self.training is True and self.distillation_output is True:
            x = torch.stack([x_cls, x_dist], dim=1)
        else:
            # Classifier "token" as an average of both tokens (during normal training or inference)
            x = (x_cls + x_dist) / 2

        return x

    def adjust_size(self, new_size: int) -> None:
        super().adjust_size(new_size)

        # Sort out sizes
        num_pos_tokens = self.pos_embedding.shape[1]
        num_new_tokens = (new_size // self.patch_size) ** 2
        if self.pos_embed_class is True:
            num_prefix_tokens = 2
            num_new_tokens += 2  # Adding the class and distillation tokens
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


registry.register_alias("deit_t16", DeiT, 0)
registry.register_alias("deit_s16", DeiT, 1)
registry.register_alias("deit_b16", DeiT, 2)

registry.register_weights(
    "deit_t16_il-common",
    {
        "description": "DeiT tiny model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 21.7,
                "sha256": "6b752b579cea28aaa351591bc9e17bb8c7cf946defa9f30f001f65f7cb5b3035",
            }
        },
        "net": {"network": "deit_t16", "tag": "il-common"},
    },
)
