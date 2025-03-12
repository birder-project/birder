"""
DeiT, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/deit.py

Paper "Training data-efficient image transformers & distillation through attention",
https://arxiv.org/abs/2012.12877
"""

# Reference license: Apache-2.0

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


class DeiT(BaseNet):
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
        drop_path_rate: float = self.config["drop_path_rate"]

        torch._assert(image_size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(image_size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_special_tokens = 2
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

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        # Add distillation token
        self.dist_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
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
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.embedding_size = hidden_dim
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()
        self.distillation_output = False

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

        if isinstance(self.dist_classifier, nn.Linear):
            nn.init.zeros_(self.dist_classifier.weight)
            nn.init.zeros_(self.dist_classifier.bias)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
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

        x = torch.concat([batch_class_token, batch_dist_token, x], dim=1)
        x = x + self.pos_embedding

        x = self.encoder(x)
        x = self.norm(x)
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
            # Classifier "token" as an average of both tokens (during regular training or inference)
            x = (x_cls + x_dist) / 2

        return x

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        old_size = self.size
        super().adjust_size(new_size)

        # Sort out sizes
        num_prefix_tokens = 2

        # Add back class tokens
        self.pos_embedding = nn.Parameter(
            adjust_position_embedding(
                self.pos_embedding,
                (old_size[0] // self.patch_size, old_size[1] // self.patch_size),
                (new_size[0] // self.patch_size, new_size[1] // self.patch_size),
                num_prefix_tokens,
            )
        )


registry.register_alias(
    "deit_t16",
    DeiT,
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
    "deit_s16",
    DeiT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "deit_b16",
    DeiT,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.1,
    },
)

registry.register_weights(
    "deit_t16_il-common",
    {
        "description": "DeiT tiny model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 21.7,
                "sha256": "ac124122dec9f1bceff383a6a555ca375ca1b613caf486dac3f29d87afac03b3",
            }
        },
        "net": {"network": "deit_t16", "tag": "il-common"},
    },
)
registry.register_weights(
    "deit_t16_dist-il-common",
    {
        "description": (
            "DeiT tiny model trained using distillation from a ConvNeXt v2 tiny teacher on the il-common dataset"
        ),
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 21.7,
                "sha256": "fafd0c3c65f9c35318f449f60485f640917736ee7b44056be55c2226909ffdb8",
            }
        },
        "net": {"network": "deit_t16", "tag": "dist-il-common"},
    },
)
