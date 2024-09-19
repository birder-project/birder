"""
Simple ViT, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
and
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

Paper "Better plain ViT baselines for ImageNet-1k",
https://arxiv.org/abs/2205.01580
"""

# Reference license: BSD 3-Clause and MIT

import logging
import math
from functools import partial
from typing import Optional

import torch
from torch import nn

from birder.model_registry import registry
from birder.net.base import PreTrainEncoder
from birder.net.base import pos_embedding_sin_cos_2d
from birder.net.vit import Encoder
from birder.net.vit import EncoderBlock
from birder.net.vit import PatchEmbed


# pylint: disable=invalid-name
class Simple_ViT(PreTrainEncoder):
    default_size = 224

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        assert self.net_param is not None, "must set net-param"
        net_param = int(self.net_param)

        image_size = self.size
        drop_path_rate = 0.0
        if net_param == 0:
            # Base 32 (b32)
            patch_size = 32
            num_layers = 12
            num_heads = 12
            hidden_dim = 768
            mlp_dim = 3072

        elif net_param == 1:
            # Base 16 (b16)
            patch_size = 16
            num_layers = 12
            num_heads = 12
            hidden_dim = 768
            mlp_dim = 3072

        elif net_param == 2:
            # Large 32 (l32)
            patch_size = 32
            num_layers = 24
            num_heads = 16
            hidden_dim = 1024
            mlp_dim = 4096

        elif net_param == 3:
            # Large 16 (l16)
            patch_size = 16
            num_layers = 24
            num_heads = 16
            hidden_dim = 1024
            mlp_dim = 4096

        elif net_param == 4:
            # Huge 14 (h14)
            patch_size = 14
            num_layers = 32
            num_heads = 16
            hidden_dim = 1280
            mlp_dim = 5120

        else:
            raise ValueError(f"net_param = {net_param} not supported")

        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_special_tokens = 0
        self.num_classes = num_classes
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

        # Add positional embedding
        pos_embedding = pos_embedding_sin_cos_2d(
            h=image_size // patch_size,
            w=image_size // patch_size,
            dim=hidden_dim,
            num_special_tokens=self.num_special_tokens,
        )
        self.pos_embedding = nn.Parameter(pos_embedding, requires_grad=False)

        self.encoder = Encoder(num_layers, num_heads, hidden_dim, mlp_dim, dropout=0.0, attention_dropout=0.0, dpr=dpr)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
        )

        self.embedding_size = hidden_dim
        self.classifier = self.create_classifier()

        self.encoding_size = hidden_dim * (image_size // patch_size) ** 2
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
        if isinstance(self.conv_proj, nn.Conv2d) is True:
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        if isinstance(self.classifier, nn.Linear) is True:
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    def masked_encoding(
        self, x: torch.Tensor, mask_ratio: float, _mask_token: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        # Add pos embedding
        x = x + self.pos_embedding

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

        # Apply transformer
        x = self.encoder(x)

        return (x, mask, ids_restore)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_proj(x)
        x = self.patch_embed(x)
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        return self.features(x)

    def adjust_size(self, new_size: int) -> None:
        super().adjust_size(new_size)

        # Sort out sizes
        num_pos_tokens = self.pos_embedding.shape[0]
        num_new_tokens = (new_size // self.patch_size) ** 2
        if num_new_tokens == num_pos_tokens:
            return

        pos_embedding = pos_embedding_sin_cos_2d(
            h=new_size // self.patch_size,
            w=new_size // self.patch_size,
            dim=self.hidden_dim,
            num_special_tokens=self.num_special_tokens,
        )
        self.pos_embedding = nn.Parameter(pos_embedding, requires_grad=False)

        logging.info(f"Resized position embedding: {num_pos_tokens} to {num_new_tokens}")


registry.register_alias("simple_vit_b32", Simple_ViT, 0)
registry.register_alias("simple_vit_b16", Simple_ViT, 1)
registry.register_alias("simple_vit_l32", Simple_ViT, 2)
registry.register_alias("simple_vit_l16", Simple_ViT, 3)
registry.register_alias("simple_vit_h14", Simple_ViT, 4)
