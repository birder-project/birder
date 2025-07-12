"""
Simple ViT, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
and
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

Paper "Better plain ViT baselines for ImageNet-1k",
https://arxiv.org/abs/2205.01580
"""

# Reference license: BSD 3-Clause and MIT

import math
from functools import partial
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torch import nn

from birder.model_registry import registry
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenOmissionResultType
from birder.net.base import pos_embedding_sin_cos_2d
from birder.net.vit import Encoder
from birder.net.vit import EncoderBlock
from birder.net.vit import PatchEmbed


# pylint: disable=invalid-name
class Simple_ViT(PreTrainEncoder, MaskedTokenOmissionMixin):
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
        drop_path_rate = 0.0
        patch_size: int = self.config["patch_size"]
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_dim: int = self.config["mlp_dim"]

        torch._assert(image_size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(image_size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_special_tokens = 0
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

        # Add positional embedding
        pos_embedding = pos_embedding_sin_cos_2d(
            h=image_size[0] // patch_size,
            w=image_size[1] // patch_size,
            dim=hidden_dim,
            num_special_tokens=self.num_special_tokens,
        )
        self.pos_embedding = nn.Parameter(pos_embedding, requires_grad=False)

        self.encoder = Encoder(num_layers, num_heads, hidden_dim, mlp_dim, dropout=0.0, attention_dropout=0.0, dpr=dpr)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(1),
        )

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

    def _get_pos_embed(self, H: int, W: int) -> torch.Tensor:
        if self.dynamic_size is False:
            return self.pos_embedding

        if H == self.size[0] and W == self.size[1]:
            return self.pos_embedding

        return pos_embedding_sin_cos_2d(
            h=H // self.patch_size,
            w=W // self.patch_size,
            dim=self.hidden_dim,
            num_special_tokens=self.num_special_tokens,
        ).to(self.pos_embedding.device)

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

        # Add pos embedding
        x = x + self._get_pos_embed(H, W)

        # Mask tokens
        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(2)))

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

            result["embedding"] = x.mean(dim=1)

        return result

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        (H, W) = x.shape[-2:]
        x = self.conv_proj(x)
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(H, W)
        x = self.encoder(x)
        x = self.norm(x)

        return x

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = x.permute(0, 2, 1)
        return self.features(x)

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        assert new_size[0] % self.patch_size == 0, "Input shape indivisible by patch size!"
        assert new_size[1] % self.patch_size == 0, "Input shape indivisible by patch size!"

        super().adjust_size(new_size)

        # Sort out sizes
        pos_embedding = pos_embedding_sin_cos_2d(
            h=new_size[0] // self.patch_size,
            w=new_size[1] // self.patch_size,
            dim=self.hidden_dim,
            num_special_tokens=self.num_special_tokens,
        )
        self.pos_embedding = nn.Parameter(pos_embedding, requires_grad=False)


registry.register_model_config(
    "simple_vit_s32",
    Simple_ViT,
    config={"patch_size": 32, "num_layers": 12, "num_heads": 6, "hidden_dim": 384, "mlp_dim": 1536},
)
registry.register_model_config(
    "simple_vit_s16",
    Simple_ViT,
    config={"patch_size": 16, "num_layers": 12, "num_heads": 6, "hidden_dim": 384, "mlp_dim": 1536},
)
registry.register_model_config(
    "simple_vit_s14",
    Simple_ViT,
    config={"patch_size": 14, "num_layers": 12, "num_heads": 6, "hidden_dim": 384, "mlp_dim": 1536},
)
registry.register_model_config(
    "simple_vit_m32",
    Simple_ViT,
    config={"patch_size": 32, "num_layers": 12, "num_heads": 8, "hidden_dim": 512, "mlp_dim": 2048},
)
registry.register_model_config(
    "simple_vit_m16",
    Simple_ViT,
    config={"patch_size": 16, "num_layers": 12, "num_heads": 8, "hidden_dim": 512, "mlp_dim": 2048},
)
registry.register_model_config(
    "simple_vit_m14",
    Simple_ViT,
    config={"patch_size": 14, "num_layers": 12, "num_heads": 8, "hidden_dim": 512, "mlp_dim": 2048},
)
registry.register_model_config(
    "simple_vit_b32",
    Simple_ViT,
    config={"patch_size": 32, "num_layers": 12, "num_heads": 12, "hidden_dim": 768, "mlp_dim": 3072},
)
registry.register_model_config(
    "simple_vit_b16",
    Simple_ViT,
    config={"patch_size": 16, "num_layers": 12, "num_heads": 12, "hidden_dim": 768, "mlp_dim": 3072},
)
registry.register_model_config(
    "simple_vit_b14",
    Simple_ViT,
    config={"patch_size": 14, "num_layers": 12, "num_heads": 12, "hidden_dim": 768, "mlp_dim": 3072},
)
registry.register_model_config(
    "simple_vit_l32",
    Simple_ViT,
    config={"patch_size": 32, "num_layers": 24, "num_heads": 16, "hidden_dim": 1024, "mlp_dim": 4096},
)
registry.register_model_config(
    "simple_vit_l16",
    Simple_ViT,
    config={"patch_size": 16, "num_layers": 24, "num_heads": 16, "hidden_dim": 1024, "mlp_dim": 4096},
)
registry.register_model_config(
    "simple_vit_l14",
    Simple_ViT,
    config={"patch_size": 14, "num_layers": 24, "num_heads": 16, "hidden_dim": 1024, "mlp_dim": 4096},
)
registry.register_model_config(
    "simple_vit_h16",
    Simple_ViT,
    config={"patch_size": 16, "num_layers": 32, "num_heads": 16, "hidden_dim": 1280, "mlp_dim": 5120},
)
registry.register_model_config(
    "simple_vit_h14",
    Simple_ViT,
    config={"patch_size": 14, "num_layers": 32, "num_heads": 16, "hidden_dim": 1280, "mlp_dim": 5120},
)
registry.register_model_config(  # From "Scaling Vision Transformers"
    "simple_vit_g14",
    Simple_ViT,
    config={"patch_size": 14, "num_layers": 40, "num_heads": 16, "hidden_dim": 1408, "mlp_dim": 6144},
)
