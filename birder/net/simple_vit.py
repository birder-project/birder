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
from birder.net._vit_configs import BASE
from birder.net._vit_configs import GIANT
from birder.net._vit_configs import HUGE
from birder.net._vit_configs import LARGE
from birder.net._vit_configs import MEDIUM
from birder.net._vit_configs import SMALL
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenOmissionResultType
from birder.net.base import normalize_out_indices
from birder.net.base import pos_embedding_sin_cos_2d
from birder.net.vit import Encoder
from birder.net.vit import EncoderBlock
from birder.net.vit import PatchEmbed


# pylint: disable=invalid-name,too-many-instance-attributes
class Simple_ViT(DetectorBackbone, PreTrainEncoder, MaskedTokenOmissionMixin):
    block_group_regex = r"encoder\.block\.(\d+)"

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, config=config, size=size)
        assert self.config is not None, "must set config"

        image_size = self.size
        patch_size: int = self.config["patch_size"]
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_dim: int = self.config["mlp_dim"]
        out_indices: Optional[list[int]] = self.config.get("out_indices", None)
        drop_path_rate: float = self.config["drop_path_rate"]

        torch._assert(image_size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(image_size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_special_tokens = 0
        self.out_indices = normalize_out_indices(out_indices, num_layers)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # Stochastic depth decay rule

        self.conv_proj = nn.Conv2d(
            self.input_channels,
            hidden_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
        )
        self.patch_embed = PatchEmbed()

        # Add positional embedding
        pos_embedding = pos_embedding_sin_cos_2d(
            h=image_size[0] // patch_size,
            w=image_size[1] // patch_size,
            dim=hidden_dim,
            num_special_tokens=self.num_special_tokens,
        )
        self.pos_embedding = nn.Buffer(pos_embedding)

        self.encoder = Encoder(num_layers, num_heads, hidden_dim, mlp_dim, dropout=0.0, attention_dropout=0.0, dpr=dpr)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(1),
        )

        num_return_stages = len(self.out_indices) if self.out_indices is not None else 1
        self.return_stages = [f"stage{stage_idx + 1}" for stage_idx in range(num_return_stages)]
        self.return_channels = [hidden_dim] * num_return_stages
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
        H, W = x.shape[-2:]

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
        H, W = x.shape[-2:]
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

    def transform_to_backbone(self) -> None:
        super().transform_to_backbone()
        self.norm = nn.Identity()

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        x = self.conv_proj(x)
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(H, W)

        if self.out_indices is None:
            xs = [self.encoder(x)]
        else:
            xs = self.encoder.forward_features(x, out_indices=self.out_indices)

        out: dict[str, torch.Tensor] = {}
        for stage_name, stage_x in zip(self.return_stages, xs):
            stage_x = stage_x[:, self.num_special_tokens :]
            stage_x = stage_x.permute(0, 2, 1)
            B, C, _ = stage_x.size()
            stage_x = stage_x.reshape(B, C, H // self.patch_size, W // self.patch_size)
            out[stage_name] = stage_x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.conv_proj.parameters():
            param.requires_grad_(False)

        for idx, module in enumerate(self.encoder.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad_(False)

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        assert new_size[0] % self.patch_size == 0, "Input shape indivisible by patch size!"
        assert new_size[1] % self.patch_size == 0, "Input shape indivisible by patch size!"

        super().adjust_size(new_size)

        # Sort out sizes
        with torch.no_grad():
            pos_embedding = pos_embedding_sin_cos_2d(
                h=new_size[0] // self.patch_size,
                w=new_size[1] // self.patch_size,
                dim=self.hidden_dim,
                num_special_tokens=self.num_special_tokens,
                device=self.pos_embedding.device,
            )

        self.pos_embedding = nn.Buffer(pos_embedding)

    def set_causal_attention(self, is_causal: bool = True) -> None:
        self.encoder.set_causal_attention(is_causal)


registry.register_model_config(
    "simple_vit_s32",
    Simple_ViT,
    config={"patch_size": 32, **SMALL},
)
registry.register_model_config(
    "simple_vit_s16",
    Simple_ViT,
    config={"patch_size": 16, **SMALL},
)
registry.register_model_config(
    "simple_vit_s14",
    Simple_ViT,
    config={"patch_size": 14, **SMALL},
)
registry.register_model_config(
    "simple_vit_m32",
    Simple_ViT,
    config={"patch_size": 32, **MEDIUM},
)
registry.register_model_config(
    "simple_vit_m16",
    Simple_ViT,
    config={"patch_size": 16, **MEDIUM},
)
registry.register_model_config(
    "simple_vit_m14",
    Simple_ViT,
    config={"patch_size": 14, **MEDIUM},
)
registry.register_model_config(
    "simple_vit_b32",
    Simple_ViT,
    config={"patch_size": 32, **BASE},  # Override the BASE definition
)
registry.register_model_config(
    "simple_vit_b16",
    Simple_ViT,
    config={"patch_size": 16, **BASE},
)
registry.register_model_config(
    "simple_vit_b14",
    Simple_ViT,
    config={"patch_size": 14, **BASE},
)
registry.register_model_config(
    "simple_vit_l32",
    Simple_ViT,
    config={"patch_size": 32, **LARGE},
)
registry.register_model_config(
    "simple_vit_l16",
    Simple_ViT,
    config={"patch_size": 16, **LARGE},
)
registry.register_model_config(
    "simple_vit_l14",
    Simple_ViT,
    config={"patch_size": 14, **LARGE},
)
registry.register_model_config(
    "simple_vit_h16",
    Simple_ViT,
    config={"patch_size": 16, **HUGE},
)
registry.register_model_config(
    "simple_vit_h14",
    Simple_ViT,
    config={"patch_size": 14, **HUGE},
)
registry.register_model_config(  # From "Scaling Vision Transformers"
    "simple_vit_g14",
    Simple_ViT,
    config={"patch_size": 14, **GIANT},
)
