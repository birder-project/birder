"""
Paper "FlexiViT: One Model for All Patch Sizes", https://arxiv.org/abs/2212.08013
"""

# Reference license: Apache-2.0

import logging
import random
from functools import partial
from typing import Any
from typing import Optional

import torch
from torch import nn

from birder.model_registry import registry
from birder.net._vit_configs import BASE
from birder.net._vit_configs import SMALL
from birder.net.flexivit import flex_proj
from birder.net.flexivit import get_patch_sizes
from birder.net.flexivit import interpolate_proj
from birder.net.rope_vit import MAEDecoderBlock
from birder.net.rope_vit import RoPE_ViT
from birder.net.vit import adjust_position_embedding

logger = logging.getLogger(__name__)


class RoPE_FlexiViT(RoPE_ViT):
    default_size = (240, 240)

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        # Registered models are dynamically created subclasses, with their base config
        # stored as a class attribute before this initializer runs.
        # The config argument contains only per-call overrides, so check both sources
        # before applying RoPE FlexiViT's default.
        pos_embed_config_key = "pos_embed_special_tokens"
        registered_config = getattr(type(self), "config", None)
        has_registered_setting = registered_config is not None and pos_embed_config_key in registered_config
        has_call_override = config is not None and pos_embed_config_key in config
        if has_registered_setting is False and has_call_override is False:
            config = dict(config or {})
            config[pos_embed_config_key] = False

        super().__init__(input_channels, num_classes, config=config, size=size)
        assert self.config is not None, "must set config"

        if isinstance(self.conv_proj, nn.Conv2d) is False:
            raise ValueError("RoPE FlexiViT only supports the standard patchify stem")

        self.min_patch_size: int = self.config.get("min_patch_size", 8)
        self.max_patch_size: int = self.config.get("max_patch_size", 48)
        self.patch_size_list = get_patch_sizes(self.min_patch_size, self.max_patch_size, self.size)
        self.set_dynamic_size()

    def _get_pos_embed(self, H: int, W: int, patch_size: Optional[int] = None) -> Optional[torch.Tensor]:
        if self.pos_embedding is None:
            return None

        if patch_size is None:
            patch_size = self.patch_size

        if H == self.size[0] and W == self.size[1] and patch_size == self.patch_size:
            return self.pos_embedding

        return adjust_position_embedding(
            self.pos_embedding,
            (self.size[0] // self.patch_size, self.size[1] // self.patch_size),
            (H // patch_size, W // patch_size),
            self.num_special_tokens if self.pos_embed_special_tokens is True else 0,
            interpolation_mode=self.pos_embed_interpolation_mode,
            antialias=self.pos_embed_antialias,
        )

    def _get_rope_embed(self, H: int, W: int, patch_size: Optional[int] = None) -> torch.Tensor:
        if patch_size is None:
            patch_size = self.patch_size

        return self.rope.get_pos_embed((H // patch_size, W // patch_size))

    # pylint: disable-next=arguments-renamed
    def forward_features(  # type: ignore[override]
        self,
        x: torch.Tensor,
        patch_size: Optional[int] = None,
        return_input_embedding: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training is True and patch_size is None and not torch.jit.is_tracing() and not torch.jit.is_scripting():
            patch_size = random.choice(self.patch_size_list)

        H, W = x.shape[-2:]

        # Reshape and permute the input tensor
        x = flex_proj(x, self.conv_proj.weight, self.conv_proj.bias, patch_size)
        x = self.patch_embed(x)

        patch_embedding = x
        pos_embedding = self._get_pos_embed(H, W, patch_size=patch_size)

        if pos_embedding is not None and self.pos_embed_special_tokens is False:
            x = x + pos_embedding

        # Expand special tokens to batch size and prepend in order [REG..., CLS, PATCH...]
        special_tokens: list[torch.Tensor] = []
        if self.reg_tokens is not None:
            special_tokens.append(self.reg_tokens.expand(x.size(0), -1, -1))
        if self.class_token is not None:
            special_tokens.append(self.class_token.expand(x.size(0), -1, -1))
        if len(special_tokens) > 0:
            x = torch.concat(special_tokens + [x], dim=1)

        if return_input_embedding is True:
            if len(special_tokens) > 0:
                input_embedding = torch.concat(special_tokens + [patch_embedding], dim=1)
            else:
                input_embedding = patch_embedding
        else:
            input_embedding = None  # For TorchScript compatibility

        if pos_embedding is not None and self.pos_embed_special_tokens is True:
            x = x + pos_embedding

        x = self.encoder(x, self._get_rope_embed(H, W, patch_size=patch_size), attn_mask=attn_mask)
        x = self.norm(x)

        if return_input_embedding is True and input_embedding is not None:
            return torch.stack([input_embedding, x], dim=-1)

        return x

    def embedding(self, x: torch.Tensor, patch_size: Optional[int] = None) -> torch.Tensor:
        return self.embedding_from_features(self.forward_features(x, patch_size))

    def forward(self, x: torch.Tensor, patch_size: Optional[int] = None) -> torch.Tensor:
        x = self.embedding(x, patch_size)
        return self.classify(x)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        if dynamic_size is False:
            raise ValueError("RoPE FlexiViT only supports dynamic mode")

        super().set_dynamic_size(dynamic_size)

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)
        self.patch_size_list = get_patch_sizes(self.min_patch_size, self.max_patch_size, self.size)

    def adjust_patch_size(self, patch_size: int) -> None:
        if self.patch_size == patch_size:
            return

        assert self.size[0] % patch_size == 0, "Input shape indivisible by patch size!"
        assert self.size[1] % patch_size == 0, "Input shape indivisible by patch size!"

        logger.debug(f"Setting patch size to: {patch_size}")
        with torch.no_grad():
            conv_proj_weight = interpolate_proj(self.conv_proj.weight, patch_size)

        self.conv_proj.weight = nn.Parameter(conv_proj_weight)
        self.conv_proj.kernel_size = (patch_size, patch_size)
        self.conv_proj.stride = (patch_size, patch_size)
        if self.pos_embedding is not None:
            # Adjust pos_embedding accordingly
            if self.pos_embed_special_tokens is True:
                num_prefix_tokens = self.num_special_tokens
            else:
                num_prefix_tokens = 0

            with torch.no_grad():
                pos_embedding = adjust_position_embedding(
                    self.pos_embedding,
                    (self.size[0] // self.patch_size, self.size[1] // self.patch_size),
                    (self.size[0] // patch_size, self.size[1] // patch_size),
                    num_prefix_tokens,
                    interpolation_mode=self.pos_embed_interpolation_mode,
                )

            self.pos_embedding = nn.Parameter(pos_embedding)

        grid_size = (self.size[0] // patch_size, self.size[1] // patch_size)
        self.rope.set_grid_size(grid_size)

        # Define adjusted decoder block
        self.decoder_block = partial(
            MAEDecoderBlock,
            16,
            num_special_tokens=self.num_special_tokens,
            activation_layer=self.act_layer,
            grid_size=grid_size,
            rope_grid_indexing=self.rope_grid_indexing,
            rope_grid_offset=self.rope_grid_offset,
            rope_temperature=self.rope_temperature,
            layer_scale_init_value=self.layer_scale_init_value,
            norm_layer=self.norm_layer,
            norm_layer_eps=self.norm_layer_eps,
            mlp_layer=self.decoder_mlp_layer,
            rope_style=self.rope_style,
            rope_rot_type=self.rope_rot_type,
        )

        self.patch_size = patch_size
        self.max_stride = patch_size
        self.stem_stride = patch_size

    def load_rope_vit_weights(self, state_dict: dict[str, Any]) -> None:
        if self.pos_embedding is None:
            state_dict.pop("pos_embedding", None)
            self.load_state_dict(state_dict, strict=True)
            return

        num_special_tokens = 0
        if "class_token" in state_dict:
            num_special_tokens += 1

        if "reg_tokens" in state_dict:
            num_special_tokens += state_dict["reg_tokens"].size(1)

        pos_embedding = state_dict["pos_embedding"]
        seq_length = (self.size[0] // self.patch_size) * (self.size[1] // self.patch_size)
        vit_pos_embed_special_tokens = pos_embedding.size(1) != seq_length

        # Adjust pos_embedding
        if self.pos_embed_special_tokens is False and vit_pos_embed_special_tokens is True:
            logger.debug("Folding RoPE ViT special-token positional embeddings into the learned tokens")
            special_pos_embedding = pos_embedding[:, :num_special_tokens, :]
            special_token_offset = 0
            if "reg_tokens" in state_dict:
                num_reg_tokens = state_dict["reg_tokens"].size(1)
                state_dict["reg_tokens"] = (
                    state_dict["reg_tokens"]
                    + special_pos_embedding[:, special_token_offset : special_token_offset + num_reg_tokens, :]
                )
                special_token_offset += num_reg_tokens

            if "class_token" in state_dict:
                state_dict["class_token"] = (
                    state_dict["class_token"]
                    + special_pos_embedding[:, special_token_offset : special_token_offset + 1, :]
                )

            pos_embedding = pos_embedding[:, num_special_tokens:, :]

        state_dict["pos_embedding"] = pos_embedding

        self.load_state_dict(state_dict, strict=True)


# For the model naming convention see rope_vit.py

registry.register_model_config(
    "rope_flexivit_s16",
    RoPE_FlexiViT,
    config={"patch_size": 16, **SMALL},
)
registry.register_model_config(
    "rope_flexivit_b16",
    RoPE_FlexiViT,
    config={"patch_size": 16, **BASE},
)

# With registers
####################

registry.register_model_config(
    "rope_flexivit_reg1_s16",
    RoPE_FlexiViT,
    config={"patch_size": 16, **SMALL, "num_reg_tokens": 1},
)
registry.register_model_config(
    "rope_flexivit_reg4_b16",
    RoPE_FlexiViT,
    config={"patch_size": 16, **BASE, "num_reg_tokens": 4},
)
registry.register_model_config(
    "rope_flexivit_reg4_b16_avg",
    RoPE_FlexiViT,
    config={"patch_size": 16, **BASE, "num_reg_tokens": 4, "class_token": False},
)
registry.register_model_config(
    "rope_flexivit_reg4_b16_qkn_ls",
    RoPE_FlexiViT,
    config={"patch_size": 16, **BASE, "num_reg_tokens": 4, "layer_scale_init_value": 1e-5, "qk_norm": True},
)
registry.register_model_config(
    "rope_flexivit_reg4_b16_qkn_ls_ep",
    RoPE_FlexiViT,
    config={
        "patch_size": 16,
        **BASE,
        "num_reg_tokens": 4,
        "layer_scale_init_value": 1e-5,
        "qk_norm": True,
        "attn_pool_head": True,
        "attn_pool_type": "EfficientProbing",
    },
)
