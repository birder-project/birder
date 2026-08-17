"""
Paper "FlexiViT: One Model for All Patch Sizes", https://arxiv.org/abs/2212.08013
"""

# Reference license: Apache-2.0

import logging
import random
from functools import lru_cache
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from birder.model_registry import registry
from birder.net._vit_configs import BASE
from birder.net._vit_configs import SMALL
from birder.net.vit import ViT
from birder.net.vit import adjust_position_embedding

logger = logging.getLogger(__name__)


def get_patch_sizes(min_size: int, max_size: int, input_size: tuple[int, int]) -> list[int]:
    H, W = input_size
    valid_sizes = []
    for patch_size in range(min_size, max_size + 1):
        if H % patch_size == 0 and W % patch_size == 0:
            valid_sizes.append(patch_size)

    return sorted(valid_sizes)


@torch.compiler.disable()  # type: ignore[untyped-decorator]
@lru_cache(maxsize=128)
@torch.no_grad()  # type: ignore[untyped-decorator]
def _get_resize_matrix_pinv(
    old_size: tuple[int, int],
    new_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    old_h, old_w = old_size
    old_numel = old_h * old_w

    # Resize the identity basis to construct the bilinear interpolation matrix
    basis_vectors = torch.eye(old_numel, dtype=torch.float32, device=device).reshape(old_numel, 1, old_h, old_w)
    resized_basis_vectors = F.interpolate(basis_vectors, size=new_size, mode="bilinear", align_corners=False)
    resize_matrix = resized_basis_vectors.squeeze(1).permute(1, 2, 0).reshape(new_size[0] * new_size[1], old_numel)

    # Its pseudoinverse (Moore-Penrose inverse) maps flattened kernels from old_size to new_size
    return torch.linalg.pinv(resize_matrix)  # pylint: disable=not-callable


def interpolate_proj(proj_weight: torch.Tensor, patch_size: int) -> torch.Tensor:
    orig_dtype = proj_weight.dtype
    old_size = (proj_weight.shape[-2], proj_weight.shape[-1])
    new_size = (patch_size, patch_size)
    resize_matrix_pinv = _get_resize_matrix_pinv(old_size, new_size, proj_weight.device)

    weight_resampled = proj_weight.float().flatten(2) @ resize_matrix_pinv
    weight_resampled = weight_resampled.reshape(proj_weight.shape[0], proj_weight.shape[1], *new_size).to(orig_dtype)

    return weight_resampled


def flex_proj(
    x: torch.Tensor, proj_weight: torch.Tensor, proj_bias: Optional[torch.Tensor], patch_size: Optional[int]
) -> torch.Tensor:
    if patch_size is not None and patch_size != proj_weight.shape[-1] and not torch.jit.is_scripting():
        weight_resampled = interpolate_proj(proj_weight, patch_size)
        x = F.conv2d(x, weight_resampled, proj_bias, stride=(patch_size, patch_size))  # pylint: disable=not-callable
    else:
        x = F.conv2d(x, proj_weight, proj_bias, stride=proj_weight.shape[-2:])  # pylint: disable=not-callable

    return x


class FlexiViT(ViT):
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
        # before applying FlexiViT's default.
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
            raise ValueError("FlexiViT only supports the standard patchify stem")

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
            antialias=False,
        )

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

        x = self.encoder(x, attn_mask=attn_mask)
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
            raise ValueError("FlexiViT only supports dynamic mode")

        super().set_dynamic_size(dynamic_size)

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)
        self.patch_size_list = get_patch_sizes(self.min_patch_size, self.max_patch_size, self.size)

    def adjust_patch_size(self, patch_size: int) -> None:
        if self.patch_size == patch_size:
            return

        logger.debug(f"Setting patch size to: {patch_size}")
        self.conv_proj.weight = nn.Parameter(interpolate_proj(self.conv_proj.weight, patch_size))
        self.conv_proj.kernel_size = (patch_size, patch_size)
        self.conv_proj.stride = (patch_size, patch_size)
        if self.pos_embedding is not None:
            # Adjust pos_embedding accordingly
            if self.pos_embed_special_tokens is True:
                num_prefix_tokens = self.num_special_tokens
            else:
                num_prefix_tokens = 0

            self.pos_embedding = nn.Parameter(
                adjust_position_embedding(
                    self.pos_embedding,
                    (self.size[0] // self.patch_size, self.size[1] // self.patch_size),
                    (self.size[0] // patch_size, self.size[1] // patch_size),
                    num_prefix_tokens,
                    interpolation_mode=self.pos_embed_interpolation_mode,
                )
            )

        self.patch_size = patch_size
        self.max_stride = patch_size
        self.stem_stride = patch_size

    def load_vit_weights(self, state_dict: dict[str, Any]) -> None:
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
            logger.debug("Folding ViT special-token positional embeddings into the FlexiViT learned tokens")
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


registry.register_model_config(
    "flexivit_s16",
    FlexiViT,
    config={"patch_size": 16, **SMALL},
)
registry.register_model_config(
    "flexivit_s16_ls",
    FlexiViT,
    config={"patch_size": 16, **SMALL, "layer_scale_init_value": 1e-5},
)
registry.register_model_config(
    "flexivit_b16",
    FlexiViT,
    config={"patch_size": 16, **BASE},
)

# With registers
####################

registry.register_model_config(
    "flexivit_reg1_s16",
    FlexiViT,
    config={"patch_size": 16, **SMALL, "num_reg_tokens": 1},
)
registry.register_model_config(
    "flexivit_reg1_s16_rms_ls",
    FlexiViT,
    config={
        "patch_size": 16,
        **SMALL,
        "layer_scale_init_value": 1e-5,
        "num_reg_tokens": 1,
        "norm_layer_type": "RMSNorm",
    },
)
registry.register_model_config(
    "flexivit_reg4_b16",
    FlexiViT,
    config={"patch_size": 16, **BASE, "num_reg_tokens": 4},
)
registry.register_model_config(
    "flexivit_reg8_b14_ap",
    FlexiViT,
    config={"patch_size": 14, **BASE, "num_reg_tokens": 8, "class_token": False, "attn_pool_head": True},
)

registry.register_weights(
    "flexivit_reg1_s16_rms_ls_dino-v2-il-all",
    {
        "url": "https://huggingface.co/birder-project/flexivit_reg1_s16_rms_ls_dino-v2-il-all/resolve/main",
        "description": (
            "FlexiViT Reg1 S/16 model pretrained using DINO v2 on the il-all dataset, then fine-tuned on the "
            "il-all dataset"
        ),
        "resolution": (240, 240),
        "formats": {
            "pt": {
                "file_size": 83.6,
                "sha256": "8285f4fe56401f169491cb2399d2a7c82f3a0cfbe8a5a8d3c27163024a274800",
            },
        },
        "net": {"network": "flexivit_reg1_s16_rms_ls", "tag": "dino-v2-il-all"},
    },
)
