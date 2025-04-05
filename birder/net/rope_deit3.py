"""
RoPE DeiT3, adapted from
https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed_sincos.py

Paper "Rotary Position Embedding for Vision Transformer", https://arxiv.org/abs/2403.13298

Changes from original:
* Implemented only axial RoPE (EVA style RoPE)
* Modified rotate_half (original implementation seems off)
"""

# Reference license: Apache-2.0 and Apache-2.0

import math
from typing import Any
from typing import Optional

import torch
from torch import nn

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.rope_vit import Encoder
from birder.net.rope_vit import RoPE
from birder.net.rope_vit import build_rotary_pos_embed
from birder.net.vit import PatchEmbed
from birder.net.vit import adjust_position_embedding


# pylint: disable=invalid-name,too-many-instance-attributes
class RoPE_DeiT3(DetectorBackbone):
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

        layer_scale_init_value = 1e-5
        image_size = self.size
        attention_dropout = 0.0
        dropout = 0.0
        pos_embed_class: bool = self.config.get("pos_embed_class", False)
        patch_size: int = self.config["patch_size"]
        num_layers: int = self.config["num_layers"]
        num_heads: int = self.config["num_heads"]
        hidden_dim: int = self.config["hidden_dim"]
        mlp_dim: int = self.config["mlp_dim"]
        num_reg_tokens: int = self.config.get("num_reg_tokens", 0)
        pt_grid_size: Optional[tuple[int, int]] = self.config.get("pt_grid_size", None)
        drop_path_rate: float = self.config["drop_path_rate"]

        torch._assert(image_size[0] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(image_size[1] % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_reg_tokens = num_reg_tokens
        self.num_special_tokens = 1 + self.num_reg_tokens
        self.pos_embed_class = pos_embed_class
        self.rope_temperature = 100.0

        # Cast in case config was loaded from a json (no tuples),
        # TorchScript does not accept a list when tuple expected
        if isinstance(pt_grid_size, list):
            pt_grid_size = tuple(pt_grid_size)  # type: ignore[unreachable]

        self.pt_grid_size = pt_grid_size
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
        if pos_embed_class is True:
            seq_length += 1

        # Add optional register tokens
        if self.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, self.num_reg_tokens, hidden_dim))
            if pos_embed_class is True:
                seq_length += self.num_reg_tokens
        else:
            self.reg_tokens = None

        # Add positional embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))

        # RoPE
        self.rope = RoPE(
            hidden_dim // num_heads,
            temperature=self.rope_temperature,
            grid_size=(image_size[0] // patch_size, image_size[1] // patch_size),
            pt_grid_size=self.pt_grid_size,
        )

        # Encoder
        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            self.num_special_tokens,
            dropout,
            attention_dropout,
            dpr,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.return_stages = ["neck"]  # Actually meaningless, but for completeness
        self.return_channels = [hidden_dim]
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

    def _get_pos_embed(self, H: int, W: int) -> torch.Tensor:
        if self.dynamic_size is False:
            return self.pos_embedding

        if H == self.size[0] and W == self.size[1]:
            return self.pos_embedding

        return adjust_position_embedding(
            self.pos_embedding,
            (self.size[0] // self.patch_size, self.size[1] // self.patch_size),
            (H // self.patch_size, W // self.patch_size),
            self.num_special_tokens if self.pos_embed_class is True else 0,
            antialias=False,
        )

    def _get_rope_embed(self, H: int, W: int) -> torch.Tensor:
        if self.dynamic_size is False:
            return self.rope.pos_embed

        if H == self.size[0] and W == self.size[1]:
            return self.rope.pos_embed

        return torch.concat(
            build_rotary_pos_embed(
                self.hidden_dim // self.num_heads,
                self.rope_temperature,
                grid_size=(H // self.patch_size, W // self.patch_size),
                pt_grid_size=self.pt_grid_size,
            ),
            dim=-1,
        ).to(self.rope.pos_embed.device)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        (H, W) = x.shape[-2:]
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        batch_special_tokens = self.class_token.expand(x.shape[0], -1, -1)
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            batch_special_tokens = torch.concat([batch_reg_tokens, batch_special_tokens], dim=1)

        if self.pos_embed_class is True:
            x = torch.concat([batch_special_tokens, x], dim=1)
            x = x + self._get_pos_embed(H, W)
        else:
            x = x + self._get_pos_embed(H, W)
            x = torch.concat([batch_special_tokens, x], dim=1)

        x = self.encoder(x, self._get_rope_embed(H, W))
        x = self.norm(x)

        x = x[:, self.num_special_tokens :]
        x = x.permute(0, 2, 1)
        (B, C, _) = x.size()
        x = x.reshape(B, C, self.size[0] // self.patch_size, self.size[1] // self.patch_size)

        return {self.return_stages[0]: x}

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.conv_proj.parameters():
            param.requires_grad = False

        self.pos_embedding.requires_grad = False

        for idx, module in enumerate(self.encoder.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        (H, W) = x.shape[-2:]

        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = self.patch_embed(x)

        # Expand the class token to the full batch
        batch_special_tokens = self.class_token.expand(x.shape[0], -1, -1)

        # Expand the register tokens to the full batch
        if self.reg_tokens is not None:
            batch_reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            batch_special_tokens = torch.concat([batch_reg_tokens, batch_special_tokens], dim=1)

        if self.pos_embed_class is True:
            x = torch.concat([batch_special_tokens, x], dim=1)
            x = x + self._get_pos_embed(H, W)
        else:
            x = x + self._get_pos_embed(H, W)
            x = torch.concat([batch_special_tokens, x], dim=1)

        x = self.encoder(x, self._get_rope_embed(H, W))
        x = self.norm(x)
        x = x[:, self.num_reg_tokens]

        return x

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        self.dynamic_size = dynamic_size

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        old_size = self.size
        super().adjust_size(new_size)

        # Sort out sizes
        if self.pos_embed_class is True:
            num_prefix_tokens = 1 + self.num_reg_tokens
        else:
            num_prefix_tokens = 0

        # Add back class tokens
        self.pos_embedding = nn.Parameter(
            adjust_position_embedding(
                self.pos_embedding,
                (old_size[0] // self.patch_size, old_size[1] // self.patch_size),
                (new_size[0] // self.patch_size, new_size[1] // self.patch_size),
                num_prefix_tokens,
            )
        )

        # Adjust RoPE
        self.rope = RoPE(
            self.hidden_dim // self.num_heads,
            temperature=self.rope_temperature,
            grid_size=(new_size[0] // self.patch_size, new_size[1] // self.patch_size),
            pt_grid_size=self.pt_grid_size,
        )


registry.register_alias(
    "rope_deit3_t16",
    RoPE_DeiT3,
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
    "rope_deit3_s16",
    RoPE_DeiT3,
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
    "rope_deit3_s14",
    RoPE_DeiT3,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "drop_path_rate": 0.05,
    },
)
registry.register_alias(
    "rope_deit3_m16",
    RoPE_DeiT3,
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
    "rope_deit3_m14",
    RoPE_DeiT3,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_deit3_b16",
    RoPE_DeiT3,
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
    "rope_deit3_b14",
    RoPE_DeiT3,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.2,
    },
)
registry.register_alias(
    "rope_deit3_l16",
    RoPE_DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "drop_path_rate": 0.45,
    },
)

# With registers
registry.register_alias(
    "rope_deit3_reg4_t16",
    RoPE_DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 3,
        "hidden_dim": 192,
        "mlp_dim": 768,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "rope_deit3_reg4_s16",
    RoPE_DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.05,
    },
)
registry.register_alias(
    "rope_deit3_reg4_s14",
    RoPE_DeiT3,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 6,
        "hidden_dim": 384,
        "mlp_dim": 1536,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.05,
    },
)
registry.register_alias(
    "rope_deit3_reg4_m16",
    RoPE_DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_deit3_reg4_m14",
    RoPE_DeiT3,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 8,
        "hidden_dim": 512,
        "mlp_dim": 2048,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "rope_deit3_reg4_b16",
    RoPE_DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.2,
    },
)
registry.register_alias(
    "rope_deit3_reg4_b14",
    RoPE_DeiT3,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.2,
    },
)
registry.register_alias(
    "rope_deit3_reg4_l16",
    RoPE_DeiT3,
    config={
        "patch_size": 16,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_reg_tokens": 4,
        "drop_path_rate": 0.45,
    },
)

registry.register_weights(
    "rope_deit3_reg4_t16_il-common",
    {
        "description": "RoPE DeiT3 reg4 tiny p16 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 21.5,
                "sha256": "3c0e1500d062d75f1b3c5f1aae5015c48b0736521c5289d039da133eefc3519f",
            }
        },
        "net": {"network": "rope_deit3_reg4_t16", "tag": "il-common"},
    },
)
registry.register_weights(
    "rope_deit3_reg4_m14_arabian-peninsula",
    {
        "url": "https://huggingface.co/birder-project/rope_deit3_reg4_m14_arabian-peninsula/resolve/main",
        "description": "RoPE DeiT3 reg4 medium p14 model trained on the arabian-peninsula dataset",
        "resolution": (252, 252),
        "formats": {
            "pt": {
                "file_size": 147.7,
                "sha256": "596223dde050561e2045352d4c0816ef874b9e8ccc6e5157f9e112cecfa9fb8c",
            }
        },
        "net": {"network": "rope_deit3_reg4_m14", "tag": "arabian-peninsula"},
    },
)
