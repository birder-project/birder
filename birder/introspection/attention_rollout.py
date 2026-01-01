"""
Attention Rollout for Vision Transformers, adapted from
https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py

Paper "Quantifying Attention Flow in Transformers", https://arxiv.org/abs/2005.00928
"""

# Reference license: MIT

from collections.abc import Callable
from pathlib import Path
from typing import Literal
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import nn

from birder.introspection.base import InterpretabilityResult
from birder.introspection.base import predict_class
from birder.introspection.base import preprocess_image
from birder.introspection.base import show_mask_on_image
from birder.net.vit import Encoder


# pylint: disable=too-many-locals
def compute_rollout(
    attentions: list[torch.Tensor],
    discard_ratio: float,
    head_fusion: Literal["mean", "max", "min"],
    num_special_tokens: int,
    patch_grid_shape: tuple[int, int],
) -> torch.Tensor:
    # Assume batch size = 1
    num_tokens = attentions[0].size(-1)
    device = attentions[0].device

    # Start with identity (residual)
    result = torch.eye(num_tokens, device=device)

    with torch.no_grad():
        for attention in attentions:
            # Fuse heads: [B, H, T, T] -> [B, T, T]
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(dim=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(dim=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(dim=1)[0]
            else:
                raise ValueError(f"Unsupported head_fusion: {head_fusion}")

            # attention_heads_fused: [1, T, T] (batch = 1)
            if discard_ratio > 0:
                # Work on the single batch element
                attn = attention_heads_fused[0]  # [T, T]

                # Define which positions are "non-special"
                idx = torch.arange(num_tokens, device=attn.device)
                is_special = idx < num_special_tokens
                non_special = ~is_special

                # We are only allowed to prune NON-special <-> NON-special entries
                allow = non_special[:, None] & non_special[None, :]  # [T, T]

                allowed_values = attn[allow]
                num_allowed = allowed_values.numel()
                if num_allowed > 0:
                    num_to_discard = int(num_allowed * discard_ratio)
                    if num_to_discard > 0:
                        # Drop the smallest allowed values
                        (_, low_idx) = torch.topk(allowed_values, num_to_discard, largest=False)
                        allowed_values[low_idx] = 0
                        attn[allow] = allowed_values
                        attention_heads_fused[0] = attn

            # Add residual connection and normalize
            eye = torch.eye(num_tokens, device=attention_heads_fused.device)
            a = (attention_heads_fused + eye) / 2.0  # [1, T, T]
            a = a / a.sum(dim=-1, keepdim=True)

            # Accumulate attention across layers
            result = torch.matmul(a, result)  # [1, T, T]

    rollout = result[0]  # [T, T]

    # Build final token → patch map
    if 0 < num_special_tokens < num_tokens:
        # Sources: all special tokens (0 .. num_special_tokens-1)
        # Targets: all non-special tokens (num_special_tokens .. end)
        source_to_patches = rollout[:num_special_tokens, num_special_tokens:]
        mask = source_to_patches.mean(dim=0)
    else:
        # No special tokens (or all are special): fall back to averaging over all sources
        mask = rollout.mean(dim=0)  # [T]

    # Normalize and reshape to 2D map using actual patch grid dimensions
    mask = mask / (mask.max() + 1e-8)
    (grid_h, grid_w) = patch_grid_shape
    mask = mask.reshape(grid_h, grid_w)

    return mask


class AttentionGatherer:
    def __init__(self, net: nn.Module, attention_layer_name: str) -> None:
        assert hasattr(net, "encoder") is True and isinstance(net.encoder, Encoder)

        net.encoder.set_need_attn()
        self.net = net
        self.attentions: list[torch.Tensor] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

        # Register hooks on attention layers
        for name, module in self.net.named_modules():
            if name.endswith(attention_layer_name) is True:
                handle = module.register_forward_hook(self._capture_attention)
                self.handles.append(handle)

    def _capture_attention(
        self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]
    ) -> None:
        self.attentions.append(outputs[1].cpu())

    def __call__(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        self.attentions = []
        with torch.inference_mode():
            logits = self.net(x)

        return (self.attentions, logits)

    def release(self) -> None:
        for handle in self.handles:
            handle.remove()


class AttentionRollout:
    def __init__(
        self,
        net: nn.Module,
        device: torch.device,
        transform: Callable[..., torch.Tensor],
        attention_layer_name: str = "self_attention",
        discard_ratio: float = 0.9,
        head_fusion: Literal["mean", "max", "min"] = "max",
    ) -> None:
        if not 0 <= discard_ratio <= 1:
            raise ValueError(f"discard_ratio must be in [0, 1], got {discard_ratio}")

        self.net = net.eval()
        self.device = device
        self.transform = transform
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        self.attention_gatherer = AttentionGatherer(net, attention_layer_name)

    def __call__(self, image: str | Path | Image.Image, target_class: Optional[int] = None) -> InterpretabilityResult:
        (input_tensor, rgb_img) = preprocess_image(image, self.transform, self.device)

        (attentions, logits) = self.attention_gatherer(input_tensor)

        (_, _, H, W) = input_tensor.shape
        patch_grid_shape = (H // self.net.stem_stride, W // self.net.stem_stride)

        attention_map = compute_rollout(
            attentions, self.discard_ratio, self.head_fusion, self.net.num_special_tokens, patch_grid_shape
        )
        attention_img = Image.fromarray(attention_map.numpy())
        attention_img = attention_img.resize((rgb_img.shape[1], rgb_img.shape[0]))
        attention_arr = np.array(attention_img)

        visualization = show_mask_on_image(rgb_img, attention_arr, image_weight=0.4)

        return InterpretabilityResult(
            original_image=rgb_img,
            visualization=visualization,
            raw_output=attention_arr,
            logits=logits.detach(),
            predicted_class=predict_class(logits),
        )

    def __del__(self) -> None:
        if hasattr(self, "attention_gatherer") is True:
            self.attention_gatherer.release()
