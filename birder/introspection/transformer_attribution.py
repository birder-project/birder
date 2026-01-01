"""
Transformer Attribution (Gradient-weighted Attention Rollout), adapted from
https://github.com/hila-chefer/Transformer-Explainability

Paper "Transformer Interpretability Beyond Attention Visualization", https://arxiv.org/abs/2012.09838
"""

# Reference license: MIT

from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import nn

from birder.introspection.base import InterpretabilityResult
from birder.introspection.base import predict_class
from birder.introspection.base import preprocess_image
from birder.introspection.base import show_mask_on_image
from birder.introspection.base import validate_target_class
from birder.net.vit import Encoder


def compute_attribution_rollout(
    attributions: list[tuple[torch.Tensor, torch.Tensor]], num_special_tokens: int, patch_grid_shape: tuple[int, int]
) -> torch.Tensor:
    """
    NOTE: Uses gradient norm per token instead of element-wise grad * attention multiplication.
    """

    # Assume batch size = 1
    num_tokens = attributions[0][0].size(-1)
    device = attributions[0][0].device

    result = torch.eye(num_tokens, device=device)
    with torch.no_grad():
        for attn_weights, output_grad in attributions:
            # Compute token importance from output gradient norm across embedding dimension
            token_importance = output_grad.norm(dim=-1, keepdim=True)
            token_importance = token_importance.transpose(-1, -2)

            # Weight attention patterns by token importance
            weighted_attn = attn_weights * token_importance.unsqueeze(1)

            # Fuse attention heads and apply non-negativity constraint
            relevance = weighted_attn.mean(dim=1).clamp(min=0)

            # Add residual connection and normalize
            eye = torch.eye(num_tokens, device=device)
            normalized = (relevance + eye) / 2.0
            normalized = normalized / normalized.sum(dim=-1, keepdim=True)

            # Accumulate attention across layers
            result = torch.matmul(normalized, result)

    rollout = result[0]

    if 0 < num_special_tokens:
        source_to_patches = rollout[:num_special_tokens, num_special_tokens:]
        mask = source_to_patches.mean(dim=0)
    else:
        mask = rollout.mean(dim=0)

    mask = mask / (mask.max() + 1e-8)

    (grid_h, grid_w) = patch_grid_shape
    mask = mask.reshape(grid_h, grid_w)

    return mask


class AttributionGatherer:
    def __init__(self, net: nn.Module, attention_layer_name: str) -> None:
        assert hasattr(net, "encoder") is True and isinstance(net.encoder, Encoder)

        net.encoder.set_need_attn()

        self.net = net
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._gradients: list[torch.Tensor] = []
        self._attention_weights: list[torch.Tensor] = []

        for name, module in self.net.named_modules():
            if name.endswith(attention_layer_name) is True:
                handle = module.register_forward_hook(self._capture_forward)
                self.handles.append(handle)

    def _capture_forward(
        self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, ...] | torch.Tensor
    ) -> None:
        output_tensor = output[0]
        attn_weights = output[1]

        self._attention_weights.append(attn_weights.detach())
        if output_tensor.requires_grad:

            def _store_grad(grad: torch.Tensor) -> None:
                self._gradients.append(grad.detach())

            output_tensor.register_hook(_store_grad)

    def get_captured_data(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if len(self._attention_weights) != len(self._gradients):
            raise RuntimeError(
                f"Mismatch between attention weights ({len(self._attention_weights)}) "
                f"and gradients ({len(self._gradients)}). Ensure backward() was called."
            )

        if len(self._attention_weights) == 0:
            raise RuntimeError("No attention data captured. Ensure the model has attention layers.")

        # Pair attention weights with output gradients (gradients reversed to match forward order)
        results = [(attn.cpu(), grad.cpu()) for attn, grad in zip(self._attention_weights, reversed(self._gradients))]

        # Clear storage for next forward pass
        self._gradients = []
        self._attention_weights = []

        return results

    def release(self) -> None:
        for handle in self.handles:
            handle.remove()


class TransformerAttribution:
    def __init__(
        self,
        net: nn.Module,
        device: torch.device,
        transform: Callable[..., torch.Tensor],
        attention_layer_name: str = "self_attention",
    ) -> None:
        self.net = net.eval()
        self.device = device
        self.transform = transform
        self.gatherer = AttributionGatherer(net, attention_layer_name)

    def __call__(self, image: str | Path | Image.Image, target_class: Optional[int] = None) -> InterpretabilityResult:
        (input_tensor, rgb_img) = preprocess_image(image, self.transform, self.device)
        input_tensor.requires_grad_(True)

        self.net.zero_grad()
        logits = self.net(input_tensor)

        if target_class is None:
            target_class = predict_class(logits)
        else:
            validate_target_class(target_class, logits.shape[-1])

        score = logits[0, target_class]
        score.backward()

        attribution_data = self.gatherer.get_captured_data()

        (_, _, H, W) = input_tensor.shape
        patch_grid_shape = (H // self.net.stem_stride, W // self.net.stem_stride)

        attribution_map = compute_attribution_rollout(
            attribution_data, num_special_tokens=self.net.num_special_tokens, patch_grid_shape=patch_grid_shape
        )

        attribution_img = Image.fromarray(attribution_map.numpy())
        attribution_img = attribution_img.resize((rgb_img.shape[1], rgb_img.shape[0]))
        attribution_arr = np.array(attribution_img)

        visualization = show_mask_on_image(rgb_img, attribution_arr, image_weight=0.4)

        return InterpretabilityResult(
            original_image=rgb_img,
            visualization=visualization,
            raw_output=attribution_arr,
            logits=logits.detach(),
            predicted_class=target_class,
        )

    def __del__(self) -> None:
        if hasattr(self, "gatherer") is True:
            self.gatherer.release()
