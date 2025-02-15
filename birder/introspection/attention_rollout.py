"""
Adapted from https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py
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
from birder.introspection.base import Interpreter
from birder.introspection.base import show_mask_on_image
from birder.net.vit import Encoder


def rollout(
    attentions: list[torch.Tensor],
    discard_ratio: float,
    head_fusion: Literal["mean", "max", "min"],
    num_special_tokens: int,
) -> torch.Tensor:
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise ValueError("Attention head fusion type Not supported")

            # Drop the lowest attentions, but don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            (_, indices) = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            eye = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * eye) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token and the image patches
    mask = result[0, 0, num_special_tokens:]

    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width)
    mask = mask / torch.max(mask)

    return mask


class AttentionRollout:
    def __init__(self, net: torch.nn.Module, attention_layer_name: str) -> None:
        assert hasattr(net, "encoder") is True and isinstance(net.encoder, Encoder)
        net.encoder.set_need_attn()
        self.net = net
        for name, module in self.net.named_modules():
            if name.endswith(attention_layer_name) is True:
                module.register_forward_hook(self.get_attention)

        self.attentions: list[torch.Tensor] = []

    def get_attention(
        self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]
    ) -> None:
        self.attentions.append(outputs[1].cpu())

    def __call__(
        self, x: torch.Tensor, discard_ratio: float, head_fusion: Literal["mean", "max", "min"]
    ) -> torch.Tensor:
        self.attentions = []
        with torch.inference_mode():
            self.net(x)

        return rollout(self.attentions, discard_ratio, head_fusion, self.net.num_special_tokens)


class AttentionRolloutInterpreter(Interpreter):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        transform: Callable[..., torch.Tensor],
        attention_layer_name: str,
        discard_ratio: float,
        head_fusion: Literal["mean", "max", "min"],
    ) -> None:
        super().__init__(model, device, transform)
        self.attention_rollout = AttentionRollout(model, attention_layer_name)
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion

    def interpret(self, image: str | Path | Image.Image, target_class: Optional[int] = None) -> InterpretabilityResult:
        (input_tensor, rgb_img) = self._preprocess_image(image)

        attention_map = self.attention_rollout(
            input_tensor, discard_ratio=self.discard_ratio, head_fusion=self.head_fusion
        )

        # Resize attention map to match image size
        attention_img = Image.fromarray(attention_map.numpy())
        attention_img = attention_img.resize(rgb_img.shape[:2])  # type: ignore[arg-type]
        attention_arr = np.array(attention_img)
        visualization = show_mask_on_image(rgb_img, attention_arr, image_weight=0.4)

        return InterpretabilityResult(rgb_img, visualization, raw_output=attention_arr)
