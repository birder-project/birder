"""
Capture sparse MoE routing as patch-by-expert tensors
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch import nn

from birder.data.transforms.classification import RGBType
from birder.introspection.base import preprocess_image
from birder.layers.moe import BaseSparseMoE_FFN
from birder.layers.moe import ExpertChoiceRouter
from birder.layers.moe import MoE_FFN
from birder.layers.moe import NoisyTopKRouter
from birder.layers.moe import SigmoidTopKRouter


@dataclass(frozen=True)
class MoELayerRouting:
    """
    Routing for one encoder layer, with tensors shaped (B, grid_h, grid_w, num_experts)

    All tensors are stored on CPU without autograd history. Expert IDs index the last
    dimension and are local to this layer. Only image patches are included, register and class tokens
    are excluded even when they participate in routing.

    scores contains the original router affinities for every expert: sigmoid scores for
    SigmoidTopKRouter and softmax probabilities for the other routers. V-MoE scores are
    captured before router noise, selected and weights reflect the actual noisy routing.
    selected marks the router's choices before capacity dropping. assignments marks dispatched choices, so
    V-MoE choices with zero combine weight are excluded. For the other routers, all selected
    choices are dispatched, including choices whose scores underflow to zero.

    weights contains the router's combine weights, with zero at unselected or dropped
    choices. Sigmoid top-k weights are normalized over selected experts, the other routers
    retain their softmax probabilities. Shared and special-token experts are not represented here.
    """

    router_type: str
    scores: torch.Tensor
    selected: torch.Tensor
    assignments: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class MoERoutingResult:
    """
    Model output and routing from one forward pass
    """

    logits: torch.Tensor
    layers: dict[int, MoELayerRouting]
    patch_grid_shape: tuple[int, int]
    original_image: npt.NDArray[np.float32]

    def show(
        self,
        layer_idx: Optional[int] = None,
        sample_idx: int = 0,
        *,
        page: Optional[int] = None,
        experts_per_page: int = 18,
    ) -> None:
        image = self.original_image
        if layer_idx is None:
            layer_idx = max(self.layers)

        expert_weights = self.layers[layer_idx].weights[sample_idx].numpy()
        num_experts = expert_weights.shape[-1]
        num_pages = (num_experts + experts_per_page - 1) // experts_per_page
        if page is not None and page > num_pages:
            raise ValueError(f"page must be in range [1, {num_pages}], got {page}")

        height, width = image.shape[:2]
        extent = (0, width, height, 0)
        vmax = expert_weights.max(initial=0)

        pages = range(1, num_pages + 1) if page is None else (page,)
        for current_page in pages:
            start = (current_page - 1) * experts_per_page
            end = min(start + experts_per_page, num_experts)
            page_weights = expert_weights[:, :, start:end]
            num_displayed_experts = end - start
            ncols = min(6, num_displayed_experts)
            nrows = (num_displayed_experts + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False, layout="constrained")
            fig.suptitle(
                f"Layer {layer_idx} - Expert weights\n"
                f"Page {current_page}/{num_pages} - Experts {start}-{end - 1} of {num_experts}"
            )
            for idx, ax in enumerate(axes.flat):
                ax.set_axis_off()
                if idx >= num_displayed_experts:
                    ax.set_visible(False)
                    continue

                ax.imshow(image, extent=extent, origin="upper")
                ax.imshow(
                    np.ma.masked_equal(page_weights[:, :, idx], 0),
                    extent=extent,
                    origin="upper",
                    cmap="inferno",
                    interpolation="nearest",
                    alpha=0.65,
                    vmin=0,
                    vmax=vmax,
                )
                ax.set_title(f"Expert {start + idx}")

            plt.show()


class MoERouting:
    """
    Inspect routed-expert choices and weights for an image
    """

    def __init__(
        self, net: nn.Module, device: torch.device, transform: Callable[..., torch.Tensor], rgb_stats: RGBType
    ) -> None:
        self.net = net.eval()
        self.device = device
        self.transform = transform
        self.rgb_stats = rgb_stats

    def __call__(self, image: str | Path | Image.Image) -> MoERoutingResult:
        """
        Preprocess the image and capture routed-expert choices from a forward pass
        """

        input_tensor, rgb_img = preprocess_image(image, self.transform, self.device, self.rgb_stats)
        B, _, H, W = input_tensor.shape
        patch_grid_shape = (H // self.net.patch_size, W // self.net.patch_size)
        ffns = {
            idx: block.mlp
            for idx, block in enumerate(self.net.encoder.block)
            if isinstance(block.mlp, BaseSparseMoE_FFN)
        }
        if len(ffns) == 0:
            raise ValueError("Model has no sparse MoE layers")

        scores_by_layer: dict[int, torch.Tensor] = {}
        layers: dict[int, MoELayerRouting] = {}

        def capture_scores(
            layer_idx: int, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], logits: torch.Tensor
        ) -> None:
            router: nn.Module = ffns[layer_idx].router
            if isinstance(router, SigmoidTopKRouter):
                scores = logits.float().sigmoid()
            elif isinstance(router, NoisyTopKRouter):
                scores = logits.softmax(dim=-1)
            elif isinstance(router, ExpertChoiceRouter):
                scores = logits.float().softmax(dim=-1)
            else:
                raise TypeError(f"Unsupported router type: {type(router).__name__}")

            scores_by_layer[layer_idx] = scores.float().cpu()

        def capture_routing(
            layer_idx: int, router: nn.Module, _inputs: tuple[torch.Tensor, ...], output: tuple[Any, ...]
        ) -> None:
            scores = scores_by_layer.pop(layer_idx)
            indices = output[0].cpu()
            ffn = ffns[layer_idx]
            prefix = self.net.num_special_tokens
            if isinstance(ffn, MoE_FFN) and ffn.has_special_token_experts is True:
                prefix = 0

            expected_tokens = patch_grid_shape[0] * patch_grid_shape[1] + prefix
            if isinstance(router, NoisyTopKRouter):
                combine_weights = output[2].float().cpu()

                # Undo V-MoE grouping and remove trailing padding, including multi-image training groups
                num_tokens = B * expected_tokens
                scores = scores.flatten(0, 1)[:num_tokens].reshape(B, expected_tokens, -1)
                indices = indices.flatten(0, 1)[:num_tokens].reshape(B, expected_tokens, -1)
                combine_weights = combine_weights.flatten(0, 1)[:num_tokens].reshape(B, expected_tokens, -1)
            else:
                combine_weights = output[1].float().cpu()

            selected = torch.zeros_like(scores, dtype=torch.bool)
            weights = torch.zeros_like(scores)
            if isinstance(router, ExpertChoiceRouter):
                # Each expert returns token indices, invert to the common token-major layout
                indices = indices.transpose(-2, -1)
                combine_weights = combine_weights.transpose(-2, -1)
                selected.scatter_(1, indices, True)
                weights.scatter_(1, indices, combine_weights)
            else:
                selected.scatter_(2, indices, True)
                weights.scatter_(2, indices, combine_weights)

            assignments = weights > 0 if isinstance(router, NoisyTopKRouter) else selected.clone()
            if scores.shape[:2] != (B, expected_tokens):
                raise ValueError(f"Layer {layer_idx} router tokens do not match the input patch grid")

            shape = (B, *patch_grid_shape, scores.size(-1))
            layers[layer_idx] = MoELayerRouting(
                router_type=type(router).__name__,
                scores=scores[:, prefix:].reshape(shape),
                selected=selected[:, prefix:].reshape(shape),
                assignments=assignments[:, prefix:].reshape(shape),
                weights=weights[:, prefix:].reshape(shape),
            )

        handles = []
        try:
            for layer_idx, ffn in ffns.items():
                handles.append(ffn.router.gate.register_forward_hook(partial(capture_scores, layer_idx)))
                handles.append(ffn.router.register_forward_hook(partial(capture_routing, layer_idx)))

            with torch.no_grad():
                logits = self.net(input_tensor)
        finally:
            for handle in handles:
                handle.remove()

        return MoERoutingResult(
            logits=logits.cpu(), layers=layers, patch_grid_shape=patch_grid_shape, original_image=rgb_img
        )
