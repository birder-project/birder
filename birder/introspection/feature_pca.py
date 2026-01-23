from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

from birder.introspection.base import InterpretabilityResult
from birder.introspection.base import preprocess_image
from birder.net.base import DetectorBackbone


class FeaturePCA:
    """
    Visualizes feature maps using Principal Component Analysis

    This method extracts feature maps from a specified stage of a DetectorBackbone model,
    applies PCA to reduce the channel dimension to 3 components and visualizes them as an RGB image where:
    - R channel = 1st principal component (most important)
    - G channel = 2nd principal component
    - B channel = 3rd principal component
    """

    def __init__(
        self,
        net: DetectorBackbone,
        device: torch.device,
        transform: Callable[..., torch.Tensor],
        normalize: bool = False,
        channels_last: bool = False,
        stage: Optional[str] = None,
    ) -> None:
        self.net = net.eval()
        self.device = device
        self.transform = transform
        self.normalize = normalize
        self.channels_last = channels_last
        self.stage = stage

    def __call__(self, image: str | Path | Image.Image) -> InterpretabilityResult:
        input_tensor, rgb_img = preprocess_image(image, self.transform, self.device)

        with torch.inference_mode():
            features_dict = self.net.detection_features(input_tensor)

        if self.stage is not None:
            features = features_dict[self.stage]
        else:
            features = list(features_dict.values())[-1]  # Use the last stage by default

        features_np = features.cpu().numpy()

        # Handle channels_last format (B, H, W, C) vs channels_first (B, C, H, W)
        if self.channels_last is True:
            B, H, W, C = features_np.shape
            # Already in (B, H, W, C), just reshape to (B*H*W, C)
            features_reshaped = features_np.reshape(-1, C)
        else:
            B, C, H, W = features_np.shape
            # Reshape to (spatial_points, channels) for PCA
            features_reshaped = features_np.reshape(B, C, -1)
            features_reshaped = features_reshaped.transpose(0, 2, 1)  # (B, H*W, C)
            features_reshaped = features_reshaped.reshape(-1, C)  # (B*H*W, C)

        x = features_reshaped
        if self.normalize is True:
            x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)

        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(x)
        pca_features = pca_features.reshape(B, H, W, 3)

        # Extract all 3 components (B=1)
        pca_rgb = pca_features[0]  # (H, W, 3)

        # Normalize each channel independently to [0, 1]
        for i in range(3):
            channel = pca_rgb[:, :, i]
            channel = channel - channel.min()
            channel = channel / (channel.max() + 1e-7)
            pca_rgb[:, :, i] = channel

        target_size = (input_tensor.size(-1), input_tensor.size(-2))  # PIL expects (width, height)
        pca_rgb_resized = (
            np.array(
                Image.fromarray((pca_rgb * 255).astype(np.uint8)).resize(target_size, Image.Resampling.BILINEAR)
            ).astype(np.float32)
            / 255.0
        )

        visualization = (pca_rgb_resized * 255).astype(np.uint8)

        return InterpretabilityResult(
            original_image=rgb_img,
            visualization=visualization,
            raw_output=pca_rgb.astype(np.float32),
            logits=None,
            predicted_class=None,
        )
