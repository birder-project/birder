import logging
import unittest

import numpy as np
import torch
from PIL import Image
from torch import nn

from birder.conf.settings import DEFAULT_NUM_CHANNELS
from birder.introspection import base
from birder.introspection.attention_rollout import AttentionRollout
from birder.introspection.gradcam import GradCAM
from birder.introspection.guided_backprop import GuidedBackprop
from birder.introspection.transformer_attribution import TransformerAttribution
from birder.model_registry import registry

logging.disable(logging.CRITICAL)


class _TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        return self.fc(x)


class TestIntrospectionBase(unittest.TestCase):
    def test_show_mask_on_image(self) -> None:
        img = np.random.rand(16, 16, 3).astype(np.float32)
        mask = np.random.rand(16, 16).astype(np.float32)

        result = base.show_mask_on_image(img, mask, image_weight=0.5)

        self.assertEqual(result.shape, (16, 16, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))

    def test_scale_cam_image(self) -> None:
        cam = np.random.rand(2, 8, 8).astype(np.float32)

        result = base.scale_cam_image(cam, target_size=None)

        self.assertEqual(result.shape, (2, 8, 8))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

        result_resized = base.scale_cam_image(cam, target_size=(16, 16))

        self.assertEqual(result_resized.shape, (2, 16, 16))

    def test_deprocess_image(self) -> None:
        img = np.random.randn(16, 16, 3).astype(np.float32)

        result = base.deprocess_image(img)

        self.assertEqual(result.shape, (16, 16, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))

    def test_validate_target_class(self) -> None:
        # Valid target
        base.validate_target_class(0, num_classes=2)
        base.validate_target_class(1, num_classes=2)

        # None is valid
        base.validate_target_class(None, num_classes=2)

        # Invalid: out of range
        with self.assertRaises(ValueError):
            base.validate_target_class(2, num_classes=2)

        with self.assertRaises(ValueError):
            base.validate_target_class(-1, num_classes=2)

    def test_predict_class(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 0.5]])
        pred = base.predict_class(logits)
        self.assertEqual(pred, 1)

        logits = torch.tensor([[2.0, 1.0]])
        pred = base.predict_class(logits)
        self.assertEqual(pred, 0)

    def test_preprocess_image(self) -> None:
        # Create test image
        img = Image.new("RGB", (32, 32), color=(255, 0, 0))

        def simple_transform(_x: Image.Image) -> torch.Tensor:
            return torch.rand(3, 16, 16)

        device = torch.device("cpu")
        (input_tensor, rgb_img) = base.preprocess_image(img, simple_transform, device)

        self.assertEqual(input_tensor.shape, (1, 3, 16, 16))
        self.assertEqual(rgb_img.shape, (16, 16, 3))
        self.assertTrue(np.all(rgb_img >= 0))
        self.assertTrue(np.all(rgb_img <= 1))


class TestInterpreters(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cpu")

        # Create test image
        self.test_image = Image.new("RGB", (16, 16), color=(128, 128, 128))

        def simple_transform(x: Image.Image) -> torch.Tensor:
            arr = np.array(x).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        self.transform = simple_transform

    def test_gradcam_result_structure(self) -> None:
        net = _TinyCNN()
        target_layer = net.conv2

        interpreter = GradCAM(net, self.device, self.transform, target_layer)
        result = interpreter(self.test_image, target_class=None)

        # Check result structure
        self.assertIsInstance(result.original_image, np.ndarray)
        self.assertIsInstance(result.visualization, np.ndarray)
        self.assertIsInstance(result.raw_output, np.ndarray)
        self.assertIsInstance(result.logits, torch.Tensor)
        self.assertIsInstance(result.predicted_class, int)

        # Check shapes
        self.assertEqual(len(result.original_image.shape), 3)
        self.assertEqual(len(result.visualization.shape), 3)
        self.assertEqual(result.logits.shape[-1], 2)  # type: ignore[union-attr]
        self.assertIn(result.predicted_class, [0, 1])

    def test_gradcam_with_target_class(self) -> None:
        net = _TinyCNN()
        target_layer = net.conv2

        interpreter = GradCAM(net, self.device, self.transform, target_layer)
        result = interpreter(self.test_image, target_class=1)

        self.assertEqual(result.predicted_class, 1)

    def test_gradcam_invalid_target_class(self) -> None:
        net = _TinyCNN()
        target_layer = net.conv2

        interpreter = GradCAM(net, self.device, self.transform, target_layer)

        with self.assertRaises(ValueError):
            interpreter(self.test_image, target_class=5)

        with self.assertRaises(ValueError):
            interpreter(self.test_image, target_class=-1)

    def test_guided_backprop_result_structure(self) -> None:
        net = _TinyCNN()

        interpreter = GuidedBackprop(net, self.device, self.transform)
        result = interpreter(self.test_image, target_class=None)

        # Check result structure
        self.assertIsInstance(result.original_image, np.ndarray)
        self.assertIsInstance(result.visualization, np.ndarray)
        self.assertIsInstance(result.raw_output, np.ndarray)
        self.assertIsInstance(result.logits, torch.Tensor)
        self.assertIsInstance(result.predicted_class, int)

        # Check visualization is uint8
        self.assertEqual(result.visualization.dtype, np.uint8)
        self.assertTrue(np.all(result.visualization >= 0))
        self.assertTrue(np.all(result.visualization <= 255))

    def test_guided_backprop_model_restoration(self) -> None:
        net = _TinyCNN()
        original_relu_count = sum(1 for m in net.modules() if isinstance(m, nn.ReLU))

        interpreter = GuidedBackprop(net, self.device, self.transform)
        _ = interpreter(self.test_image, target_class=0)

        # Check model is restored
        restored_relu_count = sum(1 for m in net.modules() if isinstance(m, nn.ReLU))
        self.assertEqual(original_relu_count, restored_relu_count)

    def test_attention_rollout_result_structure(self) -> None:
        net = registry.net_factory("vit_t16", DEFAULT_NUM_CHANNELS, 2, size=(160, 160))

        # Create transform that resizes to match model input
        def vit_transform(x: Image.Image) -> torch.Tensor:
            x = x.resize((160, 160))
            arr = np.array(x).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        interpreter = AttentionRollout(
            net, self.device, vit_transform, attention_layer_name="self_attention", discard_ratio=0.9, head_fusion="max"
        )
        result = interpreter(self.test_image, target_class=None)

        # Check result structure
        self.assertIsInstance(result.original_image, np.ndarray)
        self.assertIsInstance(result.visualization, np.ndarray)
        self.assertIsInstance(result.raw_output, np.ndarray)
        self.assertIsInstance(result.logits, torch.Tensor)
        self.assertIsInstance(result.predicted_class, int)

        # Check shapes
        self.assertEqual(len(result.original_image.shape), 3)
        self.assertEqual(len(result.visualization.shape), 3)
        self.assertEqual(result.logits.shape[-1], 2)  # type: ignore[union-attr]
        self.assertIn(result.predicted_class, [0, 1])

    def test_attention_rollout_no_duplicate_attentions(self) -> None:
        net = registry.net_factory("vit_t16", DEFAULT_NUM_CHANNELS, 2, size=(160, 160))

        # Create transform that resizes to match model input
        def vit_transform(x: Image.Image) -> torch.Tensor:
            x = x.resize((160, 160))
            arr = np.array(x).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        # Track forward pass count
        original_forward = net.forward
        forward_count = {"count": 0}

        def counting_forward(x: torch.Tensor) -> torch.Tensor:
            forward_count["count"] += 1
            return original_forward(x)

        net.forward = counting_forward  # type: ignore[method-assign]

        interpreter = AttentionRollout(
            net, self.device, vit_transform, attention_layer_name="self_attention", discard_ratio=0.9, head_fusion="max"
        )

        # Track attention list length during execution
        attention_gatherer = interpreter.attention_gatherer
        original_call = attention_gatherer.__class__.__call__

        attention_lengths = []

        def tracking_call(self: object, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
            result = original_call(self, x)  # type: ignore[arg-type]
            attention_lengths.append(len(result[0]))
            return result

        attention_gatherer.__class__.__call__ = tracking_call  # type: ignore[method-assign]

        # Run interpreter
        _ = interpreter(self.test_image, target_class=None)

        # Restore original methods
        attention_gatherer.__class__.__call__ = original_call  # type: ignore[method-assign]
        net.forward = original_forward  # type: ignore[method-assign]

        # Verify only one forward pass occurred
        self.assertEqual(forward_count["count"], 1, "Should only perform ONE forward pass, not two!")

        # Verify attention list wasn't polluted
        self.assertEqual(len(attention_lengths), 1, "Attention gatherer should be called exactly once")
        num_encoder_layers = len([m for m in net.modules() if hasattr(m, "self_attention")])
        self.assertEqual(
            attention_lengths[0],
            num_encoder_layers,
            f"Should have {num_encoder_layers} attention maps, one per encoder layer",
        )

    def test_transformer_attribution_result_structure(self) -> None:
        net = registry.net_factory("vit_t16", DEFAULT_NUM_CHANNELS, 2, size=(160, 160))

        def vit_transform(x: Image.Image) -> torch.Tensor:
            x = x.resize((160, 160))
            arr = np.array(x).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        interpreter = TransformerAttribution(net, self.device, vit_transform, attention_layer_name="self_attention")
        result = interpreter(self.test_image, target_class=None)

        self.assertIsInstance(result.original_image, np.ndarray)
        self.assertIsInstance(result.visualization, np.ndarray)
        self.assertIsInstance(result.raw_output, np.ndarray)
        self.assertIsInstance(result.logits, torch.Tensor)
        self.assertIsInstance(result.predicted_class, int)

        self.assertEqual(len(result.original_image.shape), 3)
        self.assertEqual(len(result.visualization.shape), 3)
        self.assertEqual(result.logits.shape[-1], 2)  # type: ignore[union-attr]
        self.assertIn(result.predicted_class, [0, 1])

    def test_transformer_attribution_with_target_class(self) -> None:
        net = registry.net_factory("vit_t16", DEFAULT_NUM_CHANNELS, 2, size=(160, 160))

        def vit_transform(x: Image.Image) -> torch.Tensor:
            x = x.resize((160, 160))
            arr = np.array(x).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        interpreter = TransformerAttribution(net, self.device, vit_transform, attention_layer_name="self_attention")
        result = interpreter(self.test_image, target_class=1)

        self.assertEqual(result.predicted_class, 1)
