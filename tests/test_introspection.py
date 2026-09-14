import logging
import unittest

import numpy as np
import torch
from PIL import Image
from torch import nn

from birder.data.transforms.classification import get_rgb_stats
from birder.data.transforms.classification import inference_preset
from birder.introspection import MoERouting
from birder.introspection import base
from birder.introspection.attention_rollout import AttentionRollout
from birder.introspection.feature_pca import FeaturePCA
from birder.introspection.gradcam import GradCAM
from birder.introspection.guided_backprop import GuidedBackprop
from birder.introspection.transformer_attribution import AttributionGatherer
from birder.introspection.transformer_attribution import TransformerAttribution
from birder.introspection.transformer_attribution import compute_attribution_rollout
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stage1 = self.relu(self.conv1(x))
        stage2 = self.relu(self.conv2(stage1))
        return {"stage1": stage1, "stage2": stage2}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
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
        input_tensor, rgb_img = base.preprocess_image(img, simple_transform, device, get_rgb_stats("neutral"))

        self.assertEqual(input_tensor.shape, (1, 3, 16, 16))
        self.assertEqual(rgb_img.shape, (16, 16, 3))
        self.assertTrue(np.all(rgb_img >= 0))
        self.assertTrue(np.all(rgb_img <= 1))


class TestInterpreters(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.rgb_stats = get_rgb_stats("neutral")

        # Create test image
        self.test_image = Image.new("RGB", (16, 16), color=(128, 128, 128))

        def simple_transform(x: Image.Image) -> torch.Tensor:
            arr = np.array(x).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        self.transform = simple_transform

    def test_attention_rollout_result_structure(self) -> None:
        net = registry.net_factory("vit_t16", 2, size=(160, 160))

        # Create transform that resizes to match model input
        def vit_transform(x: Image.Image) -> torch.Tensor:
            x = x.resize((160, 160))
            arr = np.array(x).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        interpreter = AttentionRollout(
            net,
            self.device,
            vit_transform,
            self.rgb_stats,
            attention_layer_name="attn",
            discard_ratio=0.9,
            head_fusion="max",
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
        net = registry.net_factory("vit_t16", 2, size=(160, 160))

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
            net,
            self.device,
            vit_transform,
            self.rgb_stats,
            attention_layer_name="attn",
            discard_ratio=0.9,
            head_fusion="max",
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
        num_encoder_layers = len([m for m in net.modules() if hasattr(m, "attn")])
        self.assertEqual(
            attention_lengths[0],
            num_encoder_layers,
            f"Should have {num_encoder_layers} attention maps, one per encoder layer",
        )

    def test_feature_pca_result_structure(self) -> None:
        net = _TinyCNN()

        interpreter = FeaturePCA(net, self.device, self.transform, self.rgb_stats)
        result = interpreter(self.test_image)

        self.assertIsInstance(result.original_image, np.ndarray)
        self.assertIsInstance(result.visualization, np.ndarray)
        self.assertIsInstance(result.raw_output, np.ndarray)
        self.assertIsNone(result.logits)
        self.assertIsNone(result.predicted_class)

        self.assertEqual(len(result.original_image.shape), 3)
        self.assertEqual(len(result.visualization.shape), 3)
        self.assertEqual(result.visualization.shape[-1], 3)  # RGB channels
        self.assertEqual(result.visualization.dtype, np.uint8)

        self.assertEqual(len(result.raw_output.shape), 3)
        self.assertEqual(result.raw_output.shape[-1], 3)
        self.assertEqual(result.raw_output.dtype, np.float32)

    def test_feature_pca_values_normalized(self) -> None:
        net = _TinyCNN()

        interpreter = FeaturePCA(net, self.device, self.transform, self.rgb_stats)
        result = interpreter(self.test_image)

        self.assertTrue(np.all(result.raw_output >= 0))
        self.assertTrue(np.all(result.raw_output <= 1))

        self.assertTrue(np.all(result.visualization >= 0))
        self.assertTrue(np.all(result.visualization <= 255))

    def test_gradcam_result_structure(self) -> None:
        net = _TinyCNN()
        target_layer = net.conv2

        interpreter = GradCAM(net, self.device, self.transform, self.rgb_stats, target_layer)
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

        interpreter = GradCAM(net, self.device, self.transform, self.rgb_stats, target_layer)
        result = interpreter(self.test_image, target_class=1)

        self.assertEqual(result.predicted_class, 1)

    def test_gradcam_invalid_target_class(self) -> None:
        net = _TinyCNN()
        target_layer = net.conv2

        interpreter = GradCAM(net, self.device, self.transform, self.rgb_stats, target_layer)

        with self.assertRaises(ValueError):
            interpreter(self.test_image, target_class=5)

        with self.assertRaises(ValueError):
            interpreter(self.test_image, target_class=-1)

    def test_guided_backprop_result_structure(self) -> None:
        net = _TinyCNN()

        interpreter = GuidedBackprop(net, self.device, self.transform, self.rgb_stats)
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
        net.relu = nn.ReLU(inplace=True)
        original_relu = net.relu

        interpreter = GuidedBackprop(net, self.device, self.transform, self.rgb_stats)
        _ = interpreter(self.test_image, target_class=0)

        self.assertIs(net.relu, original_relu)
        self.assertTrue(net.relu.inplace)

    def test_moe_routing_result_structure(self) -> None:
        net = registry.net_factory("vit_moe_t16_4e1s1p_2k_last1o1", 2, size=(32, 48))
        transform = inference_preset((32, 48), self.rgb_stats)
        interpreter = MoERouting(net, self.device, transform, self.rgb_stats)
        result = interpreter(self.test_image)

        self.assertIsInstance(result.logits, torch.Tensor)
        self.assertEqual(result.logits.shape, (1, 2))
        self.assertEqual(result.logits.device, self.device)
        self.assertFalse(result.logits.requires_grad)
        self.assertEqual(result.patch_grid_shape, (2, 3))
        self.assertEqual(list(result.layers), [net.num_layers - 2])
        self.assertIsInstance(result.original_image, np.ndarray)
        self.assertEqual(result.original_image.shape, (32, 48, 3))

        layer = result.layers[net.num_layers - 2]
        self.assertEqual(layer.router_type, "SigmoidTopKRouter")
        for tensor in (layer.scores, layer.selected, layer.assignments, layer.weights):
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.shape, (1, 2, 3, 4))
            self.assertEqual(tensor.device, self.device)
            self.assertFalse(tensor.requires_grad)

        torch.testing.assert_close(layer.selected, layer.assignments)
        torch.testing.assert_close(layer.assignments.sum(dim=-1), torch.full((1, 2, 3), 2, dtype=torch.int64))
        torch.testing.assert_close(layer.weights.sum(dim=-1), torch.ones(1, 2, 3))

    def test_moe_routing_expert_choice(self) -> None:
        net = registry.net_factory("rope_vit_moe_t16_4e1s_2c_last1_avg", 2, size=(32, 32))
        transform = inference_preset((32, 32), self.rgb_stats)
        interpreter = MoERouting(net, self.device, transform, self.rgb_stats)
        result = interpreter(self.test_image)

        self.assertEqual(result.patch_grid_shape, (2, 2))
        self.assertEqual(list(result.layers), [net.num_layers - 1])
        layer = result.layers[net.num_layers - 1]
        self.assertEqual(layer.router_type, "ExpertChoiceRouter")
        self.assertEqual(layer.assignments.shape, (1, 2, 2, 4))

        # Each expert selects two patches per image
        torch.testing.assert_close(layer.selected, layer.assignments)
        torch.testing.assert_close(layer.assignments.sum(dim=(1, 2)), torch.full((1, 4), 2, dtype=torch.int64))
        torch.testing.assert_close(layer.scores.sum(dim=-1), torch.ones(1, 2, 2))
        torch.testing.assert_close(layer.weights, layer.scores * layer.assignments)

    def test_moe_routing_vmoe(self) -> None:
        net = registry.net_factory(
            "vit_vmoe_vs32_8e_2k_last2s2",
            2,
            config={"moe_eval_capacity_factor": 0.25, "moe_capacity_multiple_of": None},
            size=(64, 64),
        )
        transform = inference_preset((64, 64), self.rgb_stats)
        interpreter = MoERouting(net, self.device, transform, self.rgb_stats)
        result = interpreter(self.test_image)

        self.assertEqual(result.patch_grid_shape, (2, 2))
        self.assertEqual(list(result.layers), [net.num_layers - 3, net.num_layers - 1])
        for layer in result.layers.values():
            self.assertEqual(layer.router_type, "NoisyTopKRouter")
            self.assertEqual(layer.assignments.shape, (1, 2, 2, 8))

            # Selected choices include tokens dropped by the capacity limit
            torch.testing.assert_close(layer.selected.sum(dim=-1), torch.full((1, 2, 2), 2, dtype=torch.int64))
            self.assertTrue((layer.selected & ~layer.assignments).any().item())
            torch.testing.assert_close(layer.assignments, layer.weights > 0)
            torch.testing.assert_close(layer.scores.sum(dim=-1), torch.ones(1, 2, 2))
            torch.testing.assert_close(layer.weights, layer.scores * layer.assignments)

    def test_transformer_attribution_uses_elementwise_attention_gradients(self) -> None:
        attention = torch.full((1, 2, 3, 3), 1.0 / 3.0)
        attention_gradients = torch.zeros_like(attention)

        # The first patch has opposing head gradients
        # Clamping each edge before head fusion preserves its positive contribution
        attention_gradients[0, 0, 0, 1] = 3.0
        attention_gradients[0, 1, 0, 1] = -3.0
        attention_gradients[0, :, 0, 2] = 1.0

        result = compute_attribution_rollout(
            [(attention, attention_gradients)], num_special_tokens=1, patch_grid_shape=(1, 2)
        )

        torch.testing.assert_close(result, torch.tensor([[1.0, 2.0 / 3.0]]))

    def test_transformer_attribution_captures_attention_gradients(self) -> None:
        net = registry.net_factory("vit_t16", 2, size=(32, 32))
        gatherer = AttributionGatherer(net, attention_layer_name="attn")
        self.addCleanup(gatherer.release)

        logits = net(torch.randn(1, 3, 32, 32))
        logits[0, 0].backward()
        attributions = gatherer.get_captured_data()

        self.assertEqual(len(attributions), net.num_layers)
        for attn_weights, attn_gradients in attributions:
            self.assertEqual(attn_weights.shape, (1, 3, 5, 5))
            self.assertEqual(attn_gradients.shape, attn_weights.shape)

    def test_transformer_attribution_result_structure(self) -> None:
        net = registry.net_factory("vit_t16", 2, size=(160, 160))

        def vit_transform(x: Image.Image) -> torch.Tensor:
            x = x.resize((160, 160))
            arr = np.array(x).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        interpreter = TransformerAttribution(
            net, self.device, vit_transform, self.rgb_stats, attention_layer_name="attn"
        )
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
        net = registry.net_factory("vit_t16", 2, size=(160, 160))

        def vit_transform(x: Image.Image) -> torch.Tensor:
            x = x.resize((160, 160))
            arr = np.array(x).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        interpreter = TransformerAttribution(
            net, self.device, vit_transform, self.rgb_stats, attention_layer_name="attn"
        )
        result = interpreter(self.test_image, target_class=1)

        self.assertEqual(result.predicted_class, 1)
