import logging
import unittest

import numpy as np
import torch
from torch import nn

from birder import net
from birder.inference import classification
from birder.inference import wbf
from birder.inference.data_parallel import InferenceDataParallel

logging.disable(logging.CRITICAL)


class OrderTestModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return first element of each sample as identifier
        # This lets us track if order is preserved
        batch_size = x.size(0)
        return x.view(batch_size, -1)[:, :10]


class TestInference(unittest.TestCase):
    def setUp(self) -> None:
        self.size = net.GhostNet_v2.default_size
        self.num_classes = 10
        self.model = net.GhostNet_v2(3, self.num_classes, config={"width": 1.0})
        self.model.eval()

    def test_infer_batch_default_behavior(self) -> None:
        with torch.inference_mode():
            (out, embed) = classification.infer_batch(self.model, torch.rand((1, 3, *self.size)))

        self.assertIsNone(embed)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), self.num_classes)
        self.assertAlmostEqual(out[0].sum(), 1.0, places=5)

    def test_infer_batch_return_embedding(self) -> None:
        with torch.inference_mode():
            (out, embed) = classification.infer_batch(self.model, torch.rand((1, 3, *self.size)), return_embedding=True)

        self.assertIsNotNone(embed)
        self.assertEqual(embed.shape[0], 1)  # type: ignore[union-attr]
        self.assertEqual(embed.shape[1], self.model.embedding_size)  # type: ignore[union-attr]
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), self.num_classes)
        self.assertAlmostEqual(out[0].sum(), 1.0, places=5)

    def test_infer_batch_tta(self) -> None:
        with torch.inference_mode():
            (out, embed) = classification.infer_batch(self.model, torch.rand((1, 3, *self.size)), tta=True)

        self.assertIsNone(embed)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), self.num_classes)
        self.assertAlmostEqual(out[0].sum(), 1.0, places=5)

    def test_infer_batch_return_logits(self) -> None:
        dummy_input = torch.rand((1, 3, *self.size))
        with torch.inference_mode():
            (out, embed) = classification.infer_batch(self.model, dummy_input, return_logits=True)

        self.assertIsNone(embed)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), self.num_classes)

        # Logits should NOT sum to 1
        self.assertNotAlmostEqual(out[0].sum(), 1.0, places=5)

        with torch.inference_mode():
            # Verify that the output is indeed logits by comparing to a manual forward pass
            expected_logits = self.model(dummy_input).cpu().float().numpy()

        np.testing.assert_allclose(out, expected_logits)


class TestWBF(unittest.TestCase):
    def test_weighted_boxes_fusion_merges_overlapping(self) -> None:
        boxes_list = [
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        ]
        scores_list = [torch.tensor([0.9]), torch.tensor([0.7])]
        labels_list = [torch.tensor([1]), torch.tensor([1])]

        (boxes, scores, labels) = wbf.weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=[1.0, 1.0], iou_thr=0.5
        )

        self.assertEqual(boxes.shape, (1, 4))
        self.assertEqual(labels.tolist(), [1])
        self.assertAlmostEqual(scores.item(), 0.8)
        torch.testing.assert_close(boxes[0], torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_weighted_boxes_fusion_separates_labels(self) -> None:
        boxes_list = [
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        ]
        scores_list = [torch.tensor([0.9]), torch.tensor([0.7])]
        labels_list = [torch.tensor([1]), torch.tensor([2])]

        (boxes, scores, labels) = wbf.weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.5)

        self.assertEqual(boxes.shape, (2, 4))
        self.assertEqual(labels.tolist(), [1, 2])
        self.assertAlmostEqual(scores[0].item(), 0.9)
        self.assertAlmostEqual(scores[1].item(), 0.7)

    def test_fuse_detections_wbf_batch(self) -> None:
        detections_a = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            },
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "scores": torch.tensor([0.6]),
                "labels": torch.tensor([2]),
            },
        ]
        detections_b = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "scores": torch.tensor([0.7]),
                "labels": torch.tensor([1]),
            },
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "scores": torch.tensor([0.8]),
                "labels": torch.tensor([2]),
            },
        ]

        fused = wbf.fuse_detections_wbf([detections_a, detections_b], weights=[1.0, 1.0], iou_thr=0.5)

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["labels"].tolist(), [1])
        self.assertAlmostEqual(fused[0]["scores"].item(), 0.8)
        self.assertEqual(fused[1]["labels"].tolist(), [2])
        self.assertAlmostEqual(fused[1]["scores"].item(), 0.7)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestInferenceDataParallel(unittest.TestCase):
    def setUp(self) -> None:
        self.size = net.GhostNet_v2.default_size
        self.num_classes = 10
        self.device = torch.device("cuda")
        self.num_devices = torch.cuda.device_count()

        # Create and prepare model for inference (mimics load_model with inference=True)
        self.model = net.GhostNet_v2(3, self.num_classes, config={"width": 1.0})
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires at least 2 GPUs")
    def test_basic_forward(self) -> None:
        model_parallel = InferenceDataParallel(self.model)

        # Test different batch sizes
        batch_sizes = [1, 4, 7, 16, 31]  # Include prime numbers and edge cases

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                x = torch.randn(batch_size, 3, *self.size)
                with torch.inference_mode():
                    # Single GPU reference
                    model_single = self.model.to(self.device)
                    out_single = model_single(x.to(self.device))

                    # Multi-GPU
                    out_parallel = model_parallel(x)

                # Check shapes match
                self.assertEqual(out_single.size(), out_parallel.size())

                # Check outputs are close
                diff = (out_single - out_parallel).abs().max()
                self.assertLess(diff.item(), 1e-5, f"Output mismatch for batch_size={batch_size}")

        # Test with CPU output
        model_parallel = InferenceDataParallel(self.model, output_device="cpu")
        x = torch.randn(batch_size, 3, *self.size)
        with torch.inference_mode():
            out_parallel = model_parallel(x)

        self.assertEqual(out_parallel.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires at least 2 GPUs")
    def test_output_order_preservation(self) -> None:
        model = OrderTestModel()
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        model_parallel = InferenceDataParallel(model)

        # Create input where each sample is identifiable by its values
        batch_size = 17
        x = torch.zeros(batch_size, 3, 32, 32)
        for i in range(batch_size):
            x[i, :, :, :] = float(i)  # Each sample has unique value

        with torch.inference_mode():
            # Single GPU reference
            model_single = model.to(self.device)
            out_single = model_single(x.to(self.device))

            # Multi-GPU
            out_parallel = model_parallel(x)

        # Verify exact order preservation
        for i in range(batch_size):
            self.assertAlmostEqual(out_parallel[i, 0].item(), float(i), places=6)

        # Also verify against single GPU output
        np.testing.assert_allclose(out_single.cpu().numpy(), out_parallel.cpu().numpy(), rtol=1e-6)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires at least 2 GPUs")
    def test_custom_func(self) -> None:
        model_parallel = InferenceDataParallel(self.model)

        batch_size = 8
        x = torch.randn(batch_size, 3, *self.size)
        with torch.inference_mode():
            embeddings = model_parallel.embedding(x)
            self.assertEqual(embeddings.size(0), batch_size)
            self.assertEqual(embeddings.size(1), self.model.embedding_size)

            logits = model_parallel.classify(embeddings)
            self.assertEqual(logits.size(), (batch_size, self.num_classes))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires at least 2 GPUs")
    def test_integration_with_infer_batch(self) -> None:
        model_parallel = InferenceDataParallel(self.model)

        batch_size = 4
        x = torch.randn(batch_size, 3, *self.size)
        with torch.inference_mode():
            (out, embed) = classification.infer_batch(model_parallel, x)
            self.assertIsNone(embed)
            self.assertEqual(out.shape, (batch_size, self.num_classes))

            # Test with embeddings
            (out, embed) = classification.infer_batch(model_parallel, x, return_embedding=True)
            self.assertIsNotNone(embed)
            self.assertEqual(embed.shape, (batch_size, self.model.embedding_size))  # type: ignore[union-attr]

            # Test with TTA
            (out, embed) = classification.infer_batch(model_parallel, x, tta=True)
            self.assertEqual(out.shape, (batch_size, self.num_classes))

            # Test with logits
            (out, embed) = classification.infer_batch(model_parallel, x, return_logits=True)
            self.assertEqual(out.shape, (batch_size, self.num_classes))

    def test_single_gpu_fallback(self) -> None:
        model_parallel = InferenceDataParallel(self.model, device_ids=[0])
        batch_size = 4
        x = torch.randn(batch_size, 3, *self.size)
        with torch.inference_mode():
            out = model_parallel(x)

        self.assertEqual(out.size(), (batch_size, self.num_classes))

        # Verify it produces same output as direct model
        with torch.inference_mode():
            model_single = self.model.to(self.device)
            out_direct = model_single(x.to(self.device))

        np.testing.assert_allclose(out.cpu().numpy(), out_direct.cpu().numpy(), rtol=1e-5)
