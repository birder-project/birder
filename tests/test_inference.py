import logging
import unittest

import numpy as np
import torch
from torch import nn

from birder import net
from birder.inference import classification
from birder.inference import sliding_window
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
            out, embed = classification.infer_batch(self.model, torch.rand((1, 3, *self.size)))

        self.assertIsNone(embed)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), self.num_classes)
        self.assertAlmostEqual(out[0].sum(), 1.0, places=5)

    def test_infer_batch_return_embedding(self) -> None:
        with torch.inference_mode():
            out, embed = classification.infer_batch(self.model, torch.rand((1, 3, *self.size)), return_embedding=True)

        self.assertIsNotNone(embed)
        self.assertEqual(embed.shape[0], 1)  # type: ignore[union-attr]
        self.assertEqual(embed.shape[1], self.model.embedding_size)  # type: ignore[union-attr]
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), self.num_classes)
        self.assertAlmostEqual(out[0].sum(), 1.0, places=5)

    def test_infer_batch_tta(self) -> None:
        with torch.inference_mode():
            out, embed = classification.infer_batch(self.model, torch.rand((1, 3, *self.size)), tta=True)

        self.assertIsNone(embed)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), self.num_classes)
        self.assertAlmostEqual(out[0].sum(), 1.0, places=5)

    def test_infer_batch_return_logits(self) -> None:
        dummy_input = torch.rand((1, 3, *self.size))
        with torch.inference_mode():
            out, embed = classification.infer_batch(self.model, dummy_input, return_logits=True)

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

        boxes, scores, labels = wbf.weighted_boxes_fusion(
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

        boxes, scores, labels = wbf.weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.5)

        self.assertEqual(boxes.shape, (2, 4))
        self.assertEqual(labels.tolist(), [1, 2])
        self.assertAlmostEqual(scores[0].item(), 0.45)
        self.assertAlmostEqual(scores[1].item(), 0.35)

    def test_weighted_boxes_fusion_conf_types(self) -> None:
        boxes_list = [
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.zeros((0, 4)),
        ]
        scores_list = [torch.tensor([0.9]), torch.tensor([0.3]), torch.zeros((0,))]
        labels_list = [torch.tensor([1]), torch.tensor([1]), torch.zeros((0,), dtype=torch.int64)]
        expected: dict[wbf.ConfType, float] = {
            "avg": 0.25,
            "max": 0.3,
            "box_and_model_avg": 0.25,
            "absent_model_aware_avg": 0.25,
            "cluster_avg": 0.5,
            "cluster_max": 0.9,
        }

        for conf_type, expected_score in expected.items():
            _, scores, _ = wbf.weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=[1.0, 2.0, 3.0],
                iou_thr=0.5,
                conf_type=conf_type,
            )

            self.assertAlmostEqual(scores.item(), expected_score)

    def test_weighted_boxes_fusion_avg_allows_overflow(self) -> None:
        boxes_list = [
            torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        ]
        scores_list = [torch.tensor([0.9, 0.9]), torch.tensor([0.9])]
        labels_list = [torch.tensor([1, 1]), torch.tensor([1])]

        _, bounded_scores, _ = wbf.weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, iou_thr=0.5, conf_type="avg"
        )
        _, overflow_scores, _ = wbf.weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            iou_thr=0.5,
            conf_type="avg",
            allows_overflow=True,
        )

        torch.testing.assert_close(bounded_scores, torch.tensor([0.9]))
        torch.testing.assert_close(overflow_scores, torch.tensor([1.35]))

    def test_weighted_boxes_fusion_model_aware_conf_types_track_unique_sources(self) -> None:
        boxes_list = [
            torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.zeros((0, 4)),
        ]
        scores_list = [torch.tensor([0.9, 0.9]), torch.tensor([0.9]), torch.zeros((0,))]
        labels_list = [
            torch.tensor([1, 1]),
            torch.tensor([1]),
            torch.zeros((0,), dtype=torch.int64),
        ]

        _, box_and_model_scores, _ = wbf.weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=[1.0, 1.0, 1.0],
            iou_thr=0.5,
            conf_type="box_and_model_avg",
        )
        _, absent_model_scores, _ = wbf.weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=[1.0, 1.0, 1.0],
            iou_thr=0.5,
            conf_type="absent_model_aware_avg",
        )

        self.assertAlmostEqual(box_and_model_scores.item(), 0.6)
        self.assertAlmostEqual(absent_model_scores.item(), 0.675)

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


class TestSlidingWindow(unittest.TestCase):
    def test_generate_windows_shifted_edges(self) -> None:
        windows = sliding_window._generate_windows((100, 120), (40, 50), (10, 20))

        self.assertEqual(
            windows,
            [
                (0, 0, 50, 40),
                (30, 0, 80, 40),
                (60, 0, 110, 40),
                (70, 0, 120, 40),
                (0, 30, 50, 70),
                (30, 30, 80, 70),
                (60, 30, 110, 70),
                (70, 30, 120, 70),
                (0, 60, 50, 100),
                (30, 60, 80, 100),
                (60, 60, 110, 100),
                (70, 60, 120, 100),
            ],
        )

    def test_generate_windows_covers_image(self) -> None:
        image_h = 101
        image_w = 113
        windows = sliding_window._generate_windows((image_h, image_w), (32, 40), (8, 12))

        self.assertEqual(windows[0], (0, 0, 40, 32))
        self.assertEqual(windows[-1], (73, 69, image_w, image_h))
        self.assertEqual(min(window[0] for window in windows), 0)
        self.assertEqual(min(window[1] for window in windows), 0)
        self.assertEqual(max(window[2] for window in windows), image_w)
        self.assertEqual(max(window[3] for window in windows), image_h)

    def test_generate_windows_small_image(self) -> None:
        windows = sliding_window._generate_windows((20, 30), (64, 64), (16, 16))
        self.assertEqual(windows, [(0, 0, 30, 20)])

    def test_map_window_detection_shifts_to_image_coordinates(self) -> None:
        detection = {
            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "labels": torch.tensor([2]),
            "scores": torch.tensor([0.9]),
        }

        shifted = sliding_window._map_window_detection(detection, (10, 20, 20, 30), (100, 100))

        torch.testing.assert_close(shifted["boxes"], torch.tensor([[11.0, 22.0, 13.0, 24.0]]))
        torch.testing.assert_close(detection["boxes"], torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        self.assertEqual(shifted["labels"].tolist(), [2])
        torch.testing.assert_close(shifted["scores"], torch.tensor([0.9]))

    def test_clip_detections_to_image(self) -> None:
        detection = {
            "boxes": torch.tensor([[-5.0, -2.0, 20.0, 15.0], [3.0, 4.0, 8.0, 9.0]]),
            "labels": torch.tensor([1, 2]),
            "scores": torch.tensor([0.8, 0.9]),
        }

        clipped = sliding_window._clip_detections_to_image(detection, (10, 12))

        torch.testing.assert_close(clipped["boxes"], torch.tensor([[0.0, 0.0, 12.0, 10.0], [3.0, 4.0, 8.0, 9.0]]))
        torch.testing.assert_close(detection["boxes"], torch.tensor([[-5.0, -2.0, 20.0, 15.0], [3.0, 4.0, 8.0, 9.0]]))

    def test_empty_detections(self) -> None:
        detection = {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.int64),
            "scores": torch.empty((0,)),
        }

        filtered = sliding_window._map_window_detection(detection, (10, 20, 30, 40), (100, 100))

        self.assertEqual(filtered["boxes"].shape, (0, 4))
        self.assertEqual(filtered["labels"].shape, (0,))
        self.assertEqual(filtered["scores"].shape, (0,))

    def test_merge_sliding_window_detections_empty(self) -> None:
        merged = sliding_window._merge_sliding_window_detections([], mode="none", merge_threshold=0.5)

        self.assertEqual(merged["boxes"].shape, (0, 4))
        self.assertEqual(merged["labels"].shape, (0,))
        self.assertEqual(merged["scores"].shape, (0,))

    def test_merge_sliding_window_detections_none(self) -> None:
        detections = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
            },
            {
                "boxes": torch.tensor([[2.0, 2.0, 3.0, 3.0]]),
                "labels": torch.tensor([2]),
                "scores": torch.tensor([0.8]),
            },
        ]

        merged = sliding_window._merge_sliding_window_detections(detections, mode="none", merge_threshold=0.5)

        torch.testing.assert_close(merged["boxes"], torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]))
        self.assertEqual(merged["labels"].tolist(), [1, 2])
        torch.testing.assert_close(merged["scores"], torch.tensor([0.9, 0.8]))

    def test_merge_sliding_window_detections_greedy_nmm_ios(self) -> None:
        detections = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 20.0, 20.0], [0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]]),
                "labels": torch.tensor([1, 1, 2]),
                "scores": torch.tensor([0.9, 0.8, 0.7]),
            }
        ]

        merged = sliding_window._merge_sliding_window_detections(detections, mode="greedy_nmm", merge_threshold=0.5)

        torch.testing.assert_close(merged["boxes"], torch.tensor([[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]]))
        self.assertEqual(merged["labels"].tolist(), [1, 2])
        torch.testing.assert_close(merged["scores"], torch.tensor([0.9, 0.7]))

    def test_merge_sliding_window_detections_nmm_ios_transitive(self) -> None:
        detections = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [4.0, 0.0, 14.0, 10.0], [8.0, 0.0, 18.0, 10.0]]),
                "labels": torch.tensor([1, 1, 1]),
                "scores": torch.tensor([0.9, 0.8, 0.7]),
            }
        ]

        merged = sliding_window._merge_sliding_window_detections(detections, mode="nmm", merge_threshold=0.5)

        torch.testing.assert_close(merged["boxes"], torch.tensor([[0.0, 0.0, 18.0, 10.0]]))
        self.assertEqual(merged["labels"].tolist(), [1])
        torch.testing.assert_close(merged["scores"], torch.tensor([0.9]))


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
    def test_custom_func_mapping_output(self) -> None:
        model_parallel = InferenceDataParallel(self.model)
        x = torch.randn(8, 3, *self.size)
        with torch.inference_mode():
            out_parallel = model_parallel.detection_features(x)
            model_single = self.model.to(self.device)

            chunk_sizes = []
            batch_size = x.size(0)
            base_chunk_size = batch_size // self.num_devices
            remainder = batch_size % self.num_devices
            for i in range(self.num_devices):
                chunk_sizes.append(base_chunk_size + (1 if i < remainder else 0))

            out_single: dict[str, list[torch.Tensor]] = {}
            offset = 0
            for chunk_size in chunk_sizes:
                if chunk_size == 0:
                    continue

                chunk = x[offset : offset + chunk_size].to(self.device)
                chunk_out = model_single.detection_features(chunk)
                for key, value in chunk_out.items():
                    out_single.setdefault(key, []).append(value)

                offset += chunk_size

            expected = {key: torch.concat(values, dim=0) for key, values in out_single.items()}

        self.assertEqual(set(out_parallel.keys()), set(expected.keys()))
        for key in expected:
            self.assertEqual(out_parallel[key].size(), expected[key].size())
            diff = (out_parallel[key] - expected[key]).abs().max()
            self.assertLess(diff.item(), 1e-5, f"Output mismatch for key={key}")

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires at least 2 GPUs")
    def test_integration_with_infer_batch(self) -> None:
        model_parallel = InferenceDataParallel(self.model)

        batch_size = 4
        x = torch.randn(batch_size, 3, *self.size)
        with torch.inference_mode():
            out, embed = classification.infer_batch(model_parallel, x)
            self.assertIsNone(embed)
            self.assertEqual(out.shape, (batch_size, self.num_classes))

            # Test with embeddings
            out, embed = classification.infer_batch(model_parallel, x, return_embedding=True)
            self.assertIsNotNone(embed)
            self.assertEqual(embed.shape, (batch_size, self.model.embedding_size))  # type: ignore[union-attr]

            # Test with TTA
            out, embed = classification.infer_batch(model_parallel, x, tta=True)
            self.assertEqual(out.shape, (batch_size, self.num_classes))

            # Test with logits
            out, embed = classification.infer_batch(model_parallel, x, return_logits=True)
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
