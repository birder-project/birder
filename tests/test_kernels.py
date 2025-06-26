import logging
import unittest

import torch

from birder.kernels import load_kernel

logging.disable(logging.CRITICAL)


class TestKernels(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_deformable_detr(self) -> None:
        device = torch.device("cuda")
        msda = load_kernel.load_msda()
        self.assertIsNotNone(msda)

        value = torch.rand(1, 34000, 8, 32, device=device)
        value_spatial_shapes = torch.tensor(
            [[160, 160], [80, 80], [40, 40], [20, 20]], dtype=torch.int64, device=device
        )
        value_level_start_index = torch.randint(10, (4,), dtype=torch.int64, device=device)
        sampling_locations = torch.rand(1, 34000, 8, 4, 4, 2, device=device)
        attention_weights = torch.rand(1, 34000, 8, 4, 4, device=device)
        im2col_step = 64

        output_kernel = msda.ms_deform_attn_forward(  # type: ignore
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
        )
        self.assertEqual(output_kernel.size(), (1, 34000, 256))

        with torch.amp.autocast("cuda"):
            output_kernel = msda.ms_deform_attn_forward(  # type: ignore
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
            )

        self.assertEqual(output_kernel.size(), (1, 34000, 256))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_swattention(self) -> None:
        device = torch.device("cuda")
        swattention = load_kernel.load_swattention()
        self.assertIsNotNone(swattention)

        q_norm_scaled = torch.rand(1, 2, 3136, 24, device=device)
        k_local = torch.rand(1, 2, 3136, 24, device=device)
        relative_pos_bias_local = torch.rand(2, 9, device=device)
        H = 56
        W = 56
        window_size = 3
        num_threads = 32
        attn_local = swattention.qk_rpb_forward(  # type: ignore
            q_norm_scaled, k_local, relative_pos_bias_local, H, W, window_size, num_threads
        )
        self.assertEqual(attn_local.size(), (1, 2, 3136, 9))

        with torch.amp.autocast("cuda"):
            attn_local = swattention.qk_rpb_forward(  # type: ignore
                q_norm_scaled, k_local, relative_pos_bias_local, H, W, window_size, num_threads
            )

        self.assertEqual(attn_local.size(), (1, 2, 3136, 9))

        attn_local = torch.rand(1, 2, 3136, 9, device=device)
        v_local = torch.rand(1, 2, 3136, 9, device=device)
        x_local = swattention.av_forward(attn_local, v_local, H, W, window_size, num_threads)  # type: ignore
        self.assertEqual(x_local.size(), (1, 2, 3136, 9))

        with torch.amp.autocast("cuda"):
            x_local = swattention.av_forward(attn_local, v_local, H, W, window_size, num_threads)  # type: ignore

        self.assertEqual(x_local.size(), (1, 2, 3136, 9))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_soft_nms(self) -> None:
        device = torch.device("cuda")
        soft_nms = load_kernel.load_soft_nms()
        self.assertIsNotNone(soft_nms)

        boxes = torch.tensor(
            [
                [10.0, 10.0, 20.0, 20.0],  # Box 1
                [15.0, 15.0, 25.0, 25.0],  # Box 2 (overlaps with 1)
                [30.0, 30.0, 40.0, 40.0],  # Box 3 (no overlap)
            ],
            device=device,
        )
        scores = torch.tensor([0.9, 0.8, 0.7], device=device)

        sigma = 0.5
        score_threshold = 0.001

        (updated_scores, keep) = soft_nms.soft_nms(boxes, scores, sigma, score_threshold)  # type: ignore

        # Check outputs
        self.assertIsInstance(updated_scores, torch.Tensor)
        self.assertIsInstance(keep, torch.Tensor)
        self.assertEqual(keep.dtype, torch.int64)

        # All boxes should be kept (threshold is very low)
        self.assertEqual(len(keep), 3)

        # First box should have highest score (unchanged)
        self.assertAlmostEqual(updated_scores[0].item(), 0.9, places=5)

        # Second box score should be reduced due to overlap
        self.assertLess(updated_scores[1].item(), 0.8)

        # Third box should be mostly unchanged (no overlap)
        self.assertAlmostEqual(updated_scores[2].item(), 0.7, places=3)

        # Test with identical boxes (complete overlap)
        boxes = torch.tensor(
            [
                [10.0, 10.0, 20.0, 20.0],
                [10.0, 10.0, 20.0, 20.0],
                [10.0, 10.0, 20.0, 20.0],
            ],
            device=device,
        )

        scores = torch.tensor([0.9, 0.8, 0.7], device=device)
        sigma = 0.25
        score_threshold = 0.1
        (updated_scores, keep) = soft_nms.soft_nms(boxes, scores, sigma, score_threshold)  # type: ignore

        # Only the first box should survive with high threshold
        self.assertEqual(len(keep), 1)
        self.assertEqual(keep[0].item(), 0)  # First box index
