import logging
import unittest

import torch

from birder.ops.msda import MultiScaleDeformableAttention
from birder.ops.msda import multi_scale_deformable_attention
from birder.ops.soft_nms import SoftNMS
from birder.ops.soft_nms import batched_soft_nms

logging.disable(logging.CRITICAL)


class TestOps(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_msda(self) -> None:
        device = torch.device("cuda")
        msda = MultiScaleDeformableAttention()
        self.assertTrue(msda.is_available)

        value = torch.rand(1, 34000, 8, 32, device=device)
        value_spatial_shapes = torch.tensor(
            [[160, 160], [80, 80], [40, 40], [20, 20]], dtype=torch.int64, device=device
        )
        value_level_start_index = torch.randint(10, (4,), dtype=torch.int64, device=device)
        sampling_locations = torch.rand(1, 34000, 8, 4, 4, 2, device=device)
        attention_weights = torch.rand(1, 34000, 8, 4, 4, device=device)
        im2col_step = 64

        op_kernel = msda(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
        )
        fb_kernel = multi_scale_deformable_attention(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
        )

        self.assertEqual(op_kernel.size(), fb_kernel.size())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")  # CUDA not actually required
    def test_soft_nms(self) -> None:
        soft_nms = SoftNMS()
        self.assertTrue(soft_nms.is_available)

        boxes = torch.tensor(
            [
                [10, 10, 30, 30],
                [20, 20, 40, 40],  # Overlaps with first
                [50, 50, 70, 70],  # No overlap
                [15, 15, 35, 35],  # Overlaps with first two
            ],
            dtype=torch.float32,
        )
        scores = torch.tensor([0.9, 0.8, 0.7, 0.85])
        class_ids = torch.tensor([0, 0, 1, 0])

        (op_scores, op_keep) = soft_nms(boxes, scores.clone(), class_ids)
        (fb_scores, fb_keep) = batched_soft_nms(boxes, scores.clone(), class_ids)

        torch.testing.assert_close(op_scores, fb_scores, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(op_keep, fb_keep, rtol=0, atol=0)

        # Empty input
        empty_boxes = torch.empty(0, 4)
        empty_scores = torch.empty(0)
        empty_ids = torch.empty(0, dtype=torch.int64)

        (op_scores, op_keep) = soft_nms(empty_boxes, empty_scores, empty_ids)
        (fb_scores, fb_keep) = batched_soft_nms(empty_boxes, empty_scores, empty_ids)

        self.assertEqual(len(op_keep), 0)
        self.assertEqual(len(fb_keep), 0)
