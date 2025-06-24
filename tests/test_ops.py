import logging
import unittest

import torch

from birder.ops.soft_nms import SoftNMS
from birder.ops.soft_nms import batched_soft_nms

logging.disable(logging.CRITICAL)


class TestOps(unittest.TestCase):
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
