import logging
import unittest

import numpy as np
import torch

from birder import net
from birder.inference import classification

logging.disable(logging.CRITICAL)


class TestInference(unittest.TestCase):
    def setUp(self) -> None:
        self.size = net.GhostNet_v2.default_size
        self.num_classes = 10
        self.model = net.GhostNet_v2(3, self.num_classes, net_param=1)
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
