import logging
import unittest

import torch

from birder import net
from birder.inference import classification

logging.disable(logging.CRITICAL)


class TestInference(unittest.TestCase):
    def test_predict(self) -> None:
        size = net.GhostNet_v2.default_size
        n = net.GhostNet_v2(3, 10, net_param=1)

        with self.assertRaises(RuntimeError):
            classification.infer_batch(n, torch.rand((1, 3, *size)))

        with torch.inference_mode():
            (out, embed) = classification.infer_batch(n, torch.rand((1, 3, *size)))
            self.assertIsNone(embed)
            self.assertEqual(len(out), 1)
            self.assertEqual(len(out[0]), 10)
            self.assertAlmostEqual(sum(out[0]), 1, places=5)

            (out, embed) = classification.infer_batch(n, torch.rand((1, 3, *size)), return_embedding=True)
            self.assertIsNotNone(embed)
            self.assertEqual(len(embed), 1)  # type: ignore
            self.assertEqual(len(out), 1)
            self.assertEqual(len(out[0]), 10)
            self.assertAlmostEqual(sum(out[0]), 1, places=5)

            (out, embed) = classification.infer_batch(n, torch.rand((1, 3, *size)), tta=True)
            self.assertIsNone(embed)
            self.assertEqual(len(out), 1)
            self.assertEqual(len(out[0]), 10)
            self.assertAlmostEqual(sum(out[0]), 1, places=5)
