import logging
import unittest

import torch

from birder.core import net
from birder.core.inference import inference

logging.disable(logging.CRITICAL)


class TestInference(unittest.TestCase):
    def test_predict(self) -> None:
        size = net.ResNet_v2.default_size
        n = net.ResNet_v2(3, 10, 18)

        with self.assertRaises(RuntimeError):
            inference.predict(n, torch.rand((1, 3, size, size)))

        with torch.inference_mode():
            (out, embed) = inference.predict(n, torch.rand((1, 3, size, size)))
            self.assertIsNone(embed)
            self.assertEqual(len(out), 1)
            self.assertEqual(len(out[0]), 10)
            self.assertAlmostEqual(sum(out[0]), 1, places=5)

            (out, embed) = inference.predict(n, torch.rand((1, 3, size, size)), return_embedding=True)
            self.assertIsNotNone(embed)
            self.assertEqual(len(embed), 1)  # type: ignore
            self.assertEqual(len(out), 1)
            self.assertEqual(len(out[0]), 10)
            self.assertAlmostEqual(sum(out[0]), 1, places=5)