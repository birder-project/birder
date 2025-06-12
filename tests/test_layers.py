import logging
import unittest

import torch

from birder import layers

logging.disable(logging.CRITICAL)


class TestLayers(unittest.TestCase):
    def test_ffm(self) -> None:
        swiglu_ffn = layers.FFN(8, 16)
        out = swiglu_ffn(torch.rand(2, 8))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 8))

    def test_swiglu_ffm(self) -> None:
        swiglu_ffn = layers.SwiGLU_FFN(8, 16)
        out = swiglu_ffn(torch.rand(2, 8))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 8))
