import logging
import unittest

import torch

from birder import layers

logging.disable(logging.CRITICAL)


class TestLayers(unittest.TestCase):
    def test_ffn(self) -> None:
        swiglu_ffn = layers.FFN(8, 16)
        out = swiglu_ffn(torch.rand(2, 8))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 8))

    def test_swiglu_ffn(self) -> None:
        swiglu_ffn = layers.SwiGLU_FFN(8, 16)
        out = swiglu_ffn(torch.rand(2, 8))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 8))

    def test_layer_norm(self) -> None:
        ln = layers.LayerNorm2d(16)
        out = ln(torch.rand(1, 16, 64, 64))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (1, 16, 64, 64))

    def test_layer_scale(self) -> None:
        ls = layers.LayerScale(16, 1e-5)

        # 1D
        out = ls(torch.rand(1, 64, 16))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (1, 64, 16))

        # 2D channels last
        out = ls(torch.rand(1, 64, 64, 16))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (1, 64, 64, 16))

    def test_layer_scale2d(self) -> None:
        ls = layers.LayerScale2d(16, 1e-5)
        out = ls(torch.rand(2, 16, 64, 64))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 16, 64, 64))
