import logging
import unittest

import torch

from birder import layers

logging.disable(logging.CRITICAL)


class TestLayers(unittest.TestCase):
    def test_activations(self) -> None:
        quick_gelu = layers.QuickGELU()
        out = quick_gelu(torch.rand(2, 8))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 8))

    def test_attention_pool(self) -> None:
        attention_pool = layers.MultiHeadAttentionPool(32, 2, 64, True)
        out = attention_pool(torch.rand(2, 8, 32))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 1, 32))

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

    def test_gem(self) -> None:
        fixed_gem = layers.FixedGeMPool2d(3)
        out = fixed_gem(torch.rand(2, 8, 16, 16))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 8))

        gem = layers.GeMPool2d(3)
        out = gem(torch.rand(2, 8, 16, 16))
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
