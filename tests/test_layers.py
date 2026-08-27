import logging
import unittest
from typing import Optional

import torch

from birder import layers
from birder.layers.rope import RoPE
from birder.layers.rope import build_rotary_pos_embed

logging.disable(logging.CRITICAL)


class TestLayers(unittest.TestCase):
    def test_activations(self) -> None:
        quick_gelu = layers.QuickGELU()
        out = quick_gelu(torch.rand(2, 8))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 8))

    def test_efficient_probing(self) -> None:
        efficient_probing = layers.EfficientProbing(32, 2, 64, True)
        out = efficient_probing(torch.rand(2, 8, 32))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 1, 32))

        efficient_probing = layers.EfficientProbing(32, 1, 64, True)
        out = efficient_probing(torch.rand(2, 8, 32))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 1, 32))

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

        # With norm
        swiglu_ffn = layers.SwiGLU_FFN(8, 16, norm_layer=torch.nn.LayerNorm)
        out = swiglu_ffn(torch.rand(2, 8))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 8))

    def test_soft_moe_ffn(self) -> None:
        soft_moe_ffn = layers.SoftMoE_FFN(8, 16, num_experts=3, num_slots=2)
        out = soft_moe_ffn(torch.rand(2, 5, 8))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 5, 8))

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

    def test_rope_coord_augmentations(self) -> None:
        def build_augmented_embed(
            shift_coords: Optional[float] = None,
            jitter_coords: Optional[float] = None,
            rescale_coords: Optional[float] = None,
        ) -> torch.Tensor:
            return torch.concat(
                build_rotary_pos_embed(
                    8,
                    temperature=100.0,
                    grid_size=(2, 3),
                    grid_indexing="ij",
                    grid_offset=0,
                    pt_grid_size=None,
                    rope_style="centered_separate",
                    shift_coords=shift_coords,
                    jitter_coords=jitter_coords,
                    rescale_coords=rescale_coords,
                ),
                dim=-1,
            )

        augmentations = (
            {"shift_coords": 0.25},
            {"jitter_coords": 1.25},
            {"rescale_coords": 2.0},
        )
        for augmentation in augmentations:
            with self.subTest(augmentation=augmentation):
                torch.manual_seed(0)
                first = build_augmented_embed(**augmentation)
                torch.manual_seed(0)
                second = build_augmented_embed(**augmentation)
                third = build_augmented_embed(**augmentation)

                self.assertTrue(torch.equal(first, second))
                self.assertFalse(torch.equal(second, third))
                self.assertFalse(torch.isnan(first).any())
                self.assertEqual(first.size(), (6, 16))

        rope = RoPE(
            8,
            temperature=100.0,
            grid_size=(2, 3),
            grid_indexing="ij",
            grid_offset=0,
            rope_style="centered_separate",
            shift_coords=0.25,
            jitter_coords=1.25,
            rescale_coords=2.0,
        )
        x = torch.rand(2, 4, 6, 8)
        first_q, first_k = rope(x, x)
        second_q, second_k = rope(x, x)
        self.assertTrue(torch.equal(first_q, first_k))
        self.assertTrue(torch.equal(second_q, second_k))
        self.assertFalse(torch.equal(first_q, second_q))

        rope.eval()
        first_q, first_k = rope(x, x)
        second_q, second_k = rope(x, x)
        self.assertTrue(torch.equal(first_q, first_k))
        self.assertTrue(torch.equal(first_q, second_q))
        self.assertTrue(torch.equal(first_k, second_k))

    def test_rope_batched_pos_embed(self) -> None:
        ropes = (
            RoPE(8, 100.0, (2, 3), grid_indexing="ij", grid_offset=0),
            RoPE(8, 100.0, (2, 3), grid_indexing="ij", grid_offset=1, pt_grid_size=(5, 7)),
            RoPE(8, 100.0, (2, 3), grid_indexing="xy", grid_offset=1, pt_grid_size=(5, 7)),
            RoPE(8, 100.0, (2, 3), grid_indexing="ij", grid_offset=0, rope_style="axial", rope_rot_type="interleaved"),
            RoPE(8, 100.0, (2, 3), grid_indexing="ij", grid_offset=0, rope_style="centered_separate"),
        )
        grid_sizes = torch.tensor([[2, 3], [3, 2], [1, 4]])
        max_seq_len = 6

        for rope in ropes:
            with self.subTest(rope_style=rope.rope_style, grid_indexing=rope.grid_indexing):
                batched_pos_embed = rope.get_batched_pos_embed(grid_sizes, max_seq_len)
                self.assertEqual(batched_pos_embed.size(), (3, max_seq_len, 16))

                for batch_idx, grid_size_list in enumerate(grid_sizes.tolist()):
                    grid_size = (grid_size_list[0], grid_size_list[1])
                    seq_len = grid_size[0] * grid_size[1]
                    expected = rope.get_pos_embed(grid_size)
                    torch.testing.assert_close(batched_pos_embed[batch_idx, :seq_len], expected)
