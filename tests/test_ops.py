import logging
import unittest

import torch
import torch.nn.functional as F

from birder.ops.linear_assignment import LinearAssignment
from birder.ops.linear_assignment import batch_linear_assignment
from birder.ops.msda import MultiScaleDeformableAttention
from birder.ops.msda import multi_scale_deformable_attention
from birder.ops.msda import multi_scale_deformable_attention_packed
from birder.ops.soft_nms import SoftNMS
from birder.ops.soft_nms import batched_soft_nms
from birder.ops.swattention import SWAttention_AV
from birder.ops.swattention import SWAttention_QK_RPB
from birder.ops.swattention import set_swattention_num_threads
from birder.ops.swattention import swattention_av
from birder.ops.swattention import swattention_qk_rpb

logging.disable(logging.CRITICAL)


class TestOps(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_linear_assignment(self) -> None:
        linear_assignment = LinearAssignment()

        wide_cost = torch.tensor(
            [
                [1.0, 9.0, 9.0, 9.0],
                [9.0, 2.0, 9.0, 9.0],
                [9.0, 9.0, 3.0, 4.0],
            ]
        )
        op_col4row, op_row4col = linear_assignment(wide_cost)
        fb_col4row, fb_row4col = batch_linear_assignment(wide_cost)

        torch.testing.assert_close(op_col4row, fb_col4row, rtol=0, atol=0)
        torch.testing.assert_close(op_row4col, fb_row4col, rtol=0, atol=0)

        tall_cost = torch.tensor(
            [
                [
                    [8.0, 4.0, 7.0],
                    [5.0, 2.0, 3.0],
                    [9.0, 6.0, 7.0],
                    [9.0, 4.0, 8.0],
                ],
                [
                    [3.0, 8.0, 5.0],
                    [4.0, 2.0, 6.0],
                    [7.0, 6.0, 9.0],
                    [5.0, 4.0, 3.0],
                ],
            ]
        )
        op_col4row, op_row4col = linear_assignment(tall_cost)
        fb_col4row, fb_row4col = batch_linear_assignment(tall_cost)

        torch.testing.assert_close(op_col4row, fb_col4row, rtol=0, atol=0)
        torch.testing.assert_close(op_row4col, fb_row4col, rtol=0, atol=0)

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

        src_shapes = value_spatial_shapes.tolist()
        op_kernel = msda(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
            src_shapes,
        )
        fb_kernel = multi_scale_deformable_attention(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
            src_shapes,
        )

        self.assertEqual(op_kernel.size(), fb_kernel.size())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_msda_packed(self) -> None:
        device = torch.device("cuda")
        msda = MultiScaleDeformableAttention()
        self.assertTrue(msda.is_available)

        batch_size = 80  # Not divisible by the fixed im2col step of 64
        num_queries = 5
        num_heads = 2
        head_dim = 4
        num_points = [3, 6, 3]
        max_points = max(num_points)
        total_points = sum(num_points)
        value_spatial_shapes = torch.tensor([[4, 4], [2, 2], [1, 1]], dtype=torch.int64, device=device)
        value_level_start_index = torch.tensor([0, 16, 20], dtype=torch.int64, device=device)
        num_points_per_level = torch.tensor(num_points, dtype=torch.int64, device=device)
        src_shapes = value_spatial_shapes.tolist()

        value = torch.rand(batch_size, 21, num_heads, head_dim, device=device, requires_grad=True)
        sampling_locations = torch.rand(
            batch_size, num_queries, num_heads, total_points, 2, device=device, requires_grad=True
        )
        attention_weights = torch.rand(
            batch_size, num_queries, num_heads, total_points, device=device, requires_grad=True
        )

        packed_output = msda.forward_packed(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            num_points_per_level,
            64,
            src_shapes,
            num_points,
        )

        fallback_output = multi_scale_deformable_attention_packed(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            num_points,
            64,
            src_shapes,
        )

        sampling_locations_by_level = sampling_locations.split(num_points, dim=3)
        attention_weights_by_level = attention_weights.split(num_points, dim=3)
        padded_sampling_locations = torch.stack(
            [
                F.pad(locations, (0, 0, 0, max_points - points))
                for locations, points in zip(sampling_locations_by_level, num_points)
            ],
            dim=3,
        )
        padded_attention_weights = torch.stack(
            [
                F.pad(weights, (0, max_points - points))
                for weights, points in zip(attention_weights_by_level, num_points)
            ],
            dim=3,
        )
        reference_output = multi_scale_deformable_attention(
            value,
            value_spatial_shapes,
            value_level_start_index,
            padded_sampling_locations,
            padded_attention_weights,
            64,
            src_shapes,
        )

        torch.testing.assert_close(packed_output, fallback_output, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(fallback_output, reference_output, rtol=1e-4, atol=1e-5)

        # sum() supplies an expanded, non-contiguous gradient to the custom backward.
        packed_output.sum().backward(retain_graph=True)
        self.assertIsNotNone(value.grad)
        self.assertIsNotNone(sampling_locations.grad)
        self.assertIsNotNone(attention_weights.grad)

        output_grad = torch.rand_like(packed_output)
        packed_grads = torch.autograd.grad(
            packed_output, (value, sampling_locations, attention_weights), output_grad, retain_graph=True
        )
        fallback_grads = torch.autograd.grad(
            fallback_output, (value, sampling_locations, attention_weights), output_grad, retain_graph=True
        )
        reference_grads = torch.autograd.grad(
            reference_output, (value, sampling_locations, attention_weights), output_grad
        )
        for packed_grad, fallback_grad, reference_grad in zip(packed_grads, fallback_grads, reference_grads):
            torch.testing.assert_close(packed_grad, fallback_grad, rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(fallback_grad, reference_grad, rtol=2e-4, atol=2e-5)

    def test_msda_packed_fallback(self) -> None:
        num_points = [1, 2]
        value_spatial_shapes = torch.tensor([[2, 2], [1, 1]], dtype=torch.int64)
        value_level_start_index = torch.tensor([0, 4], dtype=torch.int64)
        src_shapes = value_spatial_shapes.tolist()
        value = torch.rand(1, 5, 1, 2)
        sampling_locations = torch.rand(1, 3, 1, 3, 2)
        attention_weights = torch.rand(1, 3, 1, 3)

        fallback_output = multi_scale_deformable_attention_packed(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            num_points,
            64,
            src_shapes,
        )

        padded_sampling_locations = torch.stack(
            [
                F.pad(locations, (0, 0, 0, 2 - points))
                for locations, points in zip(sampling_locations.split(num_points, dim=3), num_points)
            ],
            dim=3,
        )
        padded_attention_weights = torch.stack(
            [
                F.pad(weights, (0, 2 - points))
                for weights, points in zip(attention_weights.split(num_points, dim=3), num_points)
            ],
            dim=3,
        )
        reference_output = multi_scale_deformable_attention(
            value,
            value_spatial_shapes,
            value_level_start_index,
            padded_sampling_locations,
            padded_attention_weights,
            64,
            src_shapes,
        )

        torch.testing.assert_close(fallback_output, reference_output, rtol=1e-5, atol=1e-6)

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

        op_scores, op_keep = soft_nms(boxes, scores.clone(), class_ids)
        fb_scores, fb_keep = batched_soft_nms(boxes, scores.clone(), class_ids)

        torch.testing.assert_close(op_scores, fb_scores, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(op_keep, fb_keep, rtol=0, atol=0)

        # Empty input
        empty_boxes = torch.empty(0, 4)
        empty_scores = torch.empty(0)
        empty_ids = torch.empty(0, dtype=torch.int64)

        op_scores, op_keep = soft_nms(empty_boxes, empty_scores, empty_ids)
        fb_scores, fb_keep = batched_soft_nms(empty_boxes, empty_scores, empty_ids)

        self.assertEqual(len(op_keep), 0)
        self.assertEqual(len(fb_keep), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_swattention(self) -> None:
        device = torch.device("cuda")
        set_swattention_num_threads(32)

        # QK RPB
        swa_qk_rpb = SWAttention_QK_RPB()
        self.assertTrue(swa_qk_rpb.is_available)

        kv = torch.rand(1, 196, 192 * 2, device=device)
        q_norm_scaled = torch.rand(1, 2, 196, 96, device=device)
        relative_pos_bias_local = torch.rand(2, 9, device=device)
        padding_mask = torch.rand(196, 9, device=device) > 0.5
        num_heads = 2
        head_dim = 96
        H = 14
        W = 14
        window_size = 3
        local_len = 9

        op_attn_local, op_v_local = swa_qk_rpb(
            kv, q_norm_scaled, relative_pos_bias_local, padding_mask, num_heads, head_dim, window_size, local_len, H, W
        )
        fb_attn_local, fb_v_local = swattention_qk_rpb(
            kv, q_norm_scaled, relative_pos_bias_local, padding_mask, num_heads, head_dim, window_size, local_len, H, W
        )
        self.assertEqual(op_attn_local.size(), fb_attn_local.size())

        op_v_local = op_v_local.contiguous()
        fb_v_local = fb_v_local.contiguous()

        # AV
        swa_av = SWAttention_AV()
        self.assertTrue(swa_av.is_available)

        q_norm = torch.rand(1, 2, 196, 24, device=device)
        attn_local = torch.rand(1, 2, 196, 9, device=device)
        learnable_tokens = torch.rand(2, 24, 9, device=device)
        learnable_bias = torch.rand(2, 1, 9, device=device)

        op_x_local = swa_av(q_norm, attn_local, op_v_local, learnable_tokens, learnable_bias, window_size, H, W)
        fb_x_local = swattention_av(q_norm, attn_local, fb_v_local, learnable_tokens, learnable_bias)

        self.assertEqual(op_x_local.size(), fb_x_local.size())
