import logging
import math
import unittest

import torch
import torch.nn.functional as F

from birder.ops.linear_assignment import LinearAssignment
from birder.ops.linear_assignment import batch_linear_assignment
from birder.ops.msda import MultiScaleDeformableAttention
from birder.ops.msda import ms_deform_attn_op
from birder.ops.msda import ms_deform_attn_packed_op
from birder.ops.msda import multi_scale_deformable_attention
from birder.ops.msda import multi_scale_deformable_attention_packed
from birder.ops.soft_nms import SoftNMS
from birder.ops.soft_nms import batched_soft_nms
from birder.ops.swattention import SWAttention_AV
from birder.ops.swattention import SWAttention_QK_RPB
from birder.ops.swattention import set_swattention_num_threads
from birder.ops.swattention import swattention_av
from birder.ops.swattention import swattention_av_op
from birder.ops.swattention import swattention_qk_rpb
from birder.ops.swattention import swattention_qk_rpb_op

logging.disable(logging.CRITICAL)


class TestLinearAssignmentOp(unittest.TestCase):
    def test_fallback_tensor_properties(self) -> None:
        cost_values = (
            (
                (1.0, 9.0, 9.0, 9.0),
                (9.0, 2.0, 9.0, 9.0),
                (9.0, 9.0, 3.0, 4.0),
            ),
            (
                (9.0, 9.0, 1.0, 9.0),
                (2.0, 9.0, 9.0, 9.0),
                (9.0, 3.0, 9.0, 4.0),
            ),
        )

        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                cost_storage = torch.empty((2, 3, 4, 2), dtype=dtype)
                cost = cost_storage[..., 0]
                cost.copy_(torch.tensor(cost_values, dtype=dtype))
                self.assertFalse(cost.is_contiguous())

                col4row, row4col = batch_linear_assignment(cost)

                self.assertEqual(col4row.shape, (2, 3))
                self.assertEqual(row4col.shape, (2, 4))
                self.assertEqual(col4row.dtype, torch.int64)
                self.assertEqual(row4col.dtype, torch.int64)
                self.assertEqual(col4row.device, cost.device)
                self.assertEqual(row4col.device, cost.device)
                self.assertTrue(col4row.is_contiguous())
                self.assertTrue(row4col.is_contiguous())

    def test_fallback_matches_expected(self) -> None:
        wide_cost = torch.tensor(
            (
                (1.0, 9.0, 9.0, 9.0),
                (9.0, 2.0, 9.0, 9.0),
                (9.0, 9.0, 3.0, 4.0),
            )
        )
        col4row, row4col = batch_linear_assignment(wide_cost)
        torch.testing.assert_close(col4row, torch.tensor((0, 1, 2)), rtol=0, atol=0)
        torch.testing.assert_close(row4col, torch.tensor((0, 1, 2, -1)), rtol=0, atol=0)

        tall_cost = torch.tensor(
            (
                (
                    (1.0, 9.0, 9.0),
                    (9.0, 2.0, 9.0),
                    (9.0, 9.0, 3.0),
                    (4.0, 4.0, 4.0),
                ),
                (
                    (9.0, 9.0, 1.0),
                    (2.0, 9.0, 9.0),
                    (9.0, 3.0, 9.0),
                    (4.0, 4.0, 4.0),
                ),
            )
        )
        col4row, row4col = batch_linear_assignment(tall_cost)
        expected_col4row = torch.tensor(((0, 1, 2, -1), (2, 0, 1, -1)))
        expected_row4col = torch.tensor(((0, 1, 2), (1, 2, 0)))
        torch.testing.assert_close(col4row, expected_col4row, rtol=0, atol=0)
        torch.testing.assert_close(row4col, expected_row4col, rtol=0, atol=0)

    def test_fallback_edge_cases(self) -> None:
        # Preserve distinctions that disappear when float64 costs are narrowed to float32
        precise_cost = torch.tensor(((1.0, 1.0), (1.0, 1.0 + 1e-10)), dtype=torch.float64)
        col4row, row4col = batch_linear_assignment(precise_cost)
        expected = torch.tensor((1, 0))
        torch.testing.assert_close(col4row, expected, rtol=0, atol=0)
        torch.testing.assert_close(row4col, expected, rtol=0, atol=0)

        for shape, col4row_shape, row4col_shape in (
            ((0, 3), (0,), (3,)),
            ((2, 3, 0), (2, 3), (2, 0)),
        ):
            with self.subTest(shape=shape):
                col4row, row4col = batch_linear_assignment(torch.empty(shape))
                torch.testing.assert_close(col4row, torch.full(col4row_shape, -1), rtol=0, atol=0)
                torch.testing.assert_close(row4col, torch.full(row4col_shape, -1), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_kernel_matches_fallback(self) -> None:
        device = torch.device("cuda")
        linear_assignment = LinearAssignment()
        self.assertTrue(linear_assignment.is_available)

        cost = torch.tensor(
            (
                (
                    (1.0, 9.0, 9.0),
                    (9.0, 2.0, 9.0),
                    (9.0, 9.0, 3.0),
                    (4.0, 4.0, 4.0),
                ),
                (
                    (9.0, 9.0, 1.0),
                    (2.0, 9.0, 9.0),
                    (9.0, 3.0, 9.0),
                    (4.0, 4.0, 4.0),
                ),
            ),
            device=device,
        )
        kernel_output = linear_assignment(cost, min_batch_for_cuda=1)
        fallback_output = batch_linear_assignment(cost)

        for tensor, reference in zip(kernel_output, fallback_output, strict=True):
            torch.testing.assert_close(tensor, reference, rtol=0, atol=0)

    def test_input_validation(self) -> None:
        linear_assignment = LinearAssignment()
        for shape in ((4,), (1, 2, 3, 4)):
            for name, operation in (
                ("operator", linear_assignment),
                ("fallback", batch_linear_assignment),
            ):
                with self.subTest(operation=name, shape=shape):
                    with self.assertRaises(ValueError):
                        operation(torch.rand(shape))


class TestMSDAOp(unittest.TestCase):
    @staticmethod
    def _metadata(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
        spatial_shapes = torch.tensor(((2, 3), (1, 2)), dtype=torch.int64, device=device)
        level_start_index = torch.tensor((0, 6), dtype=torch.int64, device=device)
        return spatial_shapes, level_start_index, [[2, 3], [1, 2]]

    @staticmethod
    def _standard_inputs(
        device: torch.device, dtype: torch.dtype, requires_grad: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=device).manual_seed(0)
        value = torch.rand((2, 8, 2, 3), dtype=dtype, device=device, generator=generator)
        sampling_locations = torch.rand((2, 4, 2, 2, 3, 2), dtype=dtype, device=device, generator=generator)
        sampling_locations.mul_(1.2).sub_(0.1)
        attention_weights = torch.rand((2, 4, 2, 2, 3), dtype=dtype, device=device, generator=generator)
        attention_weights /= attention_weights.sum(dim=(3, 4), keepdim=True)
        return (
            value.requires_grad_(requires_grad),
            sampling_locations.requires_grad_(requires_grad),
            attention_weights.requires_grad_(requires_grad),
        )

    @staticmethod
    def _packed_inputs(
        device: torch.device, dtype: torch.dtype, requires_grad: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=device).manual_seed(1)
        value = torch.rand((2, 8, 2, 3), dtype=dtype, device=device, generator=generator)
        sampling_locations = torch.rand((2, 4, 2, 4, 2), dtype=dtype, device=device, generator=generator)
        sampling_locations.mul_(1.2).sub_(0.1)
        attention_weights = torch.rand((2, 4, 2, 4), dtype=dtype, device=device, generator=generator)
        attention_weights /= attention_weights.sum(dim=3, keepdim=True)
        return (
            value.requires_grad_(requires_grad),
            sampling_locations.requires_grad_(requires_grad),
            attention_weights.requires_grad_(requires_grad),
        )

    def _assert_outputs_and_gradients_close(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> None:
        torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)

        gradient_storage = torch.linspace(
            -0.75, 0.9, steps=output.numel() * 2, dtype=output.dtype, device=output.device
        ).reshape(*output.shape, 2)
        output_gradient = gradient_storage[..., 0]
        self.assertFalse(output_gradient.is_contiguous())

        gradients = torch.autograd.grad(output, inputs, output_gradient)
        reference_gradients = torch.autograd.grad(reference, inputs, output_gradient)
        for gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
            torch.testing.assert_close(gradient, reference_gradient, rtol=rtol, atol=atol)

    def test_fallback_tensor_properties(self) -> None:
        device = torch.device("cpu")
        spatial_shapes, level_start_index, src_shapes = self._metadata(device)

        for dtype in (torch.float32, torch.float64):
            with self.subTest(operation="standard", dtype=dtype):
                value, sampling_locations, attention_weights = self._standard_inputs(device, dtype)
                output = multi_scale_deformable_attention(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    2,
                    src_shapes,
                )
                self.assertEqual(output.shape, (2, 4, 6))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.device, value.device)
                self.assertTrue(output.is_contiguous())
                self.assertTrue(torch.isfinite(output).all().item())

            with self.subTest(operation="packed", dtype=dtype):
                value, sampling_locations, attention_weights = self._packed_inputs(device, dtype)
                output = multi_scale_deformable_attention_packed(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    [1, 3],
                    2,
                    src_shapes,
                )
                self.assertEqual(output.shape, (2, 4, 6))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.device, value.device)
                self.assertTrue(output.is_contiguous())
                self.assertTrue(torch.isfinite(output).all().item())

    def test_fallback_matches_expected(self) -> None:
        spatial_shapes = torch.tensor(((1, 2), (1, 1)), dtype=torch.int64)
        level_start_index = torch.tensor((0, 2), dtype=torch.int64)
        value = torch.tensor(
            (((1.0, 10.0, 100.0), (3.0, 30.0, 300.0), (5.0, 50.0, 500.0)),), dtype=torch.float64
        ).unsqueeze(2)
        sampling_locations = torch.tensor(
            (
                (0.25, 0.5),
                (0.5, 0.5),
                (0.75, 0.5),
                (0.5, 0.5),
                (-0.5, 0.5),
                (1.5, 0.5),
            ),
            dtype=torch.float64,
        ).reshape(1, 3, 1, 2, 1, 2)
        attention_weights = torch.tensor((0.25, 0.75, 0.6, 0.4, 1.0, 1.0), dtype=torch.float64).reshape(1, 3, 1, 2, 1)

        output = multi_scale_deformable_attention(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, 1, [[1, 2], [1, 1]]
        )

        expected = torch.tensor((((4.0, 40.0, 400.0), (3.8, 38.0, 380.0), (0.0, 0.0, 0.0)),), dtype=torch.float64)
        torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)

    def test_packed_fallback_matches_standard(self) -> None:
        device = torch.device("cpu")
        spatial_shapes, level_start_index, src_shapes = self._metadata(device)
        value, sampling_locations, attention_weights = self._packed_inputs(device, torch.float64, requires_grad=True)
        num_points_per_level = [1, 3]
        max_points = max(num_points_per_level)

        packed_output = multi_scale_deformable_attention_packed(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            num_points_per_level,
            2,
            src_shapes,
        )
        padded_sampling_locations = torch.stack(
            [
                F.pad(locations, (0, 0, 0, max_points - num_points))
                for locations, num_points in zip(
                    sampling_locations.split(num_points_per_level, dim=3), num_points_per_level, strict=True
                )
            ],
            dim=3,
        )
        padded_attention_weights = torch.stack(
            [
                F.pad(weights, (0, max_points - num_points))
                for weights, num_points in zip(
                    attention_weights.split(num_points_per_level, dim=3), num_points_per_level, strict=True
                )
            ],
            dim=3,
        )
        standard_output = multi_scale_deformable_attention(
            value, spatial_shapes, level_start_index, padded_sampling_locations, padded_attention_weights, 2, src_shapes
        )

        self._assert_outputs_and_gradients_close(
            packed_output, standard_output, (value, sampling_locations, attention_weights), rtol=1e-10, atol=1e-12
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_standard_opcheck(self) -> None:
        device = torch.device("cuda")
        msda = MultiScaleDeformableAttention()
        self.assertTrue(msda.is_available)
        spatial_shapes, level_start_index, _src_shapes = self._metadata(device)
        value, sampling_locations, attention_weights = self._standard_inputs(device, torch.float64, requires_grad=True)

        torch.library.opcheck(
            ms_deform_attn_op,
            (value, spatial_shapes, level_start_index, sampling_locations, attention_weights, 2),
            rtol=1e-5,
            atol=1e-6,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_packed_opcheck(self) -> None:
        device = torch.device("cuda")
        msda = MultiScaleDeformableAttention()
        self.assertTrue(msda.is_available)
        spatial_shapes, level_start_index, _src_shapes = self._metadata(device)
        value, sampling_locations, attention_weights = self._packed_inputs(device, torch.float64, requires_grad=True)
        num_points_per_level = torch.tensor((1, 3), dtype=torch.int64, device=device)

        torch.library.opcheck(
            ms_deform_attn_packed_op,
            (value, spatial_shapes, level_start_index, sampling_locations, attention_weights, num_points_per_level, 2),
            rtol=1e-5,
            atol=1e-6,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_standard_kernel_matches_fallback(self) -> None:
        device = torch.device("cuda")
        msda = MultiScaleDeformableAttention()
        self.assertTrue(msda.is_available)
        spatial_shapes, level_start_index, src_shapes = self._metadata(device)
        value, sampling_locations, attention_weights = self._standard_inputs(device, torch.float64, requires_grad=True)

        kernel_output = msda(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, 2, src_shapes
        )
        fallback_output = multi_scale_deformable_attention(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, 2, src_shapes
        )

        self._assert_outputs_and_gradients_close(
            kernel_output, fallback_output, (value, sampling_locations, attention_weights), rtol=1e-5, atol=1e-6
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_packed_kernel_matches_fallback(self) -> None:
        device = torch.device("cuda")
        msda = MultiScaleDeformableAttention()
        self.assertTrue(msda.is_available)
        spatial_shapes, level_start_index, src_shapes = self._metadata(device)
        value, sampling_locations, attention_weights = self._packed_inputs(device, torch.float64, requires_grad=True)
        num_points_per_level = [1, 3]

        kernel_output = msda.forward_packed(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            torch.tensor(num_points_per_level, dtype=torch.int64, device=device),
            2,
            src_shapes,
            num_points_per_level,
        )
        fallback_output = multi_scale_deformable_attention_packed(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            num_points_per_level,
            2,
            src_shapes,
        )

        self._assert_outputs_and_gradients_close(
            kernel_output, fallback_output, (value, sampling_locations, attention_weights), rtol=1e-5, atol=1e-6
        )


class TestSoftNMSOp(unittest.TestCase):
    def test_fallback_tensor_properties(self) -> None:
        boxes_values = (
            (0.0, 0.0, 10.0, 10.0),
            (1.0, 1.0, 9.0, 9.0),
            (20.0, 20.0, 30.0, 30.0),
        )
        scores_values = (0.9, 0.8, 0.7)
        class_ids = torch.tensor((0, 0, 1), dtype=torch.int64)
        num_boxes = len(boxes_values)

        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                boxes_storage = torch.empty((num_boxes, 8), dtype=dtype)
                boxes = boxes_storage[:, ::2]
                boxes.copy_(torch.tensor(boxes_values, dtype=dtype))
                scores_storage = torch.empty((2 * num_boxes,), dtype=dtype)
                scores = scores_storage[::2]
                scores.copy_(torch.tensor(scores_values, dtype=dtype))
                self.assertFalse(boxes.is_contiguous())
                self.assertFalse(scores.is_contiguous())
                boxes_before = boxes.clone()
                scores_before = scores.clone()

                updated_scores, keep = batched_soft_nms(boxes, scores, class_ids, sigma=0.5, score_threshold=-1.0)

                self.assertEqual(updated_scores.shape, (num_boxes,))
                self.assertEqual(keep.shape, (num_boxes,))
                self.assertEqual(updated_scores.dtype, dtype)
                self.assertEqual(keep.dtype, torch.int64)
                self.assertEqual(updated_scores.device, scores.device)
                self.assertEqual(keep.device, boxes.device)
                self.assertTrue(updated_scores.is_contiguous())
                self.assertTrue(keep.is_contiguous())
                self.assertTrue(torch.isfinite(updated_scores).all().item())
                torch.testing.assert_close(boxes, boxes_before, rtol=0, atol=0)
                torch.testing.assert_close(scores, scores_before, rtol=0, atol=0)

    def test_fallback_matches_expected(self) -> None:
        boxes = torch.tensor(
            (
                (0.0, 0.0, 10.0, 10.0),
                (0.0, 0.0, 10.0, 10.0),
                (0.0, 0.0, 10.0, 10.0),
                (20.0, 20.0, 30.0, 30.0),
            ),
            dtype=torch.float64,
        )
        scores = torch.tensor((0.9, 0.8, 0.7, 0.6), dtype=torch.float64)
        class_ids = torch.tensor((0, 0, 1, 0), dtype=torch.int64)

        updated_scores, keep = batched_soft_nms(boxes, scores, class_ids, sigma=0.5, score_threshold=0.1)

        identical_box_iou = 100.0 / (100.0 + 1e-8)
        decayed_score = 0.8 * math.exp(-(identical_box_iou * identical_box_iou) / 0.5)
        expected_scores = torch.tensor((0.9, 0.7, 0.6, decayed_score), dtype=torch.float64)
        expected_keep = torch.tensor((0, 2, 3, 1), dtype=torch.int64)
        torch.testing.assert_close(updated_scores, expected_scores, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(keep, expected_keep, rtol=0, atol=0)

    def test_fallback_edge_cases(self) -> None:
        degenerate_boxes = torch.tensor(
            (
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0, 2.0),
                (2.0, 2.0, 3.0, 2.0),
            ),
            dtype=torch.float64,
        )
        scores = torch.tensor((0.9, 0.8, 0.7, 0.6), dtype=torch.float64)
        class_ids = torch.zeros(4, dtype=torch.int64)

        updated_scores, keep = batched_soft_nms(degenerate_boxes, scores, class_ids, sigma=0.5, score_threshold=0.0)
        torch.testing.assert_close(updated_scores, scores, rtol=0, atol=0)
        torch.testing.assert_close(keep, torch.arange(4), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(updated_scores).all().item())

        box = torch.tensor(((0.0, 0.0, 10.0, 10.0),), dtype=torch.float64)
        score = torch.tensor((0.5,), dtype=torch.float64)
        class_id = torch.tensor((7,), dtype=torch.int64)
        updated_scores, keep = batched_soft_nms(box, score, class_id, score_threshold=0.49)
        torch.testing.assert_close(updated_scores, score, rtol=0, atol=0)
        torch.testing.assert_close(keep, torch.zeros(1, dtype=torch.int64), rtol=0, atol=0)

        updated_scores, keep = batched_soft_nms(box, score, class_id, score_threshold=0.5)
        self.assertEqual(updated_scores.shape, (0,))
        self.assertEqual(keep.shape, (0,))

    def test_fallback_low_precision_uses_float32_geometry(self) -> None:
        boxes_values = (
            (1024.0, 1024.0, 1152.0, 1152.0),
            (1024.0, 1024.0, 1152.0, 1152.0),
            (896.0, 896.0, 1088.0, 1088.0),
            (896.0, 896.0, 1088.0, 1088.0),
        )
        scores_values = (0.9, 0.8, 0.7, 0.6)
        class_ids = torch.tensor((91, 91, 3, 3), dtype=torch.int64)
        reference_boxes = torch.tensor(boxes_values, dtype=torch.float32)
        reference_scores = torch.tensor(scores_values, dtype=torch.float32)
        expected_scores, expected_keep = batched_soft_nms(
            reference_boxes, reference_scores, class_ids, score_threshold=0.001
        )

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                boxes = torch.tensor(boxes_values, dtype=dtype)
                scores = torch.tensor(scores_values, dtype=dtype)
                updated_scores, keep = batched_soft_nms(boxes, scores, class_ids, score_threshold=0.001)

                torch.testing.assert_close(updated_scores, expected_scores.to(dtype), rtol=5e-3, atol=5e-3)
                torch.testing.assert_close(keep, expected_keep, rtol=0, atol=0)

    def test_empty_input(self) -> None:
        soft_nms = SoftNMS()

        for dtype in (torch.float32, torch.float64):
            boxes = torch.empty((0, 4), dtype=dtype)
            scores = torch.empty((0,), dtype=dtype)
            class_ids = torch.empty((0,), dtype=torch.int64)
            for name, operation in (("operator", soft_nms), ("fallback", batched_soft_nms)):
                with self.subTest(operation=name, dtype=dtype):
                    updated_scores, keep = operation(boxes, scores, class_ids)
                    self.assertEqual(updated_scores.shape, (0,))
                    self.assertEqual(keep.shape, (0,))
                    self.assertEqual(updated_scores.dtype, dtype)
                    self.assertEqual(keep.dtype, torch.int64)
                    self.assertEqual(updated_scores.device, scores.device)
                    self.assertEqual(keep.device, boxes.device)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_kernel_matches_fallback(self) -> None:
        soft_nms = SoftNMS()
        self.assertTrue(soft_nms.is_available)
        boxes = torch.tensor(
            (
                (2.0, 2.0, 12.0, 12.0),
                (0.0, 0.0, 10.0, 10.0),
                (20.0, 20.0, 30.0, 30.0),
                (1.0, 1.0, 9.0, 9.0),
                (0.0, 0.0, 10.0, 10.0),
            ),
            dtype=torch.float64,
            device="cuda",
        )
        scores = torch.tensor((0.75, 0.9, 0.8, 0.85, 0.7), dtype=torch.float64, device="cuda")
        class_ids = torch.tensor((0, 0, 0, 0, 1), dtype=torch.int64, device="cuda")

        kernel_scores, kernel_keep = soft_nms(boxes, scores, class_ids, sigma=0.7, score_threshold=0.3)
        fallback_scores, fallback_keep = batched_soft_nms(boxes, scores, class_ids, sigma=0.7, score_threshold=0.3)

        torch.testing.assert_close(kernel_scores, fallback_scores, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(kernel_keep, fallback_keep, rtol=0, atol=0)


class TestSWAttentionOp(unittest.TestCase):
    @staticmethod
    def _padding_mask(height: int, width: int, kernel_size: int, device: torch.device) -> torch.Tensor:
        radius = kernel_size // 2
        mask = []
        for row in range(height):
            for column in range(width):
                mask.append(
                    [
                        not (0 <= row + row_offset < height and 0 <= column + column_offset < width)
                        for row_offset in range(-radius, radius + 1)
                        for column_offset in range(-radius, radius + 1)
                    ]
                )

        return torch.tensor(mask, dtype=torch.bool, device=device)

    @staticmethod
    def _local_neighborhoods(tensor: torch.Tensor, height: int, width: int, kernel_size: int) -> torch.Tensor:
        batch_size, num_heads, _, head_dim = tensor.shape
        radius = kernel_size // 2
        neighborhoods = []
        for row in range(height):
            for column in range(width):
                local_values = []
                for row_offset in range(-radius, radius + 1):
                    for column_offset in range(-radius, radius + 1):
                        source_row = row + row_offset
                        source_column = column + column_offset
                        if 0 <= source_row < height and 0 <= source_column < width:
                            source_index = source_row * width + source_column
                            local_values.append(tensor[:, :, source_index])
                        else:
                            local_values.append(tensor.new_zeros((batch_size, num_heads, head_dim)))

                neighborhoods.append(torch.stack(local_values, dim=-1))

        return torch.stack(neighborhoods, dim=2)

    def test_fallback_tensor_properties(self) -> None:
        device = torch.device("cpu")
        batch_size, num_heads, head_dim, query_dim = 2, 3, 7, 6
        height, width = 4, 5
        num_tokens = height * width
        window_size = 3
        local_len = window_size * window_size
        padding_mask = self._padding_mask(height, width, window_size, device)

        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                generator = torch.Generator(device=device).manual_seed(0)
                kv = torch.rand(
                    (batch_size, num_tokens, 2 * num_heads * head_dim), dtype=dtype, device=device, generator=generator
                )
                q_norm_scaled = torch.rand(
                    (batch_size, num_heads, num_tokens, head_dim), dtype=dtype, device=device, generator=generator
                )
                relative_pos_bias = torch.rand((num_heads, local_len), dtype=dtype, device=device, generator=generator)

                attn_local, v_local = swattention_qk_rpb(
                    kv,
                    q_norm_scaled,
                    relative_pos_bias,
                    padding_mask,
                    num_heads,
                    head_dim,
                    window_size,
                    local_len,
                    height,
                    width,
                )
                q_norm = torch.rand(
                    (batch_size, num_heads, num_tokens, query_dim), dtype=dtype, device=device, generator=generator
                )
                learnable_tokens = torch.rand(
                    (num_heads, query_dim, local_len), dtype=dtype, device=device, generator=generator
                )
                learnable_bias = torch.rand((num_heads, 1, local_len), dtype=dtype, device=device, generator=generator)
                output = swattention_av(q_norm, attn_local.softmax(dim=-1), v_local, learnable_tokens, learnable_bias)

                self.assertEqual(attn_local.shape, (batch_size, num_heads, num_tokens, local_len))
                self.assertEqual(v_local.shape, (batch_size, num_heads, num_tokens, head_dim, local_len))
                self.assertEqual(output.shape, (batch_size, num_heads, num_tokens, head_dim))
                for tensor in (attn_local, v_local, output):
                    self.assertEqual(tensor.dtype, dtype)
                    self.assertEqual(tensor.device, device)

                self.assertTrue(attn_local.is_contiguous())
                self.assertTrue(output.is_contiguous())
                valid_mask = ~padding_mask.reshape(1, 1, num_tokens, local_len).expand_as(attn_local)
                self.assertTrue(torch.isfinite(attn_local[valid_mask]).all().item())
                self.assertTrue(torch.isneginf(attn_local[~valid_mask]).all().item())
                self.assertTrue(torch.isfinite(v_local).all().item())
                self.assertTrue(torch.isfinite(output).all().item())

    def test_qk_fallback_matches_reference(self) -> None:  # pylint: disable=too-many-locals
        device = torch.device("cpu")
        batch_size, num_heads, head_dim = 1, 2, 5
        height, width = 3, 4
        num_tokens = height * width
        window_size = 3
        local_len = window_size * window_size
        padding_mask = self._padding_mask(height, width, window_size, device)
        kv = torch.linspace(
            -0.9,
            1.1,
            steps=batch_size * num_tokens * 2 * num_heads * head_dim,
            dtype=torch.float64,
        ).reshape(batch_size, num_tokens, 2 * num_heads * head_dim)
        q_norm_scaled = torch.linspace(
            0.8,
            -0.7,
            steps=batch_size * num_heads * num_tokens * head_dim,
            dtype=torch.float64,
        ).reshape(batch_size, num_heads, num_tokens, head_dim)
        relative_pos_bias = torch.linspace(-0.2, 0.3, steps=num_heads * local_len, dtype=torch.float64).reshape(
            num_heads, local_len
        )
        kv.requires_grad_()
        q_norm_scaled.requires_grad_()
        relative_pos_bias.requires_grad_()

        attn_local, v_local = swattention_qk_rpb(
            kv,
            q_norm_scaled,
            relative_pos_bias,
            padding_mask,
            num_heads,
            head_dim,
            window_size,
            local_len,
            height,
            width,
        )

        reference_kv = kv.detach().requires_grad_(True)
        reference_q_norm_scaled = q_norm_scaled.detach().requires_grad_(True)
        reference_relative_pos_bias = relative_pos_bias.detach().requires_grad_(True)
        reference_keys, reference_values = reference_kv.chunk(2, dim=-1)
        reference_keys = reference_keys.reshape(batch_size, num_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
        reference_values = reference_values.reshape(batch_size, num_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
        reference_keys = reference_keys / reference_keys.norm(dim=-1, keepdim=True)
        local_keys = self._local_neighborhoods(reference_keys, height, width, window_size)
        reference_v_local = self._local_neighborhoods(reference_values, height, width, window_size)
        reference_attn_local = (reference_q_norm_scaled.unsqueeze(-1) * local_keys).sum(dim=3)
        reference_attn_local += reference_relative_pos_bias.reshape(1, num_heads, 1, local_len)
        reference_attn_local = reference_attn_local.masked_fill(
            padding_mask.reshape(1, 1, num_tokens, local_len), float("-inf")
        )

        torch.testing.assert_close(attn_local, reference_attn_local, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(v_local, reference_v_local, rtol=1e-10, atol=1e-12)

        attn_gradient = torch.linspace(-0.4, 0.5, steps=attn_local.numel() * 2, dtype=torch.float64).reshape(
            *attn_local.shape, 2
        )[..., 0]
        value_gradient = torch.linspace(0.6, -0.3, steps=v_local.numel() * 2, dtype=torch.float64).reshape(
            *v_local.shape, 2
        )[..., 0]
        gradients = torch.autograd.grad(
            (attn_local, v_local), (kv, q_norm_scaled, relative_pos_bias), (attn_gradient, value_gradient)
        )
        reference_gradients = torch.autograd.grad(
            (reference_attn_local, reference_v_local),
            (reference_kv, reference_q_norm_scaled, reference_relative_pos_bias),
            (attn_gradient, value_gradient),
        )
        for gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
            torch.testing.assert_close(gradient, reference_gradient, rtol=1e-10, atol=1e-12)

    def test_av_fallback_matches_reference(self) -> None:
        batch_size, num_heads, num_tokens, head_dim, query_dim, local_len = 1, 2, 12, 5, 6, 9
        q_norm = torch.linspace(
            -0.7, 0.9, steps=batch_size * num_heads * num_tokens * query_dim, dtype=torch.float64
        ).reshape(batch_size, num_heads, num_tokens, query_dim)
        attn_local = torch.linspace(
            0.1, 0.8, steps=batch_size * num_heads * num_tokens * local_len, dtype=torch.float64
        ).reshape(batch_size, num_heads, num_tokens, local_len)
        v_local = torch.linspace(
            0.8, -0.6, steps=batch_size * num_heads * num_tokens * head_dim * local_len, dtype=torch.float64
        ).reshape(batch_size, num_heads, num_tokens, head_dim, local_len)
        learnable_tokens = torch.linspace(
            -0.3, 0.4, steps=num_heads * query_dim * local_len, dtype=torch.float64
        ).reshape(num_heads, query_dim, local_len)
        learnable_bias = torch.linspace(-0.2, 0.25, steps=num_heads * local_len, dtype=torch.float64).reshape(
            num_heads, 1, local_len
        )
        inputs = (q_norm, attn_local, v_local, learnable_tokens, learnable_bias)
        for tensor in inputs:
            tensor.requires_grad_()

        output = swattention_av(*inputs)

        reference_inputs = tuple(tensor.detach().requires_grad_(True) for tensor in inputs)
        reference_q_norm, reference_attn_local, reference_v_local, reference_tokens, reference_bias = reference_inputs
        dynamic_attention = (
            reference_q_norm.unsqueeze(-1) * reference_tokens.reshape(1, num_heads, 1, query_dim, local_len)
        ).sum(dim=3)
        reference_output = (
            (dynamic_attention + reference_bias.reshape(1, num_heads, 1, local_len) + reference_attn_local).unsqueeze(3)
            * reference_v_local
        ).sum(dim=-1)

        torch.testing.assert_close(output, reference_output, rtol=1e-10, atol=1e-12)
        output_gradient = torch.linspace(-0.5, 0.6, steps=output.numel() * 2, dtype=torch.float64).reshape(
            *output.shape, 2
        )[..., 0]
        gradients = torch.autograd.grad(output, inputs, output_gradient)
        reference_gradients = torch.autograd.grad(reference_output, reference_inputs, output_gradient)
        for gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
            torch.testing.assert_close(gradient, reference_gradient, rtol=1e-10, atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_autocast_dtype_and_gradients(self) -> None:  # pylint: disable=too-many-locals
        device = torch.device("cuda")
        set_swattention_num_threads(32)
        swattention_qk = SWAttention_QK_RPB()
        swattention_av_kernel = SWAttention_AV()
        self.assertTrue(swattention_qk.is_available)
        self.assertTrue(swattention_av_kernel.is_available)

        batch_size, num_heads, head_dim, query_dim = 1, 2, 5, 4
        height, width = 3, 4
        num_tokens = height * width
        window_size = 3
        local_len = window_size * window_size
        padding_mask = self._padding_mask(height, width, window_size, device)
        amp_dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported() is True:
            amp_dtypes.append(torch.bfloat16)

        for amp_dtype in amp_dtypes:
            with self.subTest(amp_dtype=amp_dtype):
                generator = torch.Generator(device=device).manual_seed(3)
                kv = torch.rand(
                    (batch_size, num_tokens, 2 * num_heads * head_dim),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                    requires_grad=True,
                )
                q_norm_scaled = torch.rand(
                    (batch_size, num_heads, num_tokens, head_dim),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                    requires_grad=True,
                )
                relative_pos_bias = torch.rand(
                    (num_heads, local_len), dtype=torch.float32, device=device, generator=generator, requires_grad=True
                )
                q_norm = torch.rand(
                    (batch_size, num_heads, num_tokens, query_dim),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                    requires_grad=True,
                )
                learnable_tokens = torch.rand(
                    (num_heads, query_dim, local_len),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                    requires_grad=True,
                )
                learnable_bias = torch.rand(
                    (num_heads, 1, local_len),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                    requires_grad=True,
                )

                with torch.autocast("cuda", dtype=amp_dtype):
                    attn_local, v_local = swattention_qk(
                        kv,
                        q_norm_scaled,
                        relative_pos_bias,
                        padding_mask,
                        num_heads,
                        head_dim,
                        window_size,
                        local_len,
                        height,
                        width,
                    )
                    output = swattention_av_kernel(
                        q_norm,
                        attn_local.softmax(dim=-1),
                        v_local,
                        learnable_tokens,
                        learnable_bias,
                        window_size,
                        height,
                        width,
                    )

                for tensor in (attn_local, v_local, output):
                    self.assertEqual(tensor.dtype, amp_dtype)

                output.float().square().mean().backward()
                fp32_inputs = (kv, q_norm_scaled, relative_pos_bias, q_norm, learnable_tokens, learnable_bias)
                for tensor in fp32_inputs:
                    self.assertIsNotNone(tensor.grad)
                    self.assertEqual(tensor.grad.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(tensor.grad).all().item())

                # The wrapper applies the same casting policy when the extension
                # is unavailable and the pure-PyTorch fallback is selected.
                swattention_qk.is_available = False
                swattention_av_kernel.is_available = False
                with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
                    fallback_attn, fallback_v_local = swattention_qk(
                        kv,
                        q_norm_scaled,
                        relative_pos_bias,
                        padding_mask,
                        num_heads,
                        head_dim,
                        window_size,
                        local_len,
                        height,
                        width,
                    )
                    fallback_output = swattention_av_kernel(
                        q_norm,
                        fallback_attn.softmax(dim=-1),
                        fallback_v_local,
                        learnable_tokens,
                        learnable_bias,
                        window_size,
                        height,
                        width,
                    )

                for tensor in (fallback_attn, fallback_v_local, fallback_output):
                    self.assertEqual(tensor.dtype, amp_dtype)

                swattention_qk.is_available = True
                swattention_av_kernel.is_available = True

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_qk_opcheck(self) -> None:
        set_swattention_num_threads(32)
        swattention_qk = SWAttention_QK_RPB()
        self.assertTrue(swattention_qk.is_available)
        device = torch.device("cuda")
        height, width = 4, 5
        kernel_size = 3
        feature_shape = (2, 3, height * width, 7)
        query = torch.rand(feature_shape, dtype=torch.float64, device=device, requires_grad=True)
        key = torch.rand(feature_shape, dtype=torch.float64, device=device, requires_grad=True)
        relative_position_bias = torch.rand(
            (feature_shape[1], kernel_size * kernel_size), dtype=torch.float64, device=device, requires_grad=True
        )

        torch.library.opcheck(
            swattention_qk_rpb_op,
            (query, key, relative_position_bias, height, width, kernel_size),
            rtol=1e-6,
            atol=1e-7,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_av_opcheck(self) -> None:
        set_swattention_num_threads(32)
        swattention_av_kernel = SWAttention_AV()
        self.assertTrue(swattention_av_kernel.is_available)
        device = torch.device("cuda")
        height, width = 4, 5
        kernel_size = 3
        feature_shape = (2, 3, height * width, 7)
        attention = torch.rand(
            (*feature_shape[:3], kernel_size * kernel_size), dtype=torch.float64, device=device, requires_grad=True
        )
        value = torch.rand(feature_shape, dtype=torch.float64, device=device, requires_grad=True)

        torch.library.opcheck(swattention_av_op, (attention, value, height, width, kernel_size), rtol=1e-6, atol=1e-7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_qk_kernel_matches_fallback(self) -> None:
        device = torch.device("cuda")
        set_swattention_num_threads(32)
        swattention_qk = SWAttention_QK_RPB()
        self.assertTrue(swattention_qk.is_available)
        batch_size, num_heads, head_dim = 2, 3, 7
        height, width = 4, 5
        num_tokens = height * width
        window_size = 3
        local_len = window_size * window_size
        generator = torch.Generator(device=device).manual_seed(1)
        kv = torch.rand(
            (batch_size, num_tokens, 2 * num_heads * head_dim), dtype=torch.float64, device=device, generator=generator
        ).requires_grad_()
        q_norm_scaled = torch.rand(
            (batch_size, num_heads, num_tokens, head_dim), dtype=torch.float64, device=device, generator=generator
        ).requires_grad_()
        relative_pos_bias = torch.rand(
            (num_heads, local_len), dtype=torch.float64, device=device, generator=generator
        ).requires_grad_()
        padding_mask = self._padding_mask(height, width, window_size, device)

        kernel_attn, _ = swattention_qk(
            kv,
            q_norm_scaled,
            relative_pos_bias,
            padding_mask,
            num_heads,
            head_dim,
            window_size,
            local_len,
            height,
            width,
        )
        fallback_attn, _ = swattention_qk_rpb(
            kv,
            q_norm_scaled,
            relative_pos_bias,
            padding_mask,
            num_heads,
            head_dim,
            window_size,
            local_len,
            height,
            width,
        )

        torch.testing.assert_close(kernel_attn, fallback_attn, rtol=1e-6, atol=1e-7)
        output_gradient = torch.linspace(
            -0.4, 0.5, steps=kernel_attn.numel() * 2, dtype=torch.float64, device=device
        ).reshape(*kernel_attn.shape, 2)[..., 0]
        inputs = (kv, q_norm_scaled, relative_pos_bias)
        kernel_gradients = torch.autograd.grad(kernel_attn, inputs, output_gradient)
        fallback_gradients = torch.autograd.grad(fallback_attn, inputs, output_gradient)
        for kernel_gradient, fallback_gradient in zip(kernel_gradients, fallback_gradients, strict=True):
            torch.testing.assert_close(kernel_gradient, fallback_gradient, rtol=1e-6, atol=1e-7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_av_kernel_matches_fallback(self) -> None:  # pylint: disable=too-many-locals
        device = torch.device("cuda")
        set_swattention_num_threads(32)
        swattention_qk = SWAttention_QK_RPB()
        swattention_av_kernel = SWAttention_AV()
        self.assertTrue(swattention_qk.is_available)
        self.assertTrue(swattention_av_kernel.is_available)
        batch_size, num_heads, head_dim, query_dim = 2, 3, 7, 6
        height, width = 4, 5
        num_tokens = height * width
        window_size = 3
        local_len = window_size * window_size
        generator = torch.Generator(device=device).manual_seed(2)
        kv = torch.rand(
            (batch_size, num_tokens, 2 * num_heads * head_dim), dtype=torch.float64, device=device, generator=generator
        ).requires_grad_()
        q_norm_scaled = torch.rand(
            (batch_size, num_heads, num_tokens, head_dim), dtype=torch.float64, device=device, generator=generator
        )
        relative_pos_bias = torch.rand((num_heads, local_len), dtype=torch.float64, device=device, generator=generator)
        padding_mask = self._padding_mask(height, width, window_size, device)
        _, kernel_v_local = swattention_qk(
            kv,
            q_norm_scaled,
            relative_pos_bias,
            padding_mask,
            num_heads,
            head_dim,
            window_size,
            local_len,
            height,
            width,
        )
        _, fallback_v_local = swattention_qk_rpb(
            kv,
            q_norm_scaled,
            relative_pos_bias,
            padding_mask,
            num_heads,
            head_dim,
            window_size,
            local_len,
            height,
            width,
        )
        q_norm = torch.rand(
            (batch_size, num_heads, num_tokens, query_dim), dtype=torch.float64, device=device, generator=generator
        ).requires_grad_()
        attn_local = torch.rand(
            (batch_size, num_heads, num_tokens, local_len), dtype=torch.float64, device=device, generator=generator
        )
        attn_local /= attn_local.sum(dim=-1, keepdim=True)
        attn_local.requires_grad_()
        learnable_tokens = torch.rand(
            (num_heads, query_dim, local_len), dtype=torch.float64, device=device, generator=generator
        ).requires_grad_()
        learnable_bias = torch.rand(
            (num_heads, 1, local_len), dtype=torch.float64, device=device, generator=generator
        ).requires_grad_()

        kernel_output = swattention_av_kernel(
            q_norm,
            attn_local,
            kernel_v_local.contiguous(),
            learnable_tokens,
            learnable_bias,
            window_size,
            height,
            width,
        )
        fallback_output = swattention_av(
            q_norm, attn_local, fallback_v_local.contiguous(), learnable_tokens, learnable_bias
        )

        torch.testing.assert_close(kernel_output, fallback_output, rtol=1e-6, atol=1e-7)
        output_gradient = torch.linspace(
            -0.5, 0.6, steps=kernel_output.numel() * 2, dtype=torch.float64, device=device
        ).reshape(*kernel_output.shape, 2)[..., 0]
        inputs = (kv, q_norm, attn_local, learnable_tokens, learnable_bias)
        kernel_gradients = torch.autograd.grad(kernel_output, inputs, output_gradient)
        fallback_gradients = torch.autograd.grad(fallback_output, inputs, output_gradient)
        for kernel_gradient, fallback_gradient in zip(kernel_gradients, fallback_gradients, strict=True):
            torch.testing.assert_close(kernel_gradient, fallback_gradient, rtol=1e-6, atol=1e-7)
