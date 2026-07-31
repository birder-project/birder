import logging
import math
import os
import unittest
from itertools import permutations
from types import ModuleType
from unittest.mock import patch

import torch
import torch.nn.functional as F

from birder.kernels import load_kernel

logging.disable(logging.CRITICAL)


class TestKernelLoading(unittest.TestCase):
    def test_global_enablement(self) -> None:
        with (
            patch.object(load_kernel, "_CUSTOM_KERNELS_ENABLED", True),
            patch.object(load_kernel, "_DISABLED_CUSTOM_KERNELS", set()),
            patch.dict(os.environ, {"DISABLE_CUSTOM_KERNELS": "0"}),
        ):
            self.assertTrue(load_kernel.is_custom_kernels_enabled())
            self.assertTrue(load_kernel.is_custom_kernels_enabled("test_kernel"))

            load_kernel.set_custom_kernels_enabled(False)
            self.assertFalse(load_kernel.is_custom_kernels_enabled())
            self.assertFalse(load_kernel.is_custom_kernels_enabled("test_kernel"))

            load_kernel.set_custom_kernels_enabled(True)
            with patch.dict(os.environ, {"DISABLE_CUSTOM_KERNELS": "1"}):
                self.assertFalse(load_kernel.is_custom_kernels_enabled())
                self.assertFalse(load_kernel.is_custom_kernels_enabled("test_kernel"))

    def test_individual_enablement(self) -> None:
        with (
            patch.object(load_kernel, "_CUSTOM_KERNELS_ENABLED", True),
            patch.object(load_kernel, "_DISABLED_CUSTOM_KERNELS", set()),
            patch.dict(
                os.environ,
                {"DISABLE_CUSTOM_KERNELS": "0", "DISABLE_CUSTOM_KERNELS_TEST_KERNEL": "0"},
            ),
        ):
            self.assertTrue(load_kernel.is_custom_kernels_enabled("test_kernel"))
            self.assertTrue(load_kernel.is_custom_kernels_enabled("other_kernel"))

            load_kernel.set_custom_kernel_enabled("test_kernel", False)
            self.assertFalse(load_kernel.is_custom_kernels_enabled("test_kernel"))
            self.assertTrue(load_kernel.is_custom_kernels_enabled("other_kernel"))

            load_kernel.set_custom_kernel_enabled("test_kernel", True)
            self.assertTrue(load_kernel.is_custom_kernels_enabled("test_kernel"))

            with patch.dict(os.environ, {"DISABLE_CUSTOM_KERNELS_TEST_KERNEL": "1"}):
                self.assertFalse(load_kernel.is_custom_kernels_enabled("test_kernel"))
                self.assertTrue(load_kernel.is_custom_kernels_enabled("other_kernel"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestLinearAssignmentKernel(unittest.TestCase):
    linear_assignment: ModuleType

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        linear_assignment = load_kernel.load_linear_assignment()
        if linear_assignment is None:
            raise RuntimeError("Linear assignment kernel failed to load")

        cls.linear_assignment = linear_assignment

    @staticmethod
    def _supported_dtypes() -> tuple[torch.dtype, ...]:
        return (torch.float32, torch.float64)

    @staticmethod
    def _minimum_cost_columns(cost: torch.Tensor) -> tuple[int, ...]:
        num_rows, num_columns = cost.shape
        return min(
            permutations(range(num_columns), num_rows),
            key=lambda columns: sum(cost[row_idx, column_idx].item() for row_idx, column_idx in enumerate(columns)),
        )

    def _assert_matches_reference(self, cost: torch.Tensor) -> None:
        col4row, row4col = self.linear_assignment.batch_linear_assignment(cost)

        cost_cpu = cost.cpu()
        batch_size, num_rows, num_columns = cost_cpu.shape
        reference_col4row = torch.full((batch_size, num_rows), -1, dtype=torch.int64)
        reference_row4col = torch.full((batch_size, num_columns), -1, dtype=torch.int64)

        for batch_idx in range(batch_size):
            if num_rows <= num_columns:
                best_columns = self._minimum_cost_columns(cost_cpu[batch_idx])
                for row_idx, column_idx in enumerate(best_columns):
                    reference_col4row[batch_idx, row_idx] = column_idx
                    reference_row4col[batch_idx, column_idx] = row_idx

            else:
                best_rows = self._minimum_cost_columns(cost_cpu[batch_idx].transpose(0, 1))
                for column_idx, row_idx in enumerate(best_rows):
                    reference_col4row[batch_idx, row_idx] = column_idx
                    reference_row4col[batch_idx, column_idx] = row_idx

        torch.testing.assert_close(col4row.cpu(), reference_col4row, rtol=0, atol=0)
        torch.testing.assert_close(row4col.cpu(), reference_row4col, rtol=0, atol=0)

    def test_tensor_properties(self) -> None:
        device = torch.device("cuda")

        for dtype in self._supported_dtypes():
            for num_rows, num_columns in ((3, 5), (5, 3)):
                with self.subTest(dtype=dtype, shape=(3, num_rows, num_columns)):
                    generator = torch.Generator(device=device).manual_seed(0)
                    cost_storage = torch.rand(
                        (3, num_rows, num_columns, 2), dtype=dtype, device=device, generator=generator
                    )
                    cost = cost_storage[..., 0]
                    self.assertFalse(cost.is_contiguous())

                    col4row, row4col = self.linear_assignment.batch_linear_assignment(cost)

                    self.assertEqual(col4row.shape, (3, num_rows))
                    self.assertEqual(row4col.shape, (3, num_columns))
                    self.assertEqual(col4row.dtype, torch.int64)
                    self.assertEqual(row4col.dtype, torch.int64)
                    self.assertEqual(col4row.device, cost.device)
                    self.assertEqual(row4col.device, cost.device)
                    self.assertTrue(col4row.is_contiguous())
                    self.assertTrue(row4col.is_contiguous())

    def test_matches_exhaustive_reference(self) -> None:
        device = torch.device("cuda")
        wide_cost_values = (
            (
                (1.0, 9.0, 9.0, 9.0),
                (9.0, 2.0, 9.0, 9.0),
                (9.0, 9.0, 3.0, 4.0),
            ),
            (
                (4.0, -1.0, 3.0, 8.0),
                (2.0, 0.0, 5.0, 7.0),
                (3.0, 2.0, -2.0, 6.0),
            ),
        )
        tall_cost_values = (
            (
                (8.0, 4.0, 7.0),
                (5.0, 2.0, 3.0),
                (9.0, 6.0, 7.0),
                (9.0, 4.0, 8.0),
            ),
            (
                (3.0, 8.0, 5.0),
                (4.0, -2.0, 6.0),
                (7.0, 6.0, 9.0),
                (5.0, 4.0, -3.0),
            ),
        )

        for dtype in self._supported_dtypes():
            for name, cost_values in (("wide", wide_cost_values), ("tall", tall_cost_values)):
                with self.subTest(dtype=dtype, shape=name):
                    cost = torch.tensor(cost_values, dtype=dtype, device=device)
                    self._assert_matches_reference(cost)

        # Preserve distinctions that disappear when float64 costs are narrowed to float32
        precise_cost = torch.tensor((((1.0, 1.0), (1.0, 1.0 + 1e-10)),), dtype=torch.float64, device=device)
        self._assert_matches_reference(precise_cost)

    def test_deterministic_tie_breaking(self) -> None:
        device = torch.device("cuda")
        for dtype in self._supported_dtypes():
            for num_rows, num_columns in ((3, 5), (5, 3)):
                with self.subTest(dtype=dtype, shape=(num_rows, num_columns)):
                    cost = torch.ones((17, num_rows, num_columns), dtype=dtype, device=device)
                    col4row, row4col = self.linear_assignment.batch_linear_assignment(cost)

                    assignment_size = min(num_rows, num_columns)
                    expected_col4row = torch.full((17, num_rows), -1, dtype=torch.int64, device=device)
                    expected_row4col = torch.full((17, num_columns), -1, dtype=torch.int64, device=device)
                    diagonal = torch.arange(assignment_size, dtype=torch.int64, device=device)
                    expected_col4row[:, :assignment_size] = diagonal
                    expected_row4col[:, :assignment_size] = diagonal

                    torch.testing.assert_close(col4row, expected_col4row, rtol=0, atol=0)
                    torch.testing.assert_close(row4col, expected_row4col, rtol=0, atol=0)

    def test_input_validation(self) -> None:
        device = torch.device("cuda")

        shape: tuple[int, ...]
        for shape in ((3, 4), (1, 2, 3, 4)):
            with self.subTest(shape=shape):
                with self.assertRaises(RuntimeError):
                    self.linear_assignment.batch_linear_assignment(torch.rand(shape, device=device))

        for dtype in (torch.float16, torch.bfloat16, torch.int64):
            with self.subTest(dtype=dtype):
                with self.assertRaises(RuntimeError):
                    self.linear_assignment.batch_linear_assignment(torch.ones((1, 2, 3), dtype=dtype, device=device))

        # Empty inputs
        for shape in ((0, 3, 5), (2, 0, 5), (2, 5, 0), (2, 0, 0)):
            with self.subTest(shape=shape):
                cost = torch.empty(shape, device=device)
                self._assert_matches_reference(cost)

    def test_uses_current_cuda_stream(self) -> None:
        cost = torch.tensor(
            (
                ((1.0, 9.0, 9.0, 9.0), (9.0, 2.0, 9.0, 9.0), (9.0, 9.0, 3.0, 4.0)),
                ((4.0, -1.0, 3.0, 8.0), (2.0, 0.0, 5.0, 7.0), (3.0, 2.0, -2.0, 6.0)),
            ),
            device="cuda",
        )
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            expected_col4row, expected_row4col = self.linear_assignment.batch_linear_assignment(cost)

        stream.synchronize()
        expected_col4row = expected_col4row.cpu()
        expected_row4col = expected_row4col.cpu()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            col4row, row4col = self.linear_assignment.batch_linear_assignment(cost)

        # Erase writes from kernels that may have run outside the captured stream
        col4row.fill_(-123)
        row4col.fill_(-123)

        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(col4row.cpu(), expected_col4row, rtol=0, atol=0)
        torch.testing.assert_close(row4col.cpu(), expected_row4col, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestMSDAKernel(unittest.TestCase):
    msda: ModuleType

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        msda = load_kernel.load_msda()
        if msda is None:
            raise RuntimeError("MSDA kernel failed to load")

        cls.msda = msda

    @staticmethod
    def _supported_dtypes() -> tuple[torch.dtype, ...]:
        return (torch.bfloat16, torch.float16, torch.float32, torch.float64)

    def test_standard_tensor_properties(self) -> None:
        device = torch.device("cuda")
        im2col_step = 2  # Batch size 3 exercises the remainder batch.
        spatial_shapes = torch.tensor(((2, 3), (1, 2)), dtype=torch.int64, device=device)
        level_start_index = torch.tensor((0, 6), dtype=torch.int64, device=device)

        for dtype in self._supported_dtypes():
            with self.subTest(dtype=dtype):
                generator = torch.Generator(device=device).manual_seed(0)
                value = torch.rand((3, 8, 2, 4), dtype=dtype, device=device, generator=generator)
                sampling_locations = torch.rand((3, 2, 2, 2, 2, 2), dtype=dtype, device=device, generator=generator)
                attention_weights = torch.rand((3, 2, 2, 2, 2), dtype=dtype, device=device, generator=generator)

                output = self.msda.ms_deform_attn_forward(
                    value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
                )

                self.assertEqual(output.shape, (3, 2, 8))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.device, value.device)
                self.assertTrue(output.is_contiguous())
                self.assertTrue(torch.isfinite(output).all().item())

                grad_output = torch.rand(output.shape, dtype=dtype, device=device, generator=generator)
                grad_value, grad_sampling_locations, grad_attention_weights = self.msda.ms_deform_attn_backward(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    grad_output,
                    im2col_step,
                )

                for gradient, source in zip(
                    (grad_value, grad_sampling_locations, grad_attention_weights),
                    (value, sampling_locations, attention_weights),
                    strict=True,
                ):
                    self.assertEqual(gradient.shape, source.shape)
                    self.assertEqual(gradient.dtype, dtype)
                    self.assertEqual(gradient.device, value.device)
                    self.assertTrue(gradient.is_contiguous())
                    self.assertTrue(torch.isfinite(gradient).all().item())

    def test_standard_matches_reference(self) -> None:  # pylint: disable=too-many-locals
        device = torch.device("cuda")
        im2col_step = 2  # Batch size 3 exercises the remainder batch.
        generator = torch.Generator(device=device).manual_seed(1)
        spatial_shapes = torch.tensor(((2, 3), (1, 2)), dtype=torch.int64, device=device)
        level_start_index = torch.tensor((0, 6), dtype=torch.int64, device=device)
        value = torch.rand((3, 8, 2, 4), dtype=torch.float64, device=device, generator=generator)
        sampling_locations = torch.rand((3, 2, 2, 2, 2, 2), dtype=torch.float64, device=device, generator=generator)
        sampling_locations.mul_(1.2).sub_(0.1)
        attention_weights = torch.rand((3, 2, 2, 2, 2), dtype=torch.float64, device=device, generator=generator)
        attention_weights /= attention_weights.sum(dim=(3, 4), keepdim=True)

        output = self.msda.ms_deform_attn_forward(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
        )

        self.assertEqual(output.shape, (3, 2, 8))
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(output.device, value.device)

        # Build an independent PyTorch reference by splitting value into feature levels and sampling each level
        reference_value = value.cpu().requires_grad_(True)
        reference_sampling_locations = sampling_locations.cpu().requires_grad_(True)
        reference_attention_weights = attention_weights.cpu().requires_grad_(True)
        reference_value_by_level = reference_value.permute(0, 2, 3, 1).contiguous().split((6, 2), dim=-1)
        reference_attention_weights_reshaped = reference_attention_weights.transpose(1, 2).reshape(6, 1, 2, 2, 2)
        reference_output = reference_value.new_zeros((6, 4, 2))
        for level_id, (height, width) in enumerate(((2, 3), (1, 2))):
            value_level = reference_value_by_level[level_id].reshape(6, 4, height, width)
            sampling_grid = (
                (2 * reference_sampling_locations[:, :, :, level_id] - 1).transpose(1, 2).reshape(6, 2, 2, 2)
            )
            sampled_value = F.grid_sample(
                value_level, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            reference_output += (sampled_value * reference_attention_weights_reshaped[:, :, :, level_id, :]).sum(-1)

        reference_output = reference_output.reshape(3, 8, 2).transpose(1, 2).contiguous()
        torch.testing.assert_close(output.cpu(), reference_output, rtol=1e-6, atol=1e-7)

        # Compare the raw backward kernel with autograd gradients from the PyTorch reference
        grad_output = torch.linspace(-0.75, 0.9, steps=output.numel(), dtype=torch.float64, device=device).reshape_as(
            output
        )
        gradients = self.msda.ms_deform_attn_backward(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, grad_output, im2col_step
        )
        reference_gradients = torch.autograd.grad(
            reference_output,
            (reference_value, reference_sampling_locations, reference_attention_weights),
            grad_output.cpu(),
        )
        for gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
            torch.testing.assert_close(gradient.cpu(), reference_gradient, rtol=1e-5, atol=1e-6)

    def test_packed_tensor_properties(self) -> None:
        device = torch.device("cuda")
        im2col_step = 2  # Batch size 3 exercises the remainder batch.
        spatial_shapes = torch.tensor(((2, 3), (1, 2)), dtype=torch.int64, device=device)
        level_start_index = torch.tensor((0, 6), dtype=torch.int64, device=device)
        num_points_per_level = torch.tensor((1, 3), dtype=torch.int64, device=device)

        for dtype in self._supported_dtypes():
            with self.subTest(dtype=dtype):
                generator = torch.Generator(device=device).manual_seed(2)
                value = torch.rand((3, 8, 2, 3), dtype=dtype, device=device, generator=generator)
                sampling_locations = torch.rand((3, 2, 2, 4, 2), dtype=dtype, device=device, generator=generator)
                attention_weights = torch.rand((3, 2, 2, 4), dtype=dtype, device=device, generator=generator)

                output = self.msda.ms_deform_attn_packed_forward(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    num_points_per_level,
                    im2col_step,
                )

                self.assertEqual(output.shape, (3, 2, 6))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.device, value.device)
                self.assertTrue(output.is_contiguous())
                self.assertTrue(torch.isfinite(output).all().item())

                grad_output = torch.rand(output.shape, dtype=dtype, device=device, generator=generator)
                grad_value, grad_sampling_locations, grad_attention_weights = self.msda.ms_deform_attn_packed_backward(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    num_points_per_level,
                    grad_output,
                    im2col_step,
                )

                for gradient, source in zip(
                    (grad_value, grad_sampling_locations, grad_attention_weights),
                    (value, sampling_locations, attention_weights),
                    strict=True,
                ):
                    self.assertEqual(gradient.shape, source.shape)
                    self.assertEqual(gradient.dtype, dtype)
                    self.assertEqual(gradient.device, value.device)
                    self.assertTrue(gradient.is_contiguous())
                    self.assertTrue(torch.isfinite(gradient).all().item())

    def test_packed_matches_reference(self) -> None:  # pylint: disable=too-many-locals
        device = torch.device("cuda")
        im2col_step = 2  # Batch size 3 exercises the remainder batch.
        generator = torch.Generator(device=device).manual_seed(3)
        spatial_shapes = torch.tensor(((2, 3), (1, 2)), dtype=torch.int64, device=device)
        level_start_index = torch.tensor((0, 6), dtype=torch.int64, device=device)
        num_points_per_level = torch.tensor((1, 3), dtype=torch.int64, device=device)
        value = torch.rand((3, 8, 2, 3), dtype=torch.float64, device=device, generator=generator)
        sampling_locations = torch.rand((3, 2, 2, 4, 2), dtype=torch.float64, device=device, generator=generator)
        sampling_locations.mul_(1.2).sub_(0.1)
        attention_weights = torch.rand((3, 2, 2, 4), dtype=torch.float64, device=device, generator=generator)
        attention_weights /= attention_weights.sum(dim=3, keepdim=True)

        output = self.msda.ms_deform_attn_packed_forward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            num_points_per_level,
            im2col_step,
        )

        self.assertEqual(output.shape, (3, 2, 6))
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(output.device, value.device)

        reference_value = value.cpu().requires_grad_(True)
        reference_sampling_locations = sampling_locations.cpu().requires_grad_(True)
        reference_attention_weights = attention_weights.cpu().requires_grad_(True)
        reference_value_by_level = reference_value.permute(0, 2, 3, 1).flatten(0, 1).split((6, 2), dim=-1)
        reference_sampling_locations_by_level = reference_sampling_locations.permute(0, 2, 1, 3, 4).split((1, 3), dim=3)
        sampled_values = []
        for level_id, (height, width) in enumerate(((2, 3), (1, 2))):
            value_level = reference_value_by_level[level_id].reshape(6, 3, height, width)
            sampling_grid = (2 * reference_sampling_locations_by_level[level_id] - 1).flatten(0, 1)
            sampled_values.append(
                F.grid_sample(value_level, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            )

        reference_attention_weights_reshaped = reference_attention_weights.permute(0, 2, 1, 3).reshape(6, 1, 2, 4)
        reference_output = (torch.concat(sampled_values, dim=-1) * reference_attention_weights_reshaped).sum(-1)
        reference_output = reference_output.reshape(3, 6, 2).transpose(1, 2).contiguous()
        torch.testing.assert_close(output.cpu(), reference_output, rtol=1e-6, atol=1e-7)

        grad_output = torch.linspace(-0.75, 0.9, steps=output.numel(), dtype=torch.float64, device=device).reshape_as(
            output
        )
        gradients = self.msda.ms_deform_attn_packed_backward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            num_points_per_level,
            grad_output,
            im2col_step,
        )
        reference_gradients = torch.autograd.grad(
            reference_output,
            (reference_value, reference_sampling_locations, reference_attention_weights),
            grad_output.cpu(),
        )
        for gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
            torch.testing.assert_close(gradient.cpu(), reference_gradient, rtol=1e-5, atol=1e-6)

    def test_input_validation(self) -> None:
        device = torch.device("cuda")
        im2col_step = 1
        value = torch.rand((1, 5, 1, 4), device=device)
        spatial_shapes = torch.tensor(((2, 2), (1, 1)), dtype=torch.int64, device=device)
        level_start_index = torch.tensor((0, 4), dtype=torch.int64, device=device)
        sampling_locations = torch.rand((1, 1, 1, 2, 2, 2), device=device)
        attention_weights = torch.rand((1, 1, 1, 2, 2), device=device)
        valid_inputs = [value, spatial_shapes, level_start_index, sampling_locations, attention_weights]
        input_names = ("value", "spatial_shapes", "level_start_index", "sampling_locations", "attention_weights")

        # Every forward input must be contiguous
        for index, name in enumerate(input_names):
            with self.subTest(input=name):
                tensor = valid_inputs[index]
                storage = torch.empty((*tensor.shape, 2), dtype=tensor.dtype, device=tensor.device)
                noncontiguous_tensor = storage[..., 0]
                noncontiguous_tensor.copy_(tensor)
                invalid_inputs = valid_inputs.copy()
                invalid_inputs[index] = noncontiguous_tensor
                with self.assertRaises(RuntimeError):
                    self.msda.ms_deform_attn_forward(
                        invalid_inputs[0],
                        invalid_inputs[1],
                        invalid_inputs[2],
                        invalid_inputs[3],
                        invalid_inputs[4],
                        im2col_step,
                    )

        # The backward kernel also requires a contiguous output gradient
        grad_output_storage = torch.empty((1, 1, 4, 2), device=device)
        noncontiguous_grad_output = grad_output_storage[..., 0]
        with self.assertRaises(RuntimeError):
            self.msda.ms_deform_attn_backward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                noncontiguous_grad_output,
                im2col_step,
            )

        # All inputs must be on the same device
        with self.assertRaises(RuntimeError):
            self.msda.ms_deform_attn_forward(
                value,
                spatial_shapes.cpu(),
                level_start_index,
                sampling_locations,
                attention_weights,
                im2col_step,
            )

    def test_packed_input_validation(self) -> None:
        device = torch.device("cuda")
        im2col_step = 1
        value = torch.rand((1, 5, 1, 4), device=device)
        spatial_shapes = torch.tensor(((2, 2), (1, 1)), dtype=torch.int64, device=device)
        level_start_index = torch.tensor((0, 4), dtype=torch.int64, device=device)
        sampling_locations = torch.rand((1, 1, 1, 3, 2), device=device)
        attention_weights = torch.rand((1, 1, 1, 3), device=device)
        num_points_per_level = torch.tensor((1, 2), dtype=torch.int64, device=device)

        # Sampling locations must use the packed shape
        with self.assertRaises(RuntimeError):
            self.msda.ms_deform_attn_packed_forward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations.unsqueeze(3),
                attention_weights,
                num_points_per_level,
                im2col_step,
            )

        # Attention weights must use the packed shape
        with self.assertRaises(RuntimeError):
            self.msda.ms_deform_attn_packed_forward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights.unsqueeze(3),
                num_points_per_level,
                im2col_step,
            )

        # Point counts must contain one entry per feature level
        with self.assertRaises(RuntimeError):
            self.msda.ms_deform_attn_packed_forward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                num_points_per_level[:1].contiguous(),
                im2col_step,
            )

        # Sampling locations and attention weights must contain the same number of points
        with self.assertRaises(RuntimeError):
            self.msda.ms_deform_attn_packed_forward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights[..., :-1].contiguous(),
                num_points_per_level,
                im2col_step,
            )

        # Point counts must be contiguous
        point_storage = torch.empty((2, 2), dtype=torch.int64, device=device)
        noncontiguous_num_points = point_storage[:, 0]
        noncontiguous_num_points.copy_(num_points_per_level)
        with self.assertRaises(RuntimeError):
            self.msda.ms_deform_attn_packed_forward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                noncontiguous_num_points,
                im2col_step,
            )

    def test_uses_current_cuda_stream(self) -> None:
        im2col_step = 1
        value = torch.arange(1, 21, dtype=torch.float32, device="cuda").reshape(1, 5, 1, 4)
        spatial_shapes = torch.tensor(((2, 2), (1, 1)), dtype=torch.int64, device="cuda")
        level_start_index = torch.tensor((0, 4), dtype=torch.int64, device="cuda")
        sampling_locations = torch.full((1, 1, 1, 2, 2, 2), 0.5, device="cuda")
        attention_weights = torch.ones((1, 1, 1, 2, 2), device="cuda")
        grad_output = torch.ones((1, 1, 4), device="cuda")
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            expected_output = self.msda.ms_deform_attn_forward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                im2col_step,
            )
            expected_gradients = self.msda.ms_deform_attn_backward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                grad_output,
                im2col_step,
            )

        stream.synchronize()
        expected_output = expected_output.cpu()
        expected_gradients = tuple(gradient.cpu() for gradient in expected_gradients)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = self.msda.ms_deform_attn_forward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                im2col_step,
            )
            gradients = self.msda.ms_deform_attn_backward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                grad_output,
                im2col_step,
            )

        # Erase writes from kernels that may have run outside the captured stream
        output.fill_(-123.0)
        for gradient in gradients:
            gradient.fill_(-123.0)

        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(output.cpu(), expected_output)
        for gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(gradient.cpu(), expected_gradient)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestSoftNMSKernel(unittest.TestCase):
    soft_nms: ModuleType

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        soft_nms = load_kernel.load_soft_nms()
        if soft_nms is None:
            raise RuntimeError("Soft-NMS kernel failed to load")

        cls.soft_nms = soft_nms

    @staticmethod
    def _supported_dtypes() -> tuple[torch.dtype, ...]:
        return (torch.bfloat16, torch.float16, torch.float32, torch.float64)

    def test_tensor_properties(self) -> None:
        boxes_values = (
            (2.0, 2.0, 12.0, 12.0),
            (0.0, 0.0, 10.0, 10.0),
            (20.0, 20.0, 30.0, 30.0),
            (1.0, 1.0, 9.0, 9.0),
            (0.0, 0.0, 10.0, 10.0),
        )
        scores_values = (0.75, 0.9, 0.8, 0.85, 0.7)
        num_boxes = len(boxes_values)

        for dtype in self._supported_dtypes():
            with self.subTest(dtype=dtype):
                boxes_storage = torch.empty((num_boxes, 8), dtype=dtype, device="cuda")
                boxes = boxes_storage[:, ::2]
                boxes.copy_(torch.tensor(boxes_values, dtype=dtype, device="cuda"))
                scores_storage = torch.empty((2 * num_boxes,), dtype=dtype, device="cuda")
                scores = scores_storage[::2]
                scores.copy_(torch.tensor(scores_values, dtype=dtype, device="cuda"))
                class_storage = torch.empty((2 * num_boxes,), dtype=torch.int64, device="cuda")
                class_ids = class_storage[::2]
                class_ids.copy_(torch.tensor((0, 0, 0, 0, 1), dtype=torch.int64, device="cuda"))
                self.assertFalse(boxes.is_contiguous())
                self.assertFalse(scores.is_contiguous())
                self.assertFalse(class_ids.is_contiguous())
                boxes_before = boxes.clone()
                scores_before = scores.clone()
                class_ids_before = class_ids.clone()

                updated_scores, keep = self.soft_nms.soft_nms(boxes, scores, class_ids, 0.5, -1.0)

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
                torch.testing.assert_close(class_ids, class_ids_before, rtol=0, atol=0)

    def test_matches_reference(self) -> None:  # pylint: disable=too-many-locals
        sigma = 0.5
        score_threshold = 0.3
        boxes_values = (
            (2.0, 2.0, 12.0, 12.0),
            (0.0, 0.0, 10.0, 10.0),
            (20.0, 20.0, 30.0, 30.0),
            (1.0, 1.0, 9.0, 9.0),
            (0.0, 0.0, 10.0, 10.0),
        )
        scores_values = (0.75, 0.9, 0.8, 0.85, 0.7)

        # Build an independent scalar reference that follows Gaussian Soft-NMS
        reference_boxes = [list(box) for box in boxes_values]
        reference_scores = list(scores_values)
        reference_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in reference_boxes]
        reference_indices = list(range(len(reference_boxes)))
        for idx in range(len(reference_boxes) - 1):
            max_idx = max(range(idx + 1, len(reference_scores)), key=reference_scores.__getitem__)
            if reference_scores[idx] < reference_scores[max_idx]:
                reference_boxes[idx], reference_boxes[max_idx] = reference_boxes[max_idx], reference_boxes[idx]
                reference_scores[idx], reference_scores[max_idx] = reference_scores[max_idx], reference_scores[idx]
                reference_areas[idx], reference_areas[max_idx] = reference_areas[max_idx], reference_areas[idx]
                reference_indices[idx], reference_indices[max_idx] = reference_indices[max_idx], reference_indices[idx]

            box = reference_boxes[idx]
            for remaining_idx in range(idx + 1, len(reference_boxes)):
                remaining_box = reference_boxes[remaining_idx]
                intersection_width = max(min(box[2], remaining_box[2]) - max(box[0], remaining_box[0]), 0.0)
                intersection_height = max(min(box[3], remaining_box[3]) - max(box[1], remaining_box[1]), 0.0)
                intersection = intersection_width * intersection_height
                if intersection > 0:
                    union = reference_areas[idx] + reference_areas[remaining_idx] - intersection
                    iou = intersection / (union + 1e-8)
                else:
                    iou = 0.0

                reference_scores[remaining_idx] *= math.exp(-(iou * iou) / sigma)

        expected_scores = torch.tensor(
            [score for score in reference_scores if score > score_threshold], dtype=torch.float64
        )
        expected_keep = torch.tensor(
            [
                index
                for score, index in zip(reference_scores, reference_indices, strict=True)
                if score > score_threshold
            ],
            dtype=torch.int64,
        )
        tolerances = {
            torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1e-2),
            torch.float32: (1e-5, 1e-6),
            torch.float64: (1e-10, 1e-12),
        }
        num_boxes = len(boxes_values)

        for dtype in self._supported_dtypes():
            with self.subTest(dtype=dtype):
                boxes_storage = torch.empty((num_boxes, 8), dtype=dtype, device="cuda")
                boxes = boxes_storage[:, ::2]
                boxes.copy_(torch.tensor(boxes_values, dtype=dtype, device="cuda"))
                scores_storage = torch.empty((2 * num_boxes,), dtype=dtype, device="cuda")
                scores = scores_storage[::2]
                scores.copy_(torch.tensor(scores_values, dtype=dtype, device="cuda"))
                class_ids = torch.zeros(num_boxes, dtype=torch.int64, device="cuda")

                updated_scores, keep = self.soft_nms.soft_nms(boxes, scores, class_ids, sigma, score_threshold)

                rtol, atol = tolerances[dtype]
                torch.testing.assert_close(updated_scores.cpu(), expected_scores.to(dtype=dtype), rtol=rtol, atol=atol)
                torch.testing.assert_close(keep.cpu(), expected_keep, rtol=0, atol=0)

    def test_single_box_thresholding(self) -> None:
        boxes = torch.tensor(((0.0, 0.0, 10.0, 10.0),), dtype=torch.float64, device="cuda")
        scores = torch.tensor((0.5,), dtype=torch.float64, device="cuda")
        class_ids = torch.tensor((7,), dtype=torch.int64, device="cuda")

        updated_scores, keep = self.soft_nms.soft_nms(boxes, scores, class_ids, 0.5, 0.49)
        torch.testing.assert_close(updated_scores, scores, rtol=0, atol=0)
        torch.testing.assert_close(keep, torch.zeros(1, dtype=torch.int64, device="cuda"), rtol=0, atol=0)

        updated_scores, keep = self.soft_nms.soft_nms(boxes, scores, class_ids, 0.5, 0.5)
        self.assertEqual(updated_scores.shape, (0,))
        self.assertEqual(keep.shape, (0,))

    def test_degenerate_boxes(self) -> None:
        boxes_values = (
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 2.0),
            (2.0, 2.0, 3.0, 2.0),
        )
        scores_values = (0.9, 0.8, 0.7, 0.6)
        expected_scores = torch.tensor(scores_values, dtype=torch.float64)
        expected_keep = torch.arange(4, dtype=torch.int64)

        boxes = torch.tensor(boxes_values, dtype=torch.float64, device="cuda")
        scores = torch.tensor(scores_values, dtype=torch.float64, device="cuda")
        class_ids = torch.zeros(4, dtype=torch.int64, device="cuda")

        updated_scores, keep = self.soft_nms.soft_nms(boxes, scores, class_ids, 0.5, 0.0)

        torch.testing.assert_close(updated_scores.cpu(), expected_scores, rtol=0, atol=0)
        torch.testing.assert_close(keep.cpu(), expected_keep, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(updated_scores).all().item())

    def test_class_aware_suppression(self) -> None:
        boxes = torch.tensor(
            (
                (1024.0, 1024.0, 1152.0, 1152.0),
                (1024.0, 1024.0, 1152.0, 1152.0),
                (1024.0, 1024.0, 1152.0, 1152.0),
                (1024.0, 1024.0, 1152.0, 1152.0),
            ),
            device="cuda",
        )
        scores = torch.tensor((0.9, 0.8, 0.7, 0.6), device="cuda")
        class_ids = torch.tensor((91, 91, 90, 90), dtype=torch.int64, device="cuda")

        updated_scores, keep = self.soft_nms.soft_nms(boxes, scores, class_ids, 0.5, 0.1)

        decay = math.exp(-2.0)
        expected_scores = torch.tensor((0.9, 0.7, 0.8 * decay), device="cuda")
        expected_keep = torch.tensor((0, 2, 1), dtype=torch.int64, device="cuda")
        torch.testing.assert_close(updated_scores, expected_scores)
        torch.testing.assert_close(keep, expected_keep, rtol=0, atol=0)

    def test_equal_scores_preserve_input_order(self) -> None:
        boxes = torch.tensor(
            (
                (0.0, 0.0, 10.0, 10.0),
                (20.0, 20.0, 30.0, 30.0),
                (40.0, 40.0, 50.0, 50.0),
                (60.0, 60.0, 70.0, 70.0),
            ),
            device="cuda",
        )
        scores = torch.full((4,), 0.5, device="cuda")
        class_ids = torch.tensor((2, 1, 2, 1), dtype=torch.int64, device="cuda")

        updated_scores, keep = self.soft_nms.soft_nms(boxes, scores, class_ids, 0.5, 0.1)

        torch.testing.assert_close(updated_scores, scores, rtol=0, atol=0)
        torch.testing.assert_close(keep, torch.arange(4, dtype=torch.int64, device="cuda"), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "multiple CUDA devices not available")
    def test_uses_input_cuda_device(self) -> None:
        boxes = torch.tensor(
            ((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)),
            device="cuda:1",
        )
        scores = torch.tensor((0.9, 0.8), device="cuda:1")
        class_ids = torch.zeros(2, dtype=torch.int64, device="cuda:1")

        with torch.cuda.device(0):
            updated_scores, keep = self.soft_nms.soft_nms(boxes, scores, class_ids, 0.5, 0.1)

        self.assertEqual(updated_scores.device, torch.device("cuda:1"))
        self.assertEqual(keep.device, torch.device("cuda:1"))
        torch.testing.assert_close(updated_scores, torch.tensor((0.9, 0.8 * math.exp(-2.0)), device="cuda:1"))
        torch.testing.assert_close(keep, torch.tensor((0, 1), dtype=torch.int64, device="cuda:1"), rtol=0, atol=0)

    def test_low_precision_uses_float32_calculation(self) -> None:
        boxes_values = (
            (1024.0, 1024.0, 1152.0, 1152.0),
            (1024.0, 1024.0, 1152.0, 1152.0),
            (896.0, 896.0, 1088.0, 1088.0),
            (896.0, 896.0, 1088.0, 1088.0),
        )
        scores_values = (0.9, 0.8, 0.7, 0.6)
        class_ids = torch.tensor((91, 91, 3, 3), dtype=torch.int64, device="cuda")
        reference_boxes = torch.tensor(boxes_values, dtype=torch.float32, device="cuda")
        reference_scores = torch.tensor(scores_values, dtype=torch.float32, device="cuda")
        expected_scores, expected_keep = self.soft_nms.soft_nms(
            reference_boxes, reference_scores, class_ids, 0.5, 0.001
        )

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                boxes = torch.tensor(boxes_values, dtype=dtype, device="cuda")
                scores = torch.tensor(scores_values, dtype=dtype, device="cuda")
                updated_scores, keep = self.soft_nms.soft_nms(boxes, scores, class_ids, 0.5, 0.001)

                torch.testing.assert_close(updated_scores, expected_scores.to(dtype), rtol=5e-3, atol=5e-3)
                torch.testing.assert_close(keep, expected_keep, rtol=0, atol=0)

    def test_input_validation(self) -> None:
        device = torch.device("cuda")
        for shape in ((4,), (2, 5), (1, 2, 3)):
            with self.subTest(shape=shape):
                boxes = torch.rand(shape, device=device)
                scores = torch.rand(shape[0], device=device)
                class_ids = torch.zeros(shape[0], dtype=torch.int64, device=device)
                with self.assertRaises(RuntimeError):
                    self.soft_nms.soft_nms(boxes, scores, class_ids, 0.5, 0.1)

        boxes = torch.rand((2, 4), device=device)
        invalid_score_inputs = (
            (torch.empty((0, 4), device=device), torch.tensor(0.5, device=device)),
            (torch.rand((1, 4), device=device), torch.tensor(0.5, device=device)),
            (boxes, torch.rand((2, 1), device=device)),
            (boxes, torch.rand(1, device=device)),
            (boxes, torch.rand(3, device=device)),
        )
        for invalid_boxes, invalid_scores in invalid_score_inputs:
            with self.subTest(boxes_shape=invalid_boxes.shape, scores_shape=invalid_scores.shape):
                class_ids = torch.zeros(invalid_boxes.size(0), dtype=torch.int64, device=device)
                with self.assertRaises(RuntimeError):
                    self.soft_nms.soft_nms(invalid_boxes, invalid_scores, class_ids, 0.5, 0.1)

        with self.assertRaises(RuntimeError):
            self.soft_nms.soft_nms(
                boxes,
                torch.ones(2, dtype=torch.int64, device=device),
                torch.zeros(2, dtype=torch.int64, device=device),
                0.5,
                0.1,
            )

        scores = torch.rand(2, device=device)
        with self.assertRaises(RuntimeError):
            self.soft_nms.soft_nms(
                torch.ones((2, 4), dtype=torch.int64, device=device),
                scores,
                torch.zeros(2, dtype=torch.int64, device=device),
                0.5,
                0.1,
            )

        invalid_class_ids = (
            torch.zeros((2, 1), dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.zeros(3, dtype=torch.int64, device=device),
            torch.zeros(2, dtype=torch.float32, device=device),
        )
        for class_ids in invalid_class_ids:
            with self.subTest(class_ids_shape=class_ids.shape, class_ids_dtype=class_ids.dtype):
                with self.assertRaises(RuntimeError):
                    self.soft_nms.soft_nms(boxes, scores, class_ids, 0.5, 0.1)

        with self.assertRaises(RuntimeError):
            self.soft_nms.soft_nms(
                boxes, scores.to(torch.float64), torch.zeros(2, dtype=torch.int64, device=device), 0.5, 0.1
            )

        with self.assertRaises(RuntimeError):
            self.soft_nms.soft_nms(boxes, scores, torch.zeros(2, dtype=torch.int64, device=device), 0.0, 0.1)

        # Empty inputs
        for dtype in self._supported_dtypes():
            with self.subTest(dtype=dtype):
                boxes = torch.empty((0, 4), dtype=dtype, device="cuda")
                scores = torch.empty((0,), dtype=dtype, device="cuda")
                class_ids = torch.empty((0,), dtype=torch.int64, device="cuda")

                updated_scores, keep = self.soft_nms.soft_nms(boxes, scores, class_ids, 0.5, 0.1)

                self.assertEqual(updated_scores.shape, (0,))
                self.assertEqual(keep.shape, (0,))
                self.assertEqual(updated_scores.dtype, dtype)
                self.assertEqual(keep.dtype, torch.int64)
                self.assertEqual(updated_scores.device, scores.device)
                self.assertEqual(keep.device, boxes.device)

    def test_uses_current_cuda_stream(self) -> None:
        boxes_values = torch.tensor(
            (
                (0.0, 0.0, 10.0, 10.0),
                (0.0, 0.0, 10.0, 10.0),
                (20.0, 20.0, 30.0, 30.0),
            ),
            device="cuda",
        )
        scores_values = torch.tensor((0.9, 0.8, 0.7), device="cuda")
        class_ids_values = torch.tensor((0, 0, 1), dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()

        boxes = torch.empty_like(boxes_values)
        scores = torch.empty_like(scores_values)
        class_ids = torch.empty_like(class_ids_values)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            boxes.copy_(boxes_values)
            scores.copy_(scores_values)
            class_ids.copy_(class_ids_values)
            updated_scores, keep = self.soft_nms.soft_nms(boxes, scores, class_ids, 0.5, 0.1)

        stream.synchronize()
        expected_scores = torch.tensor((0.9, 0.8 * math.exp(-2.0), 0.7), device="cuda")
        expected_scores, expected_order = expected_scores.sort(descending=True, stable=True)
        expected_keep = torch.tensor((0, 1, 2), dtype=torch.int64, device="cuda")[expected_order]
        torch.testing.assert_close(updated_scores, expected_scores)
        torch.testing.assert_close(keep, expected_keep, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestSWAttentionKernel(unittest.TestCase):
    swattention: ModuleType

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        swattention = load_kernel.load_swattention()
        if swattention is None:
            raise RuntimeError("SWAttention kernel failed to load")

        cls.swattention = swattention

    @staticmethod
    def _supported_dtypes() -> tuple[torch.dtype, ...]:
        return (torch.bfloat16, torch.float16, torch.float32, torch.float64)

    @staticmethod
    def _local_neighborhoods(tensor: torch.Tensor, height: int, width: int, kernel_size: int) -> torch.Tensor:
        batch_size, num_heads, num_tokens, head_dim = tensor.shape
        tensor_2d = tensor.permute(0, 1, 3, 2).reshape(batch_size * num_heads, head_dim, height, width)
        return (
            F.unfold(tensor_2d, kernel_size=kernel_size, padding=kernel_size // 2)
            .reshape(batch_size, num_heads, head_dim, kernel_size * kernel_size, num_tokens)
            .permute(0, 1, 4, 3, 2)
        )

    @staticmethod
    def _valid_attention_mask(height: int, width: int, kernel_size: int, device: torch.device) -> torch.Tensor:
        return (
            F.unfold(
                torch.ones((1, 1, height, width), device=device),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            .transpose(1, 2)
            .reshape(1, 1, height * width, kernel_size * kernel_size)
            .bool()
        )

    def _assert_tensor_properties(self, tensor: torch.Tensor, shape: tuple[int, ...], reference: torch.Tensor) -> None:
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.dtype, reference.dtype)
        self.assertEqual(tensor.device, reference.device)
        self.assertTrue(tensor.is_contiguous())

    def _assert_close_for_dtype(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        rtol, atol = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (5e-2, 5e-2),
            torch.float32: (1e-5, 1e-6),
            torch.float64: (1e-10, 1e-12),
        }[actual.dtype]
        torch.testing.assert_close(actual.cpu(), expected.to(dtype=actual.dtype), rtol=rtol, atol=atol)

    def test_qk_tensor_properties(self) -> None:
        device = torch.device("cuda")
        height, width = 4, 5
        kernel_size = 3
        cuda_threads = 32
        feature_shape = (2, 6, height * width, 7)
        attention_shape = (*feature_shape[:3], kernel_size * kernel_size)

        for dtype in self._supported_dtypes():
            with self.subTest(dtype=dtype):
                generator = torch.Generator(device=device).manual_seed(0)
                queries = torch.rand(feature_shape, dtype=dtype, device=device, generator=generator)
                keys = torch.rand(feature_shape, dtype=dtype, device=device, generator=generator)
                rpb = torch.rand(
                    (feature_shape[1], attention_shape[3]), dtype=dtype, device=device, generator=generator
                )
                d_attn_weight = torch.rand(attention_shape, dtype=dtype, device=device, generator=generator)

                attn_weight = self.swattention.qk_forward(queries, keys, height, width, kernel_size, cuda_threads)
                attn_weight_rpb = self.swattention.qk_rpb_forward(
                    queries, keys, rpb, height, width, kernel_size, cuda_threads
                )
                d_queries, d_keys = self.swattention.qk_backward(
                    d_attn_weight, queries, keys, height, width, kernel_size, cuda_threads
                )
                d_queries_rpb, d_keys_rpb, d_rpb = self.swattention.qk_rpb_backward(
                    d_attn_weight, queries, keys, height, width, kernel_size, cuda_threads
                )

                for output in (attn_weight, attn_weight_rpb):
                    self._assert_tensor_properties(output, attention_shape, queries)
                    valid_mask = self._valid_attention_mask(height, width, kernel_size, output.device).expand_as(output)
                    self.assertTrue(torch.isfinite(output[valid_mask]).all().item())
                    self.assertTrue(torch.isneginf(output[~valid_mask]).all().item())

                for gradient in (d_queries, d_keys, d_queries_rpb, d_keys_rpb):
                    self._assert_tensor_properties(gradient, feature_shape, queries)
                    self.assertTrue(torch.isfinite(gradient).all().item())

                self._assert_tensor_properties(d_rpb, rpb.shape, rpb)
                self.assertTrue(torch.isfinite(d_rpb).all().item())

    def test_qk_matches_reference(self) -> None:
        device = torch.device("cuda")
        height, width = 4, 5
        kernel_size = 3
        cuda_threads = 32
        feature_shape = (2, 6, height * width, 7)
        attention_shape = (*feature_shape[:3], kernel_size * kernel_size)

        for dtype in self._supported_dtypes():
            with self.subTest(dtype=dtype):
                queries = torch.linspace(-0.7, 0.9, steps=math.prod(feature_shape), dtype=torch.float64, device=device)
                queries = queries.to(dtype=dtype).reshape(feature_shape)
                keys = torch.linspace(0.8, -0.6, steps=queries.numel(), dtype=torch.float64, device=device)
                keys = keys.to(dtype=dtype).reshape(feature_shape)
                grad_output = torch.linspace(
                    -0.4, 0.5, steps=math.prod(attention_shape), dtype=torch.float64, device=device
                )
                grad_output = grad_output.to(dtype=dtype).reshape(attention_shape)

                output = self.swattention.qk_forward(queries, keys, height, width, kernel_size, cuda_threads)
                gradients = self.swattention.qk_backward(
                    grad_output, queries, keys, height, width, kernel_size, cuda_threads
                )

                reference_queries = queries.cpu().to(dtype=torch.float64).requires_grad_(True)
                reference_keys = keys.cpu().to(dtype=torch.float64).requires_grad_(True)
                local_keys = self._local_neighborhoods(reference_keys, height, width, kernel_size)
                reference_output = (reference_queries.unsqueeze(3) * local_keys).sum(dim=-1)
                valid_mask = self._valid_attention_mask(height, width, kernel_size, reference_output.device)
                reference_output = reference_output.masked_fill(~valid_mask, float("-inf"))
                reference_gradients = torch.autograd.grad(
                    reference_output,
                    (reference_queries, reference_keys),
                    grad_output.cpu().to(dtype=torch.float64),
                )

                self._assert_close_for_dtype(output, reference_output)
                for gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
                    self._assert_close_for_dtype(gradient, reference_gradient)

    def test_qk_rpb_matches_reference(self) -> None:
        device = torch.device("cuda")
        height, width = 4, 5
        kernel_size = 3
        cuda_threads = 32
        feature_shape = (2, 6, height * width, 7)
        attention_shape = (*feature_shape[:3], kernel_size * kernel_size)

        for dtype in self._supported_dtypes():
            with self.subTest(dtype=dtype):
                queries = torch.linspace(-0.6, 0.8, steps=math.prod(feature_shape), dtype=torch.float64, device=device)
                queries = queries.to(dtype=dtype).reshape(feature_shape)
                keys = torch.linspace(0.7, -0.5, steps=queries.numel(), dtype=torch.float64, device=device)
                keys = keys.to(dtype=dtype).reshape(feature_shape)
                rpb = torch.linspace(
                    -0.2, 0.3, steps=feature_shape[1] * attention_shape[3], dtype=torch.float64, device=device
                )
                rpb = rpb.to(dtype=dtype).reshape(feature_shape[1], attention_shape[3])
                grad_output = torch.linspace(
                    -0.3, 0.4, steps=math.prod(attention_shape), dtype=torch.float64, device=device
                )
                grad_output = grad_output.to(dtype=dtype).reshape(attention_shape)

                output = self.swattention.qk_rpb_forward(queries, keys, rpb, height, width, kernel_size, cuda_threads)
                gradients = self.swattention.qk_rpb_backward(
                    grad_output, queries, keys, height, width, kernel_size, cuda_threads
                )

                reference_queries = queries.cpu().to(dtype=torch.float64).requires_grad_(True)
                reference_keys = keys.cpu().to(dtype=torch.float64).requires_grad_(True)
                reference_rpb = rpb.cpu().to(dtype=torch.float64).requires_grad_(True)
                local_keys = self._local_neighborhoods(reference_keys, height, width, kernel_size)
                reference_output = (reference_queries.unsqueeze(3) * local_keys).sum(dim=-1)
                reference_output = reference_output + reference_rpb.reshape(1, feature_shape[1], 1, -1)
                valid_mask = self._valid_attention_mask(height, width, kernel_size, reference_output.device)
                reference_output = reference_output.masked_fill(~valid_mask, float("-inf"))
                reference_gradients = torch.autograd.grad(
                    reference_output,
                    (reference_queries, reference_keys, reference_rpb),
                    grad_output.cpu().to(dtype=torch.float64),
                )

                self._assert_close_for_dtype(output, reference_output)
                for gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
                    self._assert_close_for_dtype(gradient, reference_gradient)

    def test_av_tensor_properties(self) -> None:
        device = torch.device("cuda")
        height, width = 4, 5
        kernel_size = 3
        cuda_threads = 32
        feature_shape = (2, 6, height * width, 7)
        attention_shape = (*feature_shape[:3], kernel_size * kernel_size)

        for dtype in self._supported_dtypes():
            with self.subTest(dtype=dtype):
                generator = torch.Generator(device=device).manual_seed(1)
                attn_weight = torch.rand(attention_shape, dtype=dtype, device=device, generator=generator)
                values = torch.rand(feature_shape, dtype=dtype, device=device, generator=generator)
                d_output = torch.rand(feature_shape, dtype=dtype, device=device, generator=generator)

                output = self.swattention.av_forward(attn_weight, values, height, width, kernel_size, cuda_threads)
                d_attn_weight, d_values = self.swattention.av_backward(
                    d_output, attn_weight, values, height, width, kernel_size, cuda_threads
                )

                self._assert_tensor_properties(output, feature_shape, values)
                self._assert_tensor_properties(d_attn_weight, attention_shape, attn_weight)
                self._assert_tensor_properties(d_values, feature_shape, values)
                for tensor in (output, d_attn_weight, d_values):
                    self.assertTrue(torch.isfinite(tensor).all().item())

    def test_av_matches_reference(self) -> None:
        device = torch.device("cuda")
        height, width = 4, 5
        kernel_size = 3
        cuda_threads = 32
        feature_shape = (2, 6, height * width, 7)
        attention_shape = (*feature_shape[:3], kernel_size * kernel_size)

        for dtype in self._supported_dtypes():
            with self.subTest(dtype=dtype):
                attn_weight = torch.linspace(
                    -0.4, 0.6, steps=math.prod(attention_shape), dtype=torch.float64, device=device
                )
                attn_weight = attn_weight.to(dtype=dtype).reshape(attention_shape)
                values = torch.linspace(0.7, -0.5, steps=math.prod(feature_shape), dtype=torch.float64, device=device)
                values = values.to(dtype=dtype).reshape(feature_shape)
                grad_output = torch.linspace(-0.3, 0.5, steps=values.numel(), dtype=torch.float64, device=device)
                grad_output = grad_output.to(dtype=dtype).reshape(feature_shape)

                output = self.swattention.av_forward(attn_weight, values, height, width, kernel_size, cuda_threads)
                gradients = self.swattention.av_backward(
                    grad_output, attn_weight, values, height, width, kernel_size, cuda_threads
                )

                reference_attn_weight = attn_weight.cpu().to(dtype=torch.float64).requires_grad_(True)
                reference_values = values.cpu().to(dtype=torch.float64).requires_grad_(True)
                local_values = self._local_neighborhoods(reference_values, height, width, kernel_size)
                reference_output = (reference_attn_weight.unsqueeze(-1) * local_values).sum(dim=3)
                reference_gradients = torch.autograd.grad(
                    reference_output,
                    (reference_attn_weight, reference_values),
                    grad_output.cpu().to(dtype=torch.float64),
                )

                self._assert_close_for_dtype(output, reference_output)
                for gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
                    self._assert_close_for_dtype(gradient, reference_gradient)

    def test_low_precision_uses_fp32_accumulation(self) -> None:
        device = torch.device("cuda")
        cuda_threads = 32

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype, operation="qk"):
                queries = torch.tensor([[[[2048.0, 1.0, -2048.0]]]], dtype=dtype, device=device)
                keys = torch.ones_like(queries)
                output = self.swattention.qk_forward(queries, keys, 1, 1, 1, cuda_threads)
                torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)

            with self.subTest(dtype=dtype, operation="av"):
                attn_weight = torch.zeros((1, 1, 3, 9), dtype=dtype, device=device)
                attn_weight[0, 0, 1, 3:6] = torch.tensor((2048.0, 1.0, -2048.0), dtype=dtype, device=device)
                values = torch.ones((1, 1, 3, 1), dtype=dtype, device=device)
                output = self.swattention.av_forward(attn_weight, values, 1, 3, 3, cuda_threads)
                torch.testing.assert_close(output[0, 0, 1, 0], output.new_tensor(1.0), rtol=0, atol=0)

            with self.subTest(dtype=dtype, operation="rpb_backward"):
                grad_output = torch.tensor((2048.0, 1.0, -2048.0), dtype=dtype, device=device).reshape(1, 1, 3, 1)
                queries = torch.zeros((1, 1, 3, 1), dtype=dtype, device=device)
                keys = torch.zeros_like(queries)
                _, _, d_rpb = self.swattention.qk_rpb_backward(grad_output, queries, keys, 1, 3, 1, cuda_threads)
                torch.testing.assert_close(d_rpb, torch.ones_like(d_rpb), rtol=0, atol=0)

    def test_input_validation(self) -> None:
        device = torch.device("cuda")
        height, width = 3, 4
        kernel_size = 3
        cuda_threads = 32
        feature_shape = (1, 2, height * width, 3)
        attention_shape = (*feature_shape[:3], kernel_size * kernel_size)
        queries = torch.rand(feature_shape, device=device)
        keys = torch.rand(feature_shape, device=device)
        rpb = torch.rand((feature_shape[1], attention_shape[3]), device=device)
        d_attn_weight = torch.rand(attention_shape, device=device)
        attn_weight = torch.rand(attention_shape, device=device)
        values = torch.rand(feature_shape, device=device)
        d_output = torch.rand(feature_shape, device=device)
        operations = (
            ("qk_forward", self.swattention.qk_forward, [queries, keys], ("queries", "keys")),
            (
                "qk_backward",
                self.swattention.qk_backward,
                [d_attn_weight, queries, keys],
                ("d_attn_weight", "queries", "keys"),
            ),
            (
                "qk_rpb_forward",
                self.swattention.qk_rpb_forward,
                [queries, keys, rpb],
                ("queries", "keys", "rpb"),
            ),
            (
                "qk_rpb_backward",
                self.swattention.qk_rpb_backward,
                [d_attn_weight, queries, keys],
                ("d_attn_weight", "queries", "keys"),
            ),
            ("av_forward", self.swattention.av_forward, [attn_weight, values], ("attn_weight", "values")),
            (
                "av_backward",
                self.swattention.av_backward,
                [d_output, attn_weight, values],
                ("d_output", "attn_weight", "values"),
            ),
        )

        # Every tensor input must be contiguous
        for operation_name, operation, inputs, input_names in operations:
            for input_idx, input_name in enumerate(input_names):
                with self.subTest(operation=operation_name, noncontiguous_input=input_name):
                    invalid_inputs = inputs.copy()
                    invalid_inputs[input_idx] = torch.stack((inputs[input_idx], inputs[input_idx]), dim=-1)[..., 0]
                    with self.assertRaises(RuntimeError):
                        operation(*invalid_inputs, height, width, kernel_size, cuda_threads)

        # Every tensor input must be on the same device
        for operation_name, operation, inputs, input_names in operations:
            for input_idx, input_name in enumerate(input_names):
                with self.subTest(operation=operation_name, cpu_input=input_name):
                    invalid_inputs = inputs.copy()
                    invalid_inputs[input_idx] = inputs[input_idx].cpu()
                    with self.assertRaises(RuntimeError):
                        operation(*invalid_inputs, height, width, kernel_size, cuda_threads)

        # Raw entry points require one homogeneous floating-point dtype
        for operation_name, operation, inputs, input_names in operations:
            for input_idx, input_name in enumerate(input_names):
                with self.subTest(operation=operation_name, mixed_dtype_input=input_name):
                    invalid_inputs = inputs.copy()
                    invalid_inputs[input_idx] = inputs[input_idx].to(dtype=torch.float16)
                    with self.assertRaises(RuntimeError):
                        operation(*invalid_inputs, height, width, kernel_size, cuda_threads)

        # Every entry point validates the token grid
        for operation_name, operation, inputs, _input_names in operations:
            with self.subTest(operation=operation_name, validation="token_grid"):
                with self.assertRaises(RuntimeError):
                    operation(*inputs, height, width + 1, kernel_size, cuda_threads)

        # Every QK entry point validates all query/key dimensions
        for dimension in range(4):
            mismatched_shape = list(feature_shape)
            mismatched_shape[dimension] += 1
            mismatched_keys = torch.rand(mismatched_shape, device=device)
            qk_shape_cases = (
                ("qk_forward", self.swattention.qk_forward, [queries, mismatched_keys]),
                ("qk_backward", self.swattention.qk_backward, [d_attn_weight, queries, mismatched_keys]),
                ("qk_rpb_forward", self.swattention.qk_rpb_forward, [queries, mismatched_keys, rpb]),
                ("qk_rpb_backward", self.swattention.qk_rpb_backward, [d_attn_weight, queries, mismatched_keys]),
            )
            for operation_name, operation, inputs in qk_shape_cases:
                with self.subTest(operation=operation_name, mismatched_key_dimension=dimension):
                    with self.assertRaises(RuntimeError):
                        operation(*inputs, height, width, kernel_size, cuda_threads)

        invalid_attention = torch.rand((*attention_shape[:3], attention_shape[3] + 1), device=device)
        attention_shape_cases = (
            ("qk_backward", self.swattention.qk_backward, [invalid_attention, queries, keys]),
            ("qk_rpb_backward", self.swattention.qk_rpb_backward, [invalid_attention, queries, keys]),
            ("av_forward", self.swattention.av_forward, [invalid_attention, values]),
            ("av_backward", self.swattention.av_backward, [d_output, invalid_attention, values]),
        )
        for operation_name, operation, inputs in attention_shape_cases:
            with self.subTest(operation=operation_name, validation="attention_shape"):
                with self.assertRaises(RuntimeError):
                    operation(*inputs, height, width, kernel_size, cuda_threads)

        for invalid_rpb in (
            torch.rand((feature_shape[1] + 1, attention_shape[3]), device=device),
            torch.rand((feature_shape[1], attention_shape[3] + 1), device=device),
        ):
            with self.subTest(rpb_shape=invalid_rpb.shape):
                with self.assertRaises(RuntimeError):
                    self.swattention.qk_rpb_forward(
                        queries, keys, invalid_rpb, height, width, kernel_size, cuda_threads
                    )

        with self.assertRaises(RuntimeError):
            self.swattention.av_backward(
                d_output[..., :-1].contiguous(),
                attn_weight,
                values,
                height,
                width,
                kernel_size,
                cuda_threads,
            )

        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                integer_features = torch.ones(feature_shape, dtype=dtype, device=device)
                integer_attention = torch.ones(attention_shape, dtype=dtype, device=device)
                with self.assertRaises(RuntimeError):
                    self.swattention.qk_forward(
                        integer_features, integer_features, height, width, kernel_size, cuda_threads
                    )
                with self.assertRaises(RuntimeError):
                    self.swattention.av_forward(
                        integer_attention, integer_features, height, width, kernel_size, cuda_threads
                    )

    def test_uses_current_cuda_stream(self) -> None:
        device = torch.device("cuda")
        height, width = 2, 3
        kernel_size = 3
        cuda_threads = 32
        feature_shape = (1, 2, height * width, 3)
        attention_shape = (*feature_shape[:3], kernel_size * kernel_size)
        queries = torch.linspace(-0.5, 0.7, steps=math.prod(feature_shape), device=device)
        queries = queries.reshape(feature_shape)
        keys = torch.linspace(0.6, -0.4, steps=queries.numel(), device=device).reshape(feature_shape)
        rpb = torch.linspace(-0.2, 0.3, steps=feature_shape[1] * attention_shape[3], device=device)
        rpb = rpb.reshape(feature_shape[1], attention_shape[3])
        d_attn_weight = torch.linspace(-0.3, 0.4, steps=math.prod(attention_shape), device=device)
        d_attn_weight = d_attn_weight.reshape(attention_shape)
        attn_weight = torch.linspace(-0.4, 0.5, steps=d_attn_weight.numel(), device=device).reshape(attention_shape)
        values = torch.linspace(0.7, -0.6, steps=queries.numel(), device=device).reshape(feature_shape)
        d_output = torch.linspace(-0.2, 0.3, steps=values.numel(), device=device).reshape(feature_shape)
        torch.cuda.synchronize()

        def run_all_entry_points() -> tuple[torch.Tensor, ...]:
            qk_output = self.swattention.qk_forward(queries, keys, height, width, kernel_size, cuda_threads)
            qk_gradients = self.swattention.qk_backward(
                d_attn_weight, queries, keys, height, width, kernel_size, cuda_threads
            )
            qk_rpb_output = self.swattention.qk_rpb_forward(
                queries, keys, rpb, height, width, kernel_size, cuda_threads
            )
            qk_rpb_gradients = self.swattention.qk_rpb_backward(
                d_attn_weight, queries, keys, height, width, kernel_size, cuda_threads
            )
            av_output = self.swattention.av_forward(attn_weight, values, height, width, kernel_size, cuda_threads)
            av_gradients = self.swattention.av_backward(
                d_output, attn_weight, values, height, width, kernel_size, cuda_threads
            )
            return (
                qk_output,
                qk_gradients[0],
                qk_gradients[1],
                qk_rpb_output,
                qk_rpb_gradients[0],
                qk_rpb_gradients[1],
                qk_rpb_gradients[2],
                av_output,
                av_gradients[0],
                av_gradients[1],
            )

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            expected_outputs = run_all_entry_points()

        stream.synchronize()
        expected_outputs = tuple(output.cpu() for output in expected_outputs)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = run_all_entry_points()

        # Erase writes from kernels that may have run outside the captured stream
        for output in outputs:
            output.fill_(-123.0)

        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        for output, expected_output in zip(outputs, expected_outputs, strict=True):
            torch.testing.assert_close(output.cpu(), expected_output)
