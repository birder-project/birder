import logging
import unittest
from typing import Optional

import torch

from birder import layers
from birder.layers.moe import _empty_moe_training_output
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

    def test_grouped_expert_conversion(self) -> None:
        for bias in (False, True):
            with self.subTest(bias=bias):
                experts = torch.nn.ModuleList(
                    [layers.SwiGLU_FFN(7, 11, bias=bias, dropout=0.2) for _ in range(3)]
                ).double()
                experts.eval()
                for expert in experts:
                    expert.fc1_x.weight.requires_grad_(False)

                grouped = layers.group_experts(experts)
                restored = layers.ungroup_experts(grouped)

                self.assertFalse(grouped.training)
                self.assertFalse(restored.training)
                self.assertEqual(grouped.drop1.p, 0.2)
                self.assertEqual(grouped.fc1_g.weight.dtype, torch.float64)
                for expected, actual in zip(experts.parameters(), restored.parameters(), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual.requires_grad, expected.requires_grad)

                inputs = torch.randn(3, 5, 7, dtype=torch.float64)
                expected = torch.stack([expert(expert_inputs) for expert, expert_inputs in zip(experts, inputs)])
                torch.testing.assert_close(grouped(inputs), expected)
                actual = torch.stack(
                    [expert(expert_inputs) for expert, expert_inputs in zip(restored, inputs, strict=True)]
                )
                torch.testing.assert_close(actual, expected)

    def test_soft_moe_ffn(self) -> None:
        soft_moe_ffn = layers.SoftMoE_FFN(8, 16, num_experts=3, num_slots=2)
        out = soft_moe_ffn(torch.rand(2, 5, 8))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.size(), (2, 5, 8))

    def test_vmoe_ffn(self) -> None:
        vmoe_ffn = layers.VMoE_FFN(
            4,
            8,
            num_experts=2,
            top_k=1,
            capacity_factor=1.0,
            capacity_multiple_of=None,
            router_noise_std=0.0,
            router_g_shard_loss_weight=1.0,
            router_importance_loss_weight=0.0,
            router_load_loss_weight=0.0,
        )
        inputs = torch.rand(8, 3, 4, requires_grad=True)
        token_mask = torch.tensor(
            [[True, True, False], [True, False, True], [True, True, True], [False, True, True]] * 2
        )

        # Test that training metadata is opt-in and preserves the masked output
        plain_output = vmoe_ffn(inputs, token_mask=token_mask)
        output, training_output = vmoe_ffn(inputs, token_mask=token_mask, return_moe_training_output=True)
        self.assertIsInstance(plain_output, torch.Tensor)
        torch.testing.assert_close(output, plain_output)
        self.assertEqual(output.size(), inputs.size())
        torch.testing.assert_close(output[~token_mask], torch.zeros_like(output[~token_mask]))
        self.assertEqual(training_output["auxiliary_loss"].ndim, 0)
        self.assertEqual(training_output["expert_loads"].size(), (0, 0))

        # Test gradient flow through the experts and auxiliary loss
        (output.sum() + training_output["auxiliary_loss"]).backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(vmoe_ffn.router.gate.weight.grad)
        self.assertTrue(torch.isfinite(vmoe_ffn.router.gate.weight.grad).all().item())

        # Test that evaluation returns empty metadata without changing the output
        vmoe_ffn.eval()
        plain_output = vmoe_ffn(inputs.detach(), token_mask=token_mask)
        output, training_output = vmoe_ffn(inputs.detach(), token_mask=token_mask, return_moe_training_output=True)
        torch.testing.assert_close(output, plain_output)
        expected_training_output = _empty_moe_training_output(output)
        self.assertEqual(training_output.keys(), expected_training_output.keys())
        for key, expected in expected_training_output.items():
            torch.testing.assert_close(training_output[key], expected, msg=key)

    def test_moe_ffn(self) -> None:
        moe_ffn = layers.MoE_FFN(
            4,
            8,
            num_routed_experts=3,
            num_shared_experts=1,
            top_k=2,
            router_bias_update_speed=0.1,
        )
        inputs = torch.rand(2, 4, 4, requires_grad=True)
        token_mask = torch.tensor([[True, True, False, True], [True, False, True, False]])

        # Test that training metadata is opt-in and preserves the masked output
        plain_output = moe_ffn(inputs, token_mask=token_mask)
        output, training_output = moe_ffn(inputs, token_mask=token_mask, return_moe_training_output=True)
        self.assertIsInstance(plain_output, torch.Tensor)
        torch.testing.assert_close(output, plain_output)
        self.assertEqual(output.size(), inputs.size())
        torch.testing.assert_close(output[~token_mask], torch.zeros_like(output[~token_mask]))
        torch.testing.assert_close(training_output["auxiliary_loss"], torch.tensor(0.0))
        self.assertEqual(training_output["expert_loads"].size(), (1, 3))
        self.assertEqual(training_output["expert_loads"].sum().item(), token_mask.sum().item() * 2)

        # Test gradient flow through the shared and routed experts
        (output.sum() + training_output["auxiliary_loss"]).backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(moe_ffn.router.gate.weight.grad)
        self.assertTrue(torch.isfinite(moe_ffn.router.gate.weight.grad).all().item())

        # Test that expert loads drive the router bias update
        initial_expert_bias = moe_ffn.router.expert_bias.clone()
        expert_load = training_output["expert_loads"][0]
        moe_ffn.update_expert_bias(expert_load)
        expected_update = 0.1 * torch.sign(expert_load.float().mean() - expert_load)
        expected_expert_bias = initial_expert_bias + expected_update
        expected_expert_bias.sub_(expected_expert_bias.mean())
        torch.testing.assert_close(moe_ffn.router.expert_bias, expected_expert_bias)

        # Test that evaluation returns empty metadata without changing the output
        moe_ffn.eval()
        plain_output = moe_ffn(inputs.detach(), token_mask=token_mask)
        output, training_output = moe_ffn(inputs.detach(), token_mask=token_mask, return_moe_training_output=True)
        torch.testing.assert_close(output, plain_output)
        expected_training_output = _empty_moe_training_output(output)
        self.assertEqual(training_output.keys(), expected_training_output.keys())
        for key, expected in expected_training_output.items():
            torch.testing.assert_close(training_output[key], expected, msg=key)

    def test_moe_ffn_expert_choice(self) -> None:
        moe_ffn = layers.MoE_FFN(
            4,
            8,
            num_routed_experts=3,
            num_shared_experts=1,
            routing_type="expert_choice",
            expert_choice_capacity_factor=1.5,
        )
        inputs = torch.rand(2, 4, 4, requires_grad=True)
        token_mask = torch.tensor([[True, True, False, True], [True, False, True, False]])

        # Test expert-choice routing independently per input group
        plain_output = moe_ffn(inputs, token_mask=token_mask)
        separate_output = torch.concat(
            [moe_ffn(inputs[idx : idx + 1], token_mask=token_mask[idx : idx + 1]) for idx in range(inputs.size(0))]
        )
        output, training_output = moe_ffn(inputs, token_mask=token_mask, return_moe_training_output=True)
        self.assertIsInstance(moe_ffn.router, layers.ExpertChoiceRouter)
        torch.testing.assert_close(output, plain_output)
        torch.testing.assert_close(output, separate_output)
        self.assertEqual(output.size(), inputs.size())
        torch.testing.assert_close(output[~token_mask], torch.zeros_like(output[~token_mask]))

        # Expert choice is intrinsically load balanced and emits no training metadata
        expected_training_output = _empty_moe_training_output(output)
        self.assertEqual(training_output.keys(), expected_training_output.keys())
        for key, expected in expected_training_output.items():
            torch.testing.assert_close(training_output[key], expected, msg=key)

        # Test gradient flow through the shared and routed experts
        output.sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(moe_ffn.router.gate.weight.grad)
        self.assertTrue(torch.isfinite(moe_ffn.router.gate.weight.grad).all().item())

        # Test that evaluation preserves the output and empty metadata
        moe_ffn.eval()
        plain_output = moe_ffn(inputs.detach(), token_mask=token_mask)
        output, training_output = moe_ffn(inputs.detach(), token_mask=token_mask, return_moe_training_output=True)
        torch.testing.assert_close(output, plain_output)
        expected_training_output = _empty_moe_training_output(output)
        self.assertEqual(training_output.keys(), expected_training_output.keys())
        for key, expected in expected_training_output.items():
            torch.testing.assert_close(training_output[key], expected, msg=key)

    def test_moe_ffn_special_token_experts(self) -> None:
        inputs = torch.rand(2, 5, 4)
        for routing_type in ("token_choice", "expert_choice"):
            with self.subTest(routing_type=routing_type):
                moe_ffn = layers.MoE_FFN(
                    4,
                    8,
                    num_routed_experts=3,
                    num_shared_experts=1,
                    num_special_token_experts=2,
                    routing_type=routing_type,
                    top_k=2,
                    expert_choice_capacity_factor=1.5,
                )
                with torch.no_grad():
                    output = moe_ffn(inputs, num_special_tokens=2)
                    expected_special_output = torch.stack(
                        [expert(inputs[:, :2]) for expert in moe_ffn.special_token_experts]
                    ).sum(dim=0)
                    expected_special_output += moe_ffn.shared_experts[0](inputs[:, :2])

                torch.testing.assert_close(output[:, :2], expected_special_output)
                self.assertEqual(output.size(), inputs.size())

    def test_noisy_top_k_router(self) -> None:
        router = layers.NoisyTopKRouter(
            2,
            3,
            top_k=2,
            noise_std=0.0,
            eval_capacity_factor=4.0,
            capacity_multiple_of=None,
        )
        with torch.no_grad():
            router.gate.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))

        router.eval()
        inputs = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]], requires_grad=True)

        # Test deterministic top-k routing and combine weights
        expert_indices, buffer_indices, combine_weights, _ = router(inputs)
        logits = router.gate(inputs)
        expected_indices = logits.topk(2, dim=-1).indices
        expected_weights = logits.softmax(dim=-1).gather(-1, expected_indices)

        torch.testing.assert_close(expert_indices, expected_indices)
        torch.testing.assert_close(combine_weights, expected_weights)
        self.assertEqual(buffer_indices.size(), expected_indices.size())

        # Test that evaluation metadata is opt-in and empty
        actual_indices, actual_buffers, actual_weights, training_output = router(
            inputs, return_moe_training_output=True
        )
        torch.testing.assert_close(actual_indices, expert_indices)
        torch.testing.assert_close(actual_buffers, buffer_indices)
        torch.testing.assert_close(actual_weights, combine_weights)
        expected_training_output = _empty_moe_training_output(inputs)
        self.assertEqual(training_output.keys(), expected_training_output.keys())
        for key, expected in expected_training_output.items():
            torch.testing.assert_close(training_output[key], expected, msg=key)

        # Test gradient flow through the gate
        combine_weights.sum().backward()
        self.assertIsNotNone(router.gate.weight.grad)
        self.assertTrue(torch.isfinite(router.gate.weight.grad).all().item())

    def test_noisy_top_k_router_token_mask(self) -> None:
        router = layers.NoisyTopKRouter(
            2,
            2,
            top_k=1,
            noise_std=0.0,
            capacity_factor=1.0,
            capacity_multiple_of=None,
            g_shard_loss_weight=1.0,
            importance_loss_weight=1.0,
        )
        with torch.no_grad():
            router.gate.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

        inputs = torch.tensor([[[10.0, 0.0], [3.0, 0.0], [10.0, 0.0], [2.0, 0.0]]])
        token_mask = torch.tensor([[False, True, False, True]])

        # Test that masked tokens neither consume capacity nor affect auxiliary losses
        _, buffer_index, combine_weights, training_output = router(
            inputs, token_mask=token_mask, return_moe_training_output=True
        )
        _, _, _, expected_training_output = router(inputs[:, token_mask[0]], return_moe_training_output=True)

        torch.testing.assert_close(buffer_index[token_mask], torch.tensor([[0], [1]]))
        torch.testing.assert_close(combine_weights[~token_mask], torch.zeros(2, 1))
        self.assertTrue(torch.all(combine_weights[token_mask] > 0).item())
        for key, expected in expected_training_output.items():
            torch.testing.assert_close(training_output[key], expected, msg=key)

    def test_expert_choice_router(self) -> None:
        router = layers.ExpertChoiceRouter(2, 2, capacity_factor=1.0)
        with torch.no_grad():
            router.gate.weight.copy_(torch.eye(2))

        inputs = torch.tensor([[[4.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 3.0]]], requires_grad=True)

        # Test that each expert selects its highest-affinity tokens up to its fixed capacity
        token_indices, expert_weights = router(inputs)
        expected_indices = torch.tensor([[[0, 1], [2, 3]]])
        expert_scores = router.gate(inputs).softmax(dim=-1).transpose(-2, -1)
        expected_weights = expert_scores.gather(-1, expected_indices)

        torch.testing.assert_close(token_indices, expected_indices)
        torch.testing.assert_close(expert_weights, expected_weights)
        self.assertEqual(token_indices.size(), (1, 2, 2))
        self.assertEqual(router._capacity(5), 3)

        # Test gradient flow through the gate
        expert_weights.sum().backward()
        self.assertIsNotNone(router.gate.weight.grad)
        self.assertTrue(torch.isfinite(router.gate.weight.grad).all().item())

    def test_expert_choice_router_token_mask(self) -> None:
        router = layers.ExpertChoiceRouter(2, 2, capacity_factor=1.5)
        with torch.no_grad():
            router.gate.weight.copy_(torch.eye(2))

        inputs = torch.tensor([[[10.0, 0.0], [3.0, 0.0], [0.0, 10.0], [0.0, 3.0]]])
        token_mask = torch.tensor([[False, True, False, True]])

        # Test that masked tokens are ineligible and underfilled capacity has zero weight
        token_indices, expert_weights = router(inputs, token_mask=token_mask)
        selected_token_mask = token_mask.unsqueeze(-2).expand(1, 2, 4).gather(-1, token_indices)
        expert_scores = router.gate(inputs).softmax(dim=-1).transpose(-2, -1)
        expected_weights = expert_scores.gather(-1, token_indices) * selected_token_mask

        torch.testing.assert_close(expert_weights, expected_weights)
        torch.testing.assert_close(selected_token_mask.sum(dim=-1), torch.tensor([[2, 2]]))
        torch.testing.assert_close(expert_weights[~selected_token_mask], torch.zeros(2))
        for expert_idx in range(2):
            selected_indices = token_indices[0, expert_idx, selected_token_mask[0, expert_idx]].sort().values
            torch.testing.assert_close(selected_indices, torch.tensor([1, 3]))

        # Test an entirely masked group
        _, all_masked_weights = router(inputs, token_mask=torch.zeros_like(token_mask))
        torch.testing.assert_close(all_masked_weights, torch.zeros_like(all_masked_weights))

    def test_sigmoid_top_k_router(self) -> None:
        router = layers.SigmoidTopKRouter(2, 3, top_k=2)
        with torch.no_grad():
            router.gate.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
            router.expert_bias.copy_(torch.tensor([0.0, 0.0, 2.0]))

        inputs = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]], requires_grad=True)
        token_mask = torch.tensor([[True, False]])

        # Test bias-adjusted expert selection and normalized masked weights
        expert_indices, expert_weights = router(inputs, token_mask=token_mask)

        expected_indices = torch.tensor([[[2, 0], [2, 1]]])
        affinity_scores = router.gate(inputs).sigmoid()
        expected_weights = affinity_scores.gather(-1, expected_indices)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
        expected_weights = expected_weights * token_mask.unsqueeze(-1)

        torch.testing.assert_close(expert_indices, expected_indices)
        torch.testing.assert_close(expert_weights, expected_weights)
        torch.testing.assert_close(expert_weights[token_mask].sum(dim=-1), torch.ones(1))
        torch.testing.assert_close(expert_weights[~token_mask], torch.zeros(1, 2))

        # Test gradient flow through the gate
        expert_weights[0, 0, 0].backward()
        self.assertIsNotNone(router.gate.weight.grad)
        self.assertGreater(router.gate.weight.grad.abs().sum().item(), 0.0)

        # Test weight normalization without a token mask
        _, normalized_weights = router(inputs.detach())
        torch.testing.assert_close(normalized_weights.sum(dim=-1), torch.ones((1, 2)))

    def test_sigmoid_top_k_router_bias_update(self) -> None:
        router = layers.SigmoidTopKRouter(2, 3, bias_update_speed=0.1)

        # Test that overloaded experts are biased down and unused experts up,
        # without retaining the common offset from a non-zero-sum update.
        expert_load = torch.tensor([7, 2, 0])
        router.update_expert_bias(expert_load)

        expected_expert_bias = 0.1 * torch.sign(expert_load.float().mean() - expert_load)
        expected_expert_bias.sub_(expected_expert_bias.mean())
        torch.testing.assert_close(router.expert_bias, expected_expert_bias)
        torch.testing.assert_close(router.expert_bias.mean(), torch.tensor(0.0), atol=1e-8, rtol=0)

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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestGroupedMoE(unittest.TestCase):
    def test_masked_and_partially_empty_routes(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        token_mask = torch.tensor(
            [[True, True, False, True, False], [False, False, False, False, False]],
            device=device,
        )
        expected_loads = torch.tensor([[3, 3, 0, 0]], device=device)
        for amp in (False, True):
            with self.subTest(amp=amp):
                dtype = torch.float32 if amp is True else torch.bfloat16
                moe_ffn = layers.MoE_FFN(
                    32,
                    64,
                    bias=True,
                    num_routed_experts=4,
                    num_shared_experts=0,
                    grouped_token_choice=True,
                    top_k=2,
                ).to(device=device, dtype=dtype)
                with torch.no_grad():
                    moe_ffn.router.gate.weight.zero_()
                    moe_ffn.router.expert_bias.copy_(torch.tensor([2.0, 1.0, -1.0, -2.0], device=device))

                inputs = torch.randn(2, 5, 32, device=device, dtype=dtype, requires_grad=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    output, training_output = moe_ffn(
                        inputs,
                        token_mask=token_mask,
                        return_moe_training_output=True,
                    )

                self.assertEqual(output.size(), inputs.size())
                self.assertTrue(torch.isfinite(output).all().item())
                torch.testing.assert_close(output[~token_mask], torch.zeros_like(output[~token_mask]), rtol=0, atol=0)
                torch.testing.assert_close(training_output["expert_loads"], expected_loads)

                upstream = torch.linspace(0.1, 1.0, output.numel(), device=device, dtype=output.dtype).reshape_as(
                    output
                )
                output.backward(upstream)
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all().item())
                torch.testing.assert_close(inputs.grad[~token_mask], torch.zeros_like(inputs.grad[~token_mask]))

                used_expert_grad_sum = 0.0
                for parameter in moe_ffn.routed_experts.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    torch.testing.assert_close(parameter.grad[2:], torch.zeros_like(parameter.grad[2:]), rtol=0, atol=0)
                    used_expert_grad_sum += parameter.grad[:2].float().abs().sum().item()

                self.assertGreater(used_expert_grad_sum, 0.0)

    def test_globally_empty_routes(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        moe_ffn = layers.MoE_FFN(
            32,
            64,
            bias=True,
            num_routed_experts=4,
            num_shared_experts=0,
            grouped_token_choice=True,
            top_k=2,
        ).to(device=device, dtype=torch.bfloat16)
        inputs = torch.randn(2, 5, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        token_mask = torch.zeros(2, 5, device=device, dtype=torch.bool)

        output, training_output = moe_ffn(
            inputs,
            token_mask=token_mask,
            return_moe_training_output=True,
        )
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
        torch.testing.assert_close(
            training_output["expert_loads"],
            torch.zeros((1, 4), device=device, dtype=torch.int64),
            rtol=0,
            atol=0,
        )

        output.backward(torch.ones_like(output))
        self.assertIsNotNone(inputs.grad)
        torch.testing.assert_close(inputs.grad, torch.zeros_like(inputs.grad), rtol=0, atol=0)
        for parameter in moe_ffn.routed_experts.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), rtol=0, atol=0)
