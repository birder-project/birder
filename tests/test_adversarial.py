import logging
import unittest

import torch
from torch import nn

from birder.adversarial import base
from birder.adversarial.deepfool import DeepFool
from birder.adversarial.fgsm import FGSM
from birder.adversarial.pgd import PGD
from birder.adversarial.simba import SimBA
from birder.data.transforms.classification import RGBType
from birder.data.transforms.classification import get_rgb_stats

logging.disable(logging.CRITICAL)

_RGB_STATS = get_rgb_stats("neutral")
NORM_EPS = 1e-6


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(x))


class TestAdversarialBase(unittest.TestCase):
    def test_pixel_eps_to_normalized(self) -> None:
        rgb_stats: RGBType = {"mean": (0.5, 0.5, 0.5), "std": (0.25, 0.5, 1.0)}
        eps = 0.1
        expected = torch.tensor([0.4, 0.2, 0.1]).view(1, 3, 1, 1)
        actual = base.pixel_eps_to_normalized(eps, rgb_stats)
        self.assertTrue(torch.allclose(actual, expected))

        eps_tensor = torch.tensor([0.1, 0.2, 0.3])
        expected = torch.tensor([0.4, 0.4, 0.3]).view(1, 3, 1, 1)
        actual = base.pixel_eps_to_normalized(eps_tensor, rgb_stats)
        self.assertTrue(torch.allclose(actual, expected))

    def test_clamp_normalized(self) -> None:
        rgb_stats: RGBType = {"mean": (0.5, 0.5, 0.5), "std": (0.25, 0.5, 1.0)}
        (min_val, max_val) = base.normalized_bounds(rgb_stats)

        inputs = torch.zeros((1, 3, 2, 2))
        inputs[0, :, 0, 0] = (min_val - 1.0).view(3)
        inputs[0, :, 1, 1] = (max_val + 1.0).view(3)

        clamped = base.clamp_normalized(inputs, rgb_stats)
        self.assertTrue(torch.all(clamped >= min_val))
        self.assertTrue(torch.all(clamped <= max_val))
        self.assertTrue(torch.allclose(clamped[0, :, 0, 0], min_val.view(3)))
        self.assertTrue(torch.allclose(clamped[0, :, 1, 1], max_val.view(3)))

    def test_validate_target(self) -> None:
        # Valid target
        target = torch.tensor([0])
        result = base.validate_target(target, batch_size=1, num_classes=2, device=torch.device("cpu"))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.shape[0], 1)

        # None target
        result = base.validate_target(None, batch_size=1, num_classes=2, device=torch.device("cpu"))
        self.assertIsNone(result)

        # Invalid: out of range
        target = torch.tensor([3])
        with self.assertRaises(ValueError):
            base.validate_target(target, batch_size=1, num_classes=2, device=torch.device("cpu"))

        # Invalid: wrong batch size
        target = torch.tensor([0, 1])
        with self.assertRaises(ValueError):
            base.validate_target(target, batch_size=1, num_classes=2, device=torch.device("cpu"))

    def test_attack_success(self) -> None:
        # Untargeted attack success (prediction changed)
        logits = torch.tensor([[2.0, 1.0]])
        adv_logits = torch.tensor([[1.0, 2.0]])
        success = base.attack_success(logits, adv_logits, targeted=False)
        self.assertTrue(success.item())

        # Untargeted attack failure (prediction unchanged)
        logits = torch.tensor([[2.0, 1.0]])
        adv_logits = torch.tensor([[2.5, 1.0]])
        success = base.attack_success(logits, adv_logits, targeted=False)
        self.assertFalse(success.item())

        # Targeted attack success
        logits = torch.tensor([[2.0, 1.0]])
        adv_logits = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([1])
        success = base.attack_success(logits, adv_logits, targeted=True, target=target)
        self.assertTrue(success.item())

        # Targeted attack failure
        logits = torch.tensor([[2.0, 1.0]])
        adv_logits = torch.tensor([[2.5, 1.0]])
        target = torch.tensor([1])
        success = base.attack_success(logits, adv_logits, targeted=True, target=target)
        self.assertFalse(success.item())

        with self.assertRaises(ValueError):
            base.attack_success(logits, adv_logits, targeted=True, target=None)


class TestAdversarialAttacks(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.net = _TinyNet()
        self.inputs = torch.rand((1, 3, 4, 4))

    def test_fgsm_bounds(self) -> None:
        attack = FGSM(self.net, eps=0.5, rgb_stats=_RGB_STATS)
        result = attack(self.inputs, target=None)

        # Check pixel bounds
        (min_val, max_val) = base.normalized_bounds(_RGB_STATS)
        self.assertTrue(torch.all(result.adv_inputs >= min_val))
        self.assertTrue(torch.all(result.adv_inputs <= max_val))

        # Check perturbation magnitude (allowing small numerical error)
        eps_norm = base.pixel_eps_to_normalized(0.5, _RGB_STATS)
        self.assertTrue(torch.all(result.perturbation.abs() <= eps_norm + NORM_EPS))

    def test_fgsm_targeted(self) -> None:
        target = torch.tensor([1])
        attack = FGSM(self.net, eps=0.5, rgb_stats=_RGB_STATS)
        result = attack(self.inputs, target=target)
        self.assertIsNotNone(result.success)

    def test_pgd_bounds(self) -> None:
        attack = PGD(self.net, eps=0.4, steps=3, step_size=0.2, random_start=True, rgb_stats=_RGB_STATS)
        result = attack(self.inputs, target=None)

        # Check pixel bounds
        (min_val, max_val) = base.normalized_bounds(_RGB_STATS)
        self.assertTrue(torch.all(result.adv_inputs >= min_val))
        self.assertTrue(torch.all(result.adv_inputs <= max_val))

        # Check perturbation magnitude
        eps_norm = base.pixel_eps_to_normalized(0.4, _RGB_STATS)
        self.assertTrue(torch.all(result.perturbation.abs() <= eps_norm + NORM_EPS))

    def test_pgd_default_step_size(self) -> None:
        attack = PGD(self.net, eps=0.3, steps=10, rgb_stats=_RGB_STATS)
        self.assertAlmostEqual(attack.step_size, 0.03, places=6)

    def test_pgd_targeted(self) -> None:
        target = torch.tensor([1])
        attack = PGD(self.net, eps=0.3, steps=2, step_size=0.2, rgb_stats=_RGB_STATS)
        result = attack(self.inputs, target=target)
        self.assertIsNotNone(result.success)

    def test_deepfool_bounds(self) -> None:
        attack = DeepFool(self.net, num_classes=2, max_iter=3, overshoot=0.02, rgb_stats=_RGB_STATS)
        result = attack(self.inputs, target=None)

        # Check pixel bounds
        (min_val, max_val) = base.normalized_bounds(_RGB_STATS)
        self.assertTrue(torch.all(result.adv_inputs >= min_val))
        self.assertTrue(torch.all(result.adv_inputs <= max_val))

    def test_deepfool_validation(self) -> None:
        with self.assertRaises(ValueError):
            DeepFool(self.net, num_classes=1, rgb_stats=_RGB_STATS)

        with self.assertRaises(ValueError):
            DeepFool(self.net, num_classes=2, max_iter=0, rgb_stats=_RGB_STATS)

        with self.assertRaises(ValueError):
            DeepFool(self.net, num_classes=2, overshoot=-0.1, rgb_stats=_RGB_STATS)

    def test_deepfool_targeted(self) -> None:
        target = torch.tensor([1])
        attack = DeepFool(self.net, num_classes=2, max_iter=2, rgb_stats=_RGB_STATS)
        result = attack(self.inputs, target=target)
        self.assertIsNotNone(result.success)

    def test_simba_bounds(self) -> None:
        attack = SimBA(self.net, step_size=0.1, max_iter=10, rgb_stats=_RGB_STATS)
        result = attack(self.inputs, target=None)

        (min_val, max_val) = base.normalized_bounds(_RGB_STATS)
        self.assertTrue(torch.all(result.adv_inputs >= min_val))
        self.assertTrue(torch.all(result.adv_inputs <= max_val))

    def test_simba_query_counting(self) -> None:
        attack = SimBA(self.net, step_size=0.1, max_iter=5, rgb_stats=_RGB_STATS)
        result = attack(self.inputs, target=None)

        self.assertIsNotNone(result.num_queries)
        self.assertGreater(result.num_queries, 0)  # type: ignore[arg-type]

    def test_simba_targeted(self) -> None:
        target = torch.tensor([1])
        attack = SimBA(self.net, step_size=0.1, max_iter=5, rgb_stats=_RGB_STATS)
        result = attack(self.inputs, target=target)
        self.assertIsNotNone(result.success)

    def test_batch_processing(self) -> None:
        batch_inputs = torch.rand((2, 3, 4, 4))

        fgsm = FGSM(self.net, eps=0.1, rgb_stats=_RGB_STATS)
        result = fgsm(batch_inputs, target=None)
        self.assertEqual(result.adv_inputs.shape[0], 2)
        self.assertEqual(result.success.shape[0], 2)  # type: ignore[union-attr]

        pgd = PGD(self.net, eps=0.3, steps=2, step_size=0.2, rgb_stats=_RGB_STATS)
        result = pgd(batch_inputs, target=None)
        self.assertEqual(result.adv_inputs.shape[0], 2)
        self.assertEqual(result.success.shape[0], 2)  # type: ignore[union-attr]

        deepfool = DeepFool(self.net, num_classes=2, max_iter=2, rgb_stats=_RGB_STATS)
        result = deepfool(batch_inputs, target=None)
        self.assertEqual(result.adv_inputs.shape[0], 2)
        self.assertEqual(result.success.shape[0], 2)  # type: ignore[union-attr]

        simba = SimBA(self.net, step_size=0.1, max_iter=5, rgb_stats=_RGB_STATS)
        result = simba(batch_inputs, target=None)
        self.assertEqual(result.adv_inputs.shape[0], 2)
        self.assertEqual(result.success.shape[0], 2)  # type: ignore[union-attr]

    def test_attack_result_structure(self) -> None:
        attack = FGSM(self.net, eps=0.1, rgb_stats=_RGB_STATS)
        result = attack(self.inputs, target=None)

        self.assertIsInstance(result.adv_inputs, torch.Tensor)
        self.assertIsInstance(result.adv_logits, torch.Tensor)
        self.assertIsInstance(result.perturbation, torch.Tensor)
        self.assertIsInstance(result.success, torch.Tensor)

        self.assertEqual(result.adv_inputs.shape, self.inputs.shape)
        self.assertEqual(result.perturbation.shape, self.inputs.shape)
