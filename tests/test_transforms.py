import logging
import typing
import unittest

import torch
from torchvision.transforms import v2

from birder.transforms import classification
from birder.transforms import detection

logging.disable(logging.CRITICAL)


class TestTransforms(unittest.TestCase):
    def test_classification(self) -> None:
        # Get rgb
        for rgb_mode in typing.get_args(classification.RGBMode):
            rgb_stats = classification.get_rgb_stats(rgb_mode)
            self.assertIsInstance(rgb_stats, dict)
            self.assertIn("mean", rgb_stats)
            self.assertIn("std", rgb_stats)
            self.assertEqual(len(rgb_stats["mean"]), 3)
            self.assertEqual(len(rgb_stats["std"]), 3)

        # Get mixup / cutmix
        mixup_cutmix: v2.Transform = classification.get_mixup_cutmix(0.5, 5, True)
        self.assertIsInstance(mixup_cutmix, v2.Transform)
        self.assertEqual(len(mixup_cutmix.transforms), 3)  # identity, mixup, cutmix
        self.assertIsInstance(mixup_cutmix.transforms[0], v2.Identity)

        mixup_cutmix = classification.get_mixup_cutmix(None, 5, False)
        self.assertIsInstance(mixup_cutmix, v2.Transform)
        self.assertEqual(len(mixup_cutmix.transforms), 1)  # Only identity
        self.assertIsInstance(mixup_cutmix.transforms[0], v2.Identity)

        # Mixup module
        mixup = classification.RandomMixup(5, 0.2, 1.0)
        (samples, targets) = mixup(torch.rand((2, 3, 96, 96)), torch.tensor([0, 1], dtype=torch.int64))
        self.assertSequenceEqual(targets.size(), (2, 5))
        self.assertSequenceEqual(samples.size(), (2, 3, 96, 96))
        repr(mixup)

        # Presets
        classification.training_preset((256, 256), 0, classification.get_rgb_stats("none"))
        classification.training_preset((256, 256), 3, classification.get_rgb_stats("none"))
        classification.inference_preset((256, 256), classification.get_rgb_stats("none"), 0.9)

    def test_detection(self) -> None:
        images = detection.batch_images(
            [
                torch.ones((3, 10, 10)),
                torch.ones((3, 12, 12)),
            ],
            size_divisible=4,
        )

        self.assertSequenceEqual(images.size(), (2, 3, 12, 12))
        self.assertEqual(images[0][0][0][10].item(), 0)
        self.assertEqual(images[0][0][10][0].item(), 0)
        self.assertEqual(images[0][0][9][9].item(), 1)

        # Presets
        detection.training_preset((256, 256), 0, classification.get_rgb_stats("none"))
        detection.inference_preset((256, 256), classification.get_rgb_stats("none"))
