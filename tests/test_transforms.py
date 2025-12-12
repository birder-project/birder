import logging
import typing
import unittest
from typing import Any

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from birder.data.transforms import classification
from birder.data.transforms import detection
from birder.data.transforms import mosaic

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
        classification.training_preset((256, 256), "birder", 0, classification.get_rgb_stats("none"))
        classification.training_preset((256, 256), "birder", 8, classification.get_rgb_stats("birder"))
        classification.training_preset((256, 256), "3aug", 3, classification.get_rgb_stats("none"))
        classification.inference_preset((256, 256), classification.get_rgb_stats("none"), 0.9)
        classification.inference_preset((256, 256), classification.get_rgb_stats("none"), 0.9, True)

    def test_detection(self) -> None:
        # Presets
        detection.training_preset((256, 256), "birder", 0, classification.get_rgb_stats("none"), False, False)
        detection.training_preset((256, 256), "birder", 5, classification.get_rgb_stats("birder"), False, True)
        detection.training_preset((256, 256), "ssd", 0, classification.get_rgb_stats("none"), True, False)
        detection.training_preset((256, 256), "multiscale", 0, classification.get_rgb_stats("none"), False, False)
        detection.InferenceTransform((256, 256), classification.get_rgb_stats("birder"), False)


class TestMosaic(unittest.TestCase):
    def _create_dummy_data(self) -> tuple[list[Image.Image], list[dict[str, Any]]]:
        images = []
        targets = []
        for i in range(4):
            # Create 100x100 images with different colors
            img = Image.new("RGB", (100, 100), color=(i * 50, 100, 100))
            images.append(img)

            # Create a box covering the center 50x50 area
            boxes = torch.tensor([[25.0, 25.0, 75.0, 75.0]], dtype=torch.float32)
            labels = torch.tensor([i + 1], dtype=torch.int64)

            # Wrap in tv_tensors as the pipeline expects
            boxes = tv_tensors.BoundingBoxes(boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(100, 100))
            targets.append({"boxes": boxes, "labels": labels})

        return (images, targets)

    def test_mosaic_random_center(self) -> None:
        (images, targets) = self._create_dummy_data()
        output_size = (300, 300)

        (out_img, out_target) = mosaic.mosaic_random_center(images, targets, output_size, fill_value=(114, 114, 114))

        self.assertEqual(out_img.size, output_size)
        self.assertIsInstance(out_target["boxes"], tv_tensors.BoundingBoxes)
        self.assertEqual(out_target["boxes"].canvas_size, (output_size[1], output_size[0]))  # H, W

        # Verify boxes are within bounds
        if len(out_target["boxes"]) > 0:
            self.assertTrue((out_target["boxes"][:, 0] >= 0).all())
            self.assertTrue((out_target["boxes"][:, 1] >= 0).all())
            self.assertTrue((out_target["boxes"][:, 2] <= output_size[0]).all())
            self.assertTrue((out_target["boxes"][:, 3] <= output_size[1]).all())

        # Empty targets
        empty_targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,))} for _ in range(4)]
        (_, out_target_empty) = mosaic.mosaic_random_center(images, empty_targets, output_size, fill_value=0)
        self.assertEqual(len(out_target_empty["boxes"]), 0)
        self.assertEqual(len(out_target_empty["labels"]), 0)

    def test_mosaic_fixed_grid(self) -> None:
        (images, targets) = self._create_dummy_data()
        output_size = (300, 300)

        # Crop to square
        (out_img, out_target) = mosaic.mosaic_fixed_grid(
            images, targets, output_size, fill_value=114, crop_to_square=True
        )

        self.assertEqual(out_img.size, output_size)
        self.assertIsInstance(out_target["boxes"], tv_tensors.BoundingBoxes)

        # Aspect ratio limit
        (out_img_ar, _) = mosaic.mosaic_fixed_grid(
            images, targets, output_size, fill_value=114, crop_to_square=False, max_aspect_ratio=1.5
        )
        self.assertEqual(out_img_ar.size, output_size)

        # Missing keys handling
        empty_targets: list[dict[str, Any]] = [{} for _ in range(4)]
        (_, out_target_empty) = mosaic.mosaic_fixed_grid(images, empty_targets, output_size, fill_value=0)
        self.assertEqual(out_target_empty["boxes"].shape, (0, 4))
