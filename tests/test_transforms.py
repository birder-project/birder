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
from birder.data.transforms import naflex

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
        self.assertSequenceEqual(mixup_cutmix.p, [1 / 3, 1 / 3, 1 / 3])

        mixup_cutmix = classification.get_mixup_cutmix(0.5, 5, True, prob=0.6)
        self.assertSequenceEqual(mixup_cutmix.p, [0.4, 0.3, 0.3])

        mixup_choice = classification.get_mixup_cutmix(0.5, 5, False, prob=1.0)
        self.assertSequenceEqual(mixup_choice.p, [0.0, 1.0])  # type: ignore[attr-defined]

        mixup_cutmix = classification.get_mixup_cutmix(None, 5, False)
        self.assertIsInstance(mixup_cutmix, v2.Transform)
        self.assertEqual(len(mixup_cutmix.transforms), 1)  # Only identity
        self.assertIsInstance(mixup_cutmix.transforms[0], v2.Identity)

        with self.assertRaisesRegex(ValueError, "Probability must be in range"):
            classification.get_mixup_cutmix(0.5, 5, False, prob=1.1)

        # Mixup module
        mixup = classification.RandomMixup(5, 0.2, 1.0)
        samples, targets = mixup(torch.rand((2, 3, 96, 96)), torch.tensor([0, 1], dtype=torch.int64))
        self.assertSequenceEqual(targets.size(), (2, 5))
        self.assertSequenceEqual(samples.size(), (2, 3, 96, 96))
        repr(mixup)

        # Presets
        classification.training_preset((256, 256), "birder", 0, classification.get_rgb_stats("centered"))
        classification.training_preset((256, 256), "birder", 8, classification.get_rgb_stats("birder"))
        classification.training_preset((256, 256), "3aug", 3, classification.get_rgb_stats("centered"))
        classification.training_preset((256, 256), "clip", 0, classification.get_rgb_stats("clip"))
        preset = classification.training_preset(
            (256, 256),
            "birder",
            1,
            classification.get_rgb_stats("centered"),
            resize_min_scale=0.05,
            resize_max_scale=0.3,
        )
        self.assertEqual(preset.transforms[1].transform[0].scale, (0.05, 0.3))  # type: ignore[attr-defined]
        classification.inference_preset((256, 256), classification.get_rgb_stats("centered"), 0.9)
        classification.inference_preset((256, 256), classification.get_rgb_stats("centered"), 0.9, True)

    def test_detection(self) -> None:
        # Presets
        detection.training_preset(
            (256, 256), "birder", 0, classification.get_rgb_stats("centered"), dynamic_size=False, multiscale=False
        )
        detection.training_preset(
            (256, 256), "birder", 5, classification.get_rgb_stats("birder"), dynamic_size=False, multiscale=True
        )
        detection.training_preset(
            (256, 256), "ssd", 0, classification.get_rgb_stats("centered"), dynamic_size=True, multiscale=False
        )
        detection.training_preset(
            (256, 256), "multiscale", 0, classification.get_rgb_stats("centered"), dynamic_size=False, multiscale=False
        )
        detection.training_preset(
            (256, 256), "deim", 0, classification.get_rgb_stats("centered"), dynamic_size=False, multiscale=False
        )
        detection.InferenceTransform((256, 256), classification.get_rgb_stats("birder"), False)

        # Multiscale
        sizes = detection.build_multiscale_sizes()
        self.assertEqual(sizes, (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800))
        if len(sizes) > 1:
            self.assertEqual(sizes[1] - sizes[0], detection.MULTISCALE_STEP)

        self.assertEqual(detection.build_multiscale_sizes(481, max_size=513), (512,))
        self.assertEqual(detection.build_multiscale_sizes(500, max_size=620), (512, 544, 576, 608))

        # DETR intermediate sizes
        self.assertEqual(detection.resolve_detr_intermediate_sizes((640, 640)), (400, 500, 600))
        self.assertEqual(detection.resolve_detr_intermediate_sizes((1280, 1280)), (800, 1000, 1200))

    def test_fixed_size_crop(self) -> None:
        transform = detection.FixedSizeCrop((4, 4), [0.0, 0.0, 0.0])

        # Smaller inputs are padded on the right and bottom without moving boxes
        image = tv_tensors.Image(torch.ones((3, 3, 3), dtype=torch.uint8))
        boxes = tv_tensors.BoundingBoxes(
            [[0, 0, 1, 1], [1, 1, 3, 3]],
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(3, 3),
        )
        output_image, output_target = transform(image, {"boxes": boxes})

        expected_image = torch.zeros((3, 4, 4), dtype=torch.uint8)
        expected_image[:, :3, :3] = 1
        torch.testing.assert_close(output_image, expected_image)
        torch.testing.assert_close(output_target["boxes"], boxes)
        self.assertEqual(output_target["boxes"].canvas_size, (4, 4))

        # A centered box remains aligned with its pixels for every possible crop offset
        image = torch.zeros((3, 5, 5), dtype=torch.uint8)
        image[:, 1:4, 1:4] = 1
        image = tv_tensors.Image(image)
        boxes = tv_tensors.BoundingBoxes(
            [[1, 1, 4, 4]],
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(5, 5),
        )
        output_image, output_target = transform(image, {"boxes": boxes})

        left, top, right, bottom = output_target["boxes"][0].int().tolist()
        self.assertEqual((right - left, bottom - top), (3, 3))
        self.assertGreaterEqual(min(left, top), 0)
        self.assertLessEqual(max(right, bottom), 4)
        self.assertEqual(output_target["boxes"].canvas_size, (4, 4))

        expected_image = torch.zeros((3, 4, 4), dtype=torch.uint8)
        expected_image[:, top:bottom, left:right] = 1
        torch.testing.assert_close(output_image, expected_image)


class TestNaFlex(unittest.TestCase):
    def test_get_sequence_lengths(self) -> None:
        self.assertEqual(naflex.get_sequence_lengths((256, 256), 16), (256,))
        self.assertEqual(naflex.get_sequence_lengths((224, 320), 16), (280,))
        self.assertEqual(naflex.get_sequence_lengths((256, 256), 16, (192, 256)), (144, 256))

        for image_size, patch_size, sizes in (
            ((255, 256), 16, None),
            ((256, 256), 0, None),
            ((256, 256), 16, ()),
            ((256, 256), 16, (192, 192)),
            ((256, 256), 16, (191,)),
        ):
            with self.subTest(image_size=image_size, patch_size=patch_size, sizes=sizes):
                with self.assertRaises(ValueError):
                    naflex.get_sequence_lengths(image_size, patch_size, sizes)

    def test_resolve_patch_grid(self) -> None:
        self.assertEqual(naflex.resolve_patch_grid((256, 256), 256), (16, 16))
        self.assertEqual(naflex.resolve_patch_grid((100, 1000), 256), (5, 50))
        self.assertEqual(naflex.resolve_patch_grid((1000, 100), 256), (50, 5))
        self.assertEqual(naflex.resolve_patch_grid((256, 256), 257), (16, 16))
        self.assertEqual(naflex.resolve_patch_grid((256, 256), 1), (1, 1))

        for image_size in ((123, 321), (321, 123), (17, 19), (1, 1000)):
            grid_h, grid_w = naflex.resolve_patch_grid(image_size, 256)
            self.assertGreaterEqual(grid_h, 1)
            self.assertGreaterEqual(grid_w, 1)
            self.assertLessEqual(grid_h * grid_w, 256)

        with self.assertRaises(ValueError):
            naflex.resolve_patch_grid((0, 256), 256)
        with self.assertRaises(ValueError):
            naflex.resolve_patch_grid((256, -1), 256)
        with self.assertRaises(ValueError):
            naflex.resolve_patch_grid((256, 256), 0)

    def test_native_aspect_ratio_resize(self) -> None:
        image = torch.arange(3 * 100 * 1000, dtype=torch.float32).reshape(3, 100, 1000)
        transform = naflex.NativeAspectRatioResize(16, 256)
        output = transform(image)

        self.assertEqual(output.shape, (3, 80, 800))
        self.assertEqual(output.shape[-2] % 16, 0)
        self.assertEqual(output.shape[-1] % 16, 0)
        self.assertLessEqual((output.shape[-2] // 16) * (output.shape[-1] // 16), 256)

        with self.assertRaises(ValueError):
            naflex.NativeAspectRatioResize(0, 256)
        with self.assertRaises(ValueError):
            naflex.NativeAspectRatioResize(16, 0)

    def test_random_crop_with_scale_and_relative_ratio(self) -> None:
        image = torch.rand((3, 80, 40))
        transform = naflex.RandomCropWithScaleAndRelativeRatio((0.25, 0.25), relative_ratio=(1.0, 1.0))
        output = transform(image)
        self.assertEqual(output.shape, (3, 40, 20))

        image = torch.rand((3, 40, 80))
        output = transform(image)
        self.assertEqual(output.shape, (3, 20, 40))

        image = torch.rand((3, 80, 40))
        transform = naflex.RandomCropWithScaleAndRelativeRatio((0.25, 0.25), relative_ratio=(2.0, 2.0))
        output = transform(image)
        self.assertEqual(output.shape, (3, 28, 28))

        with self.assertRaises(ValueError):
            naflex.RandomCropWithScaleAndRelativeRatio((0.0, 1.0))
        with self.assertRaises(ValueError):
            naflex.RandomCropWithScaleAndRelativeRatio((0.8, 0.5))
        with self.assertRaises(ValueError):
            naflex.RandomCropWithScaleAndRelativeRatio((0.5, 1.1))
        with self.assertRaises(ValueError):
            naflex.RandomCropWithScaleAndRelativeRatio((0.5, 1.0), relative_ratio=(0.0, 1.0))
        with self.assertRaises(ValueError):
            naflex.RandomCropWithScaleAndRelativeRatio((0.5, 1.0), relative_ratio=(2.0, 1.0))

    def test_presets(self) -> None:
        image = Image.new("RGB", (1000, 100), color=(255, 128, 0))
        rgb_stats = classification.get_rgb_stats("neutral")

        transform = naflex.training_preset(16, 256, "birder", 0, rgb_stats)
        output = transform(image)
        self.assertEqual(output.shape, (3, 80, 800))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())

        transform = naflex.training_preset(
            16,
            256,
            "birder",
            4,
            rgb_stats,
            resize_min_scale=0.25,
            resize_max_scale=0.25,
            relative_resize_ratio=(1.0, 1.0),
            re_prob=0.0,
        )
        self.assertEqual(transform.transforms[1].relative_ratio, (1.0, 1.0))  # type: ignore[attr-defined]
        output = transform(image)
        self.assertEqual(output.shape[0], 3)
        self.assertEqual(output.shape[-2] % 16, 0)
        self.assertEqual(output.shape[-1] % 16, 0)
        self.assertLessEqual((output.shape[-2] // 16) * (output.shape[-1] // 16), 256)
        self.assertTrue(torch.isfinite(output).all())

        with self.assertRaises(ValueError):
            naflex.training_preset(16, 256, "birder", 11, rgb_stats)

        image = Image.new("RGB", (321, 123), color=(20, 40, 60))
        for aug_type in typing.get_args(classification.AugType):
            transform = naflex.training_preset(
                16,
                256,
                aug_type,
                4,
                rgb_stats,
                resize_min_scale=0.25,
                resize_max_scale=0.25,
                re_prob=0.0,
            )
            output = transform(image)
            self.assertEqual(output.shape[0], 3)
            self.assertEqual(output.shape[-2] % 16, 0)
            self.assertEqual(output.shape[-1] % 16, 0)
            self.assertLessEqual((output.shape[-2] // 16) * (output.shape[-1] // 16), 256)
            self.assertTrue(torch.isfinite(output).all())

        transform = naflex.inference_preset(16, 256, classification.get_rgb_stats("centered"))
        output_1 = transform(image)
        output_2 = transform(image)
        grid_h, grid_w = naflex.resolve_patch_grid((123, 321), 256)
        self.assertEqual(output_1.shape, (3, grid_h * 16, grid_w * 16))
        self.assertLessEqual(grid_h * grid_w, 256)
        torch.testing.assert_close(output_1, output_2)


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
        images, targets = self._create_dummy_data()
        output_size = (300, 300)

        out_img, out_target = mosaic.mosaic_random_center(images, targets, output_size, fill_value=(114, 114, 114))

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
        _, out_target_empty = mosaic.mosaic_random_center(images, empty_targets, output_size, fill_value=0)
        self.assertEqual(len(out_target_empty["boxes"]), 0)
        self.assertEqual(len(out_target_empty["labels"]), 0)

    def test_mosaic_fixed_grid(self) -> None:
        images, targets = self._create_dummy_data()
        output_size = (300, 300)

        # Crop to square
        out_img, out_target = mosaic.mosaic_fixed_grid(
            images, targets, output_size, fill_value=114, crop_to_square=True
        )

        self.assertEqual(out_img.size, output_size)
        self.assertIsInstance(out_target["boxes"], tv_tensors.BoundingBoxes)

        # Aspect ratio limit
        out_img_ar, _ = mosaic.mosaic_fixed_grid(
            images, targets, output_size, fill_value=114, crop_to_square=False, max_aspect_ratio=1.5
        )
        self.assertEqual(out_img_ar.size, output_size)

        # Missing keys handling
        empty_targets: list[dict[str, Any]] = [{} for _ in range(4)]
        _, out_target_empty = mosaic.mosaic_fixed_grid(images, empty_targets, output_size, fill_value=0)
        self.assertEqual(out_target_empty["boxes"].shape, (0, 4))
