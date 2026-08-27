import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from birder.data.collators import detection
from birder.data.collators import naflex

logging.disable(logging.CRITICAL)


class TestCollators(unittest.TestCase):
    def test_detection(self) -> None:
        images, masks, size_list = detection.batch_images(
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

        assert masks is not None
        self.assertTrue(torch.all(masks[0, :10, :10] == False))  # pylint: disable=singleton-comparison # noqa: E712
        self.assertTrue(torch.all(masks[0, 11:, 11:] == True))  # pylint: disable=singleton-comparison # noqa: E712
        self.assertTrue(torch.all(masks[1] == False))  # pylint: disable=singleton-comparison # noqa: E712

        self.assertEqual(size_list[0], (10, 10))
        self.assertEqual(size_list[1], (12, 12))

    def test_detection_no_padding_omits_masks(self) -> None:
        images, masks, size_list = detection.batch_images(
            [
                torch.ones((3, 12, 12)),
                torch.ones((3, 12, 12)),
            ],
            size_divisible=4,
        )

        self.assertSequenceEqual(images.size(), (2, 3, 12, 12))
        self.assertIsNone(masks)
        self.assertEqual(size_list[0], (12, 12))
        self.assertEqual(size_list[1], (12, 12))

    def test_batch_random_resize_collator_scales_boxes(self) -> None:
        collator = detection.BatchRandomResizeCollator(0, (32, 32))
        collator.sizes = [20]

        image = torch.zeros((3, 10, 20))
        boxes = torch.tensor([[2.0, 1.0, 10.0, 5.0]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)
        batch = [(image, {"boxes": boxes, "labels": labels})]

        images, targets, masks, size_list = collator(batch)

        # Collator pads to size_divisible=32
        self.assertSequenceEqual(images.size(), (1, 3, 32, 32))
        self.assertEqual(size_list[0], (20, 20))
        self.assertTrue(torch.all(masks[:, :20, :20] == False))  # pylint: disable=singleton-comparison # noqa: E712

        expected = torch.tensor([[2.0, 2.0, 10.0, 10.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(targets[0]["boxes"], expected))


class TestNaFlexBatchSpec(unittest.TestCase):
    def test_resolve_batch_specs(self) -> None:
        specs = naflex.resolve_naflex_batch_specs((8, 12), canonical_patch_size=2)
        self.assertSequenceEqual(specs, (naflex.NaFlexBatchSpec(2, 24),))

        specs = naflex.resolve_naflex_batch_specs(
            (384, 384),
            canonical_patch_size=16,
            naflex_sizes=(224, 256, 320, 384),
            naflex_patch_sizes=(10, 14, 16, 20),
        )

        self.assertSequenceEqual(
            specs,
            (
                naflex.NaFlexBatchSpec(10, 501),
                naflex.NaFlexBatchSpec(14, 256),
                naflex.NaFlexBatchSpec(14, 334),
                naflex.NaFlexBatchSpec(14, 522),
                naflex.NaFlexBatchSpec(16, 196),
                naflex.NaFlexBatchSpec(16, 256),
                naflex.NaFlexBatchSpec(16, 400),
                naflex.NaFlexBatchSpec(16, 576),
                naflex.NaFlexBatchSpec(20, 125),
                naflex.NaFlexBatchSpec(20, 163),
                naflex.NaFlexBatchSpec(20, 256),
                naflex.NaFlexBatchSpec(20, 368),
            ),
        )

        with self.assertRaisesRegex(ValueError, "unmatched patch sizes \\[10\\]"):
            naflex.resolve_naflex_batch_specs(
                (384, 384),
                canonical_patch_size=16,
                naflex_sizes=(256, 320, 384),
                naflex_patch_sizes=(10, 16),
            )

    def test_resolve_batch_specs_preserves_resolution_multiplicity(self) -> None:
        specs = naflex.resolve_naflex_batch_specs(
            (80, 80),
            canonical_patch_size=16,
            naflex_sizes=(48, 64, 80),
            naflex_patch_sizes=(48,),
        )

        self.assertSequenceEqual(
            specs,
            (
                naflex.NaFlexBatchSpec(48, 1),
                naflex.NaFlexBatchSpec(48, 1),
                naflex.NaFlexBatchSpec(48, 2),
            ),
        )


class TestNaFlexBatchSchedule(unittest.TestCase):
    def test_batch_schedule_is_deterministic(self) -> None:
        specs = (
            naflex.NaFlexBatchSpec(2, 4),
            naflex.NaFlexBatchSpec(2, 6),
            naflex.NaFlexBatchSpec(2, 9),
        )
        schedule = naflex.NaFlexBatchSchedule(specs, seed=17)

        epoch_three = [schedule.select(3, batch_idx) for batch_idx in range(10)]
        epoch_four = [schedule.select(4, batch_idx) for batch_idx in range(10)]

        self.assertSequenceEqual(epoch_three, [specs[idx] for idx in (0, 1, 2, 0, 2, 0, 1, 2, 2, 2)])
        self.assertSequenceEqual(epoch_four, [specs[idx] for idx in (1, 2, 1, 1, 2, 0, 1, 1, 2, 1)])
        self.assertSequenceEqual(
            [schedule.select(3, batch_idx) for batch_idx in range(10)],
            epoch_three,
        )

    def test_batch_schedule_selects_patch_size_first(self) -> None:
        specs = (
            naflex.NaFlexBatchSpec(10, 501),
            naflex.NaFlexBatchSpec(16, 196),
            naflex.NaFlexBatchSpec(16, 256),
            naflex.NaFlexBatchSpec(16, 576),
        )
        schedule = naflex.NaFlexBatchSchedule(specs, seed=17)

        with patch.object(torch, "randint", side_effect=(torch.tensor(1), torch.tensor(2))) as randint:
            selected_spec = schedule.select(0, 0)

        self.assertEqual(selected_spec, specs[3])
        self.assertSequenceEqual([call.args[0] for call in randint.call_args_list], [2, 3])

    def test_batch_schedule_preserves_spec_multiplicity(self) -> None:
        spec_1 = naflex.NaFlexBatchSpec(48, 1)
        spec_2 = naflex.NaFlexBatchSpec(48, 2)
        schedule = naflex.NaFlexBatchSchedule((spec_1, spec_1, spec_2), seed=17)

        with patch.object(torch, "randint", return_value=torch.tensor(1)) as randint:
            selected_spec = schedule.select(0, 0)

        self.assertEqual(selected_spec, spec_1)
        self.assertEqual(randint.call_args.args[0], 3)


class TestNaFlexCollator(unittest.TestCase):
    def test_variable_size_batch(self) -> None:
        patch_size = 2
        collator = naflex.NaFlexTrainingCollator(patch_size)
        image_a = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)
        image_b = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 4)

        (patches, grid_sizes, valid_mask), targets = collator([(image_a, 3), (image_b, 7)])

        self.assertEqual(patches.size(), (2, 6, 8))
        torch.testing.assert_close(grid_sizes, torch.tensor([[2, 3], [1, 2]]))
        torch.testing.assert_close(
            valid_mask,
            torch.tensor(
                [
                    [True, True, True, True, True, True],
                    [True, True, False, False, False, False],
                ]
            ),
        )
        torch.testing.assert_close(targets, torch.tensor([3, 7]))
        torch.testing.assert_close(patches[~valid_mask], torch.zeros_like(patches[~valid_mask]))

    def test_patch_projection_matches_conv2d(self) -> None:
        patch_size = 2
        collator = naflex.NaFlexTrainingCollator(patch_size)
        image_a = torch.randn((2, 4, 6))
        image_b = torch.randn((2, 2, 4))
        conv = nn.Conv2d(2, 5, kernel_size=patch_size, stride=patch_size)

        (patches, _, valid_mask), _ = collator([(image_a, 0), (image_b, 1)])
        projected = F.linear(patches, conv.weight.flatten(1), conv.bias)  # pylint: disable=not-callable

        for image, batch_projection, mask in zip((image_a, image_b), projected, valid_mask):
            expected = conv(image.unsqueeze(0)).flatten(2).transpose(1, 2).squeeze(0)
            torch.testing.assert_close(batch_projection[mask], expected)

    def test_mixup_random_region(self) -> None:
        collator = naflex.NaFlexMixupTrainingCollator(1, num_classes=3, alpha=1.0, p=1.0)
        image_a = torch.arange(4 * 10, dtype=torch.float32).reshape(1, 4, 10)
        image_b = (torch.arange(8 * 5, dtype=torch.float32) + 100).reshape(1, 8, 5)

        with (
            patch.object(collator._lambda_dist, "sample", return_value=torch.tensor(0.25)),
            patch.object(collator, "_random_offset", side_effect=[0, 5, 4, 0, 3, 0, 0, 2]),
        ):
            (patches, grid_sizes, valid_mask), targets = collator([(image_a, 0), (image_b, 1)])

        expected_a = image_a.squeeze(0).clone()
        expected_a[:, 5:10].mul_(0.25).add_(image_b.squeeze(0)[4:8, :], alpha=0.75)
        expected_b = image_b.squeeze(0).clone()
        expected_b[3:7, :].mul_(0.25).add_(image_a.squeeze(0)[:, 2:7], alpha=0.75)

        torch.testing.assert_close(patches[0].reshape(4, 10), expected_a)
        torch.testing.assert_close(patches[1].reshape(8, 5), expected_b)
        torch.testing.assert_close(grid_sizes, torch.tensor([[4, 10], [8, 5]]))
        self.assertTrue(valid_mask.all().item())
        torch.testing.assert_close(targets, torch.tensor([[0.625, 0.375, 0.0], [0.375, 0.625, 0.0]]))

    def test_mixup_effective_lambda_and_padding(self) -> None:
        collator = naflex.NaFlexMixupTrainingCollator(1, num_classes=2, alpha=1.0, p=1.0)
        image_a = torch.zeros((1, 2, 4))
        image_b = torch.ones((1, 1, 2))

        with (
            patch.object(collator._lambda_dist, "sample", return_value=torch.tensor(0.5)),
            patch.object(collator, "_random_offset", return_value=0),
        ):
            (patches, _, valid_mask), targets = collator([(image_a, 0), (image_b, 1)])

        torch.testing.assert_close(valid_mask.sum(dim=1), torch.tensor([8, 2]))
        torch.testing.assert_close(patches[0, :2], torch.full((2, 1), 0.5))
        torch.testing.assert_close(patches[1, :2], torch.full((2, 1), 0.5))
        torch.testing.assert_close(targets, torch.tensor([[0.875, 0.125], [0.5, 0.5]]))

    def test_mixup_disabled(self) -> None:
        spec = naflex.NaFlexBatchSpec(2, 6)
        collator = naflex.NaFlexMixupTrainingCollator(None, num_classes=3, alpha=1.0, p=0.0)
        image_a = torch.randn((2, 4, 6))
        image_b = torch.randn((2, 2, 4))
        batch = [(image_a, 0), (image_b, 1)]
        scheduled_batch = [(naflex.NaFlexScheduledInput(image, spec), target) for image, target in batch]

        (patches, grid_sizes, valid_mask), targets = collator(scheduled_batch)
        (expected_patches, expected_grid_sizes, expected_valid_mask), expected_targets = naflex.NaFlexTrainingCollator(
            spec.patch_size
        )(batch)

        torch.testing.assert_close(patches, expected_patches)
        torch.testing.assert_close(grid_sizes, expected_grid_sizes)
        torch.testing.assert_close(valid_mask, expected_valid_mask)
        torch.testing.assert_close(targets, expected_targets)

    def test_inference_batch(self) -> None:
        patch_size = 2
        collator = naflex.NaFlexPathCollator(patch_size)
        image_a = torch.ones((3, 4, 6))
        image_b = torch.ones((3, 2, 4))

        paths, (patches, grid_sizes, valid_mask), targets = collator(
            [("image_a.jpg", image_a, 3), ("image_b.jpg", image_b, 7)]
        )

        self.assertSequenceEqual(paths, ["image_a.jpg", "image_b.jpg"])
        self.assertEqual(patches.size(), (2, 6, 12))
        torch.testing.assert_close(grid_sizes, torch.tensor([[2, 3], [1, 2]]))
        torch.testing.assert_close(valid_mask.sum(dim=1), torch.tensor([6, 2]))
        torch.testing.assert_close(targets, torch.tensor([3, 7]))

    def test_multi_view_batch(self) -> None:
        spec = naflex.NaFlexBatchSpec(2, 6)
        collator = naflex.NaFlexMultiViewPathCollator()
        image_a_views = (torch.ones((3, 4, 6)), torch.ones((3, 2, 4)))
        image_b_views = (torch.ones((3, 2, 4)), torch.ones((3, 6, 2)))

        paths, views, targets = collator(
            [
                ("image_a.jpg", naflex.NaFlexScheduledInput(image_a_views, spec), 3),
                ("image_b.jpg", naflex.NaFlexScheduledInput(image_b_views, spec), 7),
            ]
        )

        self.assertSequenceEqual(paths, ["image_a.jpg", "image_b.jpg"])
        self.assertEqual(len(views), 2)
        view1_patches, view1_grid_sizes, view1_valid_mask = views[0]
        view2_patches, view2_grid_sizes, view2_valid_mask = views[1]
        self.assertEqual(view1_patches.size(), (2, 6, 12))
        self.assertEqual(view2_patches.size(), (2, 3, 12))
        torch.testing.assert_close(view1_grid_sizes, torch.tensor([[2, 3], [1, 2]]))
        torch.testing.assert_close(view2_grid_sizes, torch.tensor([[1, 2], [3, 1]]))
        torch.testing.assert_close(view1_valid_mask.sum(dim=1), torch.tensor([6, 2]))
        torch.testing.assert_close(view2_valid_mask.sum(dim=1), torch.tensor([2, 3]))
        torch.testing.assert_close(targets, torch.tensor([3, 7]))


class TestNaFlexBatchProcessor(unittest.TestCase):
    def test_requires_scheduled_collator(self) -> None:
        spec = naflex.NaFlexBatchSpec(2, 4)
        with self.assertRaises(ValueError):
            naflex.NaFlexBatchProcessor(
                naflex.NaFlexTrainingCollator(2),
                (spec,),
                {spec: torch.nn.Identity()},
                seed=0,
            )

    def test_batch_uses_selected_spec(self) -> None:
        spec_1 = naflex.NaFlexBatchSpec(1, 6)
        spec_2 = naflex.NaFlexBatchSpec(2, 6)
        base_collator = naflex.NaFlexTrainingCollator()
        processor = naflex.NaFlexBatchProcessor(
            base_collator,
            (spec_1, spec_1, spec_2),
            {
                spec_1: lambda image: image[:, :2, :3],
                spec_2: lambda image: image[:, :4, :6],
            },
            seed=0,
        )
        self.assertSequenceEqual(processor.schedule.specs, (spec_1, spec_1, spec_2))
        batch = [(torch.ones((1, 8, 8)), 1), (torch.ones((1, 8, 8)), 2)]

        with patch.object(processor, "schedule") as schedule:
            schedule.select.side_effect = [spec_1, spec_2]
            batches = [processor(batch), processor(batch)]

        self.assertSequenceEqual([call.args for call in schedule.select.call_args_list], [(0, 0), (0, 1)])
        self.assertEqual([inputs[0].size() for inputs, _ in batches], [(2, 6, 1), (2, 6, 4)])
        for (_, grid_sizes, valid_mask), targets in batches:
            torch.testing.assert_close(grid_sizes, torch.tensor([[2, 3], [2, 3]]))
            self.assertTrue(valid_mask.all().item())
            torch.testing.assert_close(targets, torch.tensor([1, 2]))

    def test_multiscale_stream_preserves_batch_size(self) -> None:
        spec_4 = naflex.NaFlexBatchSpec(2, 4)
        spec_6 = naflex.NaFlexBatchSpec(2, 6)
        base_collator = naflex.NaFlexTrainingCollator()
        processor = naflex.NaFlexBatchProcessor(
            base_collator,
            (spec_4, spec_6),
            {
                spec_4: lambda image: image[:, :4, :4],
                spec_6: lambda image: image[:, :4, :6],
            },
            seed=0,
        )
        samples = [(torch.ones((1, 8, 8)), idx) for idx in range(5)]

        with patch.object(processor, "schedule") as schedule:
            schedule.select.side_effect = [spec_4, spec_6, spec_4]
            batches = list(processor.iter_batches(samples, batch_size=2, drop_last=False))

        self.assertSequenceEqual([call.args for call in schedule.select.call_args_list], [(0, 0), (0, 1), (0, 2)])
        self.assertEqual([batch[0][0].size(0) for batch in batches], [2, 2, 1])
        self.assertEqual([batch[0][0].size(1) for batch in batches], [4, 6, 4])
        torch.testing.assert_close(torch.concat([batch[1] for batch in batches]), torch.arange(5))

    def test_multiscale_stream_snapshots_epoch(self) -> None:
        spec = naflex.NaFlexBatchSpec(2, 4)
        processor = naflex.NaFlexBatchProcessor(
            naflex.NaFlexTrainingCollator(),
            (spec,),
            {spec: lambda image: image[:, :4, :4]},
            seed=0,
        )
        samples = [(torch.ones((1, 4, 4)), idx) for idx in range(4)]
        calls: list[tuple[int, int]] = []

        def select_spec(epoch: int, global_batch_idx: int) -> naflex.NaFlexBatchSpec:
            calls.append((epoch, global_batch_idx))
            return spec

        with patch.object(processor, "schedule") as schedule:
            schedule.select.side_effect = select_spec
            old_iterator = processor.iter_batches(samples, batch_size=2, drop_last=False)
            next(old_iterator)
            processor.set_epoch(1)
            next(old_iterator)
            next(processor.iter_batches(samples, batch_size=2, drop_last=False))

        self.assertSequenceEqual(calls, [(0, 0), (0, 1), (1, 0)])

    def test_multiscale_stream_maps_worker_batches_to_global_schedule(self) -> None:
        spec = naflex.NaFlexBatchSpec(2, 4)
        processor = naflex.NaFlexBatchProcessor(
            naflex.NaFlexTrainingCollator(),
            (spec,),
            {spec: lambda image: image},
            seed=0,
        )
        samples = [(torch.ones((1, 4, 4)), idx) for idx in range(2)]

        with (
            patch.object(torch.utils.data, "get_worker_info", return_value=SimpleNamespace(id=2, num_workers=4)),
            patch.object(processor, "schedule") as schedule,
        ):
            schedule.select.return_value = spec
            list(processor.iter_batches(samples, batch_size=1, drop_last=False))

        self.assertSequenceEqual([call.args for call in schedule.select.call_args_list], [(0, 2), (0, 6)])
