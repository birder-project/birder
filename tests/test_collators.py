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


class TestNaFlexSequenceLengthSchedule(unittest.TestCase):
    def test_sequence_length_schedule_is_deterministic(self) -> None:
        schedule = naflex.NaFlexSequenceLengthSchedule((4, 6, 9), seed=17)

        epoch_three = [schedule.select(3, batch_idx) for batch_idx in range(10)]
        epoch_four = [schedule.select(4, batch_idx) for batch_idx in range(10)]

        self.assertSequenceEqual(epoch_three, [4, 6, 9, 4, 9, 4, 6, 9, 9, 9])
        self.assertSequenceEqual(epoch_four, [6, 9, 6, 6, 9, 4, 6, 6, 9, 6])
        self.assertSequenceEqual(
            [schedule.select(3, batch_idx) for batch_idx in range(10)],
            epoch_three,
        )


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
        collator = naflex.NaFlexMixupTrainingCollator(2, num_classes=3, alpha=1.0, p=0.0)
        image_a = torch.randn((2, 4, 6))
        image_b = torch.randn((2, 2, 4))

        (patches, grid_sizes, valid_mask), targets = collator([(image_a, 0), (image_b, 1)])
        (expected_patches, expected_grid_sizes, expected_valid_mask), expected_targets = naflex.NaFlexTrainingCollator(
            2
        )([(image_a, 0), (image_b, 1)])

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


class TestNaFlexBatchProcessor(unittest.TestCase):
    def test_multiscale_batch_uses_one_transform(self) -> None:
        base_collator = naflex.NaFlexTrainingCollator(2)
        processor = naflex.NaFlexBatchProcessor(
            base_collator,
            {
                4: lambda image: image[:, :4, :4],
                6: lambda image: image[:, :4, :6],
            },
            seed=0,
        )
        batch = [(torch.ones((1, 8, 8)), 1), (torch.ones((1, 8, 8)), 2)]

        with patch.object(processor, "schedule") as schedule:
            schedule.select.return_value = 6
            (patches, grid_sizes, valid_mask), targets = processor(batch)

        schedule.select.assert_called_once_with(0, 0)
        self.assertEqual(patches.size(), (2, 6, 4))
        torch.testing.assert_close(grid_sizes, torch.tensor([[2, 3], [2, 3]]))
        self.assertTrue(valid_mask.all().item())
        torch.testing.assert_close(targets, torch.tensor([1, 2]))

    def test_multiscale_stream_preserves_batch_size(self) -> None:
        base_collator = naflex.NaFlexTrainingCollator(2)
        processor = naflex.NaFlexBatchProcessor(
            base_collator,
            {
                4: lambda image: image[:, :4, :4],
                6: lambda image: image[:, :4, :6],
            },
            seed=0,
        )
        samples = [(torch.ones((1, 8, 8)), idx) for idx in range(5)]

        with patch.object(processor, "schedule") as schedule:
            schedule.select.side_effect = [4, 6, 4]
            batches = list(processor.iter_batches(samples, batch_size=2, drop_last=False))

        self.assertSequenceEqual([call.args for call in schedule.select.call_args_list], [(0, 0), (0, 1), (0, 2)])
        self.assertEqual([batch[0][0].size(0) for batch in batches], [2, 2, 1])
        self.assertEqual([batch[0][0].size(1) for batch in batches], [4, 6, 4])
        torch.testing.assert_close(torch.concat([batch[1] for batch in batches]), torch.arange(5))

    def test_multiscale_stream_snapshots_epoch(self) -> None:
        processor = naflex.NaFlexBatchProcessor(
            naflex.NaFlexTrainingCollator(2),
            {4: lambda image: image[:, :4, :4]},
            seed=0,
        )
        samples = [(torch.ones((1, 4, 4)), idx) for idx in range(4)]
        calls: list[tuple[int, int]] = []

        def select_max_seq_len(epoch: int, global_batch_idx: int) -> int:
            calls.append((epoch, global_batch_idx))
            return 4

        with patch.object(processor, "schedule") as schedule:
            schedule.select.side_effect = select_max_seq_len
            old_iterator = processor.iter_batches(samples, batch_size=2, drop_last=False)
            next(old_iterator)
            processor.set_epoch(1)
            next(old_iterator)
            next(processor.iter_batches(samples, batch_size=2, drop_last=False))

        self.assertSequenceEqual(calls, [(0, 0), (0, 1), (1, 0)])

    def test_multiscale_stream_maps_worker_batches_to_global_schedule(self) -> None:
        processor = naflex.NaFlexBatchProcessor(
            naflex.NaFlexTrainingCollator(2),
            {4: lambda image: image},
            seed=0,
        )
        samples = [(torch.ones((1, 4, 4)), idx) for idx in range(2)]

        with (
            patch.object(torch.utils.data, "get_worker_info", return_value=SimpleNamespace(id=2, num_workers=4)),
            patch.object(processor, "schedule") as schedule,
        ):
            schedule.select.return_value = 4
            list(processor.iter_batches(samples, batch_size=1, drop_last=False))

        self.assertSequenceEqual([call.args for call in schedule.select.call_args_list], [(0, 2), (0, 6)])
