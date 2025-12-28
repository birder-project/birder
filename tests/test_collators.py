import logging
import unittest

import torch

from birder.data.collators import detection

logging.disable(logging.CRITICAL)


class TestTransforms(unittest.TestCase):
    def test_detection(self) -> None:
        (images, masks, size_list) = detection.batch_images(
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

        self.assertTrue(torch.all(masks[0, :10, :10] == False))  # pylint: disable=singleton-comparison # noqa: E712
        self.assertTrue(torch.all(masks[0, 11:, 11:] == True))  # pylint: disable=singleton-comparison # noqa: E712
        self.assertTrue(torch.all(masks[1] == False))  # pylint: disable=singleton-comparison # noqa: E712

        self.assertEqual(size_list[0], (10, 10))
        self.assertEqual(size_list[1], (12, 12))

    def test_batch_random_resize_collator_scales_boxes(self) -> None:
        collator = detection.BatchRandomResizeCollator(0, (32, 32))
        collator.sizes = [20]

        image = torch.zeros((3, 10, 20))
        boxes = torch.tensor([[2.0, 1.0, 10.0, 5.0]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)
        batch = [(image, {"boxes": boxes, "labels": labels})]

        (images, targets, masks, size_list) = collator(batch)

        # Collator pads to size_divisible=32
        self.assertSequenceEqual(images.size(), (1, 3, 32, 32))
        self.assertEqual(size_list[0], (20, 20))
        self.assertTrue(torch.all(masks[:, :20, :20] == False))  # pylint: disable=singleton-comparison # noqa: E712

        expected = torch.tensor([[2.0, 2.0, 10.0, 10.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(targets[0]["boxes"], expected))
