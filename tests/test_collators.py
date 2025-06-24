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
