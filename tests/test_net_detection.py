import logging
import unittest

import torch

from birder.core import net
from birder.core.net import detection

logging.disable(logging.CRITICAL)


class TestNetDetection(unittest.TestCase):
    def test_faster_rcnn(self) -> None:
        size = detection.Faster_RCNN.default_size
        backbone = net.ResNet_v2(3, 2, 50, size=size)
        n = detection.Faster_RCNN(50, backbone, size=size)
        n.eval()
        n(torch.rand((1, 3, size, size)))

        n.train()
        n(
            torch.rand((1, 3, size, size)),
            targets=[
                {
                    "boxes": torch.tensor([[10.1, 10.1, 30.2, 40.2]]),
                    "labels": torch.tensor([1]),
                }
            ],
        )
        torch.jit.script(n)

    def test_retinanet(self) -> None:
        size = detection.RetinaNet.default_size
        backbone = net.ResNet_v2(3, 2, 50, size=size)
        n = detection.RetinaNet(50, backbone, size=size)
        n.eval()
        n(torch.rand((1, 3, size, size)))

        n.train()
        n(
            torch.rand((1, 3, size, size)),
            targets=[
                {
                    "boxes": torch.tensor([[10.1, 10.1, 30.2, 40.2]]),
                    "labels": torch.tensor([1]),
                }
            ],
        )
        torch.jit.script(n)

    def test_ssd(self) -> None:
        size = detection.SSD.default_size
        backbone = net.ResNet_v2(3, 2, 50, size=size)
        n = detection.SSD(50, backbone, size=size)
        n.eval()
        n(torch.rand((1, 3, size, size)))

        n.train()
        n(
            torch.rand((1, 3, size, size)),
            targets=[
                {
                    "boxes": torch.tensor([[10.1, 10.1, 30.2, 40.2]]),
                    "labels": torch.tensor([1]),
                }
            ],
        )
        torch.jit.script(n)
