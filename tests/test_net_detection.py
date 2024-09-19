import logging
import unittest
from typing import Optional

import torch
from parameterized import parameterized

from birder.model_registry import registry
from birder.net.detection import base

logging.disable(logging.CRITICAL)


class TestBase(unittest.TestCase):
    def test_get_signature(self) -> None:
        signature = base.get_detection_signature((1, 3, 224, 224), 10)
        self.assertIn("inputs", signature)
        self.assertIn("outputs", signature)
        self.assertIn("boxes", signature["outputs"][0][0])


class TestNetDetection(unittest.TestCase):
    @parameterized.expand(  # type: ignore[misc]
        [
            ("faster_rcnn", None, ("resnet_v2", 50)),
            ("retinanet", None, ("mobilenet_v3_small", 1)),
            ("ssd", None, ("efficientnet_v2", 0)),
        ]
    )
    def test_net_detection(self, network_name: str, net_param: Optional[float], encoder: tuple[str, float]) -> None:
        backbone = registry.net_factory(encoder[0], 3, 10, encoder[1])
        n = registry.detection_net_factory(network_name, 10, backbone, net_param=net_param)
        size = n.default_size
        backbone.adjust_size(size)

        n.eval()
        out = n(torch.rand((1, 3, size, size)))
        (detections, losses) = out
        self.assertEqual(len(losses), 0)
        for detection in detections:
            for key in ["boxes", "labels", "scores"]:
                self.assertFalse(torch.isnan(detection[key]).any())

        n.train()
        out = n(
            torch.rand((1, 3, size, size)),
            targets=[
                {
                    "boxes": torch.tensor([[10.1, 10.1, 30.2, 40.2]]),
                    "labels": torch.tensor([1]),
                }
            ],
        )
        (detections, losses) = out
        self.assertGreater(len(losses), 0)
        for loss in losses.values():
            self.assertFalse(torch.isnan(loss).any())

        for detection in detections:
            for key in ["boxes", "labels", "scores"]:
                self.assertFalse(torch.isnan(detection[key]).any())

        torch.jit.script(n)
