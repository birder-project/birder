import json
import logging
import unittest

import torch
from parameterized import parameterized

from birder.conf.settings import DEFAULT_NUM_CHANNELS
from birder.data.collators.detection import batch_images
from birder.model_registry import registry
from birder.net.detection import base

logging.disable(logging.CRITICAL)


class TestBase(unittest.TestCase):
    def test_get_signature(self) -> None:
        signature = base.get_detection_signature((1, DEFAULT_NUM_CHANNELS, 224, 224), 10, dynamic=False)
        self.assertIn("dynamic", signature)
        self.assertIn("inputs", signature)
        self.assertIn("outputs", signature)
        self.assertIn("boxes", signature["outputs"][0][0])


class TestNetDetection(unittest.TestCase):
    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("deformable_detr", "fasternet_t0"),
            ("deformable_detr", "efficientvit_msft_m0"),  # 3 stage network
            ("deformable_detr_boxref", "regnet_x_200m"),
            ("detr", "regnet_y_1_6g"),
            ("efficientdet_d0", "efficientnet_v1_b0"),
            ("faster_rcnn", "resnet_v2_18"),
            ("faster_rcnn", "efficientvit_msft_m0"),  # 3 stage network
            ("fcos", "tiny_vit_5m"),
            ("fcos", "vit_s32"),  # 1 stage network
            ("retinanet", "mobilenet_v3_small_1_0"),
            ("retinanet", "efficientvit_msft_m0"),  # 3 stage network
            ("retinanet_sfp", "vit_det_m16_rms"),
            ("ssd", "efficientnet_v2_s"),
            ("ssd", "vit_s16"),  # 1 stage network
            ("ssdlite", "mobilenet_v2_0_25", (512, 512)),
            ("ssdlite", "vit_t16"),  # 1 stage network
            ("vitdet", "vit_sam_b16"),
            ("yolo_v2", "resnet_v1_18"),
            ("yolo_v3", "darknet_17"),
            ("yolo_v4", "csp_darknet_53"),
            ("yolo_v4_tiny", "efficientnet_lite0"),
        ]
    )
    def test_net_detection(
        self,
        network_name: str,
        encoder: str,
        size: tuple[int, int] = (256, 256),
    ) -> None:
        backbone = registry.net_factory(encoder, DEFAULT_NUM_CHANNELS, 10, size=size)
        n = registry.detection_net_factory(network_name, 10, backbone, size=size, export_mode=True)

        # Ensure config is serializable
        _ = json.dumps(n.config)

        # Test network
        n.eval()
        out = n(torch.rand((1, DEFAULT_NUM_CHANNELS, *size)))
        (detections, losses) = out
        self.assertEqual(len(losses), 0)
        for detection in detections:
            for key in ["boxes", "labels", "scores"]:
                self.assertFalse(torch.isnan(detection[key]).any())

        # Again in "dynamic size" mode
        (images, masks, image_sizes) = batch_images(
            [torch.rand((DEFAULT_NUM_CHANNELS, *size)), torch.rand((DEFAULT_NUM_CHANNELS, size[0] - 12, size[1] - 24))],
            size_divisible=4,
        )
        out = n(images, masks=masks, image_sizes=image_sizes)

        # Reset classifier
        n.reset_classifier(20)
        n(torch.rand((1, DEFAULT_NUM_CHANNELS, *size)))

        n.train()
        out = n(
            torch.rand((1, DEFAULT_NUM_CHANNELS, *size)),
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

        loss = sum(v for v in losses.values())
        self.assertEqual(loss.ndim, 0)

        for detection in detections:
            for key in ["boxes", "labels", "scores"]:
                self.assertFalse(torch.isnan(detection[key]).any())

        if n.scriptable is True:
            torch.jit.script(n)
        else:
            n.eval()
            torch.jit.trace(n, example_inputs=torch.rand((1, DEFAULT_NUM_CHANNELS, *size)))
            n.train()

        # Freeze
        n.eval()
        n.freeze(freeze_classifier=False)
        n(torch.rand((1, DEFAULT_NUM_CHANNELS, *size)))
