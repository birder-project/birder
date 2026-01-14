import json
import logging
import unittest

import torch
from parameterized import parameterized

from birder.common.lib import env_bool
from birder.conf.settings import DEFAULT_NUM_CHANNELS
from birder.data.collators.detection import batch_images
from birder.model_registry import registry
from birder.net.base import reparameterize_available
from birder.net.detection import base

logging.disable(logging.CRITICAL)

NET_DETECTION_TEST_CASES = [
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
    ("rt_detr_v1", "resnet_v1_50"),
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

DETECTION_DYNAMIC_SIZE_CASES = [
    ("deformable_detr", "fasternet_t0"),
    ("deformable_detr_boxref", "regnet_x_200m"),
    ("detr", "regnet_y_1_6g"),
    ("efficientdet_d0", "efficientnet_v1_b0"),
    ("faster_rcnn", "resnet_v2_18"),
    ("fcos", "resnet_d_50"),
    ("retinanet", "mobilenet_v3_small_1_0"),
    ("retinanet_sfp", "vit_s16"),
    ("rt_detr_v1", "resnet_v1_50"),
    ("ssd", "efficientnet_v2_s"),
    ("ssdlite", "mobilenet_v2_0_25", (512, 512)),
    ("vitdet", "vit_b32"),
    ("yolo_v2", "resnet_v1_18"),
    ("yolo_v3", "darknet_17"),
    ("yolo_v4", "csp_darknet_53"),
    ("yolo_v4_tiny", "efficientnet_lite0"),
]


class TestBase(unittest.TestCase):
    def test_get_signature(self) -> None:
        signature = base.get_detection_signature((1, DEFAULT_NUM_CHANNELS, 224, 224), 10, dynamic=False)
        self.assertIn("dynamic", signature)
        self.assertIn("inputs", signature)
        self.assertIn("outputs", signature)
        self.assertIn("boxes", signature["outputs"][0][0])


class TestNetDetection(unittest.TestCase):
    @parameterized.expand(NET_DETECTION_TEST_CASES)  # type: ignore[untyped-decorator]
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

        # Reparameterize
        if reparameterize_available(n) is True:
            n.reparameterize_model()
            (detections, losses) = n(torch.rand((1, DEFAULT_NUM_CHANNELS, *size)))
            self.assertEqual(len(losses), 0)
            for detection in detections:
                for key in ["boxes", "labels", "scores"]:
                    self.assertFalse(torch.isnan(detection[key]).any())

    @parameterized.expand(DETECTION_DYNAMIC_SIZE_CASES)  # type: ignore[untyped-decorator]
    def test_detection_dynamic_size(
        self,
        network_name: str,
        encoder: str,
        size: tuple[int, int] = (256, 256),
    ) -> None:
        backbone = registry.net_factory(encoder, DEFAULT_NUM_CHANNELS, 10, size=size)
        n = registry.detection_net_factory(network_name, 10, backbone, size=size)
        n.eval()
        n.set_dynamic_size()

        (detections, losses) = n(torch.rand((1, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(len(losses), 0)
        for detection in detections:
            for key in ["boxes", "labels", "scores"]:
                self.assertFalse(torch.isnan(detection[key]).any())

        size = (size[0] + 32, size[1] + 64)
        (detections, losses) = n(torch.rand((1, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(len(losses), 0)
        for detection in detections:
            for key in ["boxes", "labels", "scores"]:
                self.assertFalse(torch.isnan(detection[key]).any())

    @parameterized.expand(NET_DETECTION_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_net_detection_backward(
        self,
        network_name: str,
        encoder: str,
        size: tuple[int, int] = (256, 256),
        size_step: int = 2**5,
    ) -> None:
        backbone = registry.net_factory(encoder, DEFAULT_NUM_CHANNELS, 10, size=size)
        n = registry.detection_net_factory(network_name, 10, backbone, size=size)

        size = (size[0] + size_step, size[1] + size_step)
        n.adjust_size(size)
        for name, param in n.named_parameters():
            self.assertIsNone(param.grad, msg=f"{network_name} adjust_size set grad for {name}")
            self.assertIsNone(param.grad_fn, msg=f"{network_name} adjust_size tracked grad for {name}")

        n.train()
        (_, losses) = n(
            torch.rand((1, DEFAULT_NUM_CHANNELS, *size)),
            targets=[
                {
                    "boxes": torch.tensor([[10.1, 10.1, 30.2, 40.2]]),
                    "labels": torch.tensor([1]),
                }
            ],
        )
        self.assertGreater(len(losses), 0)
        loss = sum(v for v in losses.values())
        self.assertEqual(loss.ndim, 0)
        loss.backward()
        for name, param in n.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"{network_name} missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all().item(), msg=f"{network_name} non-finite grad for {name}")

        n.zero_grad()

        if reparameterize_available(n) is True:
            n.eval()
            n.reparameterize_model()
            n(torch.rand((1, DEFAULT_NUM_CHANNELS, *size)))
            for name, param in n.named_parameters():
                self.assertIsNone(param.grad, msg=f"{network_name} reparameterize_model set grad for {name}")
                self.assertIsNone(param.grad_fn, msg=f"{network_name} reparameterize_model tracked grad for {name}")

    @parameterized.expand(DETECTION_DYNAMIC_SIZE_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_detection_dynamic_size_backward(
        self,
        network_name: str,
        encoder: str,
        size: tuple[int, int] = (256, 256),
        size_step: int = 2**5,
    ) -> None:
        backbone = registry.net_factory(encoder, DEFAULT_NUM_CHANNELS, 10, size=size)
        n = registry.detection_net_factory(network_name, 10, backbone, size=size)
        n.train()
        n.set_dynamic_size()

        size = (size[0] + size_step, size[1] + size_step)
        (_, losses) = n(
            torch.rand((1, DEFAULT_NUM_CHANNELS, *size)),
            targets=[
                {
                    "boxes": torch.tensor([[10.1, 10.1, 30.2, 40.2]]),
                    "labels": torch.tensor([1]),
                }
            ],
        )
        self.assertGreater(len(losses), 0)
        loss = sum(v for v in losses.values())
        loss.backward()
        for name, param in n.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"{network_name} missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all().item(), msg=f"{network_name} non-finite grad for {name}")
