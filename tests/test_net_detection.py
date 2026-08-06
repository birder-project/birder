import copy
import json
import logging
import unittest

import torch
from parameterized import parameterized
from torchvision.ops import boxes as box_ops

from birder.common.lib import env_bool
from birder.conf.settings import DEFAULT_NUM_CHANNELS
from birder.data.collators.detection import batch_images
from birder.model_registry import registry
from birder.net.base import reparameterize_available
from birder.net.detection import base

logging.disable(logging.CRITICAL)

NET_DETECTION_TEST_CASES = [
    ("d_fine_n", "hgnet_v2_b0"),
    ("deformable_detr", "fasternet_t0"),
    ("deformable_detr", "efficientvit_msft_m0"),  # 3 stage network
    ("deformable_detr_boxref", "regnet_x_200m"),
    ("detr", "regnet_y_1_6g"),
    ("efficientdet_d0", "efficientnet_v1_b0"),
    ("faster_rcnn", "resnet_v2_18"),
    ("faster_rcnn", "efficientvit_msft_m0"),  # 3 stage network
    ("fcos", "tiny_vit_5m"),
    ("fcos", "vit_s32"),  # 1 stage network
    ("lw_detr", "vit_s32", (384, 384)),  # 1 stage network
    ("lw_detr", "vovnet_v1_27s", (384, 384)),
    ("lw_detr_2stg", "vit_t32", (384, 384)),  # 1 stage network
    ("lw_detr_l", "vit_s32", (384, 384)),  # 1 stage network
    ("lw_detr_l", "vovnet_v1_27s", (384, 384)),
    ("plain_detr_lite", "vit_t16"),
    ("plain_detr", "vit_s16"),
    ("retinanet", "mobilenet_v3_small_1_0"),
    ("retinanet", "efficientvit_msft_m0"),  # 3 stage network
    ("retinanet_sfp", "vit_det_m16_rms"),
    ("rf_detr_nano", "vit_t16", (384, 384)),  # 1 stage network
    ("rf_detr_nano", "resnet_v1_18", (384, 384)),
    ("rt_detr_v1", "resnet_v1_50"),
    ("rt_detr_v2", "se_resnet_d_50"),
    ("rt_detr_v2_s_dsp", "vovnet_v2_19"),
    ("rtmdet_t", "cspnext_t"),
    ("ssd", "efficientnet_v2_s", (256, 256), 2),
    ("ssd", "vit_s16", (256, 256), 2),  # 1 stage network
    ("ssdlite", "mobilenet_v2_0_25", (512, 512), 2),
    ("ssdlite", "vit_t16", (256, 256), 2),  # 1 stage network
    ("vitdet", "vit_sam_b16"),
    ("yolo_v2", "resnet_v1_18"),
    ("yolo_v3", "darknet_17"),
    ("yolo_v4", "csp_darknet_53"),
    ("yolo_v4_tiny", "efficientnet_lite0"),
]

DETECTION_DYNAMIC_SIZE_CASES = [
    ("d_fine_n", "hgnet_v2_b0"),
    ("deformable_detr", "fasternet_t0"),
    ("deformable_detr_boxref", "regnet_x_200m"),
    ("detr", "regnet_y_1_6g"),
    ("efficientdet_d0", "efficientnet_v1_b0"),
    ("faster_rcnn", "resnet_v2_18"),
    ("fcos", "resnet_d_50"),
    ("lw_detr", "vovnet_v1_27s", (384, 384)),
    ("lw_detr_2stg", "vit_t32", (384, 384)),
    ("lw_detr_l", "vit_s32", (384, 384)),
    ("plain_detr_lite", "vit_t16"),
    ("plain_detr", "vit_s16"),
    ("retinanet", "mobilenet_v3_small_1_0"),
    ("retinanet_sfp", "vit_s16"),
    ("rf_detr_nano", "vit_t16", (384, 384)),
    ("rf_detr_nano", "resnet_v1_18", (384, 384)),
    ("rt_detr_v1", "resnet_v1_50"),
    ("rt_detr_v2", "se_resnet_d_50"),
    ("rt_detr_v2_s_dsp", "vovnet_v2_19"),
    ("rtmdet_t", "cspnext_t"),
    ("ssd", "efficientnet_v2_s", (256, 256), 2),
    ("ssdlite", "mobilenet_v2_0_25", (512, 512), 2),
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

    def test_aligned_box_ious_match_torchvision(self) -> None:
        boxes1_values = [[0, 0, 2, 2], [1, 1, 5, 4], [-2, -1, 1, 3], [3, 2, 7, 8]]
        boxes2_values = [[0, 0, 2, 2], [0, 2, 4, 5], [2, 3, 4, 6], [4, 0, 6, 9]]
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                boxes1 = torch.tensor(boxes1_values, dtype=dtype)
                boxes2 = torch.tensor(boxes2_values, dtype=dtype)

                expected_iou = box_ops.box_iou(boxes1, boxes2).diag()
                expected_giou = box_ops.generalized_box_iou(boxes1, boxes2).diag()

                self.assertTrue(torch.equal(base.aligned_box_iou(boxes1, boxes2), expected_iou))
                self.assertTrue(torch.equal(base.aligned_generalized_box_iou(boxes1, boxes2), expected_giou))

    def test_aligned_box_ious_match_torchvision_gradients(self) -> None:
        boxes1_values = [[0, 0, 2, 2], [1, 1, 5, 4], [-2, -1, 1, 3], [3, 2, 7, 8]]
        boxes2_values = [[0, 0, 2, 2], [0, 2, 4, 5], [2, 3, 4, 6], [4, 0, 6, 9]]
        function_pairs = (
            (box_ops.box_iou, base.aligned_box_iou),
            (box_ops.generalized_box_iou, base.aligned_generalized_box_iou),
        )
        for reference_function, aligned_function in function_pairs:
            with self.subTest(function=aligned_function.__name__):
                reference_boxes1 = torch.tensor(boxes1_values, dtype=torch.float64, requires_grad=True)
                reference_boxes2 = torch.tensor(boxes2_values, dtype=torch.float64, requires_grad=True)
                reference_function(reference_boxes1, reference_boxes2).diag().sum().backward()

                aligned_boxes1 = torch.tensor(boxes1_values, dtype=torch.float64, requires_grad=True)
                aligned_boxes2 = torch.tensor(boxes2_values, dtype=torch.float64, requires_grad=True)
                aligned_function(aligned_boxes1, aligned_boxes2).sum().backward()

                self.assertTrue(torch.equal(aligned_boxes1.grad, reference_boxes1.grad))
                self.assertTrue(torch.equal(aligned_boxes2.grad, reference_boxes2.grad))


class TestNetDetection(unittest.TestCase):
    @parameterized.expand(NET_DETECTION_TEST_CASES)  # type: ignore[untyped-decorator]
    def test_net_detection(
        self,
        network_name: str,
        encoder: str,
        size: tuple[int, int] = (256, 256),
        batch_size: int = 1,
    ) -> None:
        backbone = registry.net_factory(encoder, 10, size=size)
        backbone_state = copy.deepcopy(backbone.state_dict())
        n = registry.detection_net_factory(network_name, 10, backbone, size=size, export_mode=True)

        # Detector construction may remove classification-only modules, but it must not
        # replace the backbone or modify any surviving feature-extractor state
        self.assertIs(n.backbone, backbone)
        detection_backbone_state = n.backbone.state_dict()
        unexpected_keys = detection_backbone_state.keys() - backbone_state.keys()
        self.assertSetEqual(set(unexpected_keys), set())
        for name, value in detection_backbone_state.items():
            torch.testing.assert_close(
                value,
                backbone_state[name],
                rtol=0,
                atol=0,
                equal_nan=True,
                msg=lambda msg, name=name: (
                    f"{network_name} modified backbone state '{name}' during construction:\n{msg}"
                ),
            )

        del backbone_state

        # Ensure config is serializable
        _ = json.dumps(n.config)

        # Test network
        n.eval()
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        detections, losses = out
        self.assertEqual(len(losses), 0)
        for detection in detections:
            for key in ["boxes", "labels", "scores"]:
                self.assertTrue(torch.isfinite(detection[key]).all())

        # Again in "dynamic size" mode
        images, masks, image_sizes = batch_images(
            [torch.rand((DEFAULT_NUM_CHANNELS, *size)), torch.rand((DEFAULT_NUM_CHANNELS, size[0] - 12, size[1] - 24))],
            size_divisible=4,
        )
        out = n(images, masks=masks, image_sizes=image_sizes)

        # Reset classifier
        n.reset_classifier(20)
        n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))

        n.train()
        out = n(
            torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)),
            targets=[
                {
                    "boxes": torch.tensor([[10.1, 10.1, 30.2, 40.2]]),
                    "labels": torch.tensor([1]),
                }
                for _ in range(batch_size)
            ],
        )
        detections, losses = out
        self.assertGreater(len(losses), 0)
        for loss in losses.values():
            self.assertTrue(torch.isfinite(loss).all())

        loss = sum(v for v in losses.values())
        self.assertEqual(loss.ndim, 0)

        for detection in detections:
            for key in ["boxes", "labels", "scores"]:
                self.assertTrue(torch.isfinite(detection[key]).all())

        # Background-only images must produce a finite, non-zero learning signal
        _, empty_losses = n(
            torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)),
            targets=[
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }
                for _ in range(batch_size)
            ],
        )
        self.assertGreater(len(empty_losses), 0)
        for empty_loss_part in empty_losses.values():
            self.assertTrue(torch.isfinite(empty_loss_part).all())

        empty_loss = sum(empty_losses.values())
        self.assertEqual(empty_loss.ndim, 0)
        self.assertTrue(empty_loss.requires_grad)
        self.assertGreater(empty_loss.detach().item(), 0.0)

        if n.scriptable is True:
            torch.jit.script(n)
        else:
            n.eval()
            torch.jit.trace(n, example_inputs=torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            n.train()

        # Freeze
        n.eval()
        n.freeze(freeze_classifier=False)
        n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))

        # Reparameterize
        if reparameterize_available(n) is True:
            n.reparameterize_model()
            detections, losses = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            self.assertEqual(len(losses), 0)
            for detection in detections:
                for key in ["boxes", "labels", "scores"]:
                    self.assertTrue(torch.isfinite(detection[key]).all())

    @parameterized.expand(NET_DETECTION_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_net_detection_meta(
        self,
        network_name: str,
        encoder: str,
        size: tuple[int, int] = (256, 256),
        _batch_size: int = 1,
    ) -> None:
        with torch.device("meta"):
            backbone = registry.net_factory(encoder, 10, size=size)
            meta_net = registry.detection_net_factory(network_name, 10, backbone, size=size)

        non_meta_tensors = [
            f"parameter '{name}': {parameter.device}"
            for name, parameter in meta_net.named_parameters()
            if parameter.is_meta is False
        ]
        non_meta_tensors.extend(
            f"buffer '{name}': {buffer.device}" for name, buffer in meta_net.named_buffers() if buffer.is_meta is False
        )
        self.assertListEqual(non_meta_tensors, [])

    @parameterized.expand(NET_DETECTION_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_net_detection_backward(
        self,
        network_name: str,
        encoder: str,
        size: tuple[int, int] = (256, 256),
        batch_size: int = 1,
        size_step: int = 2**5,
    ) -> None:
        backbone = registry.net_factory(encoder, 10, size=size)
        n = registry.detection_net_factory(network_name, 10, backbone, size=size)

        size = (size[0] + size_step, size[1] + size_step)
        n.adjust_size(size)
        for name, param in n.named_parameters():
            self.assertIsNone(param.grad, msg=f"{network_name} adjust_size set grad for {name}")
            self.assertIsNone(param.grad_fn, msg=f"{network_name} adjust_size tracked grad for {name}")

        n.train()
        _, losses = n(
            torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)),
            targets=[
                {
                    "boxes": torch.tensor([[10.1, 10.1, 30.2, 40.2]]),
                    "labels": torch.tensor([1]),
                }
                for _ in range(batch_size)
            ],
        )
        self.assertGreater(len(losses), 0)
        loss = sum(v for v in losses.values())
        self.assertEqual(loss.ndim, 0)
        loss.backward()
        for name, param in n.named_parameters():
            if param.requires_grad is False:
                continue

            self.assertIsNotNone(param.grad, msg=f"{network_name} missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all().item(), msg=f"{network_name} non-finite grad for {name}")

        n.zero_grad()

        # A batch containing no objects
        _, empty_losses = n(
            torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)),
            targets=[
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }
                for _ in range(batch_size)
            ],
        )
        empty_loss = sum(empty_losses.values())
        empty_loss.backward()
        empty_gradients = [
            param.grad for param in n.parameters() if param.requires_grad is True and param.grad is not None
        ]
        self.assertGreater(len(empty_gradients), 0)
        self.assertTrue(all(torch.isfinite(grad).all().item() for grad in empty_gradients))
        self.assertTrue(any(torch.count_nonzero(grad).item() > 0 for grad in empty_gradients))
        n.zero_grad()

        if reparameterize_available(n) is True:
            n.eval()
            n.reparameterize_model()
            n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            for name, param in n.named_parameters():
                self.assertIsNone(param.grad, msg=f"{network_name} reparameterize_model set grad for {name}")
                self.assertIsNone(param.grad_fn, msg=f"{network_name} reparameterize_model tracked grad for {name}")

    @parameterized.expand(NET_DETECTION_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_net_detection_pt2(
        self,
        network_name: str,
        encoder: str,
        size: tuple[int, int] = (256, 256),
        _batch_size: int = 1,
    ) -> None:
        backbone = registry.net_factory(encoder, 10, size=size)
        n = registry.detection_net_factory(network_name, 10, backbone, size=size, export_mode=True)
        n.eval()

        if n.exportable is True:
            # Test PT2
            with torch.no_grad():
                torch.export.export(n, (torch.randn(1, DEFAULT_NUM_CHANNELS, *size),))

    @parameterized.expand(DETECTION_DYNAMIC_SIZE_CASES)  # type: ignore[untyped-decorator]
    def test_detection_dynamic_size(
        self,
        network_name: str,
        encoder: str,
        size: tuple[int, int] = (256, 256),
        batch_size: int = 1,
    ) -> None:
        backbone = registry.net_factory(encoder, 10, size=size)
        n = registry.detection_net_factory(network_name, 10, backbone, size=size)
        n.eval()
        n.set_dynamic_size()

        detections, losses = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(len(losses), 0)
        for detection in detections:
            for key in ["boxes", "labels", "scores"]:
                self.assertTrue(torch.isfinite(detection[key]).all())

        size = (size[0] + 32, size[1] + 64)
        detections, losses = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(len(losses), 0)
        for detection in detections:
            for key in ["boxes", "labels", "scores"]:
                self.assertTrue(torch.isfinite(detection[key]).all())

    @parameterized.expand(DETECTION_DYNAMIC_SIZE_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_detection_dynamic_size_backward(
        self,
        network_name: str,
        encoder: str,
        size: tuple[int, int] = (256, 256),
        batch_size: int = 1,
        size_step: int = 2**5,
    ) -> None:
        backbone = registry.net_factory(encoder, 10, size=size)
        n = registry.detection_net_factory(network_name, 10, backbone, size=size)
        n.train()
        n.set_dynamic_size()

        size = (size[0] + size_step, size[1] + size_step)
        _, losses = n(
            torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)),
            targets=[
                {
                    "boxes": torch.tensor([[10.1, 10.1, 30.2, 40.2]]),
                    "labels": torch.tensor([1]),
                }
                for _ in range(batch_size)
            ],
        )
        self.assertGreater(len(losses), 0)
        loss = sum(v for v in losses.values())
        loss.backward()
        for name, param in n.named_parameters():
            if param.requires_grad is False:
                continue

            self.assertIsNotNone(param.grad, msg=f"{network_name} missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all().item(), msg=f"{network_name} non-finite grad for {name}")

    # @parameterized.expand(DETECTION_DYNAMIC_SIZE_CASES)  # type: ignore[untyped-decorator]
    # @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    # def test_detection_dynamic_size_pt2(
    #     self,
    #     network_name: str,
    #     encoder: str,
    #     size: tuple[int, int] = (256, 256),
    # ) -> None:
    #     backbone = registry.net_factory(encoder, 10, size=size)
    #     n = registry.detection_net_factory(network_name, 10, backbone, size=size, export_mode=True)
    #     n.eval()
    #     n.set_dynamic_size()

    #     if n.exportable is True:
    #         # Test PT2
    #         height_dim = torch.export.Dim.DYNAMIC
    #         width_dim = torch.export.Dim.DYNAMIC
    #         with torch.no_grad():
    #             torch.export.export(
    #                 n,
    #                 (torch.randn(1, DEFAULT_NUM_CHANNELS, *size),),
    #                 dynamic_shapes={"x": {2: height_dim, 3: width_dim}},
    #             )

    def test_faster_rcnn_anchor_sizes(self) -> None:
        size = (256, 256)

        backbone = registry.net_factory("efficientvit_msft_m0", 10, size=size)
        n = registry.detection_net_factory("faster_rcnn", 10, backbone, size=size)
        self.assertEqual(n.rpn.anchor_generator.sizes, [[s] for s in [128, 256, 512, 1024]])

        backbone = registry.net_factory("efficientvit_msft_m0", 10, size=size)
        delattr(backbone, "stem_stride")
        n = registry.detection_net_factory("faster_rcnn", 10, backbone, size=size)
        self.assertEqual(n.rpn.anchor_generator.sizes, [[s] for s in [128, 256, 512, 1024]])

        backbone = registry.net_factory("xcit_nano12_p16", 10, size=size)
        n = registry.detection_net_factory("faster_rcnn", 10, backbone, size=size)
        self.assertEqual(n.rpn.anchor_generator.sizes, [[s] for s in [32, 64, 128, 256, 512]])

    def test_fcos_infers_anchor_sizes_from_max_stride(self) -> None:
        size = (256, 256)

        backbone = registry.net_factory("resnet_d_50", 10, size=size)
        n = registry.detection_net_factory("fcos", 10, backbone, size=size, export_mode=True)
        self.assertEqual(n.anchor_generator.sizes, [[s] for s in [8, 16, 32, 64, 128]])

        backbone = registry.net_factory("vit_s16", 10, size=size)
        n = registry.detection_net_factory("fcos", 10, backbone, size=size, export_mode=True)
        self.assertEqual(n.anchor_generator.sizes, [[s] for s in [16, 32, 64]])

        backbone = registry.net_factory("vit_s32", 10, size=size)
        n = registry.detection_net_factory("fcos", 10, backbone, size=size, export_mode=True)
        self.assertEqual(n.anchor_generator.sizes, [[s] for s in [32, 64, 128]])
