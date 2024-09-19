import logging
import typing
import unittest
from unittest.mock import mock_open
from unittest.mock import patch

import torch

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.common import training_utils
from birder.conf import settings
from birder.net.resnext import ResNeXt
from birder.net.vit import ViT

logging.disable(logging.CRITICAL)


class TestCommon(unittest.TestCase):
    def test_lib(self) -> None:
        # Network name
        net_name = lib.get_network_name("net", net_param=None)
        self.assertEqual(net_name, "net")

        net_name = lib.get_network_name("net", net_param=1.25)
        self.assertEqual(net_name, "net_1.25")

        net_name = lib.get_network_name("net", net_param=1.25, tag="exp")
        self.assertEqual(net_name, "net_1.25_exp")

        net_name = lib.get_network_name("net", net_param=None, tag="exp")
        self.assertEqual(net_name, "net_exp")

        # MIM network name
        net_name = lib.get_mim_network_name("net", net_param=None, encoder="encoder", encoder_param=3, tag="exp")
        self.assertEqual(net_name, "net_encoder_3_exp")

        # Detection network name
        net_name = lib.get_detection_network_name(
            "net", net_param=None, backbone="back", backbone_param=3, tag="exp", backbone_tag=None
        )
        self.assertEqual(net_name, "net_back_3_exp")

        # Label from path
        label = lib.get_label_from_path("data/validation/Barn owl/000001.jpeg")
        self.assertEqual(label, "Barn owl")

        # Detection class to index (background index)
        detection_class_to_index = lib.detection_class_to_idx({"first": 0, "second": 1})
        self.assertEqual(detection_class_to_index["first"], 1)
        self.assertEqual(detection_class_to_index["second"], 2)

    def test_cli(self) -> None:
        m = mock_open(read_data=b"test data")
        with patch("builtins.open", m):
            hex_digest = cli.calc_sha256("some_file.tar.gz")
            m.assert_called_with("some_file.tar.gz", "rb")
            self.assertEqual(hex_digest, "916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9")

    def test_fs_ops(self) -> None:
        # Test model paths
        path = fs_ops.model_path("net", states=True)
        self.assertEqual(path, settings.MODELS_DIR.joinpath("net_states"))

        path = fs_ops.model_path("net")
        self.assertEqual(path, settings.MODELS_DIR.joinpath("net.pt"))

        path = fs_ops.model_path("net", quantized=True)
        self.assertEqual(path, settings.MODELS_DIR.joinpath("net_quantized.pt"))

        path = fs_ops.model_path("net", pts=True)
        self.assertEqual(path, settings.MODELS_DIR.joinpath("net.pts"))

        path = fs_ops.model_path("net", lite=True)
        self.assertEqual(path, settings.MODELS_DIR.joinpath("net.ptl"))

        path = fs_ops.model_path("net", pt2=True)
        self.assertEqual(path, settings.MODELS_DIR.joinpath("net.pt2"))

        path = fs_ops.model_path("net", epoch=17)
        self.assertEqual(path, settings.MODELS_DIR.joinpath("net_17.pt"))

    def test_training_util(self) -> None:
        # Test RA Sampler
        dataset = list(range(512))
        sampler = training_utils.RASampler(dataset, num_replicas=2, rank=0, shuffle=False, repetitions=1)
        self.assertEqual(len(sampler), 256)  # Each rank gets half the dataset
        sampler = training_utils.RASampler(dataset, num_replicas=2, rank=1, shuffle=False, repetitions=2)
        self.assertEqual(len(sampler), 256)

        sampler = training_utils.RASampler(dataset, num_replicas=2, rank=0, shuffle=False, repetitions=1)
        sample_iterator = iter(sampler)
        self.assertEqual(next(sample_iterator), 0)
        self.assertEqual(next(sample_iterator), 2)

        sampler = training_utils.RASampler(dataset, num_replicas=2, rank=0, shuffle=False, repetitions=2)
        sample_iterator = iter(sampler)
        self.assertEqual(next(sample_iterator), 0)
        self.assertEqual(next(sample_iterator), 1)

        sampler = training_utils.RASampler(dataset, num_replicas=2, rank=0, shuffle=False, repetitions=4)
        sample_iterator = iter(sampler)
        self.assertEqual(next(sample_iterator), 0)
        self.assertEqual(next(sample_iterator), 0)
        self.assertEqual(next(sample_iterator), 1)

        # Sanity check for shuffle
        sampler = training_utils.RASampler(dataset, num_replicas=2, rank=0, shuffle=True, repetitions=4)
        sampler.set_epoch(1)
        sample_iterator = iter(sampler)
        self.assertLessEqual(next(sample_iterator), 512)  # type: ignore

        # Test Optimizer Parameter Groups
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 2, bias=True),
            torch.nn.BatchNorm1d(2),
            torch.nn.Linear(2, 1, bias=False),
        )
        params = training_utils.optimizer_parameter_groups(model, 0.1)
        self.assertEqual(len(params), 5)  # Linear + bias + norm std + norm mean + linear
        self.assertEqual(params[0]["weight_decay"], 0.1)
        self.assertEqual(params[1]["weight_decay"], 0.1)
        self.assertEqual(params[2]["weight_decay"], 0.1)
        self.assertEqual(params[3]["weight_decay"], 0.1)
        self.assertEqual(params[4]["weight_decay"], 0.1)
        self.assertEqual(params[0]["lr_scale"], 1.0)
        self.assertIsInstance(params[0]["params"], torch.Tensor)

        # Test bias
        params = training_utils.optimizer_parameter_groups(model, 0.1, custom_keys_weight_decay=[("bias", 0)])
        self.assertEqual(params[0]["weight_decay"], 0.1)
        self.assertEqual(params[1]["weight_decay"], 0.0)
        self.assertEqual(params[2]["weight_decay"], 0.1)
        self.assertEqual(params[3]["weight_decay"], 0.0)
        self.assertEqual(params[4]["weight_decay"], 0.1)

        # Test norm
        params = training_utils.optimizer_parameter_groups(model, 0.1, norm_weight_decay=0)
        self.assertEqual(params[0]["weight_decay"], 0.1)
        self.assertEqual(params[1]["weight_decay"], 0.1)
        self.assertEqual(params[2]["weight_decay"], 0.0)
        self.assertEqual(params[3]["weight_decay"], 0.0)
        self.assertEqual(params[4]["weight_decay"], 0.1)

        # Test bias and norm
        params = training_utils.optimizer_parameter_groups(
            model, 0.1, norm_weight_decay=0, custom_keys_weight_decay=[("bias", 0)]
        )
        self.assertEqual(params[0]["weight_decay"], 0.1)
        self.assertEqual(params[1]["weight_decay"], 0.0)
        self.assertEqual(params[2]["weight_decay"], 0.0)
        self.assertEqual(params[3]["weight_decay"], 0.0)
        self.assertEqual(params[4]["weight_decay"], 0.1)

        # Test layer decay
        params = training_utils.optimizer_parameter_groups(model, 0, layer_decay=0.1)
        self.assertAlmostEqual(params[0]["lr_scale"], 1e-2)
        self.assertAlmostEqual(params[1]["lr_scale"], 1e-2)
        self.assertEqual(params[2]["lr_scale"], 0.1)
        self.assertEqual(params[3]["lr_scale"], 0.1)
        self.assertEqual(params[4]["lr_scale"], 1.0)

        model = ResNeXt(3, 2, net_param=50)
        params = training_utils.optimizer_parameter_groups(model, 0, layer_decay=0.1)
        self.assertEqual(params[-1]["lr_scale"], 1.0)
        self.assertEqual(params[-2]["lr_scale"], 1.0)
        self.assertEqual(params[-3]["lr_scale"], 0.1)

        model = ViT(3, 2, net_param=0)
        params = training_utils.optimizer_parameter_groups(model, 0, layer_decay=0.1)
        self.assertEqual(params[-1]["lr_scale"], 1.0)
        self.assertEqual(params[-2]["lr_scale"], 1.0)
        self.assertEqual(params[-3]["lr_scale"], 0.1)

        # Get optimizer
        for opt_type in typing.get_args(training_utils.OptimizerType):
            opt = training_utils.get_optimizer(opt_type, [{"params": []}], 0.0, 0.0, 0.0, nesterov=False)
            self.assertIsInstance(opt, torch.optim.Optimizer)

        with self.assertRaises(ValueError):
            training_utils.get_optimizer("unknown", [{"params": []}], 0.0, 0.0, 0.0, nesterov=False)  # type: ignore

        # Get scheduler
        for scheduler_type in typing.get_args(training_utils.SchedulerType):
            scheduler = training_utils.get_scheduler(scheduler_type, opt, 0, 0, 10, 0.0, 0, 0.0)
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LRScheduler)

        # Check warmup
        scheduler = training_utils.get_scheduler("step", opt, 5, 0, 10, 0.0, 0, 0.0)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.SequentialLR)

        with self.assertRaises(ValueError):
            training_utils.get_scheduler("unknown", opt, 0, 0, 10, 0.0, 0, 0.0)  # type: ignore

        # Misc
        self.assertFalse(training_utils.is_dist_available_and_initialized())
        self.assertRegex(training_utils.training_log_name("something", torch.device("cpu")), "something__")

        # Test grad norm
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 2, bias=True),
            torch.nn.BatchNorm1d(2),
            torch.nn.Linear(2, 1, bias=False),
        )
        out: torch.Tensor = model(torch.rand((2, 1)))
        grad_norm = training_utils.get_grad_norm(model.parameters())
        self.assertEqual(grad_norm, 0.0)

        loss = out**2
        loss = loss.sum()
        loss.backward()
        grad_norm = training_utils.get_grad_norm(model.parameters())
        self.assertGreater(grad_norm, 0.0)
