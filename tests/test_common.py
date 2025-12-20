# pylint: disable=protected-access

import argparse
import logging
import math
import os
import tempfile
import typing
import unittest
from collections import OrderedDict
from unittest.mock import mock_open
from unittest.mock import patch

import torch

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.common import masking
from birder.common import training_cli
from birder.common import training_utils
from birder.conf import settings
from birder.net.base import SignatureType
from birder.net.detection.base import DetectionSignatureType
from birder.net.resnext import ResNeXt
from birder.net.vit import ViT
from birder.scheduler.cooldown import CooldownLR

logging.disable(logging.CRITICAL)


class TestLib(unittest.TestCase):
    def test_lib(self) -> None:
        # Signature components
        signature: SignatureType = {
            "inputs": [{"data_shape": [0, 3, 224, 224]}],
            "outputs": [{"data_shape": [0, 371]}],
        }
        detection_signature: DetectionSignatureType = {
            "dynamic": False,
            "inputs": [{"data_shape": [0, 3, 640, 640]}],
            "outputs": ([{"boxes": [0, 4], "labels": [0], "scores": [0]}], {}),
            "num_labels": 91,
        }
        self.assertEqual(lib.get_size_from_signature(signature), (224, 224))
        self.assertEqual(lib.get_size_from_signature(detection_signature), (640, 640))
        self.assertEqual(lib.get_channels_from_signature(signature), 3)
        self.assertEqual(lib.get_channels_from_signature(detection_signature), 3)
        self.assertEqual(lib.get_num_labels_from_signature(signature), 371)
        self.assertEqual(lib.get_num_labels_from_signature(detection_signature), 91)

        # Network name
        net_name = lib.get_network_name("net")
        self.assertEqual(net_name, "net")

        net_name = lib.get_network_name("net_1_25")
        self.assertEqual(net_name, "net_1_25")

        net_name = lib.get_network_name("net", tag="exp")
        self.assertEqual(net_name, "net_exp")

        net_name = lib.get_network_name("net", tag="exp")
        self.assertEqual(net_name, "net_exp")

        # MIM network name
        net_name = lib.get_mim_network_name("net", encoder="encoder", tag="exp")
        self.assertEqual(net_name, "net_encoder_exp")

        # Detection network name
        net_name = lib.get_detection_network_name("net", backbone="back", tag="exp", backbone_tag=None)
        self.assertEqual(net_name, "net_exp_back")

        # Label from path
        label = lib.get_label_from_path("data/validation/Barn owl/000001.jpeg")
        self.assertEqual(label, "Barn owl")

        label = lib.get_label_from_path(
            "data/validation/Aves/Barn owl/000001.jpeg", hierarchical=True, root="data/validation"
        )
        self.assertEqual(label, "Aves_Barn owl")

        # Detection class to index (background index)
        detection_class_to_index = lib.detection_class_to_idx({"first": 0, "second": 1})
        self.assertEqual(detection_class_to_index["first"], 1)
        self.assertEqual(detection_class_to_index["second"], 2)


class TestCLI(unittest.TestCase):
    def test_cli(self) -> None:
        m = mock_open(read_data=b"test data")
        with patch("builtins.open", m):
            hex_digest = cli.calc_sha256("some_file.tar.gz")
            m.assert_called_with("some_file.tar.gz", "rb")
            self.assertEqual(hex_digest, "916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9")

    @unittest.skipUnless(os.environ.get("NETWORK_TESTS", False), "Avoid tests that require network access")
    def test_download_file(self) -> None:
        with tempfile.NamedTemporaryFile() as f:
            cli.download_file(
                "wget https://huggingface.co/spaces/birder-project/README/resolve/main/img_001.jpeg", f.name
            )


class TestFSOps(unittest.TestCase):
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

        path = fs_ops.model_path("net", st=True)
        self.assertEqual(path, settings.MODELS_DIR.joinpath("net.safetensors"))

        path = fs_ops.model_path("net", epoch=17)
        self.assertEqual(path, settings.MODELS_DIR.joinpath("net_17.pt"))


class TestTrainingUtils(unittest.TestCase):
    def test_misc(self) -> None:
        self.assertFalse(training_utils.is_dist_available_and_initialized())
        self.assertRegex(training_utils.training_log_name("something", torch.device("cpu")), "something__")

        # Logging context manager
        logger = logging.getLogger("birder")
        new_handler = logging.NullHandler()
        logger.addHandler(new_handler)
        last_handler = logging.NullHandler()
        logger.addHandler(last_handler)
        num_handlers = len(logger.handlers)
        original_handlers = list(logger.handlers)

        # Known handler
        with training_utils.single_handler_logging(logger, new_handler) as log:
            self.assertEqual(len(log.handlers), 1)
            log.debug("Sanity check, will not be shown")

        self.assertEqual(len(logger.handlers), num_handlers)
        self.assertEqual(original_handlers, logger.handlers)  # Verify handler order

        # Unknown handler
        unknown_handler = logging.NullHandler()
        with training_utils.single_handler_logging(logger, unknown_handler) as log:
            self.assertEqual(len(log.handlers), 1)
            log.debug("Sanity check, will not be shown")

        self.assertEqual(len(logger.handlers), num_handlers)
        self.assertEqual(original_handlers, logger.handlers)  # Verify handler order

        # Disabled
        with training_utils.single_handler_logging(logger, new_handler, enabled=False) as log:
            self.assertEqual(len(log.handlers), num_handlers)
            log.debug("Sanity check, will not be shown")

        self.assertEqual(len(logger.handlers), num_handlers)
        self.assertEqual(original_handlers, logger.handlers)  # Verify handler order

    def test_group_by_regex(self) -> None:
        layer_names = [
            "pos_embed",
            "pos_embed_win",
            "stem.proj.weight",
            "stem.proj.bias",
            "body.stage1.0.norm1.weight",
            "body.stage1.0.norm1.bias",
            "body.stage1.0.attn.qkv.weight",
            "body.stage1.0.attn.qkv.bias",
            "body.stage1.0.mlp.3.weight",
            "body.stage1.0.mlp.3.bias",
            "body.stage2.0.norm1.weight",
            "body.stage2.0.mlp.3.weight",
            "body.stage2.0.mlp.3.bias",
            "body.stage2.1.norm1.weight",
            "body.stage2.1.norm1.bias",
            "body.stage2.1.attn.qkv.weight",
            "body.stage2.1.attn.qkv.bias",
            "body.stage2.1.attn.proj.weight",
            "anything",
        ]
        pattern = r"body\.stage(\d+)\.(\d+)"
        groups = training_utils.group_by_regex(layer_names, pattern)

        # First group should be the stem
        self.assertEqual(len(groups[0]), 4)
        self.assertEqual(groups[0][-1], "stem.proj.bias")

        # Second should only include the stage1 blocks
        self.assertEqual(len(groups[1]), 6)
        self.assertEqual(groups[1][0], "body.stage1.0.norm1.weight")
        self.assertEqual(groups[1][-1], "body.stage1.0.mlp.3.bias")

        # 3rd is stage2 seq 0
        self.assertEqual(len(groups[2]), 3)
        self.assertEqual(groups[2][0], "body.stage2.0.norm1.weight")
        self.assertEqual(groups[2][-1], "body.stage2.0.mlp.3.bias")

        # 4th is stage2 seq 1
        self.assertEqual(len(groups[3]), 5)
        self.assertEqual(groups[3][0], "body.stage2.1.norm1.weight")
        self.assertEqual(groups[3][-1], "body.stage2.1.attn.proj.weight")

        # Last group, catch all of the rest
        self.assertEqual(len(groups[-1]), 1)
        self.assertEqual(groups[-1][0], "anything")

    def test_ra_sampler(self) -> None:
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

    def test_optimizer_parameter_groups(self) -> None:
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

        model = ResNeXt(3, 2, config={"units": [3, 4, 6, 3]})
        params = training_utils.optimizer_parameter_groups(model, 0, layer_decay=0.1)
        self.assertEqual(params[-1]["lr_scale"], 1.0)
        self.assertEqual(params[-2]["lr_scale"], 1.0)
        self.assertEqual(params[-3]["lr_scale"], 0.1)

        model = ViT(
            3,
            2,
            config={
                "patch_size": 32,
                "num_layers": 12,
                "num_heads": 8,
                "hidden_dim": 128,
                "mlp_dim": 512,
                "num_reg_tokens": 0,
                "drop_path_rate": 0.0,
            },
        )
        params = training_utils.optimizer_parameter_groups(model, 0, layer_decay=0.1)
        for param in params[-4:]:  # Head + norm
            self.assertEqual(param["lr_scale"], 1.0)
        for param in params[-16:-4]:  # Block 12
            self.assertAlmostEqual(param["lr_scale"], 0.1)
        for param in params[-28:-16]:  # Block 12
            self.assertAlmostEqual(param["lr_scale"], 0.01)
        for param in params[:4]:  # CLS token, positional encoding and conv_proj
            self.assertAlmostEqual(param["lr_scale"], 1e-13)

        # Test layer decay with custom options
        model = ViT(
            3,
            2,
            config={
                "patch_size": 32,
                "num_layers": 12,
                "num_heads": 8,
                "hidden_dim": 128,
                "mlp_dim": 512,
                "num_reg_tokens": 0,
                "drop_path_rate": 0.0,
            },
        )
        params = training_utils.optimizer_parameter_groups(model, 0, layer_decay=0.1, layer_decay_min_scale=0.01)
        for param in params[-4:]:  # Head + norm
            self.assertEqual(param["lr_scale"], 1.0)
        for param in params[-16:-4]:  # Block 12
            self.assertAlmostEqual(param["lr_scale"], 0.1)
        for param in params[-28:-16]:  # Block 12
            self.assertAlmostEqual(param["lr_scale"], 0.01)
        for param in params[-40:-28]:  # Block 11
            self.assertAlmostEqual(param["lr_scale"], 0.01)
        for param in params[:4]:  # CLS token, positional encoding and conv_proj
            self.assertAlmostEqual(param["lr_scale"], 0.01)

        model = ViT(
            3,
            2,
            config={
                "patch_size": 32,
                "num_layers": 12,
                "num_heads": 8,
                "hidden_dim": 128,
                "mlp_dim": 512,
                "num_reg_tokens": 0,
                "drop_path_rate": 0.0,
            },
        )
        params = training_utils.optimizer_parameter_groups(model, 0, layer_decay=0.1, layer_decay_no_opt_scale=1e-4)
        for param in params[-4:]:  # Head + norm
            self.assertTrue(param["params"].requires_grad)
        for param in params[-16:-4]:  # Block 12
            self.assertTrue(param["params"].requires_grad)
        for param in params[:4]:  # CLS token, positional encoding and conv_proj
            self.assertFalse(param["params"].requires_grad)

        # Test backbone
        model = torch.nn.Sequential(
            OrderedDict(
                {
                    "backbone": torch.nn.Sequential(
                        OrderedDict(
                            {
                                "linear": torch.nn.Linear(1, 2, bias=True),
                                "norm": torch.nn.BatchNorm1d(2),
                            }
                        )
                    ),
                    "classifier": torch.nn.Linear(2, 1, bias=False),
                }
            )
        )
        params = training_utils.optimizer_parameter_groups(model, 0, backbone_lr=0.1)
        for param in params[:4]:  # Linear + norm
            self.assertEqual(param["lr"], 0.1)
        for param in params[4:]:
            self.assertNotIn("lr", param)

        # Test bias
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 2, bias=True),
            torch.nn.BatchNorm1d(2),
            torch.nn.Linear(2, 1, bias=False),
        )
        params = training_utils.optimizer_parameter_groups(model, 0, bias_lr=0.01)
        self.assertEqual(params[1]["lr"], 0.01)
        self.assertEqual(params[3]["lr"], 0.01)

    def test_get_optimizer(self) -> None:
        parser = argparse.ArgumentParser()
        training_cli.add_optimization_args(parser)
        training_cli.add_lr_wd_args(parser)
        for opt_type in typing.get_args(training_utils.OptimizerType):
            args = parser.parse_args(["--opt", opt_type])
            opt = training_utils.get_optimizer([{"params": []}], args.lr, args)
            self.assertIsInstance(opt, torch.optim.Optimizer)

        with self.assertRaises(ValueError):
            args = argparse.Namespace(opt="unknown")
            training_utils.get_optimizer([{"params": []}], 0.001, args)

        # Check custom params
        args = argparse.Namespace(opt="adamw", lr=0.1, wd=0.1, opt_eps=1e-6, opt_betas=[0.1, 0.2])
        opt = training_utils.get_optimizer([{"params": []}], args.lr, args)
        self.assertEqual(opt.defaults["eps"], 1e-6)
        self.assertEqual(opt.defaults["betas"], [0.1, 0.2])

    def test_get_scheduler(self) -> None:
        args = argparse.Namespace(opt="sgd", lr=0.1, momentum=0.9, wd=0, nesterov=False)
        opt = training_utils.get_optimizer([{"params": []}], args.lr, args)

        parser = argparse.ArgumentParser()
        training_cli.add_training_schedule_args(parser)
        training_cli.add_lr_scheduler_args(parser)
        training_cli.add_checkpoint_args(parser)
        for scheduler_type in typing.get_args(training_utils.SchedulerType):
            args = parser.parse_args(["--lr-scheduler", scheduler_type])
            scheduler = training_utils.get_scheduler(opt, 1, args)
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LRScheduler)

        # Unknown scheduler
        args = argparse.Namespace(
            lr_scheduler="unknown",
            warmup_epochs=5,
            cooldown_epochs=0,
            resume_epoch=0,
            epochs=10,
            lr_cosine_min=0.0,
            lr_step_size=1,
            lr_steps=[],
            lr_step_gamma=0.0,
            lr_power=1.0,
        )
        with self.assertRaises(ValueError):
            training_utils.get_scheduler(opt, 1, args)

        # Resume during warmup with all phases
        # iters_per_epoch = 10 (iter mode)
        # epochs = 20 => total_steps = 200
        # warmup_epochs = 5 => warmup_steps = 50
        # cooldown_epochs = 5 => cooldown_steps = 50
        # resume_epoch = 3 => begin_step = 3 * 10 = 30
        # remaining_warmup = warmup_steps - begin_step = 20
        # remaining_cooldown = 50
        # main_steps = 200 - 30 - 20 - 50 - 1 = 99
        # milestones:
        #   [remaining_warmup, remaining_warmup + main_steps + 1]
        #   [20, 20 + 99 + 1] = [20, 120]
        args_resume_warmup = argparse.Namespace(
            lr_scheduler="cosine",
            warmup_epochs=5,
            cooldown_epochs=5,
            resume_epoch=3,
            epochs=20,
            lr_cosine_min=1e-6,
            lr_step_size=1,
            lr_steps=[],
            lr_step_gamma=0.0,
            lr_power=1.0,
            lr_warmup_decay=0.1,
        )
        scheduler = training_utils.get_scheduler(opt, iters_per_epoch=10, args=args_resume_warmup)

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        self.assertEqual(len(scheduler._schedulers), 3)  # Expect Warmup, Main, Cooldown

        # Verify types of sub-schedulers
        self.assertIsInstance(scheduler._schedulers[0], torch.optim.lr_scheduler.LinearLR)
        self.assertIsInstance(scheduler._schedulers[1], torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertIsInstance(scheduler._schedulers[2], CooldownLR)

        # Verify lengths/parameters of sub-schedulers
        self.assertEqual(scheduler._schedulers[0].total_iters, 20)
        self.assertEqual(scheduler._schedulers[1].T_max, 99)
        self.assertEqual(scheduler._schedulers[2].total_steps, 50)

        # Verify milestones
        self.assertEqual(scheduler._milestones, [20, 120])

        # Resume after warmup with all phases
        # iters_per_epoch = 1 (epoch mode)
        # epochs = 20
        # warmup_epochs = 5
        # cooldown_epochs = 5
        # resume_epoch = 7
        # remaining_warmup = 0
        # remaining_cooldown = 5
        # main_steps = 20 - 7 - 0 - 5 - 1 = 7
        # milestones:
        #   [remaining_warmup, remaining_warmup + main_steps + 1]
        #   [0, 0 + 7 + 1] = [0, 8]
        args_resume_warmup = argparse.Namespace(
            lr_scheduler="cosine",
            warmup_epochs=5,
            cooldown_epochs=5,
            resume_epoch=7,
            epochs=20,
            lr_cosine_min=1e-6,
            lr_step_size=1,
            lr_steps=[],
            lr_step_gamma=0.0,
            lr_power=1.0,
            lr_warmup_decay=0.1,
        )
        scheduler = training_utils.get_scheduler(opt, iters_per_epoch=1, args=args_resume_warmup)

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        self.assertEqual(len(scheduler._schedulers), 3)  # Expect Warmup, Main, Cooldown

        # Verify types of sub-schedulers
        self.assertIsInstance(scheduler._schedulers[0], torch.optim.lr_scheduler.LinearLR)
        self.assertIsInstance(scheduler._schedulers[1], torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertIsInstance(scheduler._schedulers[2], CooldownLR)

        # Verify lengths/parameters of sub-schedulers
        self.assertEqual(scheduler._schedulers[0].total_iters, 0)
        self.assertEqual(scheduler._schedulers[1].T_max, 7)
        self.assertEqual(scheduler._schedulers[2].total_steps, 5)

        # Verify milestones
        self.assertEqual(scheduler._milestones, [0, 8])

        # Resume during cooldown with all phases
        # iters_per_epoch = 10 (iter mode)
        # epochs = 20 => total_steps = 200
        # warmup_epochs = 5 => warmup_steps = 50
        # cooldown_epochs = 5 => cooldown_steps = 50
        # resume_epoch = 18 => begin_step = 18 * 10 = 180
        # remaining_warmup = warmup_steps - begin_step = 0
        # remaining_cooldown = 20
        # main_steps = -1
        # milestones:
        #   [remaining_warmup, remaining_warmup + main_steps + 1]
        #   [0, 0 + -1 + 1] = [0, 0]
        args_resume_warmup = argparse.Namespace(
            lr_scheduler="cosine",
            warmup_epochs=5,
            cooldown_epochs=5,
            resume_epoch=18,
            epochs=20,
            lr_cosine_min=1e-6,
            lr_step_size=1,
            lr_steps=[],
            lr_step_gamma=0.0,
            lr_power=1.0,
            lr_warmup_decay=0.1,
        )
        scheduler = training_utils.get_scheduler(opt, iters_per_epoch=10, args=args_resume_warmup)

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        self.assertEqual(len(scheduler._schedulers), 3)  # Expect Warmup, Main, Cooldown

        # Verify types of sub-schedulers
        self.assertIsInstance(scheduler._schedulers[0], torch.optim.lr_scheduler.LinearLR)
        self.assertIsInstance(scheduler._schedulers[1], torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertIsInstance(scheduler._schedulers[2], CooldownLR)

        # Verify lengths/parameters of sub-schedulers
        self.assertEqual(scheduler._schedulers[0].total_iters, 0)
        self.assertEqual(scheduler._schedulers[1].T_max, -1)
        self.assertEqual(scheduler._schedulers[2].total_steps, 20)

        # Verify milestones
        self.assertEqual(scheduler._milestones, [0, 0])

    def test_lr_scaling(self) -> None:
        args = argparse.Namespace(
            lr=0.1, batch_size=128, grad_accum_steps=1, world_size=1, lr_scale=64, lr_scale_type="linear"
        )
        lr = training_utils.scale_lr(args)
        self.assertEqual(lr, 0.2)

        args = argparse.Namespace(
            lr=0.1, batch_size=128, grad_accum_steps=2, world_size=2, lr_scale=64, lr_scale_type="linear"
        )
        lr = training_utils.scale_lr(args)
        self.assertEqual(lr, 0.8)

        args = argparse.Namespace(
            lr=0.1, batch_size=128, grad_accum_steps=2, world_size=2, lr_scale=32, lr_scale_type="sqrt"
        )
        lr = training_utils.scale_lr(args)
        self.assertEqual(lr, 0.4)

    def test_smoothed_value(self) -> None:
        smoothed_value = training_utils.SmoothedValue(window_size=4)
        smoothed_value.update(1)
        self.assertEqual(smoothed_value.value, 1)
        self.assertEqual(smoothed_value.count, 1)
        self.assertEqual(smoothed_value.total, 1)

        smoothed_value.update(2)
        self.assertEqual(smoothed_value.value, 2)
        self.assertEqual(smoothed_value.count, 2)
        self.assertEqual(smoothed_value.total, 3)

        smoothed_value.update(3)
        smoothed_value.update(4)
        self.assertEqual(smoothed_value.value, 4)
        self.assertEqual(smoothed_value.count, 4)
        self.assertEqual(smoothed_value.total, 10)
        self.assertEqual(smoothed_value.avg, 2.5)

        smoothed_value.update(5)
        self.assertEqual(smoothed_value.value, 5)
        self.assertEqual(smoothed_value.count, 5)
        self.assertEqual(smoothed_value.total, 15)
        self.assertEqual(smoothed_value.avg, 3.5)
        self.assertEqual(smoothed_value.global_avg, 3)

        smoothed_value.synchronize_between_processes(torch.device("cpu"))

    def test_accuracy(self) -> None:
        y_true = torch.tensor([0, 1, 2, 0])
        y_pred = torch.tensor([[0.9, 0.1, 0.0], [0.8, 0.1, 0.1], [0.0, 0.1, 0.9], [0.1, 0.4, 0.5]])
        self.assertAlmostEqual(training_utils.accuracy(y_true, y_pred), 2 / 4, places=6)

    def test_get_grad_norm(self) -> None:
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

    def test_cosine_scheduler(self) -> None:
        # Sanity check
        schedule = training_utils.cosine_scheduler(
            base_value=1.0, final_value=0.1, epochs=10, warmup_epochs=2.0, iter_per_epoch=5
        )
        self.assertEqual(len(schedule), 50)  # 10 epochs * 5 iter/epoch

        warmup_values = schedule[:10]
        self.assertAlmostEqual(warmup_values[0], 0.0)
        self.assertAlmostEqual(warmup_values[-1], 0.9)

        self.assertAlmostEqual(schedule[10], 1.0)
        self.assertAlmostEqual(schedule[-1], 0.1, places=2)

        # Test fractional warmup epochs
        schedule = training_utils.cosine_scheduler(
            base_value=1.0, final_value=0.0, epochs=2, warmup_epochs=0.5, iter_per_epoch=10
        )
        self.assertEqual(len(schedule), 20)

        warmup_values = schedule[:5]
        self.assertAlmostEqual(warmup_values[0], 0.0)
        self.assertTrue(warmup_values[-1] < 1.0)

        # Test schedule with zero warmup
        schedule = training_utils.cosine_scheduler(
            base_value=1.0, final_value=0.0, epochs=5, warmup_epochs=0.0, iter_per_epoch=10
        )
        self.assertEqual(len(schedule), 50)
        self.assertAlmostEqual(schedule[0], 1.0)
        self.assertAlmostEqual(schedule[-1], 0.0, places=2)

    def test_freeze_batchnorm2d(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 2, bias=True),
            torch.nn.BatchNorm1d(2),
            torch.nn.Linear(2, 1, bias=False),
        )
        model = training_utils.freeze_batchnorm2d(model)
        self.assertIsInstance(model[1], torch.nn.BatchNorm1d)  # 1d batchnorm should not change

        model = ResNeXt(3, 2, config={"units": [3, 4, 6, 3]})
        model = training_utils.freeze_batchnorm2d(model)
        for m in model.modules():
            self.assertNotIsInstance(m, torch.nn.BatchNorm2d)

    def test_replace_module(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 2, bias=True),
            torch.nn.BatchNorm1d(2),
            torch.nn.Sequential(
                torch.nn.Linear(1, 2, bias=True),
                torch.nn.ReLU(),
            ),
            torch.nn.Linear(2, 1, bias=False),
        )
        model = training_utils.replace_module(model, torch.nn.Linear, torch.nn.GELU)
        gelu_count = 0
        for m in model.modules():
            self.assertNotIsInstance(m, torch.nn.Linear)
            if isinstance(m, torch.nn.GELU):
                gelu_count += 1

        self.assertEqual(gelu_count, 3)


class TestMasking(unittest.TestCase):
    def test_mask_token_omission(self) -> None:
        x = torch.arange(1, 65)
        x = x.reshape(1, -1, 1).expand(2, -1, 80)

        (N, L, D) = x.size()  # batch, length, dim
        h = int(math.sqrt(L))
        w = h
        (mask, ids_keep, ids_restore) = masking.uniform_mask(N, h, w, mask_ratio=0.75, device=x.device)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Test x masked
        self.assertEqual(x_masked.size(0), x.size(0))
        self.assertEqual(x_masked.size(1), 16)
        self.assertEqual(x_masked.size(2), x.size(2))

        # Test mask
        self.assertEqual(mask.ndim, 2)
        self.assertEqual(mask.size(0), x.size(0))
        self.assertEqual(mask.size(1), x.size(1))
        self.assertEqual((mask == 0).sum().item(), x.size(0) * 16)
        self.assertNotEqual(mask.dtype, torch.bool)

        # Test ids
        self.assertEqual(ids_keep.size(0), x.size(0))
        self.assertEqual(ids_keep.size(1), 16)
        self.assertEqual(ids_restore.size(0), x.size(0))
        self.assertEqual(ids_restore.size(1), x.size(1))

    def test_mask_tensor(self) -> None:
        x = torch.rand(2, 8, 8, 80)
        mask = masking.uniform_mask(2, 4, 4, mask_ratio=0.5, device=x.device)[0]
        x_masked = masking.mask_tensor(x, mask, channels_last=True, patch_factor=2)

        # Test x masked
        self.assertEqual(x_masked.size(), x.size())

        # Test mask
        expected_size = x.size(1) // 2 * x.size(2) // 2
        self.assertEqual(mask.ndim, 2)
        self.assertEqual(mask.size(0), x.size(0))
        self.assertEqual(mask.size(1), expected_size)
        self.assertEqual((mask == 0).sum().item(), x.size(0) * expected_size // 2)
        self.assertNotEqual(mask.dtype, torch.bool)

        # Test mask token
        mask_token = torch.zeros(1, 1, 1, 80) * 2
        mask = masking.uniform_mask(2, 8, 8, mask_ratio=1.0, device=x.device)[0]
        x_masked = masking.mask_tensor(x, mask, channels_last=True, mask_token=mask_token)
        self.assertEqual(x_masked.size(), x.size())

        x_masked = x_masked.reshape(-1, 80)
        mask_token = mask_token.squeeze()
        for masked_token in x_masked:
            torch.testing.assert_close(masked_token, mask_token)

    def test_block_masking(self) -> None:
        generator = masking.BlockMasking((8, 8), 0, 0, 0.66, 1.5)
        mask = generator(1)
        self.assertEqual((mask == 0).sum().item(), 64)

        mask = generator(4)
        self.assertEqual((mask == 0).sum().item(), 4 * 64)

        generator = masking.BlockMasking((8, 8), 0, 32, 0.66, 1.5)
        mask = generator(1)
        self.assertGreaterEqual((mask == 0).sum().item(), 32)

    def test_roll_block_masking(self) -> None:
        generator = masking.RollBlockMasking((8, 8), 64)
        mask = generator(1)
        self.assertEqual((mask == 1).sum().item(), 64)

        mask = generator(4)
        self.assertEqual((mask == 1).sum().item(), 4 * 64)

        generator = masking.RollBlockMasking((8, 8), 32)
        mask = generator(1)
        self.assertGreaterEqual((mask == 0).sum().item(), 32)

        generator = masking.RollBlockMasking((8, 8), 0)
        mask = generator(1)
        self.assertGreaterEqual((mask == 0).sum().item(), 64)

    def test_inverse_roll_block_masking(self) -> None:
        generator = masking.InverseRollBlockMasking((8, 8), 64)
        mask = generator(1)
        self.assertEqual((mask == 1).sum().item(), 64)

        mask = generator(4)
        self.assertEqual((mask == 1).sum().item(), 4 * 64)

        generator = masking.InverseRollBlockMasking((8, 8), 32)
        mask = generator(1)
        self.assertGreaterEqual((mask == 0).sum().item(), 32)

        generator = masking.InverseRollBlockMasking((8, 8), 0)
        mask = generator(1)
        self.assertGreaterEqual((mask == 0).sum().item(), 64)

    def test_uniform_masking(self) -> None:
        generator = masking.UniformMasking((8, 8), mask_ratio=0.25)
        mask = generator(2)
        self.assertEqual((mask == 0).sum().item(), 96)
        self.assertNotEqual(mask.dtype, torch.bool)

        mask = generator(8)
        self.assertEqual((mask == 0).sum().item(), 384)
        self.assertNotEqual(mask.dtype, torch.bool)

    def test_uniform_mask_block(self) -> None:
        batch_size = 2
        h = 8
        w = 8
        min_mask_size = 2
        mask_ratio = 0.75

        (mask, ids_keep, ids_restore) = masking.uniform_mask(
            batch_size, h, w, mask_ratio=mask_ratio, min_mask_size=min_mask_size
        )

        # Test output shapes
        seq_len = h * w
        h_coarse = h // min_mask_size
        w_coarse = w // min_mask_size
        seq_len_coarse = h_coarse * w_coarse
        len_keep_coarse = int(seq_len_coarse * (1 - mask_ratio))
        len_keep = len_keep_coarse * (min_mask_size**2)

        self.assertEqual(mask.shape, (batch_size, seq_len))
        self.assertEqual(ids_keep.shape, (batch_size, len_keep))
        self.assertEqual(ids_restore.shape, (batch_size, seq_len))

        # Test that mask has block structure (2x2 blocks should have same value)
        mask_2d = mask.reshape(batch_size, h, w)
        for b in range(batch_size):
            for i in range(0, h, min_mask_size):
                for j in range(0, w, min_mask_size):
                    block = mask_2d[b, i : i + min_mask_size, j : j + min_mask_size]
                    # All values in the block should be the same
                    self.assertTrue(torch.all(block == block[0, 0]))

        # Test number of kept tokens
        self.assertEqual((mask == 0).sum().item(), batch_size * len_keep)

    def test_get_ids_keep(self) -> None:
        mask = torch.tensor(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ]
        )
        ids_keep = masking.get_ids_keep(mask)
        torch.testing.assert_close(ids_keep, torch.tensor([[0, 3], [1, 2]]))
