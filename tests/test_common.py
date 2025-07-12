import argparse
import logging
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
from birder.common import training_utils
from birder.conf import settings
from birder.net.base import SignatureType
from birder.net.detection.base import DetectionSignatureType
from birder.net.resnext import ResNeXt
from birder.net.vit import ViT

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
        self.assertEqual(net_name, "net_exp_back_3")

        # Label from path
        label = lib.get_label_from_path("data/validation/Barn owl/000001.jpeg")
        self.assertEqual(label, "Barn owl")

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
            cli.download_file("https://f000.backblazeb2.com/file/birder/data/img_001.jpeg", f.name)


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
        for opt_type in typing.get_args(training_utils.OptimizerType):
            args = argparse.Namespace(opt=opt_type, lr=0.1, momentum=0.9, wd=0, nesterov=False)
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

        for scheduler_type in typing.get_args(training_utils.SchedulerType):
            args = argparse.Namespace(
                lr_scheduler=scheduler_type,
                warmup_epochs=0,
                resume_epoch=0,
                epochs=10,
                lr_cosine_min=0.0,
                lr_step_size=1,
                lr_steps=[],
                lr_step_gamma=0.0,
                lr_power=1.0,
            )
            scheduler = training_utils.get_scheduler(opt, 1, args)
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LRScheduler)

        # Check warmup
        args = argparse.Namespace(
            lr_scheduler="step",
            warmup_epochs=5,
            resume_epoch=0,
            epochs=10,
            lr_cosine_min=0.0,
            lr_step_size=1,
            lr_steps=[],
            lr_step_gamma=0.0,
            lr_power=1.0,
        )
        scheduler = training_utils.get_scheduler(opt, 1, args)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.SequentialLR)

        args = argparse.Namespace(
            lr_scheduler="unknown",
            warmup_epochs=5,
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

        # Misc
        self.assertFalse(training_utils.is_dist_available_and_initialized())
        self.assertRegex(training_utils.training_log_name("something", torch.device("cpu")), "something__")

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


class TestMasking(unittest.TestCase):
    def test_mask_token_omission(self) -> None:
        x = torch.arange(1, 65)
        x = x.reshape(1, -1, 1).expand(2, -1, 80)

        (N, L, D) = x.size()  # batch, length, dim
        (mask, ids_keep, ids_restore) = masking.uniform_mask(N, L, mask_ratio=0.75, device=x.device)
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
        mask = masking.uniform_mask(2, 4 * 4, mask_ratio=0.5, device=x.device)[0]
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
        mask = masking.uniform_mask(2, 8 * 8, mask_ratio=1.0, device=x.device)[0]
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

    def test_get_ids_keep(self) -> None:
        mask = torch.tensor(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ]
        )
        ids_keep = masking.get_ids_keep(mask)
        torch.testing.assert_close(ids_keep, torch.tensor([[0, 3], [1, 2]]))
