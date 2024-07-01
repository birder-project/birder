import logging
import unittest

import torch

from birder.core import net
from birder.core.net import base

logging.disable(logging.CRITICAL)


class TestBase(unittest.TestCase):
    def test_make_divisible(self) -> None:
        self.assertEqual(base.make_divisible(25, 6), 24)

    def test_get_signature(self) -> None:
        signature = base.get_signature((1, 3, 224, 224), 10)
        self.assertIn("inputs", signature)
        self.assertIn("outputs", signature)


# pylint: disable=too-many-public-methods
class TestNet(unittest.TestCase):
    def test_alexnet(self) -> None:
        size = net.AlexNet.default_size
        n = net.AlexNet(3, 100)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_cait(self) -> None:
        size = net.CaiT.default_size
        n = net.CaiT(3, 100, 0)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Adjust size
        size += 2**5
        n.adjust_size(size)
        n(torch.rand((1, 3, size, size)))
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)

    def test_convnext_v1(self) -> None:
        size = net.ConvNeXt_v1.default_size
        n = net.ConvNeXt_v1(3, 100, 0)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_convnext_v2(self) -> None:
        size = net.ConvNeXt_v2.default_size
        n = net.ConvNeXt_v2(3, 100, 0)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Test pre-training encoder
        n.masked_encoding(torch.rand((1, 3, size, size)), mask_ratio=0.75)

    def test_deit(self) -> None:
        size = net.DeiT.default_size
        n = net.DeiT(3, 100, 0)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Adjust size
        size += 2**5
        n.adjust_size(size)
        n(torch.rand((1, 3, size, size)))
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)

    def test_deit3(self) -> None:
        size = net.DeiT3.default_size
        n = net.DeiT3(3, 100, 0)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Adjust size
        size += 2**5
        n.adjust_size(size)
        n(torch.rand((1, 3, size, size)))
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)

    def test_densenet(self) -> None:
        size = net.DenseNet.default_size
        n = net.DenseNet(3, 100, 121)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_edgevit(self) -> None:
        size = net.EdgeViT.default_size
        n = net.EdgeViT(3, 100, 2, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_efficientnet_v1(self) -> None:
        size = net.EfficientNet_v1.default_size
        n = net.EfficientNet_v1(3, 100, 1)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Test detection backbone
        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((1, 3, size, size)))
        prev_latent = 0
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertLessEqual(prev_latent, out[stage_name].shape[1])
            prev_latent = out[stage_name].shape[1]
            self.assertEqual(prev_latent, n.return_channels[i])

    def test_efficientnet_v2(self) -> None:
        size = net.EfficientNet_v2.default_size
        n = net.EfficientNet_v2(3, 100, 0)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Test detection backbone
        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((1, 3, size, size)))
        prev_latent = 0
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertLessEqual(prev_latent, out[stage_name].shape[1])
            prev_latent = out[stage_name].shape[1]
            self.assertEqual(prev_latent, n.return_channels[i])

    def test_focalnet(self) -> None:
        size = net.FocalNet.default_size
        n = net.FocalNet(3, 100, 0)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_ghostnet_v1(self) -> None:
        size = net.GhostNet_v1.default_size
        n = net.GhostNet_v1(3, 100, 0.5)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_ghostnet_v2(self) -> None:
        size = net.GhostNet_v2.default_size
        n = net.GhostNet_v2(3, 100, 0.5)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_inception_next(self) -> None:
        size = net.Inception_NeXt.default_size
        n = net.Inception_NeXt(3, 100, 2)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_inception_resnet_v2(self) -> None:
        size = net.Inception_ResNet_v2.default_size
        n = net.Inception_ResNet_v2(3, 100)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_inception_v3(self) -> None:
        size = net.Inception_v3.default_size
        n = net.Inception_v3(3, 100)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_inception_v4(self) -> None:
        size = net.Inception_v4.default_size
        n = net.Inception_v4(3, 100)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_maxvit(self) -> None:
        size = net.MaxViT.default_size
        n = net.MaxViT(3, 100, 0, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_mnasnet(self) -> None:
        size = net.MNASNet.default_size
        n = net.MNASNet(3, 100, 0.5, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_mobilenet_v1(self) -> None:
        size = net.MobileNet_v1.default_size
        n = net.MobileNet_v1(3, 100, 1)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_mobilenet_v2(self) -> None:
        size = net.MobileNet_v2.default_size
        n = net.MobileNet_v2(3, 100, 1)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_mobilenet_v3(self) -> None:
        size = net.MobileNet_v3.default_size
        n = net.MobileNet_v3(3, 100, 1)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Test detection backbone
        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((1, 3, size, size)))
        prev_latent = 0
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertLessEqual(prev_latent, out[stage_name].shape[1])
            prev_latent = out[stage_name].shape[1]
            self.assertEqual(prev_latent, n.return_channels[i])

    def test_mobilevit_v1(self) -> None:
        size = net.MobileViT_v1.default_size
        n = net.MobileViT_v1(3, 100, 1)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_mobilevit_v2(self) -> None:
        size = net.MobileViT_v2.default_size
        n = net.MobileViT_v2(3, 100, 1)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_next_vit(self) -> None:
        size = net.NextViT.default_size
        n = net.NextViT(3, 100, 1)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_regnet(self) -> None:
        size = net.RegNet.default_size
        n = net.RegNet(3, 100, 0.8)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_resnest(self) -> None:
        size = net.ResNeSt.default_size
        n = net.ResNeSt(3, 100, 50)

        out = n(torch.rand((2, 3, size, size)))
        self.assertEqual(out.numel(), 200)
        n.eval()
        n(torch.rand((1, 3, size, size)))
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_resnet_v2(self) -> None:
        size = net.ResNet_v2.default_size
        n = net.ResNet_v2(3, 100, 50)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Test detection backbone
        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((1, 3, size, size)))
        prev_latent = 0
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertLessEqual(prev_latent, out[stage_name].shape[1])
            prev_latent = out[stage_name].shape[1]
            self.assertEqual(prev_latent, n.return_channels[i])

    def test_resnext(self) -> None:
        size = net.ResNeXt.default_size
        n = net.ResNeXt(3, 100, 50)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Test detection backbone
        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((1, 3, size, size)))
        prev_latent = 0
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertLessEqual(prev_latent, out[stage_name].shape[1])
            prev_latent = out[stage_name].shape[1]
            self.assertEqual(prev_latent, n.return_channels[i])

    def test_se_resnet_v2(self) -> None:
        size = net.SE_ResNet_v2.default_size
        n = net.SE_ResNet_v2(3, 100, 50)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Test detection backbone
        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((1, 3, size, size)))
        prev_latent = 0
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertLessEqual(prev_latent, out[stage_name].shape[1])
            prev_latent = out[stage_name].shape[1]
            self.assertEqual(prev_latent, n.return_channels[i])

    def test_se_resnext(self) -> None:
        size = net.SE_ResNeXt.default_size
        n = net.SE_ResNeXt(3, 100, 50)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Test detection backbone
        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((1, 3, size, size)))
        prev_latent = 0
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertLessEqual(prev_latent, out[stage_name].shape[1])
            prev_latent = out[stage_name].shape[1]
            self.assertEqual(prev_latent, n.return_channels[i])

    def test_shufflenet_v1(self) -> None:
        size = net.ShuffleNet_v1.default_size
        n = net.ShuffleNet_v1(3, 100, 8)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_shufflenet_v2(self) -> None:
        size = net.ShuffleNet_v2.default_size
        n = net.ShuffleNet_v2(3, 100, 1.5)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_simple_vit(self) -> None:
        size = net.Simple_ViT.default_size
        n = net.Simple_ViT(3, 100, 1)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Adjust size
        size += 2**5
        n.adjust_size(size)
        n(torch.rand((1, 3, size, size)))
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)

    def test_squeezenet(self) -> None:
        size = net.SqueezeNet.default_size
        n = net.SqueezeNet(3, 100)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        torch.jit.script(n)

    def test_squeezenext(self) -> None:
        size = net.SqueezeNext.default_size
        n = net.SqueezeNext(3, 100, 0.5)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_swin_transformer_v1(self) -> None:
        size = net.Swin_Transformer_v1.default_size
        n = net.Swin_Transformer_v1(3, 100, 1)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Adjust size
        size += 2**5
        n.adjust_size(size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)

    def test_swin_transformer_v2(self) -> None:
        size = net.Swin_Transformer_v2.default_size
        n = net.Swin_Transformer_v2(3, 100, 0, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Adjust size
        size += 2**5
        n.adjust_size(size)
        n(torch.rand((1, 3, size, size)))
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)

        # Test pre-training encoder
        n.masked_encoding(
            torch.rand((1, 3, size, size)), mask_ratio=0.6, mask_token=torch.zeros(1, 1, 1, n.encoding_size)
        )

    def test_swin_transformer_v2_w2(self) -> None:
        size = net.Swin_Transformer_v2_w2.default_size
        n = net.Swin_Transformer_v2_w2(3, 100, 0, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Adjust size
        size += 2**5
        n.adjust_size(size)
        n(torch.rand((1, 3, size, size)))
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)

        # Test pre-training encoder
        n.masked_encoding(
            torch.rand((1, 3, size, size)), mask_ratio=0.6, mask_token=torch.zeros(1, 1, 1, n.encoding_size)
        )

    def test_vgg(self) -> None:
        size = net.Vgg.default_size
        n = net.Vgg(3, 100, 11)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Test detection backbone
        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((1, 3, size, size)))
        prev_latent = 0
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertLessEqual(prev_latent, out[stage_name].shape[1])
            prev_latent = out[stage_name].shape[1]
            self.assertEqual(prev_latent, n.return_channels[i])

    def test_vgg_reduced(self) -> None:
        size = net.Vgg_Reduced.default_size
        n = net.Vgg_Reduced(3, 100, 11)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Test detection backbone
        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((1, 3, size, size)))
        prev_latent = 0
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertLessEqual(prev_latent, out[stage_name].shape[1])
            prev_latent = out[stage_name].shape[1]
            self.assertEqual(prev_latent, n.return_channels[i])

    def test_vit(self) -> None:
        size = net.ViT.default_size
        n = net.ViT(3, 100, 0, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Adjust size
        size += 2**5
        n.adjust_size(size)
        n(torch.rand((1, 3, size, size)))
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)

        # Test pre-training encoder
        n.masked_encoding(torch.rand((1, 3, size, size)), mask_ratio=0.75)

    def test_vitreg4(self) -> None:
        size = net.ViTReg4.default_size
        n = net.ViTReg4(3, 100, 0, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

        # Adjust size
        size += 2**5
        n.adjust_size(size)
        n(torch.rand((1, 3, size, size)))
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)

        # Test pre-training encoder
        n.masked_encoding(torch.rand((1, 3, size, size)), mask_ratio=0.75)

    def test_wide_resnet(self) -> None:
        size = net.Wide_ResNet.default_size
        n = net.Wide_ResNet(3, 100, 50, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_xception(self) -> None:
        size = net.Xception.default_size
        n = net.Xception(3, 100)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)

    def test_xcit(self) -> None:
        size = net.XCiT.default_size
        n = net.XCiT(3, 100, 0, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertEqual(out.numel(), 100)
        embedding = n.embedding(torch.rand((1, 3, size, size))).flatten()
        self.assertEqual(len(embedding), n.embedding_size)
        torch.jit.script(n)
