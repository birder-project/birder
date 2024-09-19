import logging
import unittest
from typing import Optional

import torch
from parameterized import parameterized

from birder.model_registry import registry
from birder.net import base

logging.disable(logging.CRITICAL)


class TestBase(unittest.TestCase):
    def test_make_divisible(self) -> None:
        self.assertEqual(base.make_divisible(25, 6), 24)

    def test_get_signature(self) -> None:
        signature = base.get_signature((1, 3, 224, 224), 10)
        self.assertIn("inputs", signature)
        self.assertIn("outputs", signature)


class TestNet(unittest.TestCase):
    @parameterized.expand(  # type: ignore[misc]
        [
            ("alexnet", None),
            ("cait", 0),
            ("convnext_v1", 0),
            ("convnext_v2", 0),
            ("crossvit", 0, True, 1, 48),
            ("deit", 0, True),
            ("deit3", 0, True),
            ("densenet", 121),
            ("edgenext", 0),
            ("edgevit", 1),
            ("efficientformer_v1", 0),
            ("efficientformer_v2", 0),
            ("efficientnet_v1", 0),
            ("efficientnet_v2", 0),
            ("focalnet", 0),
            ("ghostnet_v1", 0),
            ("ghostnet_v2", 0),
            ("inception_next", 2),
            ("inception_resnet_v2", None),
            ("inception_v3", None),
            ("inception_v4", None),
            ("maxvit", 0),
            ("metaformer", 0),  # PoolFormer v1
            ("metaformer", 10),  # PoolFormer v2
            ("metaformer", 20),  # ConvFormer
            ("metaformer", 30),  # CAFormer
            ("mnasnet", 0.5),
            ("mobilenet_v1", 1),
            ("mobilenet_v2", 1),
            ("mobilenet_v3_large", 1),
            ("mobilenet_v3_small", 1),
            ("mobilenet_v4", 0),
            ("mobilenet_v4_hybrid", 0),
            ("mobileone", 0),
            ("mobilevit_v1", 1),
            ("mobilevit_v2", 1),
            ("nextvit", 1),
            ("rdnet", 0),
            ("regnet", 0.8),
            ("repvgg", 0),
            ("resnest", 50, False, 2),
            ("resnet_v2", 18),
            ("resnext", 50),
            ("se_resnet_v2", 50),
            ("se_resnext", 50),
            ("sequencer2d", 0),
            ("shufflenet_v1", 8),
            ("shufflenet_v2", 1),
            ("simple_vit", 1),
            ("squeezenet", None, True),
            ("squeezenext", 0.5),
            ("swin_transformer_v1", 0),
            ("swin_transformer_v2", 0),
            ("swin_transformer_v2_w2", 0),
            ("vgg", 11),
            ("vgg_reduced", 11),
            ("vit", 0),
            ("vitreg4", 0),
            ("wide_resnet", 50),
            ("xception", None),
            ("xcit", 0),
        ]
    )
    def test_net(
        self,
        network_name: str,
        net_param: Optional[float],
        skip_embedding: bool = False,
        batch_size: int = 1,
        size_step: int = 2**5,
    ) -> None:
        n = registry.net_factory(network_name, 3, 100, net_param=net_param)
        size = n.default_size

        out = n(torch.rand((batch_size, 3, size, size)))
        self.assertEqual(out.numel(), 100 * batch_size)
        self.assertFalse(torch.isnan(out).any())

        if skip_embedding is False:
            embedding = n.embedding(torch.rand((batch_size, 3, size, size))).flatten()
            self.assertEqual(len(embedding), n.embedding_size * batch_size)

        torch.jit.script(n)

        # Adjust size
        size += size_step
        n.adjust_size(size)
        out = n(torch.rand((batch_size, 3, size, size)))
        self.assertEqual(out.numel(), 100 * batch_size)
        if skip_embedding is False:
            embedding = n.embedding(torch.rand((batch_size, 3, size, size))).flatten()
            self.assertEqual(len(embedding), n.embedding_size * batch_size)

        # Reset classifier
        n.reset_classifier(200)
        out = n(torch.rand((batch_size, 3, size, size)))
        self.assertEqual(out.numel(), 200 * batch_size)

        # Reparameterize
        if base.reparameterize_available(n) is True:
            n.reparameterize_model()
            out = n(torch.rand((batch_size, 3, size, size)))
            self.assertEqual(out.numel(), 200 * batch_size)

    @parameterized.expand(  # type: ignore[misc]
        [
            ("efficientnet_v1", 0),
            ("efficientnet_v2", 0),
            ("mnasnet", 0.75),
            ("mobilenet_v2", 1),
            ("mobilenet_v3_large", 1),
            ("mobilenet_v3_small", 1),
            ("mobilenet_v4", 0),
            ("mobilenet_v4_hybrid", 0),
            ("regnet", 0.8),
            ("resnet_v2", 50),
            ("resnext", 50),
            ("se_resnet_v2", 50),
            ("se_resnext", 50),
            ("vgg", 11),
            ("vgg_reduced", 11),
        ]
    )
    def test_detection_backbone(self, network_name: str, net_param: Optional[float]) -> None:
        n = registry.net_factory(network_name, 3, 100, net_param=net_param)
        size = n.default_size

        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((1, 3, size, size)))
        prev_latent = 0
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertLessEqual(prev_latent, out[stage_name].shape[1])
            prev_latent = out[stage_name].shape[1]
            self.assertEqual(prev_latent, n.return_channels[i])

    @parameterized.expand(  # type: ignore[misc]
        [
            ("convnext_v2", 0, False),
            ("maxvit", 0, True),
            ("simple_vit", 0, False),
            ("swin_transformer_v2", 0, True),
            ("swin_transformer_v2_w2", 0, True),
            ("vit", 0, False),
            ("vitreg4", 0, False),
        ]
    )
    def test_pre_training_encoder(self, network_name: str, net_param: Optional[float], mask_token: bool) -> None:
        n = registry.net_factory(network_name, 3, 100, net_param=net_param)
        size = n.default_size

        mt = None
        if mask_token is True:
            mt = torch.zeros(1, 1, 1, n.encoding_size)

        outs = n.masked_encoding(torch.rand((1, 3, size, size)), 0.6, mt)
        for out in outs:
            self.assertFalse(torch.isnan(out).any())
