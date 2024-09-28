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
            ("alexnet"),
            ("cait_xxs24"),
            ("convmixer_768_32"),
            ("convnext_v1_tiny"),
            ("convnext_v2_atto"),
            ("crossvit_9d", None, True, 1, 48),
            ("deit_t16", 0, True),
            ("deit3_t16", 0, True),
            ("densenet_121"),
            ("edgenext_xxs"),
            ("edgevit_xxs"),
            ("efficientformer_v1_l1"),
            ("efficientformer_v2_s0"),
            ("efficientnet_v1_b0"),
            ("efficientnet_v2_s"),
            ("fastvit_t8"),
            ("focalnet_t_srf"),
            ("ghostnet_v1", 0),
            ("ghostnet_v2", 0),
            ("inception_next_t"),
            ("inception_resnet_v2", None),
            ("inception_v3", None),
            ("inception_v4", None),
            ("maxvit_t"),
            ("poolformer_v1_s12"),
            ("poolformer_v2_s12"),
            ("convformer_s18"),
            ("caformer_s18"),
            ("mnasnet", 0.5),
            ("mobilenet_v1", 1),
            ("mobilenet_v2", 1),
            ("mobilenet_v3_large", 1),
            ("mobilenet_v3_small", 1),
            ("mobilenet_v4_s"),
            ("mobilenet_v4_hybrid_m"),
            ("mobileone_s0"),
            ("mobilevit_v1_xxs"),
            ("mobilevit_v2", 1),
            ("nextvit_s"),
            ("rdnet_t"),
            ("regnet_x_200m"),
            ("regnet_y_200m"),
            ("repvgg_a0"),
            ("resnest_14", None, False, 2),
            ("resnet_v2_18"),
            ("resnext_50"),
            ("se_resnet_v2_18"),
            ("se_resnext_50"),
            ("sequencer2d_s"),
            ("shufflenet_v1", 8),
            ("shufflenet_v2", 1),
            ("simple_vit_b32"),
            ("squeezenet", None, True),
            ("squeezenext", 0.5),
            ("swin_transformer_v1_t"),
            ("swin_transformer_v2_t"),
            ("swin_transformer_v2_w2_t"),
            ("uniformer_s"),
            ("vgg_11"),
            ("vgg_reduced_11"),
            ("vit_b32"),
            ("vitreg4_b32"),
            ("wide_resnet_50"),
            ("xception"),
            ("xcit_nano16"),
        ]
    )
    def test_net(
        self,
        network_name: str,
        net_param: Optional[float] = None,
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
            ("convnext_v1_tiny"),
            ("convnext_v2_tiny"),
            ("efficientnet_v1_b0"),
            ("efficientnet_v2_s"),
            ("mnasnet", 0.75),
            ("mobilenet_v2", 1),
            ("mobilenet_v3_large", 1),
            ("mobilenet_v3_small", 1),
            ("mobilenet_v4_s"),
            ("mobilenet_v4_hybrid_m"),
            ("regnet_y_200m"),
            ("resnet_v2_18"),
            ("resnext_50"),
            ("se_resnet_v2_18"),
            ("se_resnext_50"),
            ("vgg_11"),
            ("vgg_reduced_11"),
        ]
    )
    def test_detection_backbone(self, network_name: str, net_param: Optional[float] = None) -> None:
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
            ("convnext_v2_atto", 0, False),
            ("maxvit_t", 0, True),
            ("nextvit_s", 0, True),
            ("simple_vit_b32", 0, False),
            ("swin_transformer_v2_t", 0, True),
            ("swin_transformer_v2_w2_t", 0, True),
            ("vit_b32", 0, False),
            ("vitreg4_b32", 0, False),
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

        self.assertTrue(hasattr(n, "block_group_regex"))
