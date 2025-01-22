import json
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

    def test_base_net(self) -> None:
        base_net = base.BaseNet(3, num_classes=2, size=128)
        base_net.body = torch.nn.Linear(3, 10, bias=False)
        base_net.features = torch.nn.Linear(10, 10, bias=False)
        base_net.classifier = base_net.create_classifier(embed_dim=10)

        # Test freeze
        for param in base_net.parameters():
            self.assertTrue(param.requires_grad)

        base_net.freeze()
        for param in base_net.parameters():
            self.assertFalse(param.requires_grad)

        base_net.freeze(freeze_classifier=False)
        self.assertFalse(base_net.body.weight.requires_grad)
        self.assertFalse(base_net.features.weight.requires_grad)
        self.assertTrue(base_net.classifier.weight.requires_grad)

        base_net.freeze(freeze_classifier=False, unfreeze_features=True)
        self.assertFalse(base_net.body.weight.requires_grad)
        self.assertTrue(base_net.features.weight.requires_grad)
        self.assertTrue(base_net.classifier.weight.requires_grad)

        base_net.freeze(freeze_classifier=True, unfreeze_features=True)
        self.assertFalse(base_net.body.weight.requires_grad)
        self.assertTrue(base_net.features.weight.requires_grad)
        self.assertFalse(base_net.classifier.weight.requires_grad)


class TestNet(unittest.TestCase):
    @parameterized.expand(  # type: ignore[misc]
        [
            ("alexnet"),
            ("biformer_t"),
            ("cait_xxs24"),
            ("coat_tiny"),
            ("coat_lite_tiny"),
            ("convmixer_768_32"),
            ("convnext_v1_tiny"),
            ("convnext_v2_atto"),
            ("crossvit_9d", None, True, 1, 48),
            ("csp_resnet_50"),
            ("csp_resnext_50"),
            ("csp_darknet_53"),
            ("csp_se_resnet_50"),
            ("cswin_transformer_t"),
            ("darknet_53"),
            ("davit_tiny"),
            ("deit_t16", None, True),
            ("deit3_t16", None, True),
            ("densenet_121"),
            ("dpn_92"),
            ("edgenext_xxs"),
            ("edgevit_xxs"),
            ("efficientformer_v1_l1"),
            ("efficientformer_v2_s0"),
            ("efficientnet_lite0"),
            ("efficientnet_v1_b0"),
            ("efficientnet_v2_s"),
            ("efficientvit_mit_b0"),
            ("efficientvit_mit_l1"),
            ("efficientvit_msft_m0", None, False, 2),
            ("fasternet_t0"),
            ("fastvit_t8"),
            ("fastvit_sa12"),
            ("mobileclip_i0"),
            ("focalnet_t_srf"),
            ("ghostnet_v1", 1),
            ("ghostnet_v2", 1),
            ("hiera_tiny"),
            ("hiera_abswin_tiny"),
            ("hieradet_tiny"),
            ("hornet_tiny_7x7"),
            ("hornet_tiny_gf"),
            ("iformer_s"),
            ("inception_next_t"),
            ("inception_resnet_v2"),
            ("inception_v3"),
            ("inception_v4"),
            ("levit_128"),
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
            ("mobilenet_v4_s", None, False, 2),
            ("mobilenet_v4_hybrid_m", None, False, 2),
            ("mobilenet_v4_hybrid_l", None, False, 2),  # GELU (inplace)
            ("mobileone_s0"),
            ("mobilevit_v1_xxs"),
            ("mobilevit_v2", 1),
            ("moganet_xt"),
            ("mvit_v2_t"),
            ("mvit_v2_t_cls"),
            ("nextvit_s"),
            ("nfnet_f0"),
            ("pit_t", None, True),
            ("pvt_v1_t"),
            ("pvt_v2_b0"),
            ("rdnet_t"),
            ("regnet_x_200m"),
            ("regnet_y_200m"),
            ("regnet_z_500m"),
            ("repghost", 1),
            ("repvgg_a0"),
            ("resmlp_12", None, False, 1, 0),  # No resize support
            ("resnest_14", None, False, 2),
            ("resnet_v1_18"),
            ("resnet_v2_18"),
            ("resnext_50"),
            ("se_resnet_v1_18"),
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
            ("tiny_vit_5m"),
            ("uniformer_s"),
            ("van_b0"),
            ("vgg_11"),
            ("vgg_reduced_11"),
            ("vit_b32"),
            ("vit_sam_b16"),
            ("vitreg4_b32"),
            ("wide_resnet_50"),
            ("xception"),
            ("xcit_nano12_p16"),
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

        # Ensure config is serializable
        _ = json.dumps(n.config)

        # Test network
        out = n(torch.rand((batch_size, 3, size, size)))
        self.assertEqual(out.numel(), 100 * batch_size)
        self.assertFalse(torch.isnan(out).any())

        if skip_embedding is False:
            embedding = n.embedding(torch.rand((batch_size, 3, size, size))).flatten()
            self.assertEqual(len(embedding), n.embedding_size * batch_size)

        if n.scriptable is True:
            torch.jit.script(n)
        else:
            n.eval()
            torch.jit.trace(n, example_inputs=torch.rand((batch_size, 3, size, size)))
            n.train()

        # Adjust size
        if size_step != 0:
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
            ("biformer_t"),
            ("coat_tiny"),
            ("coat_lite_tiny"),
            ("convnext_v1_tiny"),
            ("convnext_v2_tiny"),
            ("csp_resnet_50"),
            ("csp_resnext_50"),
            ("csp_darknet_53"),
            ("csp_se_resnet_50"),
            ("cswin_transformer_t"),
            ("darknet_53"),
            ("davit_tiny"),
            ("densenet_121"),
            ("edgenext_xxs"),
            ("edgevit_xxs"),
            ("efficientformer_v2_s0"),
            ("efficientnet_lite0"),
            ("efficientnet_v1_b0"),
            ("efficientnet_v2_s"),
            ("efficientvit_mit_b0"),
            ("efficientvit_mit_l1"),
            ("efficientvit_msft_m0"),
            ("fasternet_t0"),
            ("fastvit_t8"),
            ("focalnet_t_srf"),
            ("ghostnet_v1", 1),
            ("ghostnet_v2", 1),
            ("hiera_tiny"),
            ("hiera_abswin_tiny"),
            ("hieradet_tiny"),
            ("hornet_tiny_7x7"),
            ("hornet_tiny_gf"),
            ("iformer_s"),
            ("inception_next_t"),
            ("inception_resnet_v2"),
            ("inception_v3"),
            ("inception_v4"),
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
            ("mobilevit_v2", 1),
            ("moganet_xt"),
            ("mvit_v2_t"),
            ("mvit_v2_t_cls"),
            ("nextvit_s"),
            ("nfnet_f0"),
            ("pit_t"),
            ("pvt_v1_t"),
            ("pvt_v2_b0"),
            ("rdnet_t"),
            ("regnet_y_200m"),
            ("regnet_z_500m"),
            ("repghost", 1),
            ("repvgg_a0"),
            ("resnest_14", None, 2),
            ("resnet_v1_18"),
            ("resnet_v2_18"),
            ("resnext_50"),
            ("se_resnet_v1_18"),
            ("se_resnet_v2_18"),
            ("se_resnext_50"),
            ("shufflenet_v1", 8),
            ("shufflenet_v2", 1),
            ("squeezenext", 0.5),
            ("swin_transformer_v1_t"),
            ("swin_transformer_v2_t"),
            ("tiny_vit_5m"),
            ("uniformer_s"),
            ("van_b0"),
            ("vgg_11"),
            ("vgg_reduced_11"),
            ("vit_sam_b16"),
            ("wide_resnet_50"),
            ("xception"),
            ("xcit_nano12_p16", None, 1, True),
        ]
    )
    def test_detection_backbone(
        self,
        network_name: str,
        net_param: Optional[float] = None,
        batch_size: int = 1,
        allow_equal_stages: bool = False,
    ) -> None:
        n = registry.net_factory(network_name, 3, 100, net_param=net_param)
        size = n.default_size

        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((batch_size, 3, size, size)))
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertEqual(out[stage_name].shape[1], n.return_channels[i])

        prev_h = 0
        prev_w = 0
        for i, stage_name in enumerate(n.return_stages[::-1]):
            if allow_equal_stages is True:
                self.assertLessEqual(prev_h, out[stage_name].shape[2])
                self.assertLessEqual(prev_w, out[stage_name].shape[3])
            else:
                self.assertLess(prev_h, out[stage_name].shape[2])
                self.assertLess(prev_w, out[stage_name].shape[3])

            prev_h = out[stage_name].shape[2]
            prev_w = out[stage_name].shape[3]

        num_stages = len(n.return_stages)
        for idx in range(num_stages):
            n.freeze_stages(idx)

    @parameterized.expand(  # type: ignore[misc]
        [
            ("convnext_v2_atto", None, False),
            ("hiera_tiny", None, False),
            ("hiera_abswin_tiny", None, False),
            ("maxvit_t", None, True),
            ("nextvit_s", None, True),
            ("regnet_x_200m", None, False),
            ("regnet_y_200m", None, False),
            ("regnet_z_500m", None, False),
            ("simple_vit_b32", None, False),
            ("swin_transformer_v2_t", None, True),
            ("swin_transformer_v2_w2_t", None, True),
            ("vit_b32", None, False),
            ("vit_sam_b16", None, False),
            ("vitreg4_b32", None, False),
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
            if isinstance(out, (tuple, list)):  # Hierarchical MIM
                for o in out:
                    self.assertFalse(torch.isnan(o).any())
            else:
                self.assertFalse(torch.isnan(out).any())

        self.assertTrue(hasattr(n, "block_group_regex"))


class TestSpecialFunctions(unittest.TestCase):
    def test_vit_sam_weight_import(self) -> None:
        vit_b16 = registry.net_factory("vit_b16", 3, 100, size=192)
        vit_sam_b16 = registry.net_factory("vit_sam_b16", 3, 100, size=192)

        # Simple ViT
        vit_sam_b16.load_vit_weights(vit_b16.state_dict())

        vitreg4_b16 = registry.net_factory("vitreg4_b16", 3, 100, size=192)

        # ViT with register tokens
        vit_sam_b16.load_vit_weights(vitreg4_b16.state_dict())

    def test_hieradet_weight_import(self) -> None:
        hiera_abswin_tiny = registry.net_factory("hiera_abswin_tiny", 3, 100, size=192)
        hieradet_tiny = registry.net_factory("hieradet_tiny", 3, 100, size=192)

        hieradet_tiny.load_hiera_weights(hiera_abswin_tiny.state_dict())
