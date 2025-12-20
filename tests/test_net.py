import copy
import json
import logging
import unittest

import torch
from parameterized import parameterized

from birder.common.masking import uniform_mask
from birder.common.training_utils import group_by_regex
from birder.conf.settings import DEFAULT_NUM_CHANNELS
from birder.model_registry import registry
from birder.net import Hiera
from birder.net import base
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import MaskedTokenRetentionMixin

logging.disable(logging.CRITICAL)


class TestBase(unittest.TestCase):
    def test_make_divisible(self) -> None:
        self.assertEqual(base.make_divisible(25, 6), 24)

    def test_get_signature(self) -> None:
        signature = base.get_signature((1, 3, 224, 224), 10)
        self.assertIn("inputs", signature)
        self.assertIn("outputs", signature)

    def test_base_net(self) -> None:
        base_net = base.BaseNet(DEFAULT_NUM_CHANNELS, num_classes=2, size=(128, 128))
        base_net.body = torch.nn.Linear(DEFAULT_NUM_CHANNELS, 10, bias=False)
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
    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("alexnet"),
            ("biformer_t"),
            ("cait_xxs24"),
            ("cas_vit_xs"),
            ("coat_tiny"),
            ("coat_lite_tiny"),
            ("conv2former_n"),
            ("convmixer_768_32"),
            ("convnext_v1_tiny"),
            ("convnext_v2_atto"),
            ("crossformer_t"),
            ("crossvit_9d", True, True, 1, 48),
            ("csp_resnet_50"),
            ("csp_resnext_50"),
            ("csp_darknet_53"),
            ("csp_se_resnet_50"),
            ("cswin_transformer_t"),  # PT2 fails
            ("darknet_53"),
            ("davit_tiny"),
            ("deit_t16", True),
            ("deit3_t16"),
            ("deit3_reg4_t16"),
            ("densenet_121"),
            ("dpn_92"),
            ("edgenext_xxs"),
            ("edgevit_xxs"),
            ("efficientformer_v1_l1"),
            ("efficientformer_v2_s0"),
            ("efficientnet_lite0"),
            ("efficientnet_v1_b0"),
            ("efficientnet_v2_s"),
            ("efficientvim_m1", True, True),
            ("efficientvit_mit_b0"),
            ("efficientvit_mit_l1"),
            ("efficientvit_msft_m0", False, False, 2),
            ("fasternet_t0"),
            ("fastvit_t8"),
            ("fastvit_sa12"),
            ("mobileclip_v1_i0"),
            ("mobileclip_v2_i3"),
            ("flexivit_s16"),
            ("focalnet_t_srf"),
            ("ghostnet_v1_0_5"),
            ("ghostnet_v2_1_0"),
            ("groupmixformer_mobile"),
            ("hgnet_v1_tiny"),
            ("hgnet_v2_b0"),
            ("hiera_tiny"),
            ("hiera_abswin_tiny"),  # No bfloat16 support
            ("hiera_abswin_base_plus_ap"),  # No bfloat16 support
            ("hieradet_tiny"),
            ("hornet_tiny_7x7"),
            ("hornet_tiny_gf"),  # PT2 fails, no bfloat16 support
            ("iformer_s"),
            ("inception_next_t"),
            ("inception_resnet_v1"),
            ("inception_resnet_v2"),
            ("inception_v3"),
            ("inception_v4"),
            ("levit_128"),
            ("maxvit_t"),
            ("poolformer_v1_s12"),
            ("poolformer_v2_s12"),
            ("convformer_s18"),
            ("caformer_s18"),
            ("mnasnet_0_5"),
            ("mobilenet_v1_0_25"),
            ("mobilenet_v2_0_25"),
            ("mobilenet_v3_large_0_75"),
            ("mobilenet_v3_small_1_0"),
            ("mobilenet_v4_s", False, False, 2),
            ("mobilenet_v4_hybrid_m", False, False, 2),
            ("mobilenet_v4_hybrid_l", False, False, 2),  # GELU (inplace)
            ("mobileone_s0"),
            ("mobilevit_v1_xxs"),
            ("mobilevit_v2_0_25"),
            ("moganet_xt"),
            ("mvit_v2_t"),
            ("mvit_v2_t_cls"),
            ("nextvit_s"),
            ("nfnet_f0"),
            ("pit_t", True, True),
            ("pvt_v1_t"),
            ("pvt_v2_b0"),
            ("rdnet_t"),
            ("regionvit_t", False, True),
            ("regnet_x_200m"),
            ("regnet_y_200m"),
            ("regnet_z_500m"),
            ("ghostnet_v2_1_0"),
            ("repvgg_a0"),
            ("repvit_m0_6", False, False, 2),
            ("resmlp_12", False, False, 1, 0),  # No resize support
            ("resnest_14", False, False, 2),
            ("resnet_v1_18"),
            ("resnet_v2_18"),
            ("resnext_50"),
            ("rope_deit3_t16"),
            ("rope_deit3_reg4_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_i_vit_s16_pn_aps_c1"),
            ("rope_vit_reg4_b32"),
            ("rope_vit_reg4_m16_rms_avg"),
            ("rope_vit_reg8_nps_b14_ap", False, False, 1, 14),
            ("rope_vit_so150m_p14_ap", False, False, 1, 14),
            ("rope_vit_reg8_so150m_p14_swiglu_rms_avg", False, False, 1, 14),
            ("se_resnet_v1_18"),
            ("se_resnet_v2_18"),
            ("se_resnext_50"),
            ("sequencer2d_s"),
            ("shufflenet_v1_8"),
            ("shufflenet_v2_0_5"),
            ("simple_vit_s32"),
            ("smt_t"),
            ("squeezenet", True),
            ("squeezenext_0_5"),
            ("starnet_esm05"),
            ("swiftformer_xs"),
            ("swin_transformer_v1_t"),
            ("swin_transformer_v2_t"),
            ("swin_transformer_v2_w2_t"),
            ("tiny_vit_5m"),
            ("transnext_micro"),
            ("uniformer_s"),
            ("van_b0"),
            ("vgg_11"),
            ("vgg_reduced_11"),
            ("vit_s32"),
            ("vit_s16_pn"),
            ("vit_reg4_b32"),
            ("vit_reg4_m16_rms_avg"),
            ("vit_so150m_p14_ap", False, False, 1, 14),
            ("vit_reg8_so150m_p14_swiglu_avg", False, False, 1, 14),
            ("vit_parallel_s16_18x2_ls"),
            ("vit_det_b16"),
            ("vit_sam_b16"),
            ("vovnet_v1_27s"),
            ("vovnet_v2_19"),
            ("wide_resnet_50"),
            ("xception"),
            ("xcit_nano12_p16"),
        ]
    )
    def test_net(
        self,
        network_name: str,
        skip_embedding: bool = False,
        skip_features: bool = False,
        batch_size: int = 1,
        size_step: int = 2**5,
    ) -> None:
        n = registry.net_factory(network_name, DEFAULT_NUM_CHANNELS, 100)
        size = n.default_size

        # Ensure config is serializable
        _ = json.dumps(n.config)

        # Test network
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(out.numel(), 100 * batch_size)
        self.assertFalse(torch.isnan(out).any())

        if skip_embedding is False:
            embedding = n.embedding(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size))).flatten()
            self.assertEqual(len(embedding), n.embedding_size * batch_size)

        if skip_features is False:
            features = n.forward_features(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            self.assertFalse(torch.isnan(features).any())
            self.assertEqual(features.size(0), batch_size)

        # Test TorchScript support
        if n.scriptable is True:
            torch.jit.script(n)
        else:
            n.eval()
            torch.jit.trace(n, example_inputs=torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            n.train()

        # Test PT2
        # batch_dim = torch.export.Dim("batch", min=1, max=4096)
        # torch.export.export(n, (torch.randn(2, DEFAULT_NUM_CHANNELS, *size),), dynamic_shapes={"x": {0: batch_dim}})

        # Adjust size
        if size_step != 0:
            size = (size[0] + size_step, size[1] + size_step)
            n.adjust_size(size)
            out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            self.assertEqual(out.numel(), 100 * batch_size)
            if skip_embedding is False:
                embedding = n.embedding(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size))).flatten()
                self.assertEqual(len(embedding), n.embedding_size * batch_size)

        # Reset classifier
        n.reset_classifier(200)
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(out.numel(), 200 * batch_size)

        # Reparameterize
        if base.reparameterize_available(n) is True:
            n.reparameterize_model()
            out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            self.assertEqual(out.numel(), 200 * batch_size)

        # Test modified dtype
        # n.to(torch.bfloat16)
        # out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size), dtype=torch.bfloat16))
        # self.assertEqual(out.numel(), 200 * batch_size)

        # Ensure model is copyable
        n_copy = copy.deepcopy(n)
        self.assertIsNotNone(n_copy)

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("biformer_t"),
            ("cas_vit_xs"),
            ("coat_tiny"),
            ("coat_lite_tiny"),
            ("conv2former_n"),
            ("convnext_v1_tiny"),
            ("convnext_v2_tiny"),
            ("crossformer_t"),
            ("csp_resnet_50"),
            ("csp_resnext_50"),
            ("csp_darknet_53"),
            ("csp_se_resnet_50"),
            ("cswin_transformer_t"),
            ("darknet_53"),
            ("davit_tiny"),
            ("deit3_t16"),
            ("deit3_reg4_t16"),
            ("densenet_121"),
            ("edgenext_xxs"),
            ("edgevit_xxs"),
            ("efficientformer_v2_s0"),
            ("efficientnet_lite0"),
            ("efficientnet_v1_b0"),
            ("efficientnet_v2_s"),
            ("efficientvim_m1"),
            ("efficientvit_mit_b0"),
            ("efficientvit_mit_l1"),
            ("efficientvit_msft_m0"),
            ("fasternet_t0"),
            ("fastvit_t8"),
            ("mobileclip_v1_i0"),
            ("mobileclip_v2_i3"),
            ("flexivit_s16"),
            ("focalnet_t_srf"),
            ("ghostnet_v1_0_5"),
            ("ghostnet_v2_1_0"),
            ("groupmixformer_mobile"),
            ("hgnet_v1_tiny"),
            ("hgnet_v2_b0"),
            ("hiera_tiny"),
            ("hiera_abswin_tiny"),
            ("hiera_abswin_base_plus_ap"),
            ("hieradet_tiny"),
            ("hornet_tiny_7x7"),
            ("hornet_tiny_gf"),
            ("iformer_s"),
            ("inception_next_t"),
            ("inception_resnet_v1"),
            ("inception_resnet_v2"),
            ("inception_v3"),
            ("inception_v4"),
            ("maxvit_t"),
            ("poolformer_v1_s12"),
            ("poolformer_v2_s12"),
            ("convformer_s18"),
            ("caformer_s18"),
            ("mnasnet_0_5"),
            ("mobilenet_v1_0_25"),
            ("mobilenet_v2_0_25"),
            ("mobilenet_v3_large_0_75"),
            ("mobilenet_v3_small_1_0"),
            ("mobilenet_v4_s"),
            ("mobilenet_v4_hybrid_m"),
            ("mobileone_s0"),
            ("mobilevit_v2_0_25"),
            ("moganet_xt"),
            ("mvit_v2_t"),
            ("mvit_v2_t_cls"),
            ("nextvit_s"),
            ("nfnet_f0"),
            ("pit_t"),
            ("pvt_v1_t"),
            ("pvt_v2_b0"),
            ("rdnet_t"),
            ("regionvit_t"),
            ("regnet_y_200m"),
            ("regnet_z_500m"),
            ("repghost_0_5"),
            ("repvgg_a0"),
            ("repvit_m0_6"),
            ("resnest_14", 2),
            ("resnet_v1_18"),
            ("resnet_v2_18"),
            ("resnext_50"),
            ("rope_deit3_t16"),
            ("rope_deit3_reg4_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_i_vit_s16_pn_aps_c1"),
            ("rope_vit_reg4_b32"),
            ("rope_vit_reg4_m16_rms_avg"),
            ("rope_vit_reg8_nps_b14_ap"),
            ("rope_vit_so150m_p14_ap"),
            ("rope_vit_reg8_so150m_p14_swiglu_rms_avg"),
            ("se_resnet_v1_18"),
            ("se_resnet_v2_18"),
            ("se_resnext_50"),
            ("shufflenet_v1_8"),
            ("shufflenet_v2_0_5"),
            ("smt_t"),
            ("squeezenext_0_5"),
            ("starnet_esm05"),
            ("swiftformer_xs"),
            ("swin_transformer_v1_t"),
            ("swin_transformer_v2_t"),
            ("tiny_vit_5m"),
            ("transnext_micro"),
            ("uniformer_s"),
            ("van_b0"),
            ("vgg_11"),
            ("vgg_reduced_11"),
            ("vit_s32"),
            ("vit_s16_pn"),
            ("vit_reg4_b32"),
            ("vit_reg4_m16_rms_avg"),
            ("vit_so150m_p14_ap"),
            ("vit_reg8_so150m_p14_swiglu_avg"),
            ("vit_parallel_s16_18x2_ls"),
            ("vit_det_b16"),
            ("vit_sam_b16"),
            ("vovnet_v1_27s"),
            ("vovnet_v2_19"),
            ("wide_resnet_50"),
            ("xception"),
            ("xcit_nano12_p16", 1, True),
        ]
    )
    def test_detection_backbone(
        self,
        network_name: str,
        batch_size: int = 1,
        allow_equal_stages: bool = False,
    ) -> None:
        n = registry.net_factory(network_name, DEFAULT_NUM_CHANNELS, 100)
        size = n.default_size

        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertEqual(out[stage_name].shape[1], n.return_channels[i])
            self.assertFalse(torch.isnan(out[stage_name]).any())

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

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("hiera_tiny",),
            ("hiera_abswin_tiny",),
            ("hiera_abswin_base_plus_ap"),
        ]
    )
    def test_pre_training_encoder_hiera(self, network_name: str) -> None:
        n = registry.net_factory(network_name, DEFAULT_NUM_CHANNELS, 100)
        size = n.default_size

        # self.assertIsInstance(n, Hiera)
        assert isinstance(n, Hiera)

        x = torch.rand((1, DEFAULT_NUM_CHANNELS, *size))

        mask = uniform_mask(1, n.mask_spatial_shape[0], n.mask_spatial_shape[1], mask_ratio=0.6, device=x.device)[0]
        (outs, mask) = n.masked_encoding(x, mask)

        for out in outs:
            self.assertFalse(torch.isnan(out).any())

        self.assertFalse(torch.isnan(mask).any())

        self.assertTrue(hasattr(n, "block_group_regex"))
        self.assertTrue(hasattr(n, "stem_stride"))
        self.assertTrue(hasattr(n, "stem_width"))

        names = [n for n, _ in n.named_parameters()]
        groups = group_by_regex(names, n.block_group_regex)
        self.assertGreater(len(groups), 5)
        self.assertLess(len(groups), 40)

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("conv2former_n"),
            ("convnext_v1_tiny"),
            ("convnext_v2_atto"),
            ("davit_tiny"),
            ("deit3_t16"),
            ("deit3_reg4_t16"),
            ("efficientnet_lite0"),
            ("efficientnet_v1_b0"),
            ("efficientnet_v2_s"),
            ("fastvit_t8"),
            ("fastvit_sa12"),
            ("mobileclip_v1_i0"),
            ("mobileclip_v2_i3"),
            ("flexivit_s16"),
            ("focalnet_t_srf"),
            ("hieradet_tiny"),
            ("iformer_s"),
            ("maxvit_t"),
            ("poolformer_v1_s12"),
            ("poolformer_v2_s12"),
            ("convformer_s18"),
            ("caformer_s18"),
            ("mobilenet_v4_s", 2),
            ("mobilenet_v4_hybrid_m", 2),
            ("moganet_xt"),
            ("mvit_v2_t"),
            ("mvit_v2_t_cls"),
            ("nextvit_s"),
            ("rdnet_t"),
            ("regnet_x_200m"),
            ("regnet_y_200m"),
            ("regnet_z_500m"),
            ("rope_deit3_t16"),
            ("rope_deit3_reg4_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_i_vit_s16_pn_aps_c1"),
            ("rope_vit_reg4_b32"),
            ("rope_vit_reg4_m16_rms_avg"),
            ("rope_vit_reg8_nps_b14_ap"),
            ("rope_vit_so150m_p14_ap"),
            ("rope_vit_reg8_so150m_p14_swiglu_rms_avg"),
            ("swin_transformer_v2_t"),
            ("swin_transformer_v2_w2_t"),
            ("vit_s32"),
            ("vit_s16_pn"),
            ("vit_reg4_b32"),
            ("vit_reg4_m16_rms_avg"),
            ("vit_so150m_p14_ap"),
            ("vit_reg8_so150m_p14_swiglu_avg"),
            ("vit_parallel_s16_18x2_ls"),
            ("xcit_nano12_p16"),
        ]
    )
    def test_pre_training_encoder_retention(self, network_name: str, batch_size: int = 1) -> None:
        n = registry.net_factory(network_name, DEFAULT_NUM_CHANNELS, 100)
        size = n.default_size

        # self.assertIsInstance(n, MaskedTokenRetentionMixin)
        assert isinstance(n, MaskedTokenRetentionMixin)
        h = size[0] // n.max_stride
        w = size[1] // n.max_stride
        x = torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size))
        mask = uniform_mask(batch_size, h, w, mask_ratio=0.6, device=x.device)[0]

        # Test retention
        out = n.masked_encoding_retention(x, mask, return_keys="features")
        self.assertFalse(torch.isnan(out["features"]).any())
        self.assertEqual(out["features"].ndim, 4)

        # Test substitution
        out = n.masked_encoding_retention(x, mask, torch.zeros(1, 1, 1, n.stem_width))
        self.assertNotIn("embedding", out)
        self.assertFalse(torch.isnan(out["features"]).any())
        self.assertEqual(out["features"].ndim, 4)

        out = n.masked_encoding_retention(x, mask, torch.zeros(1, 1, 1, n.stem_width), return_keys="embedding")
        self.assertNotIn("features", out)
        self.assertEqual(len(out["embedding"].flatten()), n.embedding_size * batch_size)

        out = n.masked_encoding_retention(x, mask, torch.zeros(1, 1, 1, n.stem_width), return_keys="all")
        self.assertIsNotNone(out["features"])
        self.assertIsNotNone(out["embedding"])

        # Test "no mask" embedding returns the same as simple embedding
        x = torch.ones((batch_size, DEFAULT_NUM_CHANNELS, *size)) * 0.25
        n.eval()
        zero_mask = torch.zeros_like(mask)
        out = n.masked_encoding_retention(x, zero_mask, torch.ones(1, 1, 1, n.stem_width), return_keys="embedding")
        torch.testing.assert_close(out["embedding"], n.embedding(x))

        self.assertTrue(hasattr(n, "block_group_regex"))
        self.assertTrue(hasattr(n, "stem_stride"))
        self.assertTrue(hasattr(n, "stem_width"))

        names = [n for n, _ in n.named_parameters()]
        groups = group_by_regex(names, n.block_group_regex)  # type: ignore[arg-type]
        self.assertGreater(len(groups), 5)
        self.assertLessEqual(len(groups), 50)

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("deit3_t16"),
            ("deit3_reg4_t16"),
            ("flexivit_s16"),
            ("hiera_tiny", False),
            ("hiera_abswin_tiny", False),
            ("hiera_abswin_base_plus_ap", False),
            ("rope_deit3_t16"),
            ("rope_deit3_reg4_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_i_vit_s16_pn_aps_c1"),
            ("rope_vit_reg4_b32"),
            ("rope_vit_reg4_m16_rms_avg"),
            ("rope_vit_reg8_nps_b14_ap"),
            ("rope_vit_so150m_p14_ap"),
            ("rope_vit_reg8_so150m_p14_swiglu_rms_avg"),
            ("simple_vit_b32"),
            ("vit_s32"),
            ("vit_s16_pn"),
            ("vit_reg4_b32"),
            ("vit_reg4_m16_rms_avg"),
            ("vit_so150m_p14_ap"),
            ("vit_reg8_so150m_p14_swiglu_avg"),
            ("vit_parallel_s16_18x2_ls"),
        ]
    )
    def test_pre_training_encoder_omission(self, network_name: str, test_all_features: bool = True) -> None:
        n = registry.net_factory(network_name, DEFAULT_NUM_CHANNELS, 100)
        size = n.default_size

        # self.assertIsInstance(n, MaskedTokenOmissionMixin)
        assert isinstance(n, MaskedTokenOmissionMixin)
        h = size[0] // n.max_stride
        w = size[1] // n.max_stride
        x = torch.rand((1, DEFAULT_NUM_CHANNELS, *size))
        ids_keep = uniform_mask(x.size(0), h, w, mask_ratio=0.75, device=x.device)[1]
        out = n.masked_encoding_omission(x, ids_keep, return_keys="all")
        tokens = out["tokens"]
        embedding = out["embedding"]
        self.assertFalse(torch.isnan(tokens).any())
        self.assertEqual(tokens.ndim, 3)
        self.assertFalse(torch.isnan(embedding).any())
        self.assertEqual(embedding.ndim, 2)
        self.assertEqual(embedding.size(), (1, n.embedding_size))

        if test_all_features is True:
            out = n.masked_encoding_omission(x, ids_keep, return_all_features=True, return_keys="all")
            tokens = out["tokens"]
            embedding = out["embedding"]
            self.assertFalse(torch.isnan(tokens).any())
            self.assertEqual(tokens.ndim, 4)
            self.assertFalse(torch.isnan(embedding).any())
            self.assertEqual(embedding.ndim, 2)
            self.assertEqual(embedding.size(), (1, n.embedding_size))

        out = n.masked_encoding_omission(x, return_keys="all")
        tokens = out["tokens"]
        embedding = out["embedding"]
        self.assertFalse(torch.isnan(tokens).any())
        self.assertEqual(tokens.ndim, 3)
        self.assertFalse(torch.isnan(embedding).any())
        self.assertEqual(embedding.ndim, 2)

        self.assertTrue(hasattr(n, "num_special_tokens"))
        self.assertTrue(hasattr(n, "block_group_regex"))
        self.assertTrue(hasattr(n, "stem_stride"))
        self.assertTrue(hasattr(n, "stem_width"))

        names = [n for n, _ in n.named_parameters()]
        groups = group_by_regex(names, n.block_group_regex)  # type: ignore[arg-type]
        self.assertGreater(len(groups), 5)
        self.assertLess(len(groups), 40)


class TestNonSquareNet(unittest.TestCase):
    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("alexnet"),
            ("biformer_t"),
            ("cait_xxs24"),
            ("cas_vit_xs"),
            ("coat_tiny"),
            ("coat_lite_tiny"),
            ("conv2former_n"),
            ("convmixer_768_32"),
            ("convnext_v1_tiny"),
            ("convnext_v2_atto"),
            ("crossformer_t"),
            ("crossvit_9d", 1, 48, 48),
            ("csp_resnet_50"),
            ("csp_resnext_50"),
            ("csp_darknet_53"),
            ("csp_se_resnet_50"),
            ("darknet_53"),
            ("davit_tiny"),
            ("deit_t16"),
            ("deit3_t16"),
            ("deit3_reg4_t16"),
            ("densenet_121"),
            ("dpn_92"),
            ("edgenext_xxs"),
            ("edgevit_xxs"),
            ("efficientformer_v1_l1"),
            ("efficientformer_v2_s0"),
            ("efficientnet_lite0"),
            ("efficientnet_v1_b0"),
            ("efficientnet_v2_s"),
            ("efficientvim_m1"),
            ("efficientvit_mit_b0"),
            ("efficientvit_mit_l1"),
            ("efficientvit_msft_m0", 2),
            ("fasternet_t0"),
            ("fastvit_t8"),
            ("fastvit_sa12"),
            ("mobileclip_v1_i0"),
            ("mobileclip_v2_i3"),
            ("flexivit_s16"),
            ("focalnet_t_srf"),
            ("ghostnet_v1_0_5"),
            ("ghostnet_v2_1_0"),
            ("groupmixformer_mobile"),
            ("hgnet_v1_tiny"),
            ("hgnet_v2_b0"),
            ("hiera_tiny"),
            ("hiera_abswin_tiny"),
            ("hiera_abswin_base_plus_ap"),
            ("hieradet_tiny"),
            ("hornet_tiny_7x7"),
            ("hornet_tiny_gf"),
            ("iformer_s"),
            ("inception_next_t"),
            ("inception_resnet_v1"),
            ("inception_resnet_v2"),
            ("inception_v3"),
            ("inception_v4"),
            ("levit_128s"),
            ("maxvit_t"),
            ("poolformer_v1_s12"),
            ("poolformer_v2_s12"),
            ("convformer_s18"),
            ("caformer_s18"),
            ("mnasnet_0_5"),
            ("mobilenet_v1_0_25"),
            ("mobilenet_v2_0_25"),
            ("mobilenet_v3_large_0_75"),
            ("mobilenet_v3_small_1_0"),
            ("mobilenet_v4_s", 2),
            ("mobilenet_v4_hybrid_m", 2),
            ("mobileone_s0"),
            ("mobilevit_v1_xxs"),
            ("mobilevit_v2_0_25"),
            ("moganet_xt"),
            ("mvit_v2_t"),
            ("mvit_v2_t_cls"),
            ("nextvit_s"),
            ("nfnet_f0"),
            ("pit_t"),
            ("pvt_v1_t"),
            ("pvt_v2_b0"),
            ("rdnet_t"),
            ("regionvit_t"),
            ("regnet_x_200m"),
            ("regnet_y_200m"),
            ("regnet_z_500m"),
            ("repghost_0_5"),
            ("repvgg_a0"),
            ("repvit_m0_6", 2),
            ("resmlp_12", 1, 0),  # No resize support
            ("resnest_14", 2),
            ("resnet_v1_18"),
            ("resnet_v2_18"),
            ("resnext_50"),
            ("rope_deit3_t16"),
            ("rope_deit3_reg4_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_i_vit_s16_pn_aps_c1"),
            ("rope_vit_reg4_b32"),
            ("rope_vit_reg4_m16_rms_avg"),
            ("rope_vit_reg8_nps_b14_ap", 1, 14, 14),
            ("rope_vit_so150m_p14_ap", 1, 14, 14),
            ("rope_vit_reg8_so150m_p14_swiglu_rms_avg", 1, 14, 14),
            ("se_resnet_v1_18"),
            ("se_resnet_v2_18"),
            ("se_resnext_50"),
            ("sequencer2d_s"),
            ("shufflenet_v1_8"),
            ("shufflenet_v2_0_5"),
            ("simple_vit_b32"),
            ("smt_t"),
            ("squeezenet"),
            ("squeezenext_0_5"),
            ("starnet_esm05"),
            ("swiftformer_xs"),
            ("swin_transformer_v1_t"),
            ("swin_transformer_v2_t"),
            ("swin_transformer_v2_w2_t"),
            ("tiny_vit_5m"),
            ("transnext_micro"),
            ("uniformer_s"),
            ("van_b0"),
            ("vgg_11"),
            ("vgg_reduced_11"),
            ("vit_s32"),
            ("vit_s16_pn"),
            ("vit_reg4_b32"),
            ("vit_reg4_m16_rms_avg"),
            ("vit_so150m_p14_ap", 1, 14, 14),
            ("vit_reg8_so150m_p14_swiglu_avg", 1, 14, 14),
            ("vit_parallel_s16_18x2_ls"),
            ("vit_det_b16"),
            ("vit_sam_b16"),
            ("vovnet_v1_27s"),
            ("vovnet_v2_19"),
            ("wide_resnet_50"),
            ("xception"),
            ("xcit_nano12_p16"),
        ]
    )
    def test_non_square_net(
        self,
        network_name: str,
        batch_size: int = 1,
        size_step: int = 2**5,
        size_offset: int = 2**5,
    ) -> None:
        # Test resize
        n = registry.net_factory(network_name, DEFAULT_NUM_CHANNELS, 100)
        default_size = n.default_size
        if n.square_only is True:
            return

        size = (default_size[0], default_size[1] + size_step)
        n.adjust_size(size)
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(out.numel(), 100 * batch_size)

        size = (default_size[0] + size_step, default_size[1])
        n.adjust_size(size)
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(out.numel(), 100 * batch_size)

        # Test initialization
        size = (default_size[0], default_size[1] + size_offset)
        n = registry.net_factory(network_name, DEFAULT_NUM_CHANNELS, 100, size=size)
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(out.numel(), 100 * batch_size)


class TestDynamicSize(unittest.TestCase):
    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("davit_tiny"),
            ("deit3_t16"),
            ("deit3_reg4_t16"),
            ("flexivit_s16"),
            ("iformer_s"),
            ("rope_deit3_t16"),
            ("rope_deit3_reg4_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_i_vit_s16_pn_aps_c1"),
            ("rope_vit_reg4_b32"),
            ("rope_vit_reg4_m16_rms_avg"),
            ("rope_vit_reg8_nps_b14_ap", 1, 14),
            ("rope_vit_so150m_p14_ap", 1, 14),
            ("rope_vit_reg8_so150m_p14_swiglu_rms_avg", 1, 14),
            ("simple_vit_b32"),
            ("swin_transformer_v1_t"),
            ("swin_transformer_v2_t"),
            ("swin_transformer_v2_w2_t"),
            ("vit_s32"),
            ("vit_s16_pn"),
            ("vit_reg4_b32"),
            ("vit_reg4_m16_rms_avg"),
            ("vit_so150m_p14_ap", 1, 14),
            ("vit_reg8_so150m_p14_swiglu_avg", 1, 14),
            ("vit_parallel_s16_18x2_ls"),
        ]
    )
    def test_dynamic_size(
        self,
        network_name: str,
        batch_size: int = 1,
        size_step: int = 2**5,
    ) -> None:
        n = registry.net_factory(network_name, DEFAULT_NUM_CHANNELS, 100)
        default_size = n.default_size
        n.set_dynamic_size()

        # Test dynamic inference
        size = (default_size[0] + size_step, default_size[1] + size_step)
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(out.numel(), 100 * batch_size)

        if isinstance(n, base.DetectorBackbone):
            out = n.detection_features(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            for stage_name in n.return_stages:
                self.assertFalse(torch.isnan(out[stage_name]).any())


class TestSpecialFunctions(unittest.TestCase):
    def test_vit_sam_weight_import(self) -> None:
        # ViTDet

        # ViT
        vit_det_b16 = registry.net_factory("vit_det_b16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        vit_b16 = registry.net_factory("vit_b16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        vit_det_b16.load_vit_weights(vit_b16.state_dict())

        # DeiT3
        vit_det_b16_ls = registry.net_factory(
            "vit_det_b16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192), config={"layer_scale_init_value": 1e-5}
        )
        deit3_reg4_b16 = registry.net_factory("deit3_reg4_b16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        vit_det_b16_ls.load_vit_weights(deit3_reg4_b16.state_dict())

        # SAM
        vit_sam_b16 = registry.net_factory("vit_sam_b16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))

        # ViT
        vit_b16 = registry.net_factory("vit_b16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        vit_sam_b16.load_vit_weights(vit_b16.state_dict())

        # Simple ViT
        simple_vit_b16 = registry.net_factory("simple_vit_b16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        vit_sam_b16.load_vit_weights(simple_vit_b16.state_dict())

        # ViT with register tokens
        vit_reg4_b16 = registry.net_factory("vit_reg4_b16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        vit_sam_b16.load_vit_weights(vit_reg4_b16.state_dict())

    def test_hieradet_weight_import(self) -> None:
        hiera_abswin_tiny = registry.net_factory("hiera_abswin_tiny", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        hieradet_tiny = registry.net_factory("hieradet_tiny", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))

        hieradet_tiny.load_hiera_weights(hiera_abswin_tiny.state_dict())

    def test_flexivit_proj(self) -> None:
        flexivit_s16 = registry.net_factory("flexivit_s16", DEFAULT_NUM_CHANNELS, 100, size=(160, 160))

        out = flexivit_s16(torch.rand((1, DEFAULT_NUM_CHANNELS, 160, 160)), patch_size=20)
        self.assertEqual(out.numel(), 100)

    def test_flexivit_adjust_patch_size(self) -> None:
        flexivit_s16 = registry.net_factory("flexivit_s16", DEFAULT_NUM_CHANNELS, 100, size=(160, 160))

        flexivit_s16.adjust_patch_size(20)
        self.assertEqual(flexivit_s16.conv_proj.weight.shape[-2:], (20, 20))

        self.assertEqual(flexivit_s16.pos_embedding.size(1), (160 // 20) * (160 // 20))

    def test_flexivit_weight_import(self) -> None:
        # ViT
        flexivit = registry.net_factory("flexivit_s16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        vit = registry.net_factory("vit_s16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        flexivit.load_vit_weights(vit.state_dict())

        # ViT with register tokens
        flexivit = registry.net_factory("flexivit_reg1_s16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        vit = registry.net_factory("vit_reg1_s16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        flexivit.load_vit_weights(vit.state_dict())

        # ViT with AP
        flexivit = registry.net_factory("flexivit_reg8_b14_ap", DEFAULT_NUM_CHANNELS, 100, size=(196, 196))
        vit = registry.net_factory("vit_reg8_b14_ap", DEFAULT_NUM_CHANNELS, 100, size=(196, 196))
        flexivit.load_vit_weights(vit.state_dict())

        # ViT with RMS and LS
        flexivit = registry.net_factory("flexivit_reg1_s16_rms_ls", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        vit = registry.net_factory("vit_reg1_s16_rms_ls", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        flexivit.load_vit_weights(vit.state_dict())

        # DeiT3
        flexivit = registry.net_factory("flexivit_s16_ls", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        vit = registry.net_factory("deit3_s16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        flexivit.load_vit_weights(vit.state_dict())

    def test_rope_flexivit_proj(self) -> None:
        rope_flexivit_s16 = registry.net_factory("rope_flexivit_s16", DEFAULT_NUM_CHANNELS, 100, size=(160, 160))

        out = rope_flexivit_s16(torch.rand((1, DEFAULT_NUM_CHANNELS, 160, 160)), patch_size=20)
        self.assertEqual(out.numel(), 100)

    def test_rope_flexivit_adjust_patch_size(self) -> None:
        rope_flexivit_s16 = registry.net_factory("rope_flexivit_s16", DEFAULT_NUM_CHANNELS, 100, size=(160, 160))

        rope_flexivit_s16.adjust_patch_size(20)
        self.assertEqual(rope_flexivit_s16.conv_proj.weight.shape[-2:], (20, 20))

        self.assertEqual(rope_flexivit_s16.pos_embedding.size(1), (160 // 20) * (160 // 20))
        self.assertEqual(rope_flexivit_s16.rope.pos_embed.size(0), (160 // 20) * (160 // 20))

    def test_rope_flexivit_weight_import(self) -> None:
        # ViT
        flexivit = registry.net_factory("rope_flexivit_s16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        vit = registry.net_factory("rope_vit_s16", DEFAULT_NUM_CHANNELS, 100, size=(192, 192))
        flexivit.load_rope_vit_weights(vit.state_dict())
