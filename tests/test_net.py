# pylint: disable=too-many-lines

import copy
import json
import logging
import unittest

import torch
from parameterized import parameterized

from birder.common.lib import env_bool
from birder.common.masking import uniform_mask
from birder.common.training_utils import group_by_regex
from birder.conf.settings import DEFAULT_NUM_CHANNELS
from birder.model_registry import registry
from birder.net import Hiera
from birder.net import NaFlex_RoPE_ViT
from birder.net import NaFlex_ViT
from birder.net import base
from birder.net.base import MaskedTokenOmissionMixin
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.flexivit import interpolate_proj
from birder.net.vit_moe import Encoder as ViTMoEEncoder

logging.disable(logging.CRITICAL)

NET_TEST_CASES = [
    ("acb_resnet_v1_18"),
    ("dbb_resnet_v1_18"),
    ("alexnet"),
    ("biformer_t"),
    ("cait_xxs24"),
    ("cas_vit_xs"),
    ("coat_tiny"),
    ("coat_lite_tiny"),
    ("conv2former_n"),
    ("convmixer_768_32"),
    ("convnext_v1_atto"),
    ("convnext_v1_iso_small"),
    ("convnext_v2_atto"),
    ("crossformer_t"),
    ("crossvit_9d", True, True, 1, 48),
    ("csp_resnet_50"),
    ("csp_resnext_50"),
    ("csp_darknet_53"),
    ("csp_se_resnet_50"),
    ("cspnext_t"),
    ("cswin_transformer_t"),
    ("darknet_53"),
    ("davit_tiny"),
    ("davit_fl_tiny"),
    ("deit_t16", True),
    ("deit3_t16"),
    ("deit3_reg4_t16"),
    ("densenet_121"),
    ("dpn_92"),
    ("edgenext_xxs"),
    ("edgevit_xxs"),
    ("efficientformer_v1_l1"),
    ("efficientformer_v2_s0"),
    ("efficientmod_xxs"),
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
    ("gc_vit_xxt"),
    ("ghostnet_v1_0_5"),
    ("ghostnet_v2_1_0"),
    ("ghostnet_v3_0_5"),
    ("groupmixformer_mobile"),
    ("hgnet_v1_tiny"),
    ("hgnet_v2_b0"),
    ("hiera_tiny"),
    ("hiera_abswin_tiny"),  # No bfloat16 support
    ("hiera_abswin_base_plus_ap"),  # No bfloat16 support
    ("hieradet_tiny"),
    ("hieradet_d_tiny"),
    ("hornet_tiny_7x7"),
    ("hornet_tiny_gf"),  # No bfloat16 support
    ("iformer_s"),
    ("inception_next_t"),
    ("inception_resnet_v1"),
    ("inception_resnet_v2"),
    ("inception_v3"),
    ("inception_v4"),
    ("levit_128"),
    ("lit_v1_s"),
    ("lit_v1_t"),
    ("lit_v2_s"),
    ("mambaout_femto"),
    ("maxvit_t"),
    ("poolformer_v1_s12"),
    ("poolformer_v2_s12"),
    ("convformer_s18"),
    ("caformer_s18"),
    ("microvit_v1_s1", False, False, 2),
    ("microvit_v2_s1", False, False, 2),
    ("mnasnet_0_5"),
    ("mobilenet_v1_0_25"),
    ("mobilenet_v2_0_25"),
    ("mobilenet_v3_small_1_0"),
    ("mobilenet_v3_large_0_75"),
    ("mobilenet_v4_s", False, False, 2),
    ("mobilenet_v4_hybrid_m", False, False, 2),
    ("mobilenet_v4_hybrid_l", False, False, 2),  # GELU (inplace)
    ("mobileone_s0"),
    ("mobilevit_v1_xxs"),
    ("mobilevit_v2_0_25"),
    ("moganet_xt"),
    ("mvit_v1_s_d16"),
    ("mvit_v2_t"),
    ("mvit_v2_t_cls"),
    ("naflex_rope_vit_t16"),
    ("naflex_vit_t16"),
    ("nextvit_s"),
    ("nfnet_f0"),
    ("pit_t", True, True),
    ("pnasnet_mobile"),
    ("pvt_v1_t"),
    ("pvt_v2_b0"),
    ("rdnet_t"),
    ("regionvit_t", False, True),
    ("regnet_x_200m"),
    ("regnet_y_200m"),
    ("regnet_z_500m"),
    ("repghost_0_5"),
    ("replknet_31b"),
    ("repvgg_a0"),
    ("repvit_m0_6", False, False, 2),
    ("resmlp_12", False, False, 1, 0),  # No resize support
    ("resnest_14", False, False, 2),
    ("resnet_v1_18"),
    ("se_resnet_v1_18"),
    ("resnet_d_50"),
    ("resnet_v2_18"),
    ("se_resnet_v2_18"),
    ("resnext_50"),
    ("se_resnext_50"),
    ("rexnet_1_0", False, False, 2),
    ("rexnet_lite_1_0", True, False, 2),
    ("rope_deit3_t16"),
    ("rope_deit3_reg4_t16"),
    ("rope_flexivit_s16"),
    ("rope_vit_s32"),
    ("rope_vit_b16_qkn_ls"),
    ("rope_vit_b16_nf_swiglu"),
    ("rope_a_vit_s16"),
    ("rope_cs_vit_reg4_s16_nape_ls_c1"),
    ("rope_i_vit_s16_pn_aps_c1"),
    ("rope_vit_reg4_b32"),
    ("rope_vit_reg4_m16_rms_avg"),
    ("rope_vit_reg8_b14_nps_ap", False, False, 1, 14),
    ("rope_vit_so150m_p14_ap", False, False, 1, 14),
    ("rope_vit_reg8_so150m_p14_swiglu_rms_avg", False, False, 1, 14),
    ("rope_vit_s16_soft_moe_32e_4s_avg"),
    ("rope_vit5_reg4_s16"),
    ("rope_vit_moe_t16_4e1s_2k_last1_avg"),
    ("rope_vit_vmoe_vs32_8e_2k_last2s2"),
    ("rope_vit_vmoe_reg1_vs32_8e_2k_last2s2"),
    ("sequencer2d_s"),
    ("shufflenet_v1_8"),
    ("shufflenet_v2_0_5"),
    ("shvit_s1", False, False, 2),
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
    ("unireplknet_a"),
    ("van_b0"),
    ("vgg_11"),
    ("vgg_reduced_11"),
    ("vit_s32"),
    ("vit_s16_pn"),
    ("vit_b16_qkn_ls"),
    ("vit_b16_nf_swiglu"),
    ("vit_reg1_b16_nap_avg"),
    ("vit_reg4_b32"),
    ("vit_reg4_m16_rms_avg"),
    ("vit_so150m_p14_ap", False, False, 1, 14),
    ("vit_reg8_so150m_p14_swiglu_avg", False, False, 1, 14),
    ("vit_s16_soft_moe_32e_4s_avg"),
    ("vit_moe_t16_4e1s1p_2k_last1"),
    ("vit_vmoe_vs32_8e_2k_last2s2"),
    ("vit_vmoe_reg1_vs32_8e_2k_last2s2"),
    ("vit_parallel_s16_18x2_ls"),
    ("vit_det_s16"),
    ("vit_sam_b16"),
    ("vit_windowed_reg4_s14_nps_ls_avg", False, False, 1, 14),
    ("volo_d1"),
    ("vovnet_v1_27s"),
    ("vovnet_v2_19"),
    ("wide_resnet_50"),
    ("xception"),
    ("xcit_nano12_p16"),
]

DETECTION_BACKBONE_CASES = [
    ("acb_resnet_v1_18"),
    ("dbb_resnet_v1_18"),
    ("biformer_t"),
    ("cas_vit_xs"),
    ("coat_tiny"),
    ("coat_lite_tiny"),
    ("conv2former_n"),
    ("convnext_v1_atto"),
    ("convnext_v1_iso_small"),
    ("convnext_v2_atto"),
    ("crossformer_t"),
    ("csp_resnet_50"),
    ("csp_resnext_50"),
    ("csp_darknet_53"),
    ("csp_se_resnet_50"),
    ("cspnext_t"),
    ("cswin_transformer_t"),
    ("darknet_53"),
    ("davit_tiny"),
    ("davit_fl_tiny"),
    ("deit3_t16"),
    ("deit3_reg4_t16"),
    ("densenet_121"),
    ("edgenext_xxs"),
    ("edgevit_xxs"),
    ("efficientformer_v1_l1"),
    ("efficientformer_v2_s0"),
    ("efficientmod_xxs"),
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
    ("gc_vit_xxt"),
    ("ghostnet_v1_0_5"),
    ("ghostnet_v2_1_0"),
    ("ghostnet_v3_0_5"),
    ("groupmixformer_mobile"),
    ("hgnet_v1_tiny"),
    ("hgnet_v2_b0"),
    ("hiera_tiny"),
    ("hiera_abswin_tiny"),
    ("hiera_abswin_base_plus_ap"),
    ("hieradet_tiny"),
    ("hieradet_d_tiny"),
    ("hornet_tiny_7x7"),
    ("hornet_tiny_gf"),
    ("iformer_s"),
    ("inception_next_t"),
    ("inception_resnet_v1"),
    ("inception_resnet_v2"),
    ("inception_v3"),
    ("inception_v4"),
    ("lit_v1_s"),
    ("lit_v1_t"),
    ("lit_v2_s"),
    ("mambaout_femto"),
    ("maxvit_t"),
    ("poolformer_v1_s12"),
    ("poolformer_v2_s12"),
    ("convformer_s18"),
    ("caformer_s18"),
    ("microvit_v1_s1"),
    ("microvit_v2_s1"),
    ("mnasnet_0_5"),
    ("mobilenet_v1_0_25"),
    ("mobilenet_v2_0_25"),
    ("mobilenet_v3_small_1_0"),
    ("mobilenet_v3_large_0_75"),
    ("mobilenet_v4_s"),
    ("mobilenet_v4_hybrid_m"),
    ("mobileone_s0"),
    ("mobilevit_v1_xxs"),
    ("mobilevit_v2_0_25"),
    ("moganet_xt"),
    ("mvit_v1_s_d16"),
    ("mvit_v2_t"),
    ("mvit_v2_t_cls"),
    ("naflex_rope_vit_t16"),
    ("naflex_vit_t16"),
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
    ("replknet_31b"),
    ("repvgg_a0"),
    ("repvit_m0_6"),
    ("resnest_14", 2),
    ("resnet_v1_18"),
    ("se_resnet_v1_18"),
    ("resnet_d_50"),
    ("resnet_v2_18"),
    ("se_resnet_v2_18"),
    ("resnext_50"),
    ("se_resnext_50"),
    ("rexnet_1_0", 2),
    ("rexnet_lite_1_0", 2),
    ("rope_deit3_t16"),
    ("rope_deit3_reg4_t16"),
    ("rope_flexivit_s16"),
    ("rope_vit_s32"),
    ("rope_vit_b16_qkn_ls"),
    ("rope_a_vit_s16"),
    ("rope_cs_vit_reg4_s16_nape_ls_c1"),
    ("rope_i_vit_s16_pn_aps_c1"),
    ("rope_vit_reg4_b32"),
    ("rope_vit_reg4_m16_rms_avg"),
    ("rope_vit_reg8_b14_nps_ap"),
    ("rope_vit_so150m_p14_ap"),
    ("rope_vit_reg8_so150m_p14_swiglu_rms_avg"),
    ("rope_vit_s16_soft_moe_32e_4s_avg"),
    ("rope_vit5_reg4_s16"),
    ("shufflenet_v1_8"),
    ("shufflenet_v2_0_5"),
    ("shvit_s1"),
    ("smt_t"),
    ("squeezenext_0_5"),
    ("starnet_esm05"),
    ("swiftformer_xs"),
    ("swin_transformer_v1_t"),
    ("swin_transformer_v2_t"),
    ("tiny_vit_5m"),
    ("transnext_micro"),
    ("uniformer_s"),
    ("unireplknet_a"),
    ("van_b0"),
    ("vgg_11"),
    ("vgg_reduced_11"),
    ("vit_s32"),
    ("vit_s16_pn"),
    ("vit_b16_qkn_ls"),
    ("vit_reg1_b16_nap_avg"),
    ("vit_reg4_b32"),
    ("vit_reg4_m16_rms_avg"),
    ("vit_so150m_p14_ap"),
    ("vit_reg8_so150m_p14_swiglu_avg"),
    ("vit_s16_soft_moe_32e_4s_avg"),
    ("vit_parallel_s16_18x2_ls"),
    ("vit_det_b16"),
    ("vit_sam_b16"),
    ("vit_windowed_reg4_s14_nps_ls_avg"),
    ("vovnet_v1_27s"),
    ("vovnet_v2_19"),
    ("wide_resnet_50"),
    ("xception"),
    ("xcit_nano12_p16", 1, True),
]

DYNAMIC_SIZE_CASES = [
    ("davit_tiny"),
    ("davit_fl_tiny"),
    ("deit_t16"),
    ("deit3_t16"),
    ("deit3_reg4_t16"),
    ("flexivit_s16"),
    ("gc_vit_xxt"),
    ("iformer_s"),
    ("lit_v1_s"),
    ("lit_v1_t"),
    ("mvit_v1_s_d16"),
    ("naflex_rope_vit_t16"),
    ("naflex_vit_t16"),
    ("rope_deit3_t16"),
    ("rope_deit3_reg4_t16"),
    ("rope_flexivit_s16"),
    ("rope_vit_s32"),
    ("rope_vit_b16_qkn_ls"),
    ("rope_a_vit_s16"),
    ("rope_cs_vit_reg4_s16_nape_ls_c1"),
    ("rope_i_vit_s16_pn_aps_c1"),
    ("rope_vit_reg4_b32"),
    ("rope_vit_reg4_m16_rms_avg"),
    ("rope_vit_reg8_b14_nps_ap", 1, 14),
    ("rope_vit_so150m_p14_ap", 1, 14),
    ("rope_vit_reg8_so150m_p14_swiglu_rms_avg", 1, 14),
    ("rope_vit_s16_soft_moe_32e_4s_avg"),
    ("rope_vit5_reg4_s16"),
    ("rope_vit_moe_t16_4e1s_2k_last1_avg"),
    ("rope_vit_vmoe_vs32_8e_2k_last2s2"),
    ("rope_vit_vmoe_reg1_vs32_8e_2k_last2s2"),
    ("simple_vit_b32"),
    ("swin_transformer_v1_t"),
    ("swin_transformer_v2_t"),
    ("swin_transformer_v2_w2_t"),
    ("vit_s32"),
    ("vit_s16_pn"),
    ("vit_b16_qkn_ls"),
    ("vit_reg1_b16_nap_avg"),
    ("vit_reg4_b32"),
    ("vit_reg4_m16_rms_avg"),
    ("vit_so150m_p14_ap", 1, 14),
    ("vit_reg8_so150m_p14_swiglu_avg", 1, 14),
    ("vit_s16_soft_moe_32e_4s_avg"),
    ("vit_moe_t16_4e1s1p_2k_last1"),
    ("vit_vmoe_vs32_8e_2k_last2s2"),
    ("vit_vmoe_reg1_vs32_8e_2k_last2s2"),
    ("vit_parallel_s16_18x2_ls"),
    ("vit_det_s16"),
    ("vit_sam_b16"),
    ("vit_windowed_reg4_s14_nps_ls_avg"),
    ("volo_d1"),
]

META_UNSUPPORTED_NETS = {
    "transnext_micro",
}

NAFLEX_TEST_CASES = [
    ("naflex_rope_vit_t16"),
    ("naflex_vit_t16"),
]


class TestBase(unittest.TestCase):
    def test_make_divisible(self) -> None:
        self.assertEqual(base.make_divisible(25, 6), 24)

    def test_stochastic_depth_rates(self) -> None:
        self.assertListEqual(base.stochastic_depth_rates(0.2, 0), [])
        self.assertListEqual(base.stochastic_depth_rates(0.2, 1), [0.0])

        rates = base.stochastic_depth_rates(0.2, 3)
        torch.testing.assert_close(torch.tensor(rates), torch.linspace(0.0, 0.2, steps=3))

        rates_without_endpoint = base.stochastic_depth_rates(0.2, 3, endpoint=False)
        torch.testing.assert_close(torch.tensor(rates_without_endpoint), torch.linspace(0.0, 0.2, steps=4)[:-1])

    def test_staged_stochastic_depth_rates(self) -> None:
        staged_rates = base.staged_stochastic_depth_rates(0.4, [2, 1, 2])

        self.assertListEqual([len(rates) for rates in staged_rates], [2, 1, 2])
        self.assertListEqual([rate for rates in staged_rates for rate in rates], base.stochastic_depth_rates(0.4, 5))

        staged_rates_without_endpoint = base.staged_stochastic_depth_rates(0.4, [2, 1, 2], endpoint=False)
        self.assertListEqual(
            [rate for rates in staged_rates_without_endpoint for rate in rates],
            base.stochastic_depth_rates(0.4, 5, endpoint=False),
        )

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

        self.assertEqual(base_net.flatten_features(torch.rand((2, 3, 4, 5))).size(), (2, 20, 3))
        with self.assertRaises(RuntimeError):
            base_net.flatten_features(torch.rand((2, 10)))

        # Strip everything outside the forward_features path
        base_net.strip_for_forward_features()
        self.assertIsInstance(base_net.features, torch.nn.Identity)
        self.assertIsInstance(base_net.classifier, torch.nn.Identity)
        self.assertIsInstance(base_net.body, torch.nn.Linear)

    def test_base_net_mlp_head(self) -> None:
        base_net = base.BaseNet(DEFAULT_NUM_CHANNELS, num_classes=2, config={"mlp_head": True}, size=(128, 128))
        classifier = base_net.create_classifier(embed_dim=10)
        self.assertIsInstance(classifier, torch.nn.Sequential)

        classifier = base_net.create_classifier(embed_dim=10, mlp_head=False)
        self.assertIsInstance(classifier, torch.nn.Linear)

    def test_config_override_does_not_mutate_registered_config(self) -> None:
        class RegisteredNet(base.BaseNet):  # pylint: disable=abstract-method
            config = {"name": "registered", "nested": {"value": 1}}

        net = RegisteredNet(DEFAULT_NUM_CHANNELS, num_classes=2, config={"name": "override"})
        net.config["nested"]["value"] = 2
        next_net = RegisteredNet(DEFAULT_NUM_CHANNELS, num_classes=2)

        self.assertEqual(net.config["name"], "override")
        self.assertEqual(next_net.config["name"], "registered")
        self.assertEqual(next_net.config["nested"]["value"], 1)
        self.assertEqual(RegisteredNet.config["nested"]["value"], 1)


class TestNet(unittest.TestCase):
    @parameterized.expand(NET_TEST_CASES)  # type: ignore[untyped-decorator]
    def test_net(
        self,
        network_name: str,
        skip_embedding: bool = False,
        non_standard_features: bool = False,
        batch_size: int = 1,
        size_step: int = 2**5,
    ) -> None:
        n = registry.net_factory(network_name, 100)
        size = n.default_size

        self.assertIsInstance(n.feature_dim, int)
        self.assertGreater(n.feature_dim, 0)

        # Ensure config is serializable
        _ = json.dumps(n.config)

        # Test network
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(out.numel(), 100 * batch_size)
        self.assertTrue(torch.isfinite(out).all())

        if skip_embedding is False:
            embedding = n.embedding(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size))).flatten()
            self.assertEqual(len(embedding), n.embedding_size * batch_size)

        n.eval()
        inputs = torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size))
        with torch.inference_mode():
            features = n.forward_features(inputs)
            embedding_from_features = n.embedding_from_features(features)

            embedding = n.embedding(inputs)

            if non_standard_features is False:
                self.assertTrue(hasattr(n, "max_stride"))
                self.assertIsInstance(n.max_stride, int)
                self.assertGreater(n.max_stride, 0)

                # Padding can shift the final feature grid by one cell in either direction
                feature_height = size[0] // n.max_stride
                feature_width = size[1] // n.max_stride
                valid_feature_sizes = {
                    (feature_height + h_offset) * (feature_width + w_offset)
                    for h_offset in (-1, 0, 1)
                    for w_offset in (-1, 0, 1)
                    if feature_height + h_offset > 0 and feature_width + w_offset > 0
                }
                visual_features = n.flatten_features(features, include_special_tokens=False)
                self.assertIn(visual_features.size(1), valid_feature_sizes)

        n.train()
        if isinstance(embedding_from_features, torch.Tensor):
            torch.testing.assert_close(embedding_from_features, embedding)

        if non_standard_features is False:
            features = n.forward_features(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            self.assertTrue(torch.isfinite(features).all())
            self.assertEqual(features.size(0), batch_size)

            flat_features = n.flatten_features(features)
            self.assertTrue(torch.isfinite(flat_features).all())
            self.assertEqual(flat_features.size(0), batch_size)
            self.assertEqual(flat_features.size(2), n.feature_dim)

            visual_features = n.flatten_features(features, include_special_tokens=False)
            self.assertTrue(torch.isfinite(visual_features).all())
            self.assertEqual(visual_features.size(0), batch_size)
            self.assertEqual(visual_features.size(2), n.feature_dim)
            self.assertLessEqual(visual_features.size(1), flat_features.size(1))
            if hasattr(n, "num_special_tokens") is True:
                self.assertEqual(visual_features.size(1), flat_features.size(1) - n.num_special_tokens)

        else:
            features = n.forward_features(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            with self.assertRaises(RuntimeError):
                n.flatten_features(features)

        # Test TorchScript support
        if n.scriptable is True:
            torch.jit.script(n)
        else:
            n.eval()
            torch.jit.trace(n, example_inputs=torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            n.train()

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

        # Ensure model is copyable
        n_copy = copy.deepcopy(n)
        self.assertIsNotNone(n_copy)

    @parameterized.expand(NET_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_net_meta(
        self,
        network_name: str,
        _skip_embedding: bool = False,
        _non_standard_features: bool = False,
        _batch_size: int = 1,
        _size_step: int = 2**5,
    ) -> None:
        if network_name in META_UNSUPPORTED_NETS:
            self.skipTest(f"{network_name} does not support meta initialization")

        with torch.device("meta"):
            meta_net = registry.net_factory(network_name, 10)

        non_meta_tensors = [
            f"parameter '{name}': {parameter.device}"
            for name, parameter in meta_net.named_parameters()
            if parameter.is_meta is False
        ]
        non_meta_tensors.extend(
            f"buffer '{name}': {buffer.device}" for name, buffer in meta_net.named_buffers() if buffer.is_meta is False
        )
        self.assertListEqual(non_meta_tensors, [])

    @parameterized.expand(NET_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_net_backward(
        self,
        network_name: str,
        _skip_embedding: bool = False,
        _non_standard_features: bool = False,
        batch_size: int = 1,
        size_step: int = 2**5,
    ) -> None:
        n = registry.net_factory(network_name, 100)
        size = n.default_size

        # Make sure adjust_size doesn't set any gradients
        size = (size[0] + size_step, size[1] + size_step)
        n.adjust_size(size)
        for name, param in n.named_parameters():
            self.assertIsNone(param.grad, msg=f"{network_name} adjust_size set grad for {name}")
            self.assertIsNone(param.grad_fn, msg=f"{network_name} adjust_size tracked grad for {name}")

        # Make sure forward sets valid gradients
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        loss = out.sum()
        loss.backward()
        for name, param in n.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"{network_name} missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all().item(), msg=f"{network_name} non-finite grad for {name}")

        n.zero_grad()

        # Make sure reparameterization doesn't set any gradients
        if base.reparameterize_available(n) is True:
            n.reparameterize_model()
            out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            for name, param in n.named_parameters():
                self.assertIsNone(param.grad, msg=f"{network_name} reparameterize_model set grad for {name}")
                self.assertIsNone(param.grad_fn, msg=f"{network_name} reparameterize_model tracked grad for {name}")

    @parameterized.expand(NET_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_strip_for_forward_features(
        self,
        network_name: str,
        _skip_embedding: bool = False,
        _non_standard_features: bool = False,
        batch_size: int = 1,
        _size_step: int = 2**5,
    ) -> None:
        def collect_tensors(value: object) -> list[torch.Tensor]:
            if isinstance(value, torch.Tensor):
                return [value]
            if isinstance(value, dict):
                return [tensor for item in value.values() for tensor in collect_tensors(item)]
            if isinstance(value, (list, tuple)):
                return [tensor for item in value for tensor in collect_tensors(item)]

            return []

        n = registry.net_factory(network_name, 100)
        inputs = torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *n.default_size))
        n.eval()
        with torch.inference_mode():
            expected = n.forward_features(inputs)

        n.strip_for_forward_features()
        with torch.inference_mode():
            actual = n.forward_features(inputs)

        torch.testing.assert_close(actual, expected)

        n.train()
        outputs = n.forward_features(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *n.default_size)))
        tensors = collect_tensors(outputs)
        self.assertGreater(len(tensors), 0)
        loss = tensors[0].sum()
        for tensor in tensors[1:]:
            loss = loss + tensor.sum()

        loss.backward()
        for name, param in n.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"{network_name} missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all().item(), msg=f"{network_name} non-finite grad for {name}")

    @parameterized.expand(NET_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_net_pt2(
        self,
        network_name: str,
        _skip_embedding: bool = False,
        _non_standard_features: bool = False,
        _batch_size: int = 1,
        _size_step: int = 2**5,
    ) -> None:
        n = registry.net_factory(network_name, 100)
        n.eval()
        size = n.default_size

        # Test PT2
        batch_dim = torch.export.Dim.DYNAMIC
        with torch.no_grad():
            torch.export.export(n, (torch.randn(2, DEFAULT_NUM_CHANNELS, *size),), dynamic_shapes={"x": {0: batch_dim}})

    # @parameterized.expand(NET_TEST_CASES)  # type: ignore[untyped-decorator]
    # @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    # def test_net_bfloat16(
    #     self,
    #     network_name: str,
    #     _skip_embedding: bool = False,
    #     _non_standard_features: bool = False,
    #     batch_size: int = 1,
    #     _size_step: int = 2**5,
    # ) -> None:
    #     n = registry.net_factory(network_name, 100)
    #     n.eval()
    #     size = n.default_size

    #     # Test modified dtype
    #     n.to(torch.bfloat16)
    #     out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size), dtype=torch.bfloat16))
    #     self.assertEqual(out.numel(), 100 * batch_size)

    @parameterized.expand(DETECTION_BACKBONE_CASES)  # type: ignore[untyped-decorator]
    def test_detection_backbone(
        self,
        network_name: str,
        batch_size: int = 1,
        allow_equal_stages: bool = False,
    ) -> None:
        n = registry.net_factory(network_name, 100)
        size = n.default_size
        n.strip_for_detection_features()

        self.assertIsInstance(n.max_stride, int)
        self.assertEqual(len(n.return_channels), len(n.return_stages))
        out = n.detection_features(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        last_features = out[n.return_stages[-1]]
        self.assertAlmostEqual(last_features.shape[-2], size[0] // n.max_stride, delta=1)
        self.assertAlmostEqual(last_features.shape[-1], size[1] // n.max_stride, delta=1)
        for i, stage_name in enumerate(n.return_stages):
            self.assertIn(stage_name, out)
            self.assertEqual(out[stage_name].shape[1], n.return_channels[i])
            self.assertTrue(torch.isfinite(out[stage_name]).all())

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

    @parameterized.expand(DETECTION_BACKBONE_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_detection_backbone_backward(
        self,
        network_name: str,
        batch_size: int = 1,
        _allow_equal_stages: bool = False,
    ) -> None:
        n = registry.net_factory(network_name, 100)
        n.strip_for_detection_features()

        out = n.detection_features(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *n.default_size)))
        loss = sum(feature.sum() for feature in out.values())
        loss.backward()

        for name, param in n.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"{network_name} missing grad for {name}")

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("hiera_tiny",),
            ("hiera_abswin_tiny",),
            ("hiera_abswin_base_plus_ap"),
        ]
    )
    def test_pre_training_encoder_hiera(self, network_name: str) -> None:
        n = registry.net_factory(network_name, 100)
        size = n.default_size

        # self.assertIsInstance(n, Hiera)
        assert isinstance(n, Hiera)

        x = torch.rand((1, DEFAULT_NUM_CHANNELS, *size))

        mask = uniform_mask(1, n.mask_spatial_shape[0], n.mask_spatial_shape[1], mask_ratio=0.6, device=x.device)[0]
        outs, mask = n.masked_encoding(x, mask)

        for out in outs:
            self.assertTrue(torch.isfinite(out).all())

        self.assertEqual(outs[-1].size(-1), n.feature_dim)
        self.assertTrue(torch.isfinite(mask).all())

        self.assertTrue(hasattr(n, "block_group_regex"))
        self.assertTrue(hasattr(n, "stem_stride"))
        self.assertTrue(hasattr(n, "stem_width"))

        names = [n for n, _ in n.named_parameters()]
        groups = group_by_regex(names, n.block_group_regex)
        self.assertGreater(len(groups), 5)
        self.assertLess(len(groups), 40)

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("acb_resnet_v1_18"),
            ("dbb_resnet_v1_18"),
            ("biformer_t"),
            ("cait_xxs24"),
            ("conv2former_n"),
            ("convnext_v1_atto"),
            ("convnext_v1_iso_small"),
            ("convnext_v2_atto"),
            ("crossformer_t"),
            ("davit_tiny"),
            ("davit_fl_tiny"),
            ("deit3_t16"),
            ("deit3_reg4_t16"),
            ("densenet_121"),
            ("efficientnet_lite0"),
            ("efficientnet_v1_b0"),
            ("efficientnet_v2_s"),
            ("fastvit_t8"),
            ("fastvit_sa12"),
            ("mobileclip_v1_i0"),
            ("mobileclip_v2_i3"),
            ("flexivit_s16"),
            ("focalnet_t_srf"),
            ("gc_vit_xxt"),
            ("hieradet_tiny"),
            ("hieradet_d_tiny"),
            ("iformer_s"),
            ("inception_next_t"),
            ("mambaout_femto"),
            ("maxvit_t"),
            ("poolformer_v1_s12"),
            ("poolformer_v2_s12"),
            ("convformer_s18"),
            ("caformer_s18"),
            ("mobilenet_v4_s", 2),
            ("mobilenet_v4_hybrid_m", 2),
            ("mobileone_s0"),
            ("mobilevit_v1_xxs"),
            ("mobilevit_v2_0_25"),
            ("moganet_xt"),
            ("mvit_v2_t"),
            ("mvit_v2_t_cls"),
            ("naflex_rope_vit_t16"),
            ("naflex_vit_t16"),
            ("nextvit_s"),
            ("nfnet_f0"),
            ("pvt_v2_b0"),
            ("rdnet_t"),
            ("regnet_x_200m"),
            ("regnet_y_200m"),
            ("regnet_z_500m"),
            ("replknet_31b"),
            ("repvit_m0_6", 2),
            ("resnest_14", 2),
            ("resnet_v1_18"),
            ("resnet_v2_18"),
            ("resnext_50"),
            ("rexnet_1_0", 2),
            ("rope_deit3_t16"),
            ("rope_deit3_reg4_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_vit_b16_qkn_ls"),
            ("rope_vit_b16_nf_swiglu"),
            ("rope_a_vit_s16"),
            ("rope_cs_vit_reg4_s16_nape_ls_c1"),
            ("rope_i_vit_s16_pn_aps_c1"),
            ("rope_vit_reg4_b32"),
            ("rope_vit_reg4_m16_rms_avg"),
            ("rope_vit_reg8_b14_nps_ap"),
            ("rope_vit_so150m_p14_ap"),
            ("rope_vit_reg8_so150m_p14_swiglu_rms_avg"),
            ("rope_vit_s16_soft_moe_32e_4s_avg"),
            ("rope_vit5_reg4_s16"),
            ("rope_vit_moe_t16_4e1s_2k_last1_avg"),
            ("rope_vit_vmoe_vs32_8e_2k_last2s2"),
            ("rope_vit_vmoe_reg1_vs32_8e_2k_last2s2"),
            ("simple_vit_s32"),
            ("smt_t"),
            ("swin_transformer_v1_t"),
            ("swin_transformer_v2_t"),
            ("swin_transformer_v2_w2_t"),
            ("uniformer_s"),
            ("unireplknet_a"),
            ("vit_s32"),
            ("vit_s16_pn"),
            ("vit_b16_qkn_ls"),
            ("vit_b16_nf_swiglu"),
            ("vit_reg1_b16_nap_avg"),
            ("vit_reg4_b32"),
            ("vit_reg4_m16_rms_avg"),
            ("vit_so150m_p14_ap"),
            ("vit_reg8_so150m_p14_swiglu_avg"),
            ("vit_s16_soft_moe_32e_4s_avg"),
            ("vit_moe_t16_4e1s1p_2k_last1"),
            ("vit_vmoe_vs32_8e_2k_last2s2"),
            ("vit_vmoe_reg1_vs32_8e_2k_last2s2"),
            ("vit_parallel_s16_18x2_ls"),
            ("wide_resnet_50"),
            ("xcit_nano12_p16"),
        ]
    )
    def test_pre_training_encoder_retention(self, network_name: str, batch_size: int = 1) -> None:
        n = registry.net_factory(network_name, 100)
        size = n.default_size

        # self.assertIsInstance(n, MaskedTokenRetentionMixin)
        assert isinstance(n, MaskedTokenRetentionMixin)
        h = size[0] // n.max_stride
        w = size[1] // n.max_stride
        x = torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size))
        mask = uniform_mask(batch_size, h, w, mask_ratio=0.6, device=x.device)[0]

        # Test retention
        out = n.masked_encoding_retention(x, mask, return_keys="features")
        self.assertTrue(torch.isfinite(out["features"]).all())
        self.assertEqual(out["features"].ndim, 4)
        self.assertEqual(out["features"].size(1), n.feature_dim)

        # Test substitution
        out = n.masked_encoding_retention(x, mask, torch.zeros(1, 1, 1, n.stem_width))
        self.assertNotIn("embedding", out)
        self.assertTrue(torch.isfinite(out["features"]).all())
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
            ("naflex_rope_vit_t16"),
            ("naflex_vit_t16"),
            ("rope_deit3_t16"),
            ("rope_deit3_reg4_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_vit_b16_qkn_ls"),
            ("rope_vit_b16_nf_swiglu"),
            ("rope_a_vit_s16"),
            ("rope_cs_vit_reg4_s16_nape_ls_c1"),
            ("rope_i_vit_s16_pn_aps_c1"),
            ("rope_vit_reg4_b32"),
            ("rope_vit_reg4_m16_rms_avg"),
            ("rope_vit_reg8_b14_nps_ap"),
            ("rope_vit_so150m_p14_ap"),
            ("rope_vit_reg8_so150m_p14_swiglu_rms_avg"),
            ("rope_vit_s16_soft_moe_32e_4s_avg"),
            ("rope_vit5_reg4_s16"),
            ("rope_vit_moe_t16_4e1s_2k_last1_avg"),
            ("rope_vit_vmoe_vs32_8e_2k_last2s2"),
            ("rope_vit_vmoe_reg1_vs32_8e_2k_last2s2"),
            ("simple_vit_b32"),
            ("vit_s32"),
            ("vit_s16_pn"),
            ("vit_b16_qkn_ls"),
            ("vit_b16_nf_swiglu"),
            ("vit_reg1_b16_nap_avg"),
            ("vit_reg4_b32"),
            ("vit_reg4_m16_rms_avg"),
            ("vit_so150m_p14_ap"),
            ("vit_reg8_so150m_p14_swiglu_avg"),
            ("vit_s16_soft_moe_32e_4s_avg"),
            ("vit_moe_t16_4e1s1p_2k_last1"),
            ("vit_vmoe_vs32_8e_2k_last2s2"),
            ("vit_vmoe_reg1_vs32_8e_2k_last2s2"),
            ("vit_parallel_s16_18x2_ls"),
        ]
    )
    def test_pre_training_encoder_omission(self, network_name: str, test_all_features: bool = True) -> None:
        n = registry.net_factory(network_name, 100)
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
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertEqual(tokens.ndim, 3)
        self.assertEqual(tokens.size(-1), n.feature_dim)
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertEqual(embedding.ndim, 2)
        self.assertEqual(embedding.size(), (1, n.embedding_size))

        if test_all_features is True:
            out = n.masked_encoding_omission(x, ids_keep, return_all_features=True, return_keys="all")
            tokens = out["tokens"]
            embedding = out["embedding"]
            self.assertTrue(torch.isfinite(tokens).all())
            self.assertEqual(tokens.ndim, 4)
            self.assertEqual(tokens.size(-2), n.feature_dim)
            self.assertTrue(torch.isfinite(embedding).all())
            self.assertEqual(embedding.ndim, 2)
            self.assertEqual(embedding.size(), (1, n.embedding_size))

        out = n.masked_encoding_omission(x, return_keys="all")
        tokens = out["tokens"]
        embedding = out["embedding"]
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertEqual(tokens.ndim, 3)
        self.assertEqual(tokens.size(-1), n.feature_dim)
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertEqual(embedding.ndim, 2)

        self.assertTrue(hasattr(n, "num_special_tokens"))
        self.assertTrue(hasattr(n, "block_group_regex"))
        self.assertTrue(hasattr(n, "stem_stride"))
        self.assertTrue(hasattr(n, "stem_width"))

        names = [n for n, _ in n.named_parameters()]
        groups = group_by_regex(names, n.block_group_regex)  # type: ignore[arg-type]
        self.assertGreater(len(groups), 5)
        self.assertLess(len(groups), 40)

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("cait_xxs24"),
            ("conv2former_n"),
            ("convnext_v1_atto"),
            ("convnext_v2_atto"),
            ("cswin_transformer_t"),
            ("davit_tiny"),
            ("davit_fl_tiny"),
            ("deit3_t16"),
            ("efficientnet_v1_b0"),
            ("efficientnet_v2_s"),
            ("flexivit_s16"),
            ("focalnet_t_srf"),
            ("hiera_abswin_tiny"),
            ("hieradet_tiny"),
            ("maxvit_t"),
            ("moganet_xt"),
            ("naflex_rope_vit_t16"),
            ("naflex_vit_t16"),
            ("nfnet_f0"),
            ("poolformer_v1_s12"),
            ("rdnet_t"),
            ("regnet_x_200m"),
            ("replknet_31b"),
            ("rope_deit3_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_vit5_reg4_s16"),
            ("rope_vit_moe_t16_4e1s_2k_last1_avg"),
            ("rope_vit_vmoe_vs32_8e_2k_last2s2"),
            ("simple_vit_s32"),
            ("swin_transformer_v1_t"),
            ("swin_transformer_v2_t"),
            ("vit_s32"),
            ("vit_sam_b16"),
            ("vit_moe_t16_4e1s1p_2k_last1"),
            ("vit_vmoe_vs32_8e_2k_last2s2"),
            ("vit_parallel_s16_18x2_ls"),
            ("wide_resnet_50"),
        ]
    )
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_grad_checkpointing(self, network_name: str) -> None:
        batch_size = 1
        n = registry.net_factory(network_name, 100)
        size = n.default_size
        x = torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size))

        n.eval()
        with torch.no_grad():
            expected = n(x)

        for use_reentrant in (True, False):
            with self.subTest(use_reentrant=use_reentrant):
                n.zero_grad()
                n.set_grad_checkpointing(segments=4, use_reentrant=use_reentrant)
                out = n(x)

                # Verify forward with checkpointing
                self.assertEqual(out.size(), expected.size())
                self.assertTrue(torch.allclose(out, expected))
                self.assertEqual(out.numel(), 100 * batch_size)
                self.assertTrue(torch.isfinite(out).all().item(), msg=f"{network_name} non-finite output")

                # Check grads
                loss = out.sum()
                loss.backward()
                for name, param in n.named_parameters():
                    self.assertIsNotNone(param.grad, msg=f"{network_name} missing grad for {name}")
                    self.assertTrue(
                        torch.isfinite(param.grad).all().item(), msg=f"{network_name} non-finite grad for {name}"
                    )


class TestNonSquareNet(unittest.TestCase):
    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("acb_resnet_v1_18"),
            ("dbb_resnet_v1_18"),
            ("alexnet"),
            ("biformer_t"),
            ("cait_xxs24"),
            ("cas_vit_xs"),
            ("coat_tiny"),
            ("coat_lite_tiny"),
            ("conv2former_n"),
            ("convmixer_768_32"),
            ("convnext_v1_atto"),
            ("convnext_v1_iso_small"),
            ("convnext_v2_atto"),
            ("crossformer_t"),
            ("crossvit_9d", 1, 48, 48),
            ("csp_resnet_50"),
            ("csp_resnext_50"),
            ("csp_darknet_53"),
            ("csp_se_resnet_50"),
            ("cspnext_t"),
            ("darknet_53"),
            ("davit_tiny"),
            ("davit_fl_tiny"),
            ("deit_t16"),
            ("deit3_t16"),
            ("deit3_reg4_t16"),
            ("densenet_121"),
            ("dpn_92"),
            ("edgenext_xxs"),
            ("edgevit_xxs"),
            ("efficientformer_v1_l1"),
            ("efficientformer_v2_s0"),
            ("efficientmod_xxs"),
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
            ("gc_vit_xxt"),
            ("ghostnet_v1_0_5"),
            ("ghostnet_v2_1_0"),
            ("ghostnet_v3_0_5"),
            ("groupmixformer_mobile"),
            ("hgnet_v1_tiny"),
            ("hgnet_v2_b0"),
            ("hiera_tiny"),
            ("hiera_abswin_tiny"),
            ("hiera_abswin_base_plus_ap"),
            ("hieradet_tiny"),
            ("hieradet_d_tiny"),
            ("hornet_tiny_7x7"),
            ("hornet_tiny_gf"),
            ("iformer_s"),
            ("inception_next_t"),
            ("inception_resnet_v1"),
            ("inception_resnet_v2"),
            ("inception_v3"),
            ("inception_v4"),
            ("levit_128s"),
            ("lit_v1_s"),
            ("lit_v1_t"),
            ("lit_v2_s"),
            ("mambaout_femto"),
            ("maxvit_t"),
            ("poolformer_v1_s12"),
            ("poolformer_v2_s12"),
            ("convformer_s18"),
            ("caformer_s18"),
            ("microvit_v1_s1", 2),
            ("microvit_v2_s1", 2),
            ("mnasnet_0_5"),
            ("mobilenet_v1_0_25"),
            ("mobilenet_v2_0_25"),
            ("mobilenet_v3_small_1_0"),
            ("mobilenet_v3_large_0_75"),
            ("mobilenet_v4_s", 2),
            ("mobilenet_v4_hybrid_m", 2),
            ("mobileone_s0"),
            ("mobilevit_v1_xxs"),
            ("mobilevit_v2_0_25"),
            ("moganet_xt"),
            ("mvit_v1_s_d16"),
            ("mvit_v2_t"),
            ("mvit_v2_t_cls"),
            ("naflex_rope_vit_t16"),
            ("naflex_vit_t16"),
            ("nextvit_s"),
            ("nfnet_f0"),
            ("pit_t"),
            ("pnasnet_mobile"),
            ("pvt_v1_t"),
            ("pvt_v2_b0"),
            ("rdnet_t"),
            ("regionvit_t"),
            ("regnet_x_200m"),
            ("regnet_y_200m"),
            ("regnet_z_500m"),
            ("repghost_0_5"),
            ("replknet_31b"),
            ("repvgg_a0"),
            ("repvit_m0_6", 2),
            ("resmlp_12", 1, 0),  # No resize support
            ("resnest_14", 2),
            ("resnet_v1_18"),
            ("se_resnet_v1_18"),
            ("resnet_d_50"),
            ("resnet_v2_18"),
            ("se_resnet_v2_18"),
            ("resnext_50"),
            ("se_resnext_50"),
            ("rexnet_1_0", 2),
            ("rexnet_lite_1_0", 2),
            ("rope_deit3_t16"),
            ("rope_deit3_reg4_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_vit_b16_qkn_ls"),
            ("rope_a_vit_s16"),
            ("rope_cs_vit_reg4_s16_nape_ls_c1"),
            ("rope_i_vit_s16_pn_aps_c1"),
            ("rope_vit_reg4_b32"),
            ("rope_vit_reg4_m16_rms_avg"),
            ("rope_vit_reg8_b14_nps_ap", 1, 14, 14),
            ("rope_vit_so150m_p14_ap", 1, 14, 14),
            ("rope_vit_reg8_so150m_p14_swiglu_rms_avg", 1, 14, 14),
            ("rope_vit_s16_soft_moe_32e_4s_avg"),
            ("rope_vit5_reg4_s16"),
            ("rope_vit_moe_t16_4e1s_2k_last1_avg"),
            ("rope_vit_vmoe_vs32_8e_2k_last2s2"),
            ("rope_vit_vmoe_reg1_vs32_8e_2k_last2s2"),
            ("sequencer2d_s"),
            ("shufflenet_v1_8"),
            ("shufflenet_v2_0_5"),
            ("shvit_s1", 2),
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
            ("unireplknet_a"),
            ("van_b0"),
            ("vgg_11"),
            ("vgg_reduced_11"),
            ("vit_s32"),
            ("vit_s16_pn"),
            ("vit_b16_qkn_ls"),
            ("vit_reg1_b16_nap_avg"),
            ("vit_reg4_b32"),
            ("vit_reg4_m16_rms_avg"),
            ("vit_so150m_p14_ap", 1, 14, 14),
            ("vit_reg8_so150m_p14_swiglu_avg", 1, 14, 14),
            ("vit_s16_soft_moe_32e_4s_avg"),
            ("vit_moe_t16_4e1s1p_2k_last1"),
            ("vit_vmoe_vs32_8e_2k_last2s2"),
            ("vit_vmoe_reg1_vs32_8e_2k_last2s2"),
            ("vit_parallel_s16_18x2_ls"),
            ("vit_det_b16"),
            ("vit_sam_b16"),
            ("vit_windowed_reg4_s14_nps_ls_avg", 1, 14, 14),
            ("volo_d1"),
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
        n = registry.net_factory(network_name, 100)
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
        n = registry.net_factory(network_name, 100, size=size)
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(out.numel(), 100 * batch_size)


class TestDynamicSize(unittest.TestCase):
    @parameterized.expand(DYNAMIC_SIZE_CASES)  # type: ignore[untyped-decorator]
    def test_dynamic_size(
        self,
        network_name: str,
        batch_size: int = 1,
        size_step: int = 2**5,
    ) -> None:
        n = registry.net_factory(network_name, 100)
        default_size = n.default_size
        n.set_dynamic_size()

        # Test dynamic inference
        size = (default_size[0] + size_step, default_size[1] + size_step)
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        self.assertEqual(out.numel(), 100 * batch_size)

        if isinstance(n, base.DetectorBackbone):
            out = n.detection_features(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
            for stage_name in n.return_stages:
                self.assertTrue(torch.isfinite(out[stage_name]).all())

    @parameterized.expand(DYNAMIC_SIZE_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_dynamic_size_backward(
        self,
        network_name: str,
        batch_size: int = 1,
        size_step: int = 2**5,
    ) -> None:
        n = registry.net_factory(network_name, 100)
        default_size = n.default_size
        n.set_dynamic_size()

        size = (default_size[0] + size_step, default_size[1] + size_step)
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)))
        loss = out.sum()
        loss.backward()
        for name, param in n.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"{network_name} missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all().item(), msg=f"{network_name} non-finite grad for {name}")

    # @parameterized.expand(DYNAMIC_SIZE_CASES)  # type: ignore[untyped-decorator]
    # @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    # def test_dynamic_size_pt2(
    #     self,
    #     network_name: str,
    #     _batch_size: int = 1,
    #     _size_step: int = 2**5,
    # ) -> None:
    #     n = registry.net_factory(network_name, 100)
    #     n.eval()
    #     n.set_dynamic_size()
    #     size = n.default_size

    #     # Test PT2
    #     batch_dim = torch.export.Dim.DYNAMIC
    #     height_dim = torch.export.Dim.DYNAMIC
    #     width_dim = torch.export.Dim.DYNAMIC
    #     with torch.no_grad():
    #         torch.export.export(
    #             n,
    #             (torch.randn(2, DEFAULT_NUM_CHANNELS, *size),),
    #             dynamic_shapes={"x": {0: batch_dim, 2: height_dim, 3: width_dim}},
    #         )


class TestCudaAdjustSize(unittest.TestCase):
    @parameterized.expand(NET_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_adjust_size_cuda(
        self,
        network_name: str,
        _skip_embedding: bool = False,
        _non_standard_features: bool = False,
        _batch_size: int = 1,
        size_step: int = 2**5,
    ) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        n = registry.net_factory(network_name, 10).to(device)
        size = (n.default_size[0] + size_step, n.default_size[1] + size_step)
        n.adjust_size(size)

        for name, param in n.named_parameters():
            self.assertEqual(param.device, device, msg=f"{network_name} param on {param.device} for {name}")
        for name, buffer in n.named_buffers():
            self.assertEqual(buffer.device, device, msg=f"{network_name} buffer on {buffer.device} for {name}")


class TestNaFlex(unittest.TestCase):
    @parameterized.expand(NAFLEX_TEST_CASES)  # type: ignore[untyped-decorator]
    def test_image_input_parity(self, network_name: str) -> None:
        config = {
            "patch_size": 4,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 16,
            "mlp_dim": 32,
            "drop_path_rate": 0.0,
        }
        n = registry.net_factory(network_name, 10, config=config, size=(16, 16))
        n.eval()

        # self.assertIsInstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))
        assert isinstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))

        for image_size in ((8, 12), (12, 8)):
            with self.subTest(image_size=image_size):
                image = torch.rand((2, DEFAULT_NUM_CHANNELS, *image_size))
                patches = torch.nn.functional.unfold(image, kernel_size=4, stride=4).transpose(1, 2)
                grid_sizes = torch.tensor([[image_size[0] // 4, image_size[1] // 4]]).expand(image.size(0), -1)
                valid_mask = torch.ones(patches.shape[:2], dtype=torch.bool)

                with torch.inference_mode():
                    expected_features = n.forward_features(image)
                    expected_embedding = n.embedding(image)
                    features = n.forward_features(patches, grid_sizes=grid_sizes, valid_mask=valid_mask)
                    embedding = n.embedding(patches, grid_sizes, valid_mask)

                torch.testing.assert_close(features, expected_features, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(embedding, expected_embedding, atol=1e-6, rtol=1e-5)

    @parameterized.expand(NAFLEX_TEST_CASES)  # type: ignore[untyped-decorator]
    def test_patch_resampling(self, network_name: str) -> None:
        config = {
            "patch_size": 4,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 16,
            "mlp_dim": 32,
            "drop_path_rate": 0.0,
            "naflex_patch_resampling": True,
        }
        n = registry.net_factory(network_name, 10, config=config, size=(16, 16))
        n.eval()

        # self.assertIsInstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))
        assert isinstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))

        resampled_patch_size = 2
        patches = torch.rand(
            (2, 6, DEFAULT_NUM_CHANNELS * resampled_patch_size * resampled_patch_size), requires_grad=True
        )
        grid_sizes = torch.tensor([[2, 2], [2, 3]])
        valid_mask = torch.tensor([[True, True, True, True, False, False], [True, True, True, True, True, True]])
        mask = torch.tensor([[False, True, False, True, True, True], [True, False, True, False, True, False]])

        stacked = n.forward_features(
            patches,
            return_input_embedding=True,
            grid_sizes=grid_sizes,
            valid_mask=valid_mask,
        )
        input_embedding, encoded_features = stacked.unbind(dim=-1)
        with torch.no_grad():
            resampled_weight = interpolate_proj(n.conv_proj.weight, resampled_patch_size)
            expected_input_embedding = torch.nn.functional.linear(  # pylint: disable=not-callable
                patches,
                resampled_weight.flatten(1),
                n.conv_proj.bias,
            )
            expected_input_embedding = expected_input_embedding.masked_fill(~valid_mask.unsqueeze(-1), 0)

        torch.testing.assert_close(input_embedding[:, n.num_special_tokens :], expected_input_embedding)

        masked_result = n.masked_encoding_retention(
            patches,
            mask,
            return_keys="all",
            grid_sizes=grid_sizes,
            valid_mask=valid_mask,
        )
        loss = encoded_features.square().mean() + masked_result["embedding"].square().mean()
        loss.backward()

        self.assertEqual(n.conv_proj.weight.size(-2), 4)
        self.assertEqual(n.conv_proj.weight.size(-1), 4)
        self.assertIsNotNone(n.conv_proj.weight.grad)
        self.assertTrue(torch.isfinite(n.conv_proj.weight.grad).all().item())
        self.assertGreater(torch.count_nonzero(n.conv_proj.weight.grad).item(), 0)

    @parameterized.expand(NAFLEX_TEST_CASES)  # type: ignore[untyped-decorator]
    def test_masked_encoding_retention_parity(self, network_name: str) -> None:
        config = {
            "patch_size": 4,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 16,
            "mlp_dim": 32,
            "drop_path_rate": 0.0,
        }
        n = registry.net_factory(network_name, 10, config=config, size=(16, 16))
        n.eval()

        # self.assertIsInstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))
        assert isinstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))

        image = torch.rand((2, DEFAULT_NUM_CHANNELS, 8, 12))
        patches = torch.nn.functional.unfold(image, kernel_size=4, stride=4).transpose(1, 2)
        grid_sizes = torch.tensor([[2, 3]]).expand(image.size(0), -1)
        valid_mask = torch.ones(patches.shape[:2], dtype=torch.bool)
        mask = torch.tensor([[False, True, False, True, False, True], [True, False, True, False, True, False]])
        mask_token = torch.rand((1, 1, 1, n.stem_width))

        with torch.inference_mode():
            expected = n.masked_encoding_retention(image, mask, mask_token=mask_token, return_keys="all")
            result = n.masked_encoding_retention(
                patches,
                mask,
                mask_token=mask_token,
                return_keys="all",
                grid_sizes=grid_sizes,
                valid_mask=valid_mask,
            )

            zero_mask_result = n.masked_encoding_retention(
                patches,
                torch.zeros_like(mask),
                mask_token=mask_token,
                return_keys="embedding",
                grid_sizes=grid_sizes,
                valid_mask=valid_mask,
            )
            embedding = n.embedding(patches, grid_sizes, valid_mask)

        torch.testing.assert_close(result["features"], expected["features"].flatten(2), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result["embedding"], expected["embedding"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(zero_mask_result["embedding"], embedding)

    @parameterized.expand(NAFLEX_TEST_CASES)  # type: ignore[untyped-decorator]
    def test_masked_encoding_retention_valid_mask(self, network_name: str) -> None:
        config = {
            "patch_size": 4,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 16,
            "mlp_dim": 32,
            "drop_path_rate": 0.0,
        }
        n = registry.net_factory(network_name, 10, config=config, size=(16, 16))
        n.eval()

        # self.assertIsInstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))
        assert isinstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))

        patch_dim = DEFAULT_NUM_CHANNELS * 4 * 4
        patches = torch.rand((2, 6, patch_dim))
        grid_sizes = torch.tensor([[2, 2], [2, 3]])
        valid_mask = torch.tensor([[True, True, True, True, False, False], [True, True, True, True, True, True]])
        mask = torch.tensor([[False, True, False, True, True, True], [True, False, True, False, True, False]])
        mask_token = torch.rand((1, 1, 1, n.stem_width))

        with torch.inference_mode():
            result = n.masked_encoding_retention(
                patches,
                mask,
                mask_token=mask_token,
                return_keys="all",
                grid_sizes=grid_sizes,
                valid_mask=valid_mask,
            )

            changed_patches = patches.clone()
            changed_patches[0, 4:] = torch.rand_like(changed_patches[0, 4:])
            changed_mask = mask.clone()
            changed_mask[0, 4:] = ~changed_mask[0, 4:]
            changed_result = n.masked_encoding_retention(
                changed_patches,
                changed_mask,
                mask_token=mask_token,
                return_keys="all",
                grid_sizes=grid_sizes,
                valid_mask=valid_mask,
            )

            for sample_idx, seq_len in enumerate((4, 6)):
                sample_valid_mask = torch.ones((1, seq_len), dtype=torch.bool)
                expected = n.masked_encoding_retention(
                    patches[sample_idx : sample_idx + 1, :seq_len],
                    mask[sample_idx : sample_idx + 1, :seq_len],
                    mask_token=mask_token,
                    return_keys="all",
                    grid_sizes=grid_sizes[sample_idx : sample_idx + 1],
                    valid_mask=sample_valid_mask,
                )
                torch.testing.assert_close(
                    result["features"][sample_idx : sample_idx + 1, :, :seq_len],
                    expected["features"],
                )
                torch.testing.assert_close(result["embedding"][sample_idx : sample_idx + 1], expected["embedding"])

        torch.testing.assert_close(changed_result["features"], result["features"])
        torch.testing.assert_close(changed_result["embedding"], result["embedding"])
        self.assertEqual(torch.count_nonzero(result["features"][0, :, 4:]).item(), 0)

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            (network_name, pos_embed_resize_mode)
            for network_name in NAFLEX_TEST_CASES
            for pos_embed_resize_mode in ("grid_sample", "interpolate")
        ]
    )
    def test_grid_sizes(self, network_name: str, pos_embed_resize_mode: str) -> None:
        config = {
            "patch_size": 4,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 16,
            "mlp_dim": 32,
            "drop_path_rate": 0.0,
            "class_token": False,
            "naflex_pos_embed_resize_mode": pos_embed_resize_mode,
        }
        n = registry.net_factory(network_name, 10, config=config, size=(16, 16))
        n.eval()

        # self.assertIsInstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))
        assert isinstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))

        patches = torch.rand((1, 6, DEFAULT_NUM_CHANNELS * 4 * 4)).expand(2, -1, -1)
        grid_sizes = torch.tensor([[2, 3], [3, 2]])
        valid_mask = torch.ones((2, 6), dtype=torch.bool)

        with torch.inference_mode():
            features = n.forward_features(patches, grid_sizes=grid_sizes, valid_mask=valid_mask)
            for sample_idx in range(patches.size(0)):
                expected = n.forward_features(
                    patches[sample_idx : sample_idx + 1],
                    grid_sizes=grid_sizes[sample_idx : sample_idx + 1],
                    valid_mask=valid_mask[sample_idx : sample_idx + 1],
                )
                torch.testing.assert_close(features[sample_idx : sample_idx + 1], expected)

        self.assertFalse(torch.allclose(features[0], features[1]))

    @parameterized.expand(NAFLEX_TEST_CASES)  # type: ignore[untyped-decorator]
    def test_return_input_embedding(self, network_name: str) -> None:
        config = {
            "patch_size": 4,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 16,
            "mlp_dim": 32,
            "drop_path_rate": 0.0,
        }
        n = registry.net_factory(network_name, 10, config=config, size=(16, 16))
        n.eval()

        # self.assertIsInstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))
        assert isinstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))

        patches = torch.rand((2, 6, DEFAULT_NUM_CHANNELS * 4 * 4))
        grid_sizes = torch.tensor([[2, 2], [2, 3]])
        valid_mask = torch.tensor([[True, True, True, True, False, False], [True, True, True, True, True, True]])

        with torch.inference_mode():
            features = n.forward_features(patches, grid_sizes=grid_sizes, valid_mask=valid_mask)
            stacked = n.forward_features(
                patches,
                return_input_embedding=True,
                grid_sizes=grid_sizes,
                valid_mask=valid_mask,
            )

        input_embedding, encoded_features = stacked.unbind(dim=-1)
        torch.testing.assert_close(encoded_features, features)
        self.assertEqual(stacked.size(), (*features.shape, 2))
        self.assertEqual(torch.count_nonzero(input_embedding[0, n.num_special_tokens + 4 :]).item(), 0)

    @parameterized.expand(NAFLEX_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_backward(self, network_name: str) -> None:
        config = {
            "patch_size": 4,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 16,
            "mlp_dim": 32,
            "drop_path_rate": 0.0,
            "class_token": False,
        }
        n = registry.net_factory(network_name, 10, config=config, size=(16, 16))

        # self.assertIsInstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))
        assert isinstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))

        patches = torch.rand((2, 6, DEFAULT_NUM_CHANNELS * 4 * 4), requires_grad=True)
        grid_sizes = torch.tensor([[2, 2], [2, 3]])
        valid_mask = torch.tensor([[True, True, True, True, False, False], [True, True, True, True, True, True]])

        embedding = n.embedding(patches, grid_sizes, valid_mask)
        embedding.square().sum().backward()

        self.assertIsNotNone(patches.grad)
        assert patches.grad is not None
        self.assertTrue(torch.isfinite(patches.grad).all())
        self.assertGreater(torch.count_nonzero(patches.grad[valid_mask]).item(), 0)
        self.assertEqual(torch.count_nonzero(patches.grad[valid_mask.logical_not()]).item(), 0)

    @parameterized.expand(NAFLEX_TEST_CASES)  # type: ignore[untyped-decorator]
    def test_valid_mask(self, network_name: str) -> None:
        config = {
            "patch_size": 4,
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 32,
            "mlp_dim": 64,
            "drop_path_rate": 0.0,
        }
        n = registry.net_factory(network_name, 10, config=config, size=(16, 16))
        n.eval()

        # self.assertIsInstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))
        assert isinstance(n, (NaFlex_RoPE_ViT, NaFlex_ViT))

        patch_dim = DEFAULT_NUM_CHANNELS * 4 * 4
        patches = torch.rand((2, 6, patch_dim))
        grid_sizes = torch.tensor([[2, 2], [2, 3]])
        valid_mask = torch.tensor([[True, True, True, True, False, False], [True, True, True, True, True, True]])

        with torch.inference_mode():
            features = n.forward_features(patches, grid_sizes=grid_sizes, valid_mask=valid_mask)
            embedding = n.embedding(patches, grid_sizes, valid_mask)

            changed_patches = patches.clone()
            changed_patches[0, 4:] = torch.rand_like(changed_patches[0, 4:])
            changed_features = n.forward_features(
                changed_patches,
                grid_sizes=grid_sizes,
                valid_mask=valid_mask,
            )
            changed_embedding = n.embedding(changed_patches, grid_sizes, valid_mask)

            torch.testing.assert_close(changed_features, features)
            torch.testing.assert_close(changed_embedding, embedding)

            for sample_idx, seq_len in enumerate((4, 6)):
                sample_valid_mask = torch.ones((1, seq_len), dtype=torch.bool)
                expected_features = n.forward_features(
                    patches[sample_idx : sample_idx + 1, :seq_len],
                    grid_sizes=grid_sizes[sample_idx : sample_idx + 1],
                    valid_mask=sample_valid_mask,
                )
                expected_embedding = n.embedding(
                    patches[sample_idx : sample_idx + 1, :seq_len],
                    grid_sizes[sample_idx : sample_idx + 1],
                    valid_mask=sample_valid_mask,
                )

                valid_seq_len = n.num_special_tokens + seq_len
                torch.testing.assert_close(
                    features[sample_idx : sample_idx + 1, :valid_seq_len],
                    expected_features,
                )
                torch.testing.assert_close(embedding[sample_idx : sample_idx + 1], expected_embedding)


class TestSpecialFunctions(unittest.TestCase):
    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("deit_t16"),
            ("deit3_t16"),
            ("flexivit_s16"),
            ("naflex_rope_vit_t16"),
            ("naflex_vit_t16"),
            ("rope_deit3_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_vit5_reg4_s16"),
            ("rope_vit_moe_t16_4e1s_2k_last1_avg"),
            ("rope_vit_vmoe_vs32_8e_2k_last2s2"),
            ("rope_vit_vmoe_reg1_vs32_8e_2k_last2s2"),
            ("simple_vit_s32"),
            ("vit_s32"),
            ("vit_b16_qkn_ls"),
            ("vit_b16_nf_swiglu"),
            ("vit_s16_soft_moe_32e_4s_avg"),
            ("vit_moe_t16_4e1s1p_2k_last1"),
            ("vit_vmoe_vs32_8e_2k_last2s2"),
            ("vit_vmoe_reg1_vs32_8e_2k_last2s2"),
            ("vit_parallel_s16_18x2_ls"),
        ]
    )
    def test_vit_forward_features_return_input_embedding(self, network_name: str) -> None:
        n = registry.net_factory(network_name, 10)
        n.eval()
        x = torch.rand((1, DEFAULT_NUM_CHANNELS, *n.default_size))

        features = n.forward_features(x)
        stacked = n.forward_features(x, return_input_embedding=True)  # type: ignore[call-arg]

        self.assertEqual(stacked.size(), (*features.shape, 2))

        _input_embedding, encoded_features = stacked.unbind(dim=-1)
        self.assertTrue(torch.allclose(encoded_features, features))

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("deit_t16"),
            ("flexivit_s16"),
            ("naflex_rope_vit_t16"),
            ("naflex_vit_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_vit5_reg4_s16"),
            ("rope_vit_moe_t16_4e1s_2k_last1_avg"),
            ("rope_vit_vmoe_vs32_8e_2k_last2s2"),
            ("rope_vit_vmoe_reg1_vs32_8e_2k_last2s2"),
            ("simple_vit_s32"),
            ("vit_s32"),
            ("vit_moe_t16_4e1s1p_2k_last1"),
            ("vit_vmoe_vs32_8e_2k_last2s2"),
        ]
    )
    def test_vit_forward_features_attention_mask(self, network_name: str) -> None:
        n = registry.net_factory(network_name, 10)
        n.eval()
        x = torch.rand((1, DEFAULT_NUM_CHANNELS, *n.default_size))

        features = n.forward_features(x)
        attn_mask = torch.ones((features.size(1), features.size(1)), dtype=torch.bool)
        attn_mask[:, features.size(1) // 2 :] = False
        masked_features = n.forward_features(x, attn_mask=attn_mask)  # type: ignore[call-arg]

        self.assertEqual(masked_features.size(), features.size())
        self.assertFalse(torch.allclose(masked_features, features))

    def test_vit_encoder_out_indices(self) -> None:
        n = registry.net_factory("vit_s16", 10)
        n.eval()
        tokens = torch.rand([1, 64, n.embedding_size])

        all_features = n.encoder.forward_features(tokens)
        self.assertEqual(len(all_features), len(n.encoder.block))

        num_layers = len(n.encoder.block)
        out_indices = [0, num_layers // 2, num_layers - 1]

        subset_features = n.encoder.forward_features(tokens, out_indices=out_indices)
        self.assertEqual(len(subset_features), len(out_indices))
        for i, out_idx in enumerate(out_indices):
            self.assertTrue(torch.allclose(subset_features[i], all_features[out_idx]))

        empty_features = n.encoder.forward_features(tokens, out_indices=[])
        self.assertEqual(len(empty_features), 0)

    def test_vit_grad_checkpointing_and_attention_are_exclusive(self) -> None:
        n = registry.net_factory("vit_s16", 10)

        n.encoder.set_need_attn(True)
        with self.assertRaises(ValueError):
            n.set_grad_checkpointing()

        n.encoder.set_need_attn(False)
        n.set_grad_checkpointing()
        with self.assertRaises(ValueError):
            n.encoder.set_need_attn(True)

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("vit_vmoe_vs32_8e_2k_last2s2"),
            ("rope_vit_vmoe_vs32_8e_2k_last2s2"),
        ]
    )
    def test_vit_moe_grad_checkpointing_with_training_output(self, network_name: str) -> None:
        baseline = registry.net_factory(
            network_name,
            2,
            config={
                "patch_size": 16,
                "num_layers": 3,
                "num_heads": 2,
                "hidden_dim": 8,
                "mlp_dim": 16,
                "drop_path_rate": 0.0,
                "moe_last_n_layers": 1,
                "moe_num_experts": 2,
                "router_noise_std": 0.0,
                "mlp_head": False,
            },
            size=(32, 32),
        )
        with torch.no_grad():
            baseline.classifier.weight.fill_(1.0)

        baseline.train()
        checkpointed = copy.deepcopy(baseline)
        checkpointed.set_grad_checkpointing(segments=3)
        inputs = torch.rand((8, DEFAULT_NUM_CHANNELS, 32, 32))

        with torch.no_grad():
            expected_logits, expected_moe_training_output = baseline(inputs, return_moe_training_output=True)

        logits, moe_training_output = checkpointed(inputs, return_moe_training_output=True)
        self.assertTrue(torch.allclose(logits, expected_logits))
        self.assertEqual(
            set(moe_training_output),
            {"auxiliary_loss", "g_shard_loss", "importance_loss", "load_loss", "expert_loads"},
        )
        self.assertEqual(moe_training_output["expert_loads"].numel(), 0)
        for key, expected in expected_moe_training_output.items():
            self.assertTrue(torch.allclose(moe_training_output[key], expected), msg=key)

        (logits.sum() + moe_training_output["auxiliary_loss"]).backward()
        for param in (checkpointed.conv_proj.weight, checkpointed.encoder.block[2].mlp.router.gate.weight):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all().item())

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("vit_moe_t16_4e1s1p_2k_last1", 2),
            ("rope_vit_moe_t16_4e1s_2k_last1_avg", 3),
        ]
    )
    def test_vit_moe_token_choice(self, network_name: str, num_routed_experts: int) -> None:
        n = registry.net_factory(
            network_name,
            2,
            config={
                "num_layers": 4,
                "num_heads": 2,
                "hidden_dim": 8,
                "mlp_dim": 16,
                "drop_path_rate": 0.0,
                "moe_expert_width": 4,
                "moe_last_n_layers": 2,
                "router_bias_update_speed": 0.1,
            },
            size=(32, 32),
        )
        self.assertFalse(n.moe_spec.has_auxiliary_loss)
        self.assertTrue(n.moe_spec.requires_expert_bias_update)

        moe_block_indices = (2, 3)
        initial_expert_bias = torch.arange(num_routed_experts - 1, -1, -1, dtype=torch.float32) * 2.0
        with torch.no_grad():
            n.classifier.weight.fill_(1.0)
            for block_idx in moe_block_indices:
                moe_ffn = n.encoder.block[block_idx].mlp
                self.assertEqual(moe_ffn.num_routed_experts, num_routed_experts)
                self.assertEqual(len(moe_ffn.special_token_experts), n.num_special_tokens)
                moe_ffn.router.expert_bias.copy_(initial_expert_bias)

        n.train()
        checkpointed = copy.deepcopy(n)
        checkpointed.set_grad_checkpointing(segments=4, use_reentrant=True)
        inputs = torch.rand((2, DEFAULT_NUM_CHANNELS, 32, 32))

        with torch.no_grad():
            expected_output, expected_moe_training_output = n(inputs, return_moe_training_output=True)

        output, moe_training_output = checkpointed(inputs, return_moe_training_output=True)
        torch.testing.assert_close(output, expected_output)
        for key, expected in expected_moe_training_output.items():
            torch.testing.assert_close(moe_training_output[key], expected)

        expected_expert_loads = torch.zeros((len(moe_block_indices), num_routed_experts), dtype=torch.int64)
        expected_expert_loads[:, :2] = 8
        torch.testing.assert_close(moe_training_output["expert_loads"], expected_expert_loads)
        torch.testing.assert_close(moe_training_output["auxiliary_loss"], torch.tensor(0.0))

        (output.sum() + moe_training_output["auxiliary_loss"]).backward()
        for block_idx in moe_block_indices:
            router_grad = checkpointed.encoder.block[block_idx].mlp.router.gate.weight.grad
            self.assertIsNotNone(router_grad)
            self.assertTrue(torch.isfinite(router_grad).all().item())

        checkpointed.update_moe_expert_biases(moe_training_output["expert_loads"])
        expert_load = expected_expert_loads[0]
        update_direction = torch.sign(expert_load.float().mean() - expert_load)
        expected_expert_bias = initial_expert_bias + update_direction * 0.1
        expected_expert_bias.sub_(expected_expert_bias.mean())
        for block_idx in moe_block_indices:
            torch.testing.assert_close(
                checkpointed.encoder.block[block_idx].mlp.router.expert_bias, expected_expert_bias
            )

    def test_vit_moe_expert_choice(self) -> None:
        n = registry.net_factory(
            "vit_moe_t16_4e1s_2c_last1_avg",
            2,
            config={
                "num_layers": 4,
                "num_heads": 2,
                "hidden_dim": 8,
                "mlp_dim": 16,
                "drop_path_rate": 0.0,
                "moe_expert_width": 4,
                "moe_last_n_layers": 2,
            },
            size=(32, 32),
        )
        self.assertFalse(n.moe_spec.has_auxiliary_loss)
        self.assertFalse(n.moe_spec.requires_expert_bias_update)

        moe_block_indices = (2, 3)
        for block_idx in moe_block_indices:
            router = n.encoder.block[block_idx].mlp.router
            self.assertEqual(router.capacity_factor, 2.0)

        with torch.no_grad():
            n.classifier.weight.fill_(1.0)

        n.train()
        checkpointed = copy.deepcopy(n)
        checkpointed.set_grad_checkpointing(segments=4, use_reentrant=True)
        inputs = torch.rand((2, DEFAULT_NUM_CHANNELS, 32, 32))

        with torch.no_grad():
            expected_output, expected_moe_training_output = n(inputs, return_moe_training_output=True)

        output, moe_training_output = checkpointed(inputs, return_moe_training_output=True)
        torch.testing.assert_close(output, expected_output)
        for key, expected in expected_moe_training_output.items():
            torch.testing.assert_close(moe_training_output[key], expected)

        self.assertEqual(moe_training_output["expert_loads"].size(), (0, 0))
        for key in ("auxiliary_loss", "g_shard_loss", "importance_loss", "load_loss"):
            torch.testing.assert_close(moe_training_output[key], torch.tensor(0.0), msg=key)

        output.sum().backward()
        for block_idx in moe_block_indices:
            router_grad = checkpointed.encoder.block[block_idx].mlp.router.gate.weight.grad
            self.assertIsNotNone(router_grad)
            self.assertTrue(torch.isfinite(router_grad).all().item())

    def test_vit_moe_encoder_token_mask(self) -> None:
        encoder = ViTMoEEncoder(
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            mlp_dim=16,
            moe_layers=[1],
            dropout=0.0,
            attention_dropout=0.0,
            projection_dropout=0.0,
            dpr=[0.0, 0.0],
            moe_num_experts=2,
            router_noise_std=0.0,
        )
        with torch.no_grad():
            for param in encoder.block[0].parameters():
                param.zero_()
            for param in encoder.block[1].attn.parameters():
                param.zero_()

        inputs = torch.rand((2, 4, 8))
        token_mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
        encoder.eval()
        with torch.inference_mode():
            output = encoder(inputs, token_mask=token_mask)

        torch.testing.assert_close(output[~token_mask], inputs[~token_mask])
        self.assertFalse(torch.allclose(output[token_mask], inputs[token_mask]))

        encoder.train()
        checkpointed = copy.deepcopy(encoder)
        checkpointed.set_grad_checkpointing(segments=2)
        expected_output, expected_moe_training_output = encoder(
            inputs, token_mask=token_mask, return_moe_training_output=True
        )
        output, moe_training_output = checkpointed(inputs, token_mask=token_mask, return_moe_training_output=True)

        torch.testing.assert_close(output, expected_output)
        for key, expected in expected_moe_training_output.items():
            torch.testing.assert_close(moe_training_output[key], expected)

        (output.sum() + moe_training_output["auxiliary_loss"]).backward()
        router_grad = checkpointed.block[1].mlp.router.gate.weight.grad
        self.assertIsNotNone(router_grad)
        self.assertTrue(torch.isfinite(router_grad).all().item())

    def test_rope_vit_encoder_out_indices(self) -> None:
        n = registry.net_factory("rope_vit_s16_avg", 10, size=(128, 128))
        n.eval()
        tokens = torch.rand([1, 64, n.embedding_size])
        rope = n.rope.pos_embed

        all_features = n.encoder.forward_features(tokens, rope)
        self.assertEqual(len(all_features), len(n.encoder.block))

        num_layers = len(n.encoder.block)
        out_indices = [0, num_layers // 2, num_layers - 1]

        subset_features = n.encoder.forward_features(tokens, rope, out_indices=out_indices)
        self.assertEqual(len(subset_features), len(out_indices))
        for i, out_idx in enumerate(out_indices):
            self.assertTrue(torch.allclose(subset_features[i], all_features[out_idx]))

        empty_features = n.encoder.forward_features(tokens, rope, out_indices=[])
        self.assertEqual(len(empty_features), 0)

    def test_vit_sam_weight_import(self) -> None:
        # ViTDet

        # ViT
        vit_det_b16 = registry.net_factory("vit_det_b16", 100, size=(192, 192))
        vit_b16 = registry.net_factory("vit_b16", 100, size=(192, 192))
        vit_det_b16.load_vit_weights(vit_b16.state_dict())

        # DeiT3
        vit_det_b16_ls = registry.net_factory(
            "vit_det_b16", 100, size=(192, 192), config={"layer_scale_init_value": 1e-5}
        )
        deit3_reg4_b16 = registry.net_factory("deit3_reg4_b16", 100, size=(192, 192))
        vit_det_b16_ls.load_vit_weights(deit3_reg4_b16.state_dict())

        # SAM
        vit_sam_b16 = registry.net_factory("vit_sam_b16", 100, size=(192, 192))

        # ViT
        vit_b16 = registry.net_factory("vit_b16", 100, size=(192, 192))
        vit_sam_b16.load_vit_weights(vit_b16.state_dict())

        # Simple ViT
        simple_vit_b16 = registry.net_factory("simple_vit_b16", 100, size=(192, 192))
        vit_sam_b16.load_vit_weights(simple_vit_b16.state_dict())

        # ViT with register tokens
        vit_reg4_b16 = registry.net_factory("vit_reg4_b16", 100, size=(192, 192))
        vit_sam_b16.load_vit_weights(vit_reg4_b16.state_dict())

    def test_vit_windowed_state_dict_compatibility(self) -> None:
        vit_s16 = registry.net_factory("vit_s16", 100, size=(192, 192))
        vit_windowed_s16 = registry.net_factory("vit_windowed_s16", 100, size=(192, 192))

        # Supported plain ViT variants should stay exactly load-compatible.
        vit_windowed_s16.load_state_dict(vit_s16.state_dict())

    def test_hieradet_weight_import(self) -> None:
        hiera_abswin_tiny = registry.net_factory("hiera_abswin_tiny", 100, size=(192, 192))
        hieradet_tiny = registry.net_factory("hieradet_tiny", 100, size=(192, 192))

        hieradet_tiny.load_hiera_weights(hiera_abswin_tiny.state_dict())

    def test_flexivit_proj(self) -> None:
        flexivit_s16 = registry.net_factory("flexivit_s16", 100, size=(160, 160))

        out = flexivit_s16(torch.rand((1, DEFAULT_NUM_CHANNELS, 160, 160)), patch_size=20)
        self.assertEqual(out.numel(), 100)

    def test_flexivit_adjust_patch_size(self) -> None:
        flexivit_s16 = registry.net_factory("flexivit_s16", 100, size=(160, 160))

        flexivit_s16.adjust_patch_size(20)
        self.assertEqual(flexivit_s16.conv_proj.weight.shape[-2:], (20, 20))

        self.assertEqual(flexivit_s16.pos_embedding.size(1), (160 // 20) * (160 // 20))

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("flexivit_s16"),
            ("rope_flexivit_s16"),
        ]
    )
    def test_flexivit_adjust_size_patch_sizes(self, network_name: str) -> None:
        flexivit = registry.net_factory(network_name, 100, size=(240, 240))

        self.assertEqual(flexivit.patch_size_list, [8, 10, 12, 15, 16, 20, 24, 30, 40, 48])

        flexivit.adjust_size((256, 256))

        self.assertEqual(flexivit.patch_size_list, [8, 16, 32])
        for patch_size in flexivit.patch_size_list:
            self.assertEqual(flexivit.size[0] % patch_size, 0)
            self.assertEqual(flexivit.size[1] % patch_size, 0)

    def test_flexivit_weight_import(self) -> None:
        # ViT
        flexivit = registry.net_factory("flexivit_s16", 100, size=(192, 192))
        vit = registry.net_factory("vit_s16", 100, size=(192, 192))
        flexivit.load_vit_weights(vit.state_dict())

        # ViT with register tokens
        flexivit = registry.net_factory("flexivit_reg1_s16", 100, size=(192, 192))
        vit = registry.net_factory("vit_reg1_s16", 100, size=(192, 192))
        flexivit.load_vit_weights(vit.state_dict())

        # ViT with AP
        flexivit = registry.net_factory("flexivit_reg8_b14_ap", 100, size=(196, 196))
        vit = registry.net_factory("vit_reg8_b14_ap", 100, size=(196, 196))
        flexivit.load_vit_weights(vit.state_dict())

        # ViT with RMS and LS
        flexivit = registry.net_factory("flexivit_reg1_s16_rms_ls", 100, size=(192, 192))
        vit = registry.net_factory("vit_reg1_s16_rms_ls", 100, size=(192, 192))
        flexivit.load_vit_weights(vit.state_dict())

        # DeiT3
        flexivit = registry.net_factory("flexivit_s16_ls", 100, size=(192, 192))
        vit = registry.net_factory("deit3_s16", 100, size=(192, 192))
        flexivit.load_vit_weights(vit.state_dict())

    def test_rope_flexivit_proj(self) -> None:
        rope_flexivit_s16 = registry.net_factory("rope_flexivit_s16", 100, size=(160, 160))

        out = rope_flexivit_s16(torch.rand((1, DEFAULT_NUM_CHANNELS, 160, 160)), patch_size=20)
        self.assertEqual(out.numel(), 100)

    def test_rope_flexivit_adjust_patch_size(self) -> None:
        rope_flexivit_s16 = registry.net_factory("rope_flexivit_s16", 100, size=(160, 160))

        rope_flexivit_s16.adjust_patch_size(20)
        self.assertEqual(rope_flexivit_s16.conv_proj.weight.shape[-2:], (20, 20))

        self.assertEqual(rope_flexivit_s16.pos_embedding.size(1), (160 // 20) * (160 // 20))
        self.assertEqual(rope_flexivit_s16.rope.pos_embed.size(0), (160 // 20) * (160 // 20))

    def test_rope_flexivit_weight_import(self) -> None:
        # ViT
        flexivit = registry.net_factory("rope_flexivit_s16", 100, size=(192, 192))
        vit = registry.net_factory("rope_vit_s16", 100, size=(192, 192))
        flexivit.load_rope_vit_weights(vit.state_dict())

    @parameterized.expand(  # type: ignore[untyped-decorator]
        [
            ("deit_t16"),
            ("deit3_t16"),
            ("flexivit_s16"),
            ("naflex_rope_vit_t16"),
            ("naflex_vit_t16"),
            ("rope_deit3_t16"),
            ("rope_flexivit_s16"),
            ("rope_vit_s32"),
            ("rope_vit5_reg4_s16"),
            ("simple_vit_s32"),
            ("vit_s32"),
            ("vit_b16_qkn_ls"),
            ("vit_b16_nf_swiglu"),
            ("vit_moe_t16_4e1s1p_2k_last1"),
            ("vit_vmoe_vs32_8e_2k_last2s2"),
            ("vit_parallel_s16_18x2_ls"),
            ("vit_sam_b16"),
        ]
    )
    def test_set_causal_attention(self, network_name: str) -> None:
        n = registry.net_factory(network_name, 10)
        size = n.default_size
        x = torch.rand((1, DEFAULT_NUM_CHANNELS, *size))

        # Test enabling causal attention
        n.set_causal_attention(True)
        out = n(x)
        self.assertEqual(out.numel(), 10)
        self.assertTrue(torch.isfinite(out).all())

        # Test disabling causal attention
        n.set_causal_attention(False)
        out = n(x)
        self.assertEqual(out.numel(), 10)
        self.assertTrue(torch.isfinite(out).all())

    def test_set_causal_attention_soft_moe(self) -> None:
        n = registry.net_factory("vit_s16_soft_moe_32e_4s_avg", 10)

        with self.assertRaises(ValueError):
            n.set_causal_attention(True)

        n.set_causal_attention(False)

        n = registry.net_factory("rope_vit_s16_soft_moe_32e_4s_avg", 10)

        with self.assertRaises(ValueError):
            n.set_causal_attention(True)

        n.set_causal_attention(False)

    def test_set_causal_attention_expert_choice_moe(self) -> None:
        n = registry.net_factory("vit_moe_t16_4e1s_2c_last1_avg", 10)

        with self.assertRaises(ValueError):
            n.set_causal_attention(True)

        n.set_causal_attention(False)

        n = registry.net_factory("rope_vit_moe_t16_4e1s_2c_last1_avg", 10)

        with self.assertRaises(ValueError):
            n.set_causal_attention(True)

        n.set_causal_attention(False)
