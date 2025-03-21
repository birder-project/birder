import logging
import unittest

import torch

from birder.common.masking import BlockMasking
from birder.model_registry import registry
from birder.net.ssl import byol
from birder.net.ssl import dino_v1
from birder.net.ssl import ibot
from birder.net.ssl import vicreg

logging.disable(logging.CRITICAL)


class TestNetSSL(unittest.TestCase):
    def test_byol(self) -> None:
        batch_size = 2
        backbone = registry.net_factory("resnet_v1_18", 3, 0)
        encoder = byol.BYOLEncoder(backbone, 64, 128)
        net = byol.BYOL(backbone.input_channels, encoder, encoder, 64, 128)

        # Test network
        out = net(torch.rand((batch_size, 3, 96, 96)))
        self.assertFalse(torch.isnan(out).any())

    def test_dino_v1(self) -> None:
        batch_size = 4
        backbone = registry.net_factory("resnet_v2_18", 3, 0)
        dino_head = dino_v1.DINOHead(
            backbone.embedding_size,
            128,
            use_bn=True,
            norm_last_layer=True,
            num_layers=3,
            hidden_dim=512,
            bottleneck_dim=256,
        )
        net = dino_v1.DINO_v1(backbone.input_channels, backbone, dino_head)

        # Test network
        out = net(
            [
                torch.rand((batch_size, 3, 128, 128)),
                torch.rand((batch_size, 3, 128, 128)),
                torch.rand((batch_size, 3, 96, 96)),
                torch.rand((batch_size, 3, 96, 96)),
                torch.rand((batch_size, 3, 96, 96)),
                torch.rand((batch_size, 3, 96, 96)),
            ]
        )
        self.assertFalse(torch.isnan(out).any())

        teacher_out = net([torch.rand((batch_size, 3, 128, 128)), torch.rand((batch_size, 3, 128, 128))])
        dino_loss = dino_v1.DINOLoss(
            128,
            6,
            warmup_teacher_temp=0.2,
            teacher_temp=0.8,
            warmup_teacher_temp_epochs=10,
            num_epochs=100,
            student_temp=0.5,
            center_momentum=0.99,
        )
        loss = dino_loss(out, teacher_out, epoch=2)
        self.assertFalse(torch.isnan(loss).any())

    def test_ibot(self) -> None:
        batch_size = 4
        backbone = registry.net_factory("vit_b32", 3, 0)
        backbone.set_dynamic_size()
        ibot_head = ibot.iBOTHead(
            backbone.embedding_size,
            128,
            norm_last_layer=True,
            num_layers=3,
            hidden_dim=512,
            bottleneck_dim=256,
            patch_out_dim=192,
            shared_head=False,
        )
        net = ibot.iBOT(backbone.input_channels, backbone, ibot_head)

        # Test network
        images = [
            # Global
            torch.rand((batch_size, 3, 128, 128)),
            torch.rand((batch_size, 3, 128, 128)),
            # Local
            torch.rand((batch_size, 3, 96, 96)),
            torch.rand((batch_size, 3, 96, 96)),
            torch.rand((batch_size, 3, 96, 96)),
            torch.rand((batch_size, 3, 96, 96)),
        ]

        mask_generator = BlockMasking((128 // backbone.stem_stride, 128 // backbone.stem_stride), 1, 3, 0.66, 1.5)
        masks = mask_generator(batch_size * 2)

        (embedding_g, features_g) = net(torch.concat(images[:2], dim=0), masks=masks)
        self.assertFalse(torch.isnan(embedding_g).any())
        self.assertFalse(torch.isnan(features_g).any())
        self.assertEqual(features_g.size(), (batch_size * 2, (128 // 32) ** 2, 192))
        self.assertEqual(embedding_g.size(), (batch_size * 2, 128))

        (embedding_l, features_l) = net(torch.concat(images[2:], dim=0), masks=None)
        self.assertFalse(torch.isnan(embedding_l).any())
        self.assertFalse(torch.isnan(features_l).any())
        self.assertEqual(features_l.size(), (batch_size * 4, (96 // 32) ** 2, 192))
        self.assertEqual(embedding_l.size(), (batch_size * 4, 128))

        ibot_loss = ibot.iBOTLoss(
            128,
            192,
            num_global_crops=2,
            num_local_crops=4,
            warmup_teacher_temp=0.1,
            teacher_temp=0.9,
            warmup_teacher_temp2=0.2,
            teacher_temp2=0.99,
            warmup_teacher_temp_epochs=5,
            epochs=100,
            student_temp=0.5,
            center_momentum=0.98,
            center_momentum2=0.97,
            lambda1=0.2,
            lambda2=0.1,
            mim_start_epoch=1,
        )

        loss = ibot_loss.forward(
            embedding_g,
            features_g,
            torch.rand_like(embedding_g),
            torch.rand_like(features_g),
            student_local_embedding=embedding_l,
            student_mask=masks,
            epoch=2,
        )

        self.assertFalse(torch.isnan(loss["all"]).any())
        self.assertFalse(torch.isnan(loss["embedding"]).any())
        self.assertFalse(torch.isnan(loss["features"]).any())

    def test_vicreg(self) -> None:
        batch_size = 4
        backbone = registry.net_factory("resnet_v1_18", 3, 0)
        net = vicreg.VICReg(
            backbone.input_channels,
            backbone,
            mlp_dim=128,
            batch_size=batch_size,
            sim_coeff=0.1,
            std_coeff=0.1,
            cov_coeff=0.1,
        )

        # Test network
        out = net(torch.rand((batch_size, 3, 128, 128)), torch.rand((batch_size, 3, 128, 128)))
        self.assertFalse(torch.isnan(out).any())
