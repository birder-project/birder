import logging
import unittest

import torch

from birder.core import net
from birder.core.net import pretraining

logging.disable(logging.CRITICAL)


class TestNetPretrain(unittest.TestCase):
    def test_fcmae(self) -> None:
        size = pretraining.FCMAE.default_size
        encoder = net.ConvNeXt_v2(3, 2, 0, size=size)
        n = pretraining.FCMAE(encoder, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertIn("loss", out)

    def test_mae_vit(self) -> None:
        size = pretraining.MAE_ViT.default_size
        encoder = net.ViT(3, 2, 0, size=size)
        n = pretraining.MAE_ViT(encoder, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertIn("loss", out)

        # ViTReg encoder
        encoder = net.ViTReg4(3, 2, 0, size=size)
        n = pretraining.MAE_ViT(encoder, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertIn("loss", out)

    def test_simmim(self) -> None:
        size = pretraining.SimMIM.default_size
        encoder = net.Swin_Transformer_v2(3, 2, 0, size=size)
        n = pretraining.SimMIM(encoder, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertIn("loss", out)

        # Swin Transformer v2 w2 encoder
        encoder = net.Swin_Transformer_v2_w2(3, 2, 0, size=size)
        n = pretraining.SimMIM(encoder, size=size)

        out = n(torch.rand((1, 3, size, size)))
        self.assertIn("loss", out)
