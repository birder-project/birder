import logging
import unittest

import torch

from birder.kernels import load_kernel
from birder.net.detection.deformable_detr import multi_scale_deformable_attention

logging.disable(logging.CRITICAL)


class TestKernels(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_deformable_detr(self) -> None:
        device = torch.device("cuda")
        msda = load_kernel.load_msda()
        self.assertIsNotNone(msda)

        value = torch.rand(1, 34000, 8, 32, device=device)
        value_spatial_shapes = torch.tensor(
            [[160, 160], [80, 80], [40, 40], [20, 20]], dtype=torch.int64, device=device
        )
        value_level_start_index = torch.randint(10, (4,), dtype=torch.int64, device=device)
        sampling_locations = torch.rand(1, 34000, 8, 4, 4, 2, device=device)
        attention_weights = torch.rand(1, 34000, 8, 4, 4, device=device)
        im2col_step = 64

        output_kernel = msda.ms_deform_attn_forward(  # type: ignore
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
        )
        self.assertEqual(output_kernel.size(), (1, 34000, 256))

        with torch.amp.autocast("cuda"):
            output_kernel = msda.ms_deform_attn_forward(  # type: ignore
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
            )

        self.assertEqual(output_kernel.size(), (1, 34000, 256))

        output_torch = multi_scale_deformable_attention(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )
        self.assertEqual(output_torch.size(), (1, 34000, 256))
