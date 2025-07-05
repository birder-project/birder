import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-renamed
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps=self.eps)
        x = x.permute(0, 3, 1, 2)

        return x
