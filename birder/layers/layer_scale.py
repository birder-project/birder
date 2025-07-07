import torch
from torch import nn


class LayerScale(nn.Module):
    """
    Layer scale layer that suits 2d in channels last layout or 1d
    """

    def __init__(self, dim: int, init_value: float, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace is True:
            return x.mul_(self.gamma)

        return x * self.gamma


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_value: float, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_value * torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace is True:
            return x.mul_(self.gamma)

        return x * self.gamma
