"""
EfficientViM, adapted from
https://github.com/mlvlab/EfficientViM/blob/main/classification/models/EfficientViM.py

Paper "EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality",
https://arxiv.org/abs/2411.15241

Changes from original:
* No distillation heads
"""

# Reference license: MIT

from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class LayerNorm1d(nn.Module):
    """
    LayerNorm for channels of 1D tensor (B C L)
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine is True:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine is True:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class LayerNorm2d(nn.Module):
    """
    LayerNorm for channels of 2D tensor (B C H W)
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine is True:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine is True:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class Conv2dNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        groups: int = 1,
        zero_bn_init: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)

        if zero_bn_init is True:
            nn.init.zeros_(self.norm.weight)

        nn.init.zeros_(self.norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.fc1 = Conv2dNorm(in_dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.fc2 = Conv2dNorm(dim, in_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), zero_bn_init=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dNormActivation(in_dim, dim // 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            Conv2dNormActivation(dim // 8, dim // 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            Conv2dNormActivation(dim // 4, dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            Conv2dNormActivation(
                dim // 2, dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), activation_layer=None
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PatchMerging(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ratio: float) -> None:
        super().__init__()
        hidden_dim = int(out_dim * ratio)

        self.dwconv1 = Conv2dNorm(in_dim, in_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=in_dim)
        self.conv = nn.Sequential(
            Conv2dNorm(in_dim, hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            Conv2dNorm(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=hidden_dim),
            nn.ReLU(),
            SqueezeExcitation(hidden_dim, hidden_dim // 4),
            Conv2dNorm(hidden_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )
        self.dwconv2 = Conv2dNorm(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dwconv1(x)
        x = self.conv(x)
        x = x + self.dwconv2(x)

        return x


# pylint: disable=invalid-name
class HSMSSD(nn.Module):
    def __init__(self, d_model: int, ssd_expand: float, A_init_range: tuple[int, int], state_dim: int) -> None:
        super().__init__()
        self.d_inner = int(ssd_expand * d_model)
        self.state_dim = state_dim

        conv_dim = state_dim * 3
        self.bc_dt_proj = nn.Conv1d(d_model, 3 * state_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.dw = nn.Conv2d(
            conv_dim, conv_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=conv_dim, bias=False
        )
        self.hz_proj = nn.Conv1d(d_model, 2 * self.d_inner, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_proj = nn.Conv1d(self.d_inner, d_model, kernel_size=1, stride=1, padding=0, bias=False)

        A = torch.empty(state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True  # NOTE: Should this be supported ? is this a thing ?

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        bc_dt = self.dw(self.bc_dt_proj(x).view(batch_size, -1, H, W)).flatten(2)
        (B, C, dt) = torch.split(bc_dt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        A = F.softmax(dt + self.A.view(1, -1, 1), dim=-1)

        AB = A * B
        h = x @ AB.transpose(-2, -1)

        (h, z) = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        h = self.out_proj(h * self.act(z) + h * self.D)
        y = h @ C

        y = y.view(batch_size, -1, H, W).contiguous()  # + x * self.D  # B C H W

        return (y, h)


class EfficientViMBlock(nn.Module):
    def __init__(
        self, dim: int, mlp_ratio: float, ssd_expand: float, A_init_range: tuple[int, int], state_dim: int
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.mixer = HSMSSD(dim, ssd_expand=ssd_expand, A_init_range=A_init_range, state_dim=state_dim)
        self.norm = LayerNorm1d(dim)

        self.dwconv1 = Conv2dNorm(
            dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim, zero_bn_init=True
        )
        self.dwconv2 = Conv2dNorm(
            dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim, zero_bn_init=True
        )

        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))

        # LayerScale
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        # DWconv1
        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)

        # HSM-SSD
        (H, W) = x.shape[-2:]
        x_prev = x
        (x, h) = self.mixer(self.norm(x.flatten(2)), H, W)
        x = (1 - alpha[1]) * x_prev + alpha[1] * x

        # DWConv2
        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)

        # FFN
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)

        return (x, h)


class EfficientViMStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        mlp_ratio: float,
        downsample: bool,
        ssd_expand: float,
        A_init_range: tuple[int, int],
        state_dim: int,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = PatchMerging(in_dim, out_dim, ratio=4.0)
        else:
            self.downsample = nn.Identity()

        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                EfficientViMBlock(
                    dim=out_dim,
                    mlp_ratio=mlp_ratio,
                    ssd_expand=ssd_expand,
                    A_init_range=A_init_range,
                    state_dim=state_dim,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.downsample(x)
        for blk in self.blocks:
            (x, h) = blk(x)

        return (x, h)


class EfficientViM(DetectorBackbone):
    default_size = (256, 256)
    block_group_regex = r"body\.stage\d+\.blocks\.(\d+)"

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is not None, "must set config"

        mlp_ratio = 4.0
        ssd_expand = 1.0
        embed_dim: list[int] = self.config["embed_dim"]
        depths: list[int] = self.config["depths"]
        state_dim: list[int] = self.config["state_dim"]

        self.embed_dim = embed_dim

        self.stem = Stem(self.input_channels, embed_dim[0])

        num_layers = len(depths)
        prev_dim = embed_dim[0]
        return_channels: list[int] = []
        self.body = nn.ModuleList()
        for i in range(num_layers):
            self.body.add_module(
                f"stage{i+1}",
                EfficientViMStage(
                    in_dim=prev_dim,
                    out_dim=embed_dim[i],
                    depth=depths[i],
                    mlp_ratio=mlp_ratio,
                    downsample=i > 0,
                    ssd_expand=ssd_expand,
                    A_init_range=(1, 16),
                    state_dim=state_dim[i],
                ),
            )
            return_channels.append(embed_dim[i])
            prev_dim = embed_dim[i]

        # Weights for multi-stage hidden-state Fusion
        self.weights = nn.Parameter(torch.ones(4))
        self.state_norms = nn.ModuleList(
            [
                LayerNorm1d(embed_dim[0]),
                LayerNorm1d(embed_dim[1]),
                LayerNorm1d(embed_dim[2]),
            ]
        )
        self.norm = LayerNorm2d(embed_dim[2])

        self.return_stages = self.return_stages[: len(depths)]
        self.return_channels = return_channels
        self.embedding_size = embed_dim[2]
        self.state_classifiers = nn.ModuleList(
            [
                self.create_classifier(embed_dim[0]),
                self.create_classifier(embed_dim[1]),
                self.create_classifier(embed_dim[2]),
            ]
        )
        self.classifier = self.create_classifier(embed_dim[2])

        self.max_stride = 64

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, LayerNorm2d, LayerNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.state_classifiers = nn.ModuleList(
            [
                self.create_classifier(self.embed_dim[0]),
                self.create_classifier(self.embed_dim[1]),
                self.create_classifier(self.embed_dim[2]),
            ]
        )
        self.classifier = self.create_classifier(self.embed_dim[2])

    def transform_to_backbone(self) -> None:
        self.register_parameter("weights", None)
        self.state_norms = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
            ]
        )
        self.norm = nn.Identity()
        self.state_classifiers = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
            ]
        )
        self.classifier = nn.Identity()

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        out = {}
        for name, module in self.body.named_children():
            (x, _) = module(x)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.stem.parameters():
            param.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

        for idx, module in enumerate(self.norm.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.stem(x)
        hs = []
        for stage in self.body:
            (x, h) = stage(x)
            hs.append(h)

        return (x, hs)

    def embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        (x, hs) = self.forward_features(x)
        x = self.norm(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)

        return (x, hs)

    # pylint: disable=arguments-differ
    def classify(self, x: torch.Tensor, hs: list[torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        weights = F.softmax(self.weights, dim=-1)
        z = torch.zeros((x.size(0), self.num_classes), device=x.device)
        for i, (norm, classifier) in enumerate(zip(self.state_norms, self.state_classifiers)):
            h = norm(hs[i])
            h = F.adaptive_avg_pool1d(h, 1).flatten(1)  # pylint: disable=not-callable
            z = z + weights[i] * classifier(h)

        z = z + weights[3] * self.classifier(x)

        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (x, hs) = self.embedding(x)
        return self.classify(x, hs)


registry.register_model_config(
    "efficientvim_m1",
    EfficientViM,
    config={"embed_dim": [128, 192, 320], "depths": [2, 2, 2], "state_dim": [49, 25, 9]},
)
registry.register_model_config(
    "efficientvim_m2",
    EfficientViM,
    config={"embed_dim": [128, 256, 512], "depths": [2, 2, 2], "state_dim": [49, 25, 9]},
)
registry.register_model_config(
    "efficientvim_m3",
    EfficientViM,
    config={"embed_dim": [224, 320, 512], "depths": [2, 2, 2], "state_dim": [49, 25, 9]},
)
registry.register_model_config(
    "efficientvim_m4",
    EfficientViM,
    config={"embed_dim": [224, 320, 512], "depths": [3, 4, 2], "state_dim": [64, 32, 16]},
)

registry.register_weights(
    "efficientvim_m1_il-common",
    {
        "url": "https://huggingface.co/birder-project/efficientvim_m1_il-common/resolve/main",
        "description": "EfficientViM M1 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 23.4,
                "sha256": "065c039bc07b1d59f58d1fe30a162a97f30e90d909d35d0df8966abaa26d8999",
            }
        },
        "net": {"network": "efficientvim_m1", "tag": "il-common"},
    },
)
