"""
RepViT, adapted from
https://github.com/THU-MIG/RepViT/blob/main/model/repvit.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/repvit.py

Paper "RepViT: Revisiting Mobile CNN From ViT Perspective", https://arxiv.org/abs/2307.09283
"""

# Reference license: Apache-2.0 (both)

from collections import OrderedDict
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_weights
from torchvision.ops import SqueezeExcitation

from birder.common.masking import mask_tensor
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import MaskedTokenRetentionMixin
from birder.net.base import PreTrainEncoder
from birder.net.base import TokenRetentionResultType
from birder.net.base import make_divisible


class RepConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        groups: int,
        bn_weight_init: float,
        reparameterized: bool,
    ) -> None:
        super().__init__()
        self.c: nn.Conv2d
        self.reparameterized = reparameterized
        self.add_module(
            "c",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=reparameterized,
            ),
        )
        if reparameterized is False:
            self.add_module("bn", nn.BatchNorm2d(out_channels))
            nn.init.constant_(self.bn.weight, bn_weight_init)
            nn.init.zeros_(self.bn.bias)

    def reparameterize(self) -> None:
        if self.reparameterized is True:
            return

        kernel, bias = fuse_conv_bn_weights(
            self.c.weight,
            self.c.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps,
            self.bn.weight,
            self.bn.bias,
        )
        conv = self.c
        self.c = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            padding_mode=conv.padding_mode,
            device=kernel.device,
            dtype=kernel.dtype,
        )
        self.c.weight.data.copy_(kernel)
        self.c.bias.data.copy_(bias)

        # Delete unused branches
        for param in self.parameters():
            param.detach_()

        del self.bn
        self.reparameterized = True


class RepNormLinear(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, reparameterized: bool) -> None:
        super().__init__()
        self.li: nn.Module
        self.reparameterized = reparameterized
        if reparameterized is False:
            self.add_module("bn", nn.BatchNorm1d(in_dim))

        self.add_module("li", nn.Linear(in_dim, out_dim))
        nn.init.trunc_normal_(self.li.weight, std=0.02)
        nn.init.zeros_(self.li.bias)

    def reparameterize(self) -> None:
        if self.reparameterized is True:
            return

        # BatchNorm precedes the linear layer, so fuse_conv_bn_weights does not apply here.
        bn, li = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = li.weight * w[None, :]
        if li.bias is None:
            b = b @ self.li.weight.T
        else:
            b = (li.weight @ b[:, None]).view(-1) + self.li.bias

        self.li = nn.Linear(w.size(1), w.size(0), device=li.weight.device, dtype=li.weight.dtype)
        self.li.weight.data.copy_(w)
        self.li.bias.data.copy_(b)

        # Delete unused branches
        for param in self.parameters():
            param.detach_()

        del self.bn
        self.reparameterized = True


class RepVitMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, reparameterized: bool):
        super().__init__()
        self.conv1 = RepConvBN(
            in_dim,
            hidden_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            bn_weight_init=1.0,
            reparameterized=reparameterized,
        )
        self.act = nn.GELU()
        self.conv2 = RepConvBN(
            hidden_dim,
            in_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            bn_weight_init=0.0,
            reparameterized=reparameterized,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(x)))


class RepVggDW(nn.Module):
    def __init__(self, dim: int, kernel_size: int, use_se: bool, reparameterized: bool) -> None:
        super().__init__()
        self.reparameterized = reparameterized
        self.conv: nn.Conv2d | RepConvBN
        if reparameterized is True:
            self.conv = nn.Conv2d(
                dim,
                dim,
                kernel_size=(kernel_size, kernel_size),
                stride=(1, 1),
                padding=(kernel_size // 2, kernel_size // 2),
                groups=dim,
            )
        else:
            self.conv = RepConvBN(
                dim,
                dim,
                kernel_size=(kernel_size, kernel_size),
                stride=(1, 1),
                padding=(kernel_size // 2, kernel_size // 2),
                groups=dim,
                bn_weight_init=1.0,
                reparameterized=False,
            )
            self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=dim)
            self.bn = nn.BatchNorm2d(dim)

        if use_se is True:
            self.se = SqueezeExcitation(dim, make_divisible(dim * 0.25, 8, round_limit=0.0))
        else:
            self.se = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparameterized is True:
            return self.se(self.conv(x))

        return self.se(self.bn(self.conv(x) + self.conv1x1(x) + x))

    def reparameterize(self) -> None:
        if self.reparameterized is True:
            return

        assert isinstance(self.conv, RepConvBN)
        conv_bn = self.conv
        kernel, bias = self._get_kernel_bias(conv_bn)
        conv = conv_bn.c
        self.conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            padding_mode=conv.padding_mode,
            device=kernel.device,
            dtype=kernel.dtype,
        )
        self.conv.weight.data.copy_(kernel)
        self.conv.bias.data.copy_(bias)

        # Delete unused branches
        for param in self.parameters():
            param.detach_()

        del self.conv1x1
        del self.bn

        self.reparameterized = True

    def _get_kernel_bias(self, conv_bn: RepConvBN) -> tuple[torch.Tensor, torch.Tensor]:
        conv = conv_bn.c
        kernel, bias = fuse_conv_bn_weights(
            conv.weight,
            conv.bias,
            conv_bn.bn.running_mean,
            conv_bn.bn.running_var,
            conv_bn.bn.eps,
            conv_bn.bn.weight,
            conv_bn.bn.bias,
        )
        assert self.conv1x1.bias is not None

        pad_h = (conv.kernel_size[0] - self.conv1x1.kernel_size[0]) // 2
        pad_w = (conv.kernel_size[1] - self.conv1x1.kernel_size[1]) // 2
        kernel_1x1 = F.pad(self.conv1x1.weight, [pad_w, pad_w, pad_h, pad_h])

        assert conv.groups == conv.in_channels == conv.out_channels
        identity = torch.ones(
            conv.out_channels,
            1,
            1,
            1,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        identity = F.pad(identity, [pad_w, pad_w, pad_h, pad_h])

        kernel = kernel + kernel_1x1 + identity
        bias = bias + self.conv1x1.bias

        return fuse_conv_bn_weights(  # type: ignore[no-any-return]
            kernel,
            bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps,
            self.bn.weight,
            self.bn.bias,
        )


class RepViTBlock(nn.Module):
    def __init__(self, in_dim: int, mlp_ratio: float, kernel_size: int, use_se: bool, reparameterized: bool) -> None:
        super().__init__()

        self.token_mixer = RepVggDW(
            in_dim,
            kernel_size=kernel_size,
            use_se=use_se,
            reparameterized=reparameterized,
        )
        self.channel_mixer = RepVitMLP(in_dim, int(in_dim * mlp_ratio), reparameterized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mixer(x)

        identity = x
        x = self.channel_mixer(x)

        return identity + x


class RepVitDownsample(nn.Module):
    def __init__(self, in_dim: int, mlp_ratio: float, out_dim: int, kernel_size: int, reparameterized: bool) -> None:
        super().__init__()
        self.pre_block = RepViTBlock(in_dim, mlp_ratio, kernel_size, use_se=False, reparameterized=reparameterized)
        self.spatial_downsample = RepConvBN(
            in_dim,
            in_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(2, 2),
            padding=(kernel_size // 2, kernel_size // 2),
            groups=in_dim,
            bn_weight_init=1.0,
            reparameterized=reparameterized,
        )
        self.channel_downsample = RepConvBN(
            in_dim,
            out_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            bn_weight_init=1.0,
            reparameterized=reparameterized,
        )
        self.ffn = RepVitMLP(out_dim, int(out_dim * mlp_ratio), reparameterized=reparameterized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_block(x)
        x = self.spatial_downsample(x)
        x = self.channel_downsample(x)

        identity = x
        x = self.ffn(x)

        return x + identity


class RepViTStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        mlp_ratio: float,
        kernel_size: int,
        downsample: bool,
        reparameterized: bool,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = RepVitDownsample(in_dim, mlp_ratio, out_dim, kernel_size, reparameterized)
        else:
            assert in_dim == out_dim
            self.downsample = nn.Identity()

        blocks = []
        use_se = True
        for _ in range(depth):
            blocks.append(RepViTBlock(out_dim, mlp_ratio, kernel_size, use_se, reparameterized))
            use_se = not use_se

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


class RepViT(DetectorBackbone, PreTrainEncoder, MaskedTokenRetentionMixin):
    block_group_regex = r"body\.stage(\d+)\.blocks\.(\d+)"

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, config=config, size=size)
        assert self.config is not None, "must set config"
        assert "head_bias" not in self.config, "head_bias customization is not supported"
        assert "mlp_head" not in self.config, "mlp_head customization is not supported"

        self.reparameterized = False
        embed_dims: list[int] = self.config["embed_dims"]
        depths: list[int] = self.config["depths"]

        self.stem = nn.Sequential(
            RepConvBN(
                in_channels=self.input_channels,
                out_channels=embed_dims[0] // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                groups=1,
                bn_weight_init=1.0,
                reparameterized=self.reparameterized,
            ),
            nn.GELU(),
            RepConvBN(
                in_channels=embed_dims[0] // 2,
                out_channels=embed_dims[0],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                groups=1,
                bn_weight_init=1.0,
                reparameterized=self.reparameterized,
            ),
        )

        num_stages = len(depths)
        prev_dim = embed_dims[0]

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for idx in range(num_stages):
            stages[f"stage{idx+1}"] = RepViTStage(
                prev_dim,
                embed_dims[idx],
                depth=depths[idx],
                mlp_ratio=2.0,
                kernel_size=3,
                downsample=idx > 0,
                reparameterized=self.reparameterized,
            )
            return_channels.append(embed_dims[idx])
            prev_dim = embed_dims[idx]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embed_dims[-1]
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()
        self.distillation_output = False

        self.max_stride = 32
        self.stem_stride = 4
        self.stem_width = embed_dims[0]
        self.feature_dim = embed_dims[-1]

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad_(True)

            for param in self.dist_classifier.parameters():
                param.requires_grad_(True)

        if unfreeze_features is True:
            for param in self.features.parameters():
                param.requires_grad_(True)

    def transform_to_backbone(self) -> None:
        self.features = nn.Identity()
        self.classifier = nn.Identity()
        self.dist_classifier = nn.Identity()

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)

        out = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.stem.parameters():
            param.requires_grad_(False)

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad_(False)

    def masked_encoding_retention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: Optional[torch.Tensor] = None,
        return_keys: Literal["all", "features", "embedding"] = "features",
    ) -> TokenRetentionResultType:
        x = self.stem(x)
        x = mask_tensor(x, mask, patch_factor=self.max_stride // self.stem_stride, mask_token=mask_token)
        x = self.body(x)

        result: TokenRetentionResultType = {}
        if return_keys in ("all", "features"):
            result["features"] = x
        if return_keys in ("all", "embedding"):
            result["embedding"] = self.features(x)

        return result

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.features(features)

    def create_classifier(
        self, embed_dim: Optional[int] = None, head_bias: Optional[bool] = None, mlp_head: Optional[bool] = None
    ) -> nn.Module:
        assert head_bias is None, "head_bias customization is not supported"
        assert mlp_head is None, "mlp_head customization is not supported"

        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size

        return RepNormLinear(embed_dim, self.num_classes, self.reparameterized)

    def set_distillation_output(self, enable: bool = True) -> None:
        self.distillation_output = enable

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        x_cls = self.classifier(x)
        x_dist = self.dist_classifier(x)

        if self.training is True and self.distillation_output is True:
            x = torch.stack([x_cls, x_dist], dim=1)
        else:
            x = (x_cls + x_dist) / 2

        return x

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def reparameterize_model(self) -> None:
        if self.reparameterized is True:
            return

        for module in self.modules():
            if hasattr(module, "reparameterize") is True:
                module.reparameterize()

        self.reparameterized = True


registry.register_model_config("repvit_m0_6", RepViT, config={"embed_dims": [40, 80, 160, 320], "depths": [1, 1, 8, 1]})
registry.register_model_config(
    "repvit_m0_9", RepViT, config={"embed_dims": [48, 96, 192, 384], "depths": [2, 2, 14, 2]}
)
registry.register_model_config(
    "repvit_m1_0", RepViT, config={"embed_dims": [56, 112, 224, 448], "depths": [2, 2, 14, 2]}
)
registry.register_model_config(
    "repvit_m1_1", RepViT, config={"embed_dims": [64, 128, 256, 512], "depths": [2, 2, 12, 2]}
)
registry.register_model_config(
    "repvit_m1_5", RepViT, config={"embed_dims": [64, 128, 256, 512], "depths": [4, 4, 24, 4]}
)
registry.register_model_config(
    "repvit_m2_3", RepViT, config={"embed_dims": [80, 160, 320, 640], "depths": [6, 6, 34, 2]}
)
