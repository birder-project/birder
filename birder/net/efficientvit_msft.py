"""
EfficientViT (Microsoft Research), adapted from
https://github.com/microsoft/Cream/blob/main/EfficientViT/classification/model/efficientvit.py
and
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientvit_msra.py

Paper "EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention",
https://arxiv.org/abs/2305.07027

Changes from original:
* Remove batchnorm fuse
* Window size based on image size (image size // 32)
"""

# Reference license: MIT and Apache-2.0

import itertools
from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import SqueezeExcitation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import interpolate_attention_bias


class Conv2dNorm(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        hidden_dim = dim * 4
        self.conv1 = Conv2dNorm(dim, hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.act = nn.ReLU()
        self.conv2 = Conv2dNorm(
            hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=hidden_dim
        )
        self.se = SqueezeExcitation(hidden_dim, hidden_dim // 4)
        self.conv3 = Conv2dNorm(hidden_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.se(x)
        x = self.conv3(x)

        return x


class ResidualDrop(nn.Module):
    def __init__(self, m: nn.Module, drop: float) -> None:
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training is True and self.drop > 0:
            return (
                x
                + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
            )

        return x + self.m(x)


class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.pw1 = Conv2dNorm(dim, hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.act = nn.ReLU()
        self.pw2 = Conv2dNorm(hidden_dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bn_weight_init=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)

        return x


class CascadedGroupAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int,
        attn_ratio: float,
        resolution: tuple[int, int],
        kernels: list[int],
    ) -> None:
        super().__init__()
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(
                Conv2dNorm(
                    dim // (num_heads), self.key_dim * 2 + self.d, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
                )
            )
            dws.append(
                Conv2dNorm(
                    self.key_dim,
                    self.key_dim,
                    kernel_size=(kernels[i], kernels[i]),
                    stride=(1, 1),
                    padding=(kernels[i] // 2, kernels[i] // 2),
                    groups=self.key_dim,
                )
            )

        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)
        self.proj = nn.Sequential(
            nn.ReLU(),
            Conv2dNorm(self.d * num_heads, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bn_weight_init=0.0),
        )

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets: dict[tuple[int, int], int] = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)

                idxs.append(attention_offsets[offset])

        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.attention_bias_idxs = nn.Buffer(torch.LongTensor(idxs).view(N, N), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, _, H, W) = x.size()
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        attn_bias = self.attention_biases[:, self.attention_bias_idxs]
        for head_idx, (qkv, dws) in enumerate(zip(self.qkvs, self.dws)):
            if head_idx > 0:
                feat = feat + feats_in[head_idx]

            feat = qkv(feat)
            (q, k, v) = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)
            q = dws(q)
            q = q.flatten(2)
            k = k.flatten(2)
            v = v.flatten(2)

            q = q * self.scale
            attn = q.transpose(-2, -1) @ k
            attn = attn + attn_bias[head_idx]  # pylint: disable=unsubscriptable-object
            attn = attn.softmax(dim=-1)

            feat = v @ attn.transpose(-2, -1)
            feat = feat.view(B, self.d, H, W)
            feats_out.append(feat)

        x = torch.concat(feats_out, dim=1)
        x = self.proj(x)

        return x


class LocalWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int,
        attn_ratio: float,
        resolution: tuple[int, int],
        window_resolution: tuple[int, int],
        kernels: list[int],
    ) -> None:
        super().__init__()
        self.resolution = resolution
        assert window_resolution[0] > 0 and window_resolution[1] > 0, "window_size must be greater than 0"

        self.window_resolution = window_resolution
        window_resolution = (min(window_resolution[0], resolution[0]), min(window_resolution[1], resolution[1]))
        self.attn = CascadedGroupAttention(
            dim,
            key_dim,
            num_heads,
            attn_ratio=attn_ratio,
            resolution=window_resolution,
            kernels=kernels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = self.resolution[0]
        W = self.resolution[1]
        (B, C, _, _) = x.size()

        if H <= self.window_resolution[0] and W <= self.window_resolution[1]:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution[0] - H % self.window_resolution[0]) % self.window_resolution[0]
            pad_r = (self.window_resolution[1] - W % self.window_resolution[1]) % self.window_resolution[1]
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            p_h = H + pad_b
            p_w = W + pad_r
            n_h = p_h // self.window_resolution[0]
            n_w = p_w // self.window_resolution[1]

            # Window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, n_h, self.window_resolution[0], n_w, self.window_resolution[1], C).transpose(2, 3)
            x = x.reshape(B * n_h * n_w, self.window_resolution[0], self.window_resolution[1], C).permute(0, 3, 1, 2)
            x = self.attn(x)

            # Window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, n_h, n_w, self.window_resolution[0], self.window_resolution[1], C)
            x = x.transpose(2, 3).reshape(B, p_h, p_w, C)
            x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)

        return x


class EfficientVitBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int,
        attn_ratio: float,
        resolution: tuple[int, int],
        window_resolution: tuple[int, int],
        kernels: list[int],
    ) -> None:
        super().__init__()

        self.dw0 = ResidualDrop(
            Conv2dNorm(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim, bn_weight_init=0.0),
            drop=0.0,
        )
        self.ffn0 = ResidualDrop(FFN(dim, int(dim * 2)), drop=0.0)

        self.mixer = ResidualDrop(
            LocalWindowAttention(
                dim,
                key_dim,
                num_heads,
                attn_ratio=attn_ratio,
                resolution=resolution,
                window_resolution=window_resolution,
                kernels=kernels,
            ),
            drop=0.0,
        )

        self.dw1 = ResidualDrop(
            Conv2dNorm(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=dim, bn_weight_init=0.0),
            drop=0.0,
        )
        self.ffn1 = ResidualDrop(FFN(dim, int(dim * 2)), drop=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw0(x)
        x = self.ffn0(x)
        x = self.mixer(x)
        x = self.dw1(x)
        x = self.ffn1(x)

        return x


class EfficientVitStage(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        key_dim: int,
        downsample: bool,
        num_heads: int,
        attn_ratio: float,
        resolution: tuple[int, int],
        window_resolution: tuple[int, int],
        kernels: list[int],
        depth: int,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.resolution = ((resolution[0] - 1) // 2 + 1, (resolution[1] - 1) // 2 + 1)
            self.downsample = nn.Sequential(
                ResidualDrop(
                    Conv2dNorm(in_dim, in_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=in_dim),
                    drop=0.0,
                ),
                ResidualDrop(FFN(in_dim, int(in_dim * 2)), drop=0.0),
                PatchMerging(in_dim, out_dim),
                ResidualDrop(
                    Conv2dNorm(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=out_dim),
                    drop=0.0,
                ),
                ResidualDrop(FFN(out_dim, int(out_dim * 2)), drop=0.0),
            )

        else:
            assert in_dim == out_dim
            self.resolution = resolution
            self.downsample = nn.Identity()

        blocks = []
        for _ in range(depth):
            blocks.append(
                EfficientVitBlock(out_dim, key_dim, num_heads, attn_ratio, self.resolution, window_resolution, kernels)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


# pylint: disable=invalid-name
class EfficientViT_MSFT(DetectorBackbone):
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

        img_size = self.size
        key_dims = [16, 16, 16]
        embed_dims: list[int] = self.config["embed_dims"]
        depths: list[int] = self.config["depths"]
        num_heads: list[int] = self.config["num_heads"]
        kernels: list[int] = self.config["kernels"]
        window_size = [(int(img_size[0] / (2**5)), int(img_size[1] / (2**5)))] * len(depths)
        resolution = (img_size[0] // 16, img_size[1] // 16)
        attn_ratio = [embed_dims[i] / (key_dims[i] * num_heads[i]) for i in range(len(embed_dims))]

        self.stem = nn.Sequential(
            Conv2dNorm(self.input_channels, embed_dims[0] // 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            Conv2dNorm(embed_dims[0] // 8, embed_dims[0] // 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            Conv2dNorm(embed_dims[0] // 4, embed_dims[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            Conv2dNorm(embed_dims[0] // 2, embed_dims[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        num_stages = len(depths)
        prev_dim = embed_dims[0]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stage = EfficientVitStage(
                in_dim=prev_dim,
                out_dim=embed_dims[i],
                key_dim=key_dims[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                attn_ratio=attn_ratio[i],
                resolution=resolution,
                window_resolution=window_size[i],
                kernels=kernels,
                depth=depths[i],
            )
            stages[f"stage{i+1}"] = stage
            return_channels.append(embed_dims[i])
            resolution = stage.resolution
            prev_dim = embed_dims[i]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.BatchNorm1d(embed_dims[-1]),
        )
        self.return_stages = self.return_stages[: len(depths)]
        self.return_channels = return_channels
        self.embedding_size = embed_dims[-1]
        self.classifier = self.create_classifier()

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
            param.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        old_size = self.size
        super().adjust_size(new_size)

        old_resolution = (old_size[0] // 16, old_size[1] // 16)
        old_window_size = (int(old_size[0] / (2**5)), int(old_size[1] / (2**5)))
        resolution = (new_size[0] // 16, new_size[1] // 16)
        window_size = (int(new_size[0] / (2**5)), int(new_size[1] / (2**5)))
        for module in self.body.children():  # pylint: disable=too-many-nested-blocks
            if isinstance(module, EfficientVitStage):
                if not isinstance(module.downsample, nn.Identity):
                    old_resolution = ((old_resolution[0] - 1) // 2 + 1, (old_resolution[1] - 1) // 2 + 1)
                    resolution = ((resolution[0] - 1) // 2 + 1, (resolution[1] - 1) // 2 + 1)

                module.resolution = resolution
                for m in module.modules():
                    if isinstance(m, EfficientVitBlock):
                        m.mixer.m.resolution = resolution
                        m.mixer.m.window_resolution = window_size
                        old_window_resolution = (
                            min(old_window_size[0], old_resolution[0]),
                            min(old_window_size[1], old_resolution[1]),
                        )
                        window_resolution = (min(window_size[0], resolution[0]), min(window_size[1], resolution[1]))

                        # Interpolate attention_biases
                        points = list(itertools.product(range(window_resolution[0]), range(window_resolution[1])))
                        N = len(points)
                        attention_offsets: dict[tuple[int, int], int] = {}
                        idxs = []
                        for p1 in points:
                            for p2 in points:
                                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                                if offset not in attention_offsets:
                                    attention_offsets[offset] = len(attention_offsets)

                                idxs.append(attention_offsets[offset])

                        m.mixer.m.attn.attention_biases = nn.Parameter(
                            interpolate_attention_bias(
                                m.mixer.m.attn.attention_biases, old_window_resolution, window_resolution
                            )
                        )
                        m.mixer.m.attn.attention_bias_idxs = nn.Buffer(
                            torch.LongTensor(idxs).view(N, N), persistent=False
                        )


registry.register_model_config(
    "efficientvit_msft_m0",
    EfficientViT_MSFT,
    config={"embed_dims": [64, 128, 192], "depths": [1, 2, 3], "num_heads": [4, 4, 4], "kernels": [5, 5, 5, 5]},
)
registry.register_model_config(
    "efficientvit_msft_m1",
    EfficientViT_MSFT,
    config={"embed_dims": [128, 144, 192], "depths": [1, 2, 3], "num_heads": [2, 3, 3], "kernels": [7, 5, 3, 3]},
)
registry.register_model_config(
    "efficientvit_msft_m2",
    EfficientViT_MSFT,
    config={"embed_dims": [128, 192, 224], "depths": [1, 2, 3], "num_heads": [4, 3, 2], "kernels": [7, 5, 3, 3]},
)
registry.register_model_config(
    "efficientvit_msft_m3",
    EfficientViT_MSFT,
    config={"embed_dims": [128, 240, 320], "depths": [1, 2, 3], "num_heads": [4, 3, 4], "kernels": [5, 5, 5, 5]},
)
registry.register_model_config(
    "efficientvit_msft_m4",
    EfficientViT_MSFT,
    config={"embed_dims": [128, 256, 384], "depths": [1, 2, 3], "num_heads": [4, 4, 4], "kernels": [7, 5, 3, 3]},
)
registry.register_model_config(
    "efficientvit_msft_m5",
    EfficientViT_MSFT,
    config={"embed_dims": [192, 288, 384], "depths": [1, 3, 4], "num_heads": [3, 3, 4], "kernels": [7, 5, 3, 3]},
)

registry.register_weights(
    "efficientvit_msft_m0_il-common",
    {
        "description": "EfficientViT (MSFT) M0 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 8.9,
                "sha256": "f1ff44ac458b72ccdf442ce5516ee01e305d0ba3123e5512a5b1579a9ae8d422",
            }
        },
        "net": {"network": "efficientvit_msft_m0", "tag": "il-common"},
    },
)
