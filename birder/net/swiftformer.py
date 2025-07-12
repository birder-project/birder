"""
SwiftFormer, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swiftformer.py

Paper "SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications",
https://arxiv.org/abs/2303.15446
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from birder.layers import LayerScale2d
from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: tuple[int, int],
        patch_stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        return x


class ConvEncoder(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, kernel_size: tuple[int, int], drop_path: float, use_layer_scale: bool
    ) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(
            dim, dim, kernel_size, stride=(1, 1), padding=(kernel_size[0] // 2, kernel_size[1] // 2), groups=dim
        )
        self.norm = nn.BatchNorm2d(dim)
        self.pw_conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.act = nn.GELU()
        self.pw_conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.drop_path = StochasticDepth(drop_path, "row")
        if use_layer_scale is True:
            self.layer_scale = LayerScale2d(dim, init_value=1.0)
        else:
            self.layer_scale = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = self.layer_scale(x)
        x = shortcut + self.drop_path(x)

        return x


class ConvMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, in_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class EfficientAdditiveAttention(nn.Module):
    def __init__(self, in_dims: int, token_dim: int, num_heads: int) -> None:
        super().__init__()
        self.scale_factor = token_dim**-0.5
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))

        self.proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, _, H, W) = x.size()
        x = x.flatten(2).permute(0, 2, 1)

        query = F.normalize(self.to_query(x), dim=-1)
        key = F.normalize(self.to_key(x), dim=-1)

        attn = F.normalize(query @ self.w_g * self.scale_factor, dim=1)
        attn = torch.sum(attn * query, dim=1, keepdim=True)

        out = self.proj(attn * key) + query
        out = self.final(out).permute(0, 2, 1).reshape(B, -1, H, W)

        return out


class LocalRepresentation(nn.Module):
    def __init__(self, dim: int, kernel_size: tuple[int, int], drop_path: float, use_layer_scale: bool) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(
            dim, dim, kernel_size, stride=(1, 1), padding=(kernel_size[0] // 2, kernel_size[1] // 2), groups=dim
        )
        self.norm = nn.BatchNorm2d(dim)
        self.pw_conv1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.act = nn.GELU()
        self.pw_conv2 = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.drop_path = StochasticDepth(drop_path, "row")
        if use_layer_scale is True:
            self.layer_scale = LayerScale2d(dim, init_value=1.0)
        else:
            self.layer_scale = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = self.layer_scale(x)
        x = shortcut + self.drop_path(x)

        return x


class SwiftFormerBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop_rate: float, drop_path: float, use_layer_scale: bool) -> None:
        super().__init__()
        self.local_representation = LocalRepresentation(
            dim=dim, kernel_size=(3, 3), drop_path=0.0, use_layer_scale=use_layer_scale
        )
        self.attn = EfficientAdditiveAttention(dim, token_dim=dim, num_heads=1)
        self.linear = ConvMLP(dim, hidden_features=int(dim * mlp_ratio), drop=drop_rate)
        self.drop_path = StochasticDepth(drop_path, "row")
        if use_layer_scale is True:
            self.layer_scale_1 = LayerScale2d(dim, init_value=1e-5)
            self.layer_scale_2 = LayerScale2d(dim, init_value=1e-5)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_representation(x)
        x = x + self.drop_path(self.layer_scale_1(self.attn(x)))
        x = x + self.drop_path(self.layer_scale_2(self.linear(x)))

        return x


class SwiftFormerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        index: int,
        layers: list[int],
        mlp_ratio: float,
        drop_rate: float,
        drop_path_rate: float,
        use_layer_scale: bool,
        downsample: bool,
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = PatchEmbed(dim, dim_out, patch_size=(3, 3), patch_stride=(2, 2), padding=(1, 1))
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        blocks = []
        for block_idx in range(layers[index]):
            block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
            if layers[index] - block_idx <= 1:
                blocks.append(
                    SwiftFormerBlock(
                        dim_out,
                        mlp_ratio=mlp_ratio,
                        drop_rate=drop_rate,
                        drop_path=block_dpr,
                        use_layer_scale=use_layer_scale,
                    )
                )
            else:
                blocks.append(
                    ConvEncoder(
                        dim=dim_out,
                        hidden_dim=int(mlp_ratio * dim_out),
                        kernel_size=(3, 3),
                        drop_path=block_dpr,
                        use_layer_scale=use_layer_scale,
                    )
                )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


class SwiftFormer(DetectorBackbone):
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

        layers: list[int] = self.config["layers"]
        embed_dims: list[int] = self.config["embed_dims"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, embed_dims[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dims[0] // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dims[0] // 2, embed_dims[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(),
        )

        prev_dim = embed_dims[0]
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(len(layers)):
            stages[f"stage{i+1}"] = SwiftFormerStage(
                prev_dim,
                embed_dims[i],
                index=i,
                layers=layers,
                mlp_ratio=4.0,
                drop_rate=0.0,
                drop_path_rate=drop_path_rate,
                use_layer_scale=True,
                downsample=i > 0,
            )
            return_channels.append(embed_dims[i])
            prev_dim = embed_dims[i]

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.BatchNorm2d(embed_dims[-1]),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embed_dims[-1]
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()
        self.distillation_output = False

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.dist_classifier = self.create_classifier()
        self.classifier = self.create_classifier()

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad = True

            for param in self.dist_classifier.parameters():
                param.requires_grad = True

        if unfreeze_features is True:
            for param in self.features.parameters():
                param.requires_grad = True

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

    def set_distillation_output(self, enable: bool = True) -> None:
        self.distillation_output = enable

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        x_cls = self.classifier(x)
        x_dist = self.dist_classifier(x)

        if self.training is True and self.distillation_output is True:
            x = torch.stack([x_cls, x_dist], dim=1)
        else:
            # Classifier "token" as an average of both tokens (during normal training or inference)
            x = (x_cls + x_dist) / 2

        return x


registry.register_model_config(
    "swiftformer_xs",
    SwiftFormer,
    config={"layers": [3, 3, 6, 4], "embed_dims": [48, 56, 112, 220], "drop_path_rate": 0.0},
)
registry.register_model_config(
    "swiftformer_s",
    SwiftFormer,
    config={"layers": [3, 3, 9, 6], "embed_dims": [48, 64, 168, 224], "drop_path_rate": 0.0},
)
registry.register_model_config(
    "swiftformer_l1",
    SwiftFormer,
    config={"layers": [4, 3, 10, 5], "embed_dims": [48, 96, 192, 384], "drop_path_rate": 0.0},
)
registry.register_model_config(
    "swiftformer_l3",
    SwiftFormer,
    config={"layers": [4, 4, 12, 6], "embed_dims": [64, 128, 320, 512], "drop_path_rate": 0.0},
)

registry.register_weights(
    "swiftformer_xs_il-common",
    {
        "description": "SwiftFormer model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 12.3,
                "sha256": "4053b9726f888c05afb98a6d13463da35573be1a66ca347d17208a2b3f9baba6",
            }
        },
        "net": {"network": "swiftformer_xs", "tag": "il-common"},
    },
)
