"""
CrossFormer, adapted from
https://github.com/cheerss/CrossFormer/blob/crossformer/models/crossformer.py

Paper "CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention",
https://arxiv.org/abs/2108.00154

Changes from original:
* Dynamic group size based on the input size
"""

# Reference license: MIT

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class PatchEmbed(nn.Module):
    def __init__(self, patch_sizes: list[int], in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_sizes):
            if i == len(patch_sizes) - 1:
                dim = embed_dim // 2**i
            else:
                dim = embed_dim // 2 ** (i + 1)

            stride = patch_sizes[0]
            padding = (ps - patch_sizes[0]) // 2
            self.projs.append(
                nn.Conv2d(in_channels, dim, kernel_size=(ps, ps), stride=(stride, stride), padding=(padding, padding))
            )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = []
        for proj in self.projs:
            xs.append(proj(x).flatten(2).transpose(1, 2))  # B Ph*Pw C

        x = torch.concat(xs, dim=2)
        x = self.norm(x)

        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        pos_dim = dim // 4
        self.pos = nn.Sequential(
            nn.Linear(2, pos_dim),
            nn.LayerNorm(pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim),
            nn.LayerNorm(pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim),
            nn.LayerNorm(pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, num_heads),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos(x)


class Attention(nn.Module):
    def __init__(
        self, dim: int, group_size: tuple[int, int], num_heads: int, qkv_bias: bool, attn_drop: float, proj_drop: float
    ) -> None:
        super().__init__()
        self.group_size = group_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.pos = DynamicPosBias(dim // 4, self.num_heads)

        self.define_bias_table()
        self.define_relative_position_index()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def define_bias_table(self) -> None:
        position_bias_h = torch.arange(1 - self.group_size[0], self.group_size[0])
        position_bias_w = torch.arange(1 - self.group_size[1], self.group_size[1])
        biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing="ij"))  # 2, 2Wh-1, 2W2-1
        biases = biases.flatten(1).transpose(0, 1).float()
        self.biases = nn.Buffer(biases)

    def define_relative_position_index(self) -> None:
        coords_h = torch.arange(self.group_size[0])
        coords_w = torch.arange(self.group_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.group_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.group_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.group_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.relative_position_index = nn.Buffer(relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        pos = self.pos(self.biases)  # 2Wh-1 * 2Ww-1, heads
        relative_position_bias = pos[self.relative_position_index.view(-1)].view(
            self.group_size[0] * self.group_size[1], self.group_size[0] * self.group_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        group_size: tuple[int, int],
        use_lda: bool,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_drop: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.group_size = group_size
        self.use_lda = use_lda
        self.mlp_ratio = mlp_ratio
        if self.input_resolution[0] <= self.group_size[0]:
            self.use_lda = False
            self.group_size = (self.input_resolution[0], self.group_size[1])
        if self.input_resolution[1] <= self.group_size[1]:
            self.use_lda = False
            self.group_size = (self.group_size[0], self.input_resolution[1])

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            group_size=self.group_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=proj_drop)

        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (H, W) = self.input_resolution
        (B, _, C) = x.size()

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Group embeddings
        (GH, GW) = self.group_size  # pylint: disable=invalid-name
        if self.use_lda is False:
            x = x.reshape(B, H // GH, GH, W // GW, GW, C).permute(0, 1, 3, 2, 4, 5)
        else:
            x = x.reshape(B, GH, H // GH, GW, W // GW, C).permute(0, 2, 4, 1, 3, 5)

        x = x.reshape(B * H * W // (GH * GW), GH * GW, C)

        # Attention
        x = self.attn(x)  # nW*B, G*G, C

        # Un-group embeddings
        x = x.reshape(B, H // GH, W // GW, GH, GW, C)
        if self.use_lda is False:
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        else:
            x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, H, W, C)

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dim: int, patch_sizes: list[int]) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reductions = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)

        for i, ps in enumerate(patch_sizes):
            if i == len(patch_sizes) - 1:
                out_dim = 2 * dim // 2**i
            else:
                out_dim = 2 * dim // 2 ** (i + 1)

            stride = 2
            padding = (ps - stride) // 2
            self.reductions.append(
                nn.Conv2d(dim, out_dim, kernel_size=(ps, ps), stride=(stride, stride), padding=(padding, padding))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (H, W) = self.input_resolution
        (B, _, C) = x.shape

        x = self.norm(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        xs = []
        for reduction in self.reductions:
            xs.append(reduction(x).flatten(2).transpose(1, 2))

        x = torch.concat(xs, dim=2)

        return x


class CrossFormerStage(nn.Module):
    def __init__(
        self,
        prev_dim: int,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        group_size: tuple[int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        proj_drop: float,
        attn_drop: float,
        drop_path: list[float],
        downsample: bool,
        patch_sizes: list[int],
    ) -> None:
        super().__init__()
        if downsample is True:
            self.downsample = PatchMerging(input_resolution, dim=prev_dim, patch_sizes=patch_sizes)
            input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        else:
            self.downsample = nn.Identity()

        layers = []
        for i in range(depth):
            use_lda = (i % 2) != 0
            layers.append(
                CrossFormerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    group_size=group_size,
                    use_lda=use_lda,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                )
            )

        self.blocks = nn.Sequential(*layers)
        self.resolution = input_resolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


class CrossFormer(DetectorBackbone):
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

        patch_sizes = [4, 8, 16, 32]
        merge_sizes = [[2, 4], [2, 4], [2, 4]]
        embed_dim: int = self.config["embed_dim"]
        depths: list[int] = self.config["depths"]
        num_heads: list[int] = self.config["num_heads"]
        drop_path_rate: float = self.config["drop_path_rate"]
        group_size = (int(self.size[0] / (2**5)), int(self.size[1] / (2**5)))

        self.patch_sizes = patch_sizes
        self.patch_embed = PatchEmbed(patch_sizes=patch_sizes, in_channels=self.input_channels, embed_dim=embed_dim)
        patch_resolution = (self.size[0] // patch_sizes[0], self.size[1] // patch_sizes[0])

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        num_stages = len(depths)
        prev_dim = embed_dim
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            patch_sizes = merge_sizes[i - 1] if i > 0 else [0]
            dim = int(embed_dim * 2**i)
            if i == 0:
                input_resolution = patch_resolution
            else:
                input_resolution = (patch_resolution[0] // (2 ** (i - 1)), patch_resolution[1] // (2 ** (i - 1)))

            stages[f"stage{i+1}"] = CrossFormerStage(
                prev_dim=prev_dim,
                dim=dim,
                input_resolution=input_resolution,
                depth=depths[i],
                num_heads=num_heads[i],
                group_size=group_size,
                mlp_ratio=4.0,
                qkv_bias=True,
                proj_drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[i],
                downsample=i > 0,
                patch_sizes=patch_sizes,
            )
            return_channels.append(dim)
            prev_dim = dim

        last_features = int(embed_dim * 2 ** (num_stages - 1))
        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.LayerNorm(last_features),
            Permute([0, 2, 1]),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = last_features
        self.classifier = self.create_classifier()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.patch_embed(x)

        out = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.return_stages:
                (H, W) = module.resolution
                (B, _, C) = x.size()
                out[name] = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        super().adjust_size(new_size)

        new_patch_resolution = (new_size[0] // self.patch_sizes[0], new_size[1] // self.patch_sizes[0])
        input_resolution = new_patch_resolution
        for mod in self.body.modules():
            if isinstance(mod, CrossFormerStage):
                for m in mod.modules():
                    if isinstance(m, PatchMerging):
                        m.input_resolution = input_resolution
                        input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
                    elif isinstance(m, CrossFormerBlock):
                        m.input_resolution = input_resolution

                mod.resolution = input_resolution

        new_group_size = (int(new_size[0] / (2**5)), int(new_size[1] / (2**5)))
        for m in self.body.modules():
            if isinstance(m, CrossFormerBlock):
                m.group_size = new_group_size
                if m.input_resolution[0] <= m.group_size[0]:
                    m.use_lda = False
                    m.group_size = (m.input_resolution[0], m.group_size[1])
                if m.input_resolution[1] <= m.group_size[1]:
                    m.use_lda = False
                    m.group_size = (m.group_size[0], m.input_resolution[1])

            elif isinstance(m, Attention):
                m.group_size = new_group_size
                m.define_bias_table()
                m.define_relative_position_index()


registry.register_model_config(
    "crossformer_t",
    CrossFormer,
    config={"embed_dim": 64, "depths": [1, 1, 8, 6], "num_heads": [2, 4, 8, 16], "drop_path_rate": 0.1},
)
registry.register_model_config(
    "crossformer_s",
    CrossFormer,
    config={"embed_dim": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24], "drop_path_rate": 0.2},
)
registry.register_model_config(
    "crossformer_b",
    CrossFormer,
    config={"embed_dim": 96, "depths": [2, 2, 18, 2], "num_heads": [3, 6, 12, 24], "drop_path_rate": 0.3},
)
registry.register_model_config(
    "crossformer_l",
    CrossFormer,
    config={"embed_dim": 128, "depths": [2, 2, 18, 2], "num_heads": [4, 8, 16, 32], "drop_path_rate": 0.5},
)

registry.register_weights(
    "crossformer_s_arabian-peninsula256px",
    {
        "url": "https://huggingface.co/birder-project/crossformer_s_arabian-peninsula/resolve/main",
        "description": "CrossFormer small model trained on the arabian-peninsula dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 116.7,
                "sha256": "24feadef6e7921702fade34487bfcf65f0656151e2a00ede6a8ae719534d68c6",
            }
        },
        "net": {"network": "crossformer_s", "tag": "arabian-peninsula256px"},
    },
)
registry.register_weights(
    "crossformer_s_arabian-peninsula",
    {
        "url": "https://huggingface.co/birder-project/crossformer_s_arabian-peninsula/resolve/main",
        "description": "CrossFormer small model trained on the arabian-peninsula dataset",
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 118.3,
                "sha256": "94936dfb3f641cbd227b90ada937eb5055e176fa95f517711e14315b2e0b9405",
            }
        },
        "net": {"network": "crossformer_s", "tag": "arabian-peninsula"},
    },
)
