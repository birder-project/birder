"""
CrossViT, adapted from
https://github.com/IBM/CrossViT/blob/main/models/crossvit.py

Paper "CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification",
https://arxiv.org/abs/2103.14899

Changes from original:
* Removed resize per patch (224/240)
* Added dynamic size support (must be a multiply of all patches, 48 by default)
"""

# Reference license: Apache-2.0

from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.vit import EncoderBlock
from birder.net.vit import adjust_position_embedding


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: tuple[int, int], in_channels: int, embed_dim: int, multi_conv: bool) -> None:
        super().__init__()
        if multi_conv is True:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_channels, embed_dim // 4, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=(3, 3), stride=(3, 3), padding=(0, 0)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_channels, embed_dim // 4, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                )
            else:
                raise ValueError("Unsupported patch size")

        else:
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC

        return x


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, attn_drop: float, proj_drop: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, N, C) = x.shape
        # B1C -> B1H(C/H) -> BH1(C/H)
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0, scale=self.scale
        )
        x = x.transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, qkv_bias: bool, proj_drop: float, attn_drop: float, drop_path: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: list[int],
        depth: list[int],
        num_heads: list[int],
        mlp_ratio: list[float],
        qkv_bias: bool,
        proj_drop: float,
        attn_drop: float,
        drop_path: list[float],
    ) -> None:
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            layers = []
            for i in range(depth[d]):
                layers.append(
                    EncoderBlock(
                        num_heads=num_heads[d],
                        hidden_dim=dim[d],
                        mlp_dim=int(mlp_ratio[d] * dim[d]),
                        dropout=proj_drop,
                        attention_dropout=attn_drop,
                        drop_path=drop_path[i],
                        activation_layer=nn.GELU,
                    )
                )

            self.blocks.append(nn.Sequential(*layers))

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            self.projs.append(
                nn.Sequential(
                    nn.LayerNorm(dim[d]),
                    nn.GELU(),
                    nn.Linear(dim[d], dim[(d + 1) % num_branches]),
                )
            )

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:
                self.fusion.append(
                    CrossAttentionBlock(
                        dim=dim[d_],
                        num_heads=nh,
                        qkv_bias=qkv_bias,
                        proj_drop=proj_drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[-1],
                    )
                )
            else:
                layers = []
                for _ in range(depth[-1]):
                    layers.append(
                        CrossAttentionBlock(
                            dim=dim[d_],
                            num_heads=nh,
                            qkv_bias=qkv_bias,
                            proj_drop=proj_drop,
                            attn_drop=attn_drop,
                            drop_path=drop_path[-1],
                        )
                    )

                self.fusion.append(nn.Sequential(*layers))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            self.revert_projs.append(
                nn.Sequential(
                    nn.LayerNorm(dim[(d + 1) % num_branches]),
                    nn.GELU(),
                    nn.Linear(dim[(d + 1) % num_branches], dim[d]),
                )
            )

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        outs_b = []
        for i, block in enumerate(self.blocks):
            outs_b.append(block(x[i]))

        proj_cls_token: list[torch.Tensor] = []
        for i, proj in enumerate(self.projs):
            proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))

        # Cross attention
        outs = []
        for i, (fusion, revert_proj) in enumerate(zip(self.fusion, self.revert_projs)):
            tmp = torch.concat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = fusion(tmp)
            reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
            tmp = torch.concat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)

        return outs


def _compute_num_patches(img_size: list[tuple[int, int]], patches: list[int]) -> list[int]:
    return [(i[0] // p) * (i[1] // p) for i, p in zip(img_size, patches)]


class CrossViT(BaseNet):
    default_size = (240, 240)

    # pylint: disable=too-many-locals
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

        qkv_bias = True
        pos_drop_rate = 0.0
        proj_drop_rate = 0.0
        attn_drop_rate = 0.0
        patch_size = [12, 16]
        embed_dim: list[int] = self.config["embed_dim"]
        depths: list[list[int]] = self.config["depths"]
        num_heads: list[int] = self.config["num_heads"]
        mlp_ratio: list[float] = self.config["mlp_ratio"]
        multi_conv: bool = self.config["multi_conv"]
        drop_path_rate: float = self.config["drop_path_rate"]

        image_size = [(self.size[0], self.size[1])] * len(patch_size)
        num_patches = _compute_num_patches(image_size, patch_size)
        self.patch_size = patch_size
        self.num_branches = len(patch_size)
        self.embed_dim = embed_dim

        self.pos_embed = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)]
        )
        self.cls_token = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)]
        )

        self.patch_embed = nn.ModuleList()
        for p, d in zip(patch_size, embed_dim):
            self.patch_embed.append(
                PatchEmbed(
                    patch_size=(p, p),
                    in_channels=self.input_channels,
                    embed_dim=d,
                    multi_conv=multi_conv,
                )
            )

        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depths])  # pylint: disable=consider-using-generator
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # Stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for depth in depths:
            curr_depth = max(depth[:-1]) + depth[-1]
            block = MultiScaleBlock(
                embed_dim,
                depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[dpr_ptr : dpr_ptr + curr_depth],
            )
            dpr_ptr += curr_depth
            self.blocks.append(block)

        self.norm = nn.ModuleList([nn.LayerNorm(embed_dim[i]) for i in range(self.num_branches)])
        self.embedding_size = sum(self.embed_dim)
        self.classifier = nn.ModuleList()
        for i in range(self.num_branches):
            self.classifier.append(self.create_classifier(self.embed_dim[i]))

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.classifier = nn.ModuleList()
        for i in range(self.num_branches):
            self.classifier.append(self.create_classifier(self.embed_dim[i]))

    def embedding(self, x: torch.Tensor) -> list[torch.Tensor]:
        B = x.shape[0]
        xs = []
        for patch_embed, cls_tokens, pos_embed in zip(self.patch_embed, self.cls_token, self.pos_embed):
            branch = patch_embed(x)
            cls_tokens = cls_tokens.expand(B, -1, -1)
            branch = torch.concat((cls_tokens, branch), dim=1)
            branch = branch + pos_embed
            branch = self.pos_drop(branch)
            xs.append(branch)

        for block in self.blocks:
            xs = block(xs)

        xs = [norm(xs[i]) for i, norm in enumerate(self.norm)]
        out = [x[:, 0] for x in xs]

        return out

    def classify(self, x: list[torch.Tensor]) -> torch.Tensor:
        x = [classifier(x[i]) for i, classifier in enumerate(self.classifier)]
        out = torch.mean(torch.stack(x, dim=0), dim=0)

        return out

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        old_size = self.size
        super().adjust_size(new_size)

        # Sort out sizes
        for i in range(self.num_branches):
            old_h = old_size[0] // self.patch_size[i]
            old_w = old_size[1] // self.patch_size[i]
            h = new_size[0] // self.patch_size[i]
            w = new_size[1] // self.patch_size[i]
            self.pos_embed[i] = nn.Parameter(
                adjust_position_embedding(self.pos_embed[i], (old_h, old_w), (h, w), num_prefix_tokens=1)
            )


registry.register_alias(
    "crossvit_t",
    CrossViT,
    config={
        "embed_dim": [96, 192],
        "depths": [[1, 4, 0], [1, 4, 0], [1, 4, 0]],
        "num_heads": [3, 3],
        "mlp_ratio": [4.0, 4.0, 1.0],
        "multi_conv": False,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "crossvit_9",
    CrossViT,
    config={
        "embed_dim": [128, 256],
        "depths": [[1, 3, 0], [1, 3, 0], [1, 3, 0]],
        "num_heads": [4, 4],
        "mlp_ratio": [3.0, 3.0, 1.0],
        "multi_conv": False,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "crossvit_9d",
    CrossViT,
    config={
        "embed_dim": [128, 256],
        "depths": [[1, 3, 0], [1, 3, 0], [1, 3, 0]],
        "num_heads": [4, 4],
        "mlp_ratio": [3.0, 3.0, 1.0],
        "multi_conv": True,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "crossvit_s",
    CrossViT,
    config={
        "embed_dim": [192, 384],
        "depths": [[1, 4, 0], [1, 4, 0], [1, 4, 0]],
        "num_heads": [6, 6],
        "mlp_ratio": [4.0, 4.0, 1.0],
        "multi_conv": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "crossvit_15",
    CrossViT,
    config={
        "embed_dim": [192, 384],
        "depths": [[1, 5, 0], [1, 5, 0], [1, 5, 0]],
        "num_heads": [6, 6],
        "mlp_ratio": [3.0, 3.0, 1.0],
        "multi_conv": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "crossvit_15d",
    CrossViT,
    config={
        "embed_dim": [192, 384],
        "depths": [[1, 5, 0], [1, 5, 0], [1, 5, 0]],
        "num_heads": [6, 6],
        "mlp_ratio": [3.0, 3.0, 1.0],
        "multi_conv": True,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "crossvit_b",
    CrossViT,
    config={
        "embed_dim": [384, 768],
        "depths": [[1, 4, 0], [1, 4, 0], [1, 4, 0]],
        "num_heads": [12, 12],
        "mlp_ratio": [4.0, 4.0, 1.0],
        "multi_conv": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "crossvit_18",
    CrossViT,
    config={
        "embed_dim": [224, 448],
        "depths": [[1, 6, 0], [1, 6, 0], [1, 6, 0]],
        "num_heads": [7, 7],
        "mlp_ratio": [3.0, 3.0, 1.0],
        "multi_conv": False,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "crossvit_18d",
    CrossViT,
    config={
        "embed_dim": [224, 448],
        "depths": [[1, 6, 0], [1, 6, 0], [1, 6, 0]],
        "num_heads": [7, 7],
        "mlp_ratio": [3.0, 3.0, 1.0],
        "multi_conv": True,
        "drop_path_rate": 0.1,
    },
)

registry.register_weights(
    "crossvit_9d_il-common",
    {
        "description": "CrossViT 9 dagger model trained on the il-common dataset",
        "resolution": (240, 240),
        "formats": {
            "pt": {
                "file_size": 32.7,
                "sha256": "515265ed725adce09464bfd23ce612b1d1178bc22a57960db089d7148556149a",
            }
        },
        "net": {"network": "crossvit_9d", "tag": "il-common"},
    },
)
