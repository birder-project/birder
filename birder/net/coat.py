"""
CoaT, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/coat.py

Paper "Co-Scale Conv-Attentional Image Transformers", https://arxiv.org/abs/2104.06399
"""

# Reference license: Apache-2.0

from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


def insert_cls(x: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.concat((cls_tokens, x), dim=1)

    return x


def remove_cls(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:, :]


class ConvRelPosEnc(nn.Module):
    def __init__(self, head_channels: int, window: dict[int, int]) -> None:
        super().__init__()

        self.conv_list = nn.ModuleList()
        head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            # Determine padding size.
            # Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * head_channels,
                cur_head_split * head_channels,
                kernel_size=(cur_window, cur_window),
                stride=(1, 1),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * head_channels,
            )
            self.conv_list.append(cur_conv)
            head_splits.append(cur_head_split)

        self.channel_splits = [x * head_channels for x in head_splits]

    def forward(self, q: torch.Tensor, v: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        (B, num_heads, N, C) = q.size()
        (H, W) = size
        torch._assert(N == 1 + H * W, "size mismatch")  # pylint: disable=protected-access

        # Convolutional relative position encoding.
        q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]

        v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        conv_v_img_list = []
        for i, conv in enumerate(self.conv_list):
            conv_v_img_list.append(conv(v_img_list[i]))

        conv_v_img = torch.concat(conv_v_img_list, dim=1)
        conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)

        ev_hat = q_img * conv_v_img
        ev_hat = F.pad(ev_hat, (0, 0, 1, 0, 0, 0))

        return ev_hat


class FactorAttnConvRelPosEnc(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        proj_drop: float,
        shared_crpe: nn.Module,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding
        self.crpe = shared_crpe

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        (B, N, C) = x.size()

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)  # [B, h, N, Ch]

        # Factorized attention
        k_softmax = k.softmax(dim=2)
        factor_att = k_softmax.transpose(-1, -2) @ v
        factor_att = q @ factor_att

        # Convolutional relative position encoding
        crpe = self.crpe(q, v, size=size)  # [B, h, N, Ch]

        # Merge and reshape
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ConvPosEnc(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.proj = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, stride=(1, 1), padding=(kernel_size // 2, kernel_size // 2), groups=dim
        )

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        (B, N, C) = x.size()
        (H, W) = size
        torch._assert(N == 1 + H * W, "size mismatch")  # pylint: disable=protected-access

        # Extract CLS token and image tokens
        cls_token = x[:, :1]  # [B, 1, C]
        img_tokens = x[:, 1:]  # [B, H*W, C]

        # Depthwise convolution
        feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)

        # Combine with CLS token
        x = torch.concat((cls_token, x), dim=1)

        return x


class SerialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_drop: float,
        drop_path: float,
        shared_cpe: nn.Module,
        shared_crpe: nn.Module,
    ) -> None:
        super().__init__()

        # Conv-attention
        self.cpe = shared_cpe
        self.norm1 = nn.LayerNorm(dim)
        self.factor_attn_crpe = FactorAttnConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            shared_crpe=shared_crpe,
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")

        # MLP
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=proj_drop)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        # Conv-attention
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factor_attn_crpe(cur, size)
        x = x + self.drop_path(cur)

        # MLP
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)

        return x


class ParallelBlock(nn.Module):
    def __init__(
        self,
        dims: list[int],
        num_heads: int,
        mlp_ratios: list[float],
        qkv_bias: bool,
        proj_drop: float,
        drop_path: float,
        shared_crpes: list[nn.Module],
    ) -> None:
        super().__init__()

        # Conv-attention
        self.norm12 = nn.LayerNorm(dims[1])
        self.norm13 = nn.LayerNorm(dims[2])
        self.norm14 = nn.LayerNorm(dims[3])
        self.factor_attn_crpe2 = FactorAttnConvRelPosEnc(
            dims[1], num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop, shared_crpe=shared_crpes[1]
        )
        self.factor_attn_crpe3 = FactorAttnConvRelPosEnc(
            dims[2], num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop, shared_crpe=shared_crpes[2]
        )
        self.factor_attn_crpe4 = FactorAttnConvRelPosEnc(
            dims[3], num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop, shared_crpe=shared_crpes[3]
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")

        # MLP
        self.norm22 = nn.LayerNorm(dims[1])
        self.norm23 = nn.LayerNorm(dims[2])
        self.norm24 = nn.LayerNorm(dims[3])

        # In the parallel block, we assume dimensions are the same and share the linear transformation
        assert dims[1] == dims[2] == dims[3]
        assert mlp_ratios[1] == mlp_ratios[2] == mlp_ratios[3]
        self.mlp = MLP(dims[1], [int(dims[1] * mlp_ratios[1]), dims[1]], activation_layer=nn.GELU, dropout=proj_drop)

    def upsample(self, x: torch.Tensor, factor: float, size: tuple[int, int]) -> torch.Tensor:
        return self.interpolate(x, scale_factor=factor, size=size)

    def downsample(self, x: torch.Tensor, factor: float, size: tuple[int, int]) -> torch.Tensor:
        return self.interpolate(x, scale_factor=1.0 / factor, size=size)

    def interpolate(self, x: torch.Tensor, scale_factor: float, size: tuple[int, int]) -> torch.Tensor:
        (B, N, C) = x.size()
        (H, W) = size
        torch._assert(N == 1 + H * W, "size mismatch")  # pylint: disable=protected-access

        cls_token = x[:, :1, :]
        img_tokens = x[:, 1:, :]

        img_tokens = img_tokens.transpose(1, 2).reshape(B, C, H, W)
        img_tokens = F.interpolate(
            img_tokens,
            scale_factor=scale_factor,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens = img_tokens.reshape(B, C, -1).transpose(1, 2)

        out = torch.concat((cls_token, img_tokens), dim=1)

        return out

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, sizes: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (_, s2, s3, s4) = sizes
        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)
        cur2 = self.factor_attn_crpe2(cur2, size=s2)
        cur3 = self.factor_attn_crpe3(cur3, size=s3)
        cur4 = self.factor_attn_crpe4(cur4, size=s4)
        upsample3_2 = self.upsample(cur3, factor=2.0, size=s3)
        upsample4_3 = self.upsample(cur4, factor=2.0, size=s4)
        upsample4_2 = self.upsample(cur4, factor=4.0, size=s4)
        downsample2_3 = self.downsample(cur2, factor=2.0, size=s2)
        downsample3_4 = self.downsample(cur3, factor=2.0, size=s3)
        downsample2_4 = self.downsample(cur2, factor=4.0, size=s2)
        cur2 = cur2 + upsample3_2 + upsample4_2
        cur3 = cur3 + upsample4_3 + downsample2_3
        cur4 = cur4 + downsample3_4 + downsample2_4
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)

        # MLP
        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)
        cur2 = self.mlp(cur2)
        cur3 = self.mlp(cur3)
        cur4 = self.mlp(cur4)
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)

        return (x1, x2, x3, x4)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: tuple[int, int], in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=(0, 0))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)
        (H, W) = x.shape[2:4]

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return (x, (H, W))


# pylint: disable=too-many-instance-attributes
class CoaT(DetectorBackbone):
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

        patch_size = (4, 4)
        crpe_window = {3: 2, 5: 3, 7: 3}
        num_heads = 8
        qkv_bias = True
        proj_drop_rate = 0.0
        embed_dims: list[int] = self.config["embed_dims"]
        mlp_ratios: list[float] = self.config["mlp_ratios"]
        serial_depths: list[int] = self.config["serial_depths"]
        parallel_depth: int = self.config["parallel_depth"]
        drop_path_rate: float = self.config["drop_path_rate"]  # Not using the "dpr rule"

        # Patch embeddings
        self.patch_embed1 = PatchEmbed(patch_size=patch_size, in_channels=self.input_channels, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(patch_size=(2, 2), in_channels=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(patch_size=(2, 2), in_channels=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(patch_size=(2, 2), in_channels=embed_dims[2], embed_dim=embed_dims[3])

        # Class tokens
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # Convolutional position encodings
        self.cpe1 = ConvPosEnc(dim=embed_dims[0], kernel_size=3)
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], kernel_size=3)
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], kernel_size=3)
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], kernel_size=3)

        # Convolutional relative position encodings
        self.crpe1 = ConvRelPosEnc(head_channels=embed_dims[0] // num_heads, window=crpe_window)
        self.crpe2 = ConvRelPosEnc(head_channels=embed_dims[1] // num_heads, window=crpe_window)
        self.crpe3 = ConvRelPosEnc(head_channels=embed_dims[2] // num_heads, window=crpe_window)
        self.crpe4 = ConvRelPosEnc(head_channels=embed_dims[3] // num_heads, window=crpe_window)

        # Serial blocks
        self.serial_blocks1 = nn.ModuleList()
        self.serial_blocks2 = nn.ModuleList()
        self.serial_blocks3 = nn.ModuleList()
        self.serial_blocks4 = nn.ModuleList()
        for _ in range(serial_depths[0]):
            self.serial_blocks1.append(
                SerialBlock(
                    dim=embed_dims[0],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    drop_path=drop_path_rate,
                    shared_cpe=self.cpe1,
                    shared_crpe=self.crpe1,
                )
            )

        for _ in range(serial_depths[1]):
            self.serial_blocks2.append(
                SerialBlock(
                    dim=embed_dims[1],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    drop_path=drop_path_rate,
                    shared_cpe=self.cpe2,
                    shared_crpe=self.crpe2,
                )
            )

        for _ in range(serial_depths[2]):
            self.serial_blocks3.append(
                SerialBlock(
                    dim=embed_dims[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    drop_path=drop_path_rate,
                    shared_cpe=self.cpe3,
                    shared_crpe=self.crpe3,
                )
            )

        for _ in range(serial_depths[3]):
            self.serial_blocks4.append(
                SerialBlock(
                    dim=embed_dims[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    drop_path=drop_path_rate,
                    shared_cpe=self.cpe4,
                    shared_crpe=self.crpe4,
                )
            )

        # Parallel blocks
        if parallel_depth > 0:
            self.parallel_blocks = nn.ModuleList()
            for _ in range(parallel_depth):
                self.parallel_blocks.append(
                    ParallelBlock(
                        dims=embed_dims,
                        num_heads=num_heads,
                        mlp_ratios=mlp_ratios,
                        qkv_bias=qkv_bias,
                        proj_drop=proj_drop_rate,
                        drop_path=drop_path_rate,
                        shared_crpes=[self.crpe1, self.crpe2, self.crpe3, self.crpe4],
                    )
                )
        else:
            self.parallel_blocks = None

        # Norms
        if self.parallel_blocks is not None:
            self.norm2 = nn.LayerNorm(embed_dims[1])
            self.norm3 = nn.LayerNorm(embed_dims[2])
        else:
            self.norm2 = None
            self.norm3 = None

        self.norm4 = nn.LayerNorm(embed_dims[3])

        # Head
        if parallel_depth > 0:
            # CoaT series: aggregate features of last three scales for classification.
            assert embed_dims[1] == embed_dims[2] == embed_dims[3]
            self.aggregate = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        else:
            # CoaT-Lite series: use feature of last scale for classification.
            self.aggregate = None

        self.return_channels = embed_dims[1:]
        self.return_stages = self.return_stages[1:]
        self.embedding_size = embed_dims[-1]
        self.classifier = self.create_classifier()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        nn.init.trunc_normal_(self.cls_token1, std=0.02)
        nn.init.trunc_normal_(self.cls_token2, std=0.02)
        nn.init.trunc_normal_(self.cls_token3, std=0.02)
        nn.init.trunc_normal_(self.cls_token4, std=0.02)

    def transform_to_backbone(self) -> None:
        if self.norm2 is not None:
            self.norm2 = nn.Identity()
        if self.norm3 is not None:
            self.norm3 = nn.Identity()
        if self.aggregate is not None:
            self.aggregate = nn.Identity()

        self.norm4 = nn.Identity()
        self.classifier = nn.Identity()

    def _features(self, x: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        B = x.shape[0]

        # Serial blocks 1
        (x1, (h1, w1)) = self.patch_embed1(x)
        x1 = insert_cls(x1, self.cls_token1)
        for blk in self.serial_blocks1:
            x1 = blk(x1, size=(h1, w1))

        x1_no_cls = remove_cls(x1).reshape(B, h1, w1, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 2
        (x2, (h2, w2)) = self.patch_embed2(x1_no_cls)
        x2 = insert_cls(x2, self.cls_token2)
        for blk in self.serial_blocks2:
            x2 = blk(x2, size=(h2, w2))

        x2_no_cls = remove_cls(x2).reshape(B, h2, w2, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 3
        (x3, (h3, w3)) = self.patch_embed3(x2_no_cls)
        x3 = insert_cls(x3, self.cls_token3)
        for blk in self.serial_blocks3:
            x3 = blk(x3, size=(h3, w3))

        x3_no_cls = remove_cls(x3).reshape(B, h3, w3, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 4
        (x4, (h4, w4)) = self.patch_embed4(x3_no_cls)
        x4 = insert_cls(x4, self.cls_token4)
        for blk in self.serial_blocks4:
            x4 = blk(x4, size=(h4, w4))

        x4_no_cls = remove_cls(x4).reshape(B, h4, w4, -1).permute(0, 3, 1, 2).contiguous()

        if self.parallel_blocks is not None:
            # Parallel blocks
            for blk in self.parallel_blocks:
                x2 = self.cpe2(x2, (h2, w2))
                x3 = self.cpe3(x3, (h3, w3))
                x4 = self.cpe4(x4, (h4, w4))
                (x1, x2, x3, x4) = blk(x1, x2, x3, x4, sizes=[(h1, w1), (h2, w2), (h3, w3), (h4, w4)])

            x1_no_cls = remove_cls(x1).reshape(B, h1, w1, -1).permute(0, 3, 1, 2).contiguous()
            x2_no_cls = remove_cls(x2).reshape(B, h2, w2, -1).permute(0, 3, 1, 2).contiguous()
            x3_no_cls = remove_cls(x3).reshape(B, h3, w3, -1).permute(0, 3, 1, 2).contiguous()
            x4_no_cls = remove_cls(x4).reshape(B, h4, w4, -1).permute(0, 3, 1, 2).contiguous()

        return {
            "stage1": (x1, x1_no_cls),
            "stage2": (x2, x2_no_cls),
            "stage3": (x3, x3_no_cls),
            "stage4": (x4, x4_no_cls),
        }

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._features(x)
        out = {}
        for name, feat in features.items():
            if name in self.return_stages:
                out[name] = feat[1]

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        if up_to_stage >= 1:
            for param in self.patch_embed1.parameters():
                param.requires_grad = False
            for param in self.serial_blocks1.parameters():
                param.requires_grad = False
        if up_to_stage >= 2:
            for param in self.patch_embed2.parameters():
                param.requires_grad = False
            for param in self.serial_blocks2.parameters():
                param.requires_grad = False
        if up_to_stage >= 3:
            for param in self.patch_embed3.parameters():
                param.requires_grad = False
            for param in self.serial_blocks3.parameters():
                param.requires_grad = False
        if up_to_stage >= 4:
            for param in self.patch_embed4.parameters():
                param.requires_grad = False
            for param in self.serial_blocks4.parameters():
                param.requires_grad = False

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        features = self._features(x)
        if self.parallel_blocks is None:
            x4 = features["stage4"][0]
            x4 = self.norm4(x4)
            x4_cls = x4[:, 0]
            return x4_cls

        x2 = self.norm2(features["stage2"][0])
        x3 = self.norm3(features["stage3"][0])
        x4 = self.norm4(features["stage4"][0])

        x2 = self.norm2(x2)
        x3 = self.norm3(x3)
        x4 = self.norm4(x4)

        x2_cls = x2[:, :1]
        x3_cls = x3[:, :1]
        x4_cls = x4[:, :1]

        merged_cls = torch.concat((x2_cls, x3_cls, x4_cls), dim=1)
        merged_cls = self.aggregate(merged_cls).squeeze(dim=1)

        return merged_cls


registry.register_alias(
    "coat_tiny",
    CoaT,
    config={
        "embed_dims": [152, 152, 152, 152],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "serial_depths": [2, 2, 2, 2],
        "parallel_depth": 6,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "coat_mini",
    CoaT,
    config={
        "embed_dims": [152, 216, 216, 216],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "serial_depths": [2, 2, 2, 2],
        "parallel_depth": 6,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "coat_small",
    CoaT,
    config={
        "embed_dims": [152, 320, 320, 320],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "serial_depths": [2, 2, 2, 2],
        "parallel_depth": 6,
        "drop_path_rate": 0.1,
    },
)
registry.register_alias(
    "coat_lite_tiny",
    CoaT,
    config={
        "embed_dims": [64, 128, 256, 320],
        "mlp_ratios": [8.0, 8.0, 4.0, 4.0],
        "serial_depths": [2, 2, 2, 2],
        "parallel_depth": 0,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "coat_lite_mini",
    CoaT,
    config={
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8.0, 8.0, 4.0, 4.0],
        "serial_depths": [2, 2, 2, 2],
        "parallel_depth": 0,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "coat_lite_small",
    CoaT,
    config={
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8.0, 8.0, 4.0, 4.0],
        "serial_depths": [3, 4, 6, 3],
        "parallel_depth": 0,
        "drop_path_rate": 0.0,
    },
)
registry.register_alias(
    "coat_lite_medium",
    CoaT,
    config={
        "embed_dims": [128, 256, 320, 512],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "serial_depths": [3, 6, 10, 8],
        "parallel_depth": 0,
        "drop_path_rate": 0.1,
    },
)

registry.register_weights(
    "coat_tiny_il-common",
    {
        "description": "CoaT tiny model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 20.8,
                "sha256": "d6bef0c932233a48d297855e04ca421484672e4c684581c486d3cb7b2fcd0003",
            }
        },
        "net": {"network": "coat_tiny", "tag": "il-common"},
    },
)
registry.register_weights(
    "coat_lite_tiny_il-common",
    {
        "description": "CoaT lite tiny model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 21.1,
                "sha256": "275edeeccaa74547056a9a3ee09dd36b17c7c97e453030b6f62b18dd02107bf1",
            }
        },
        "net": {"network": "coat_lite_tiny", "tag": "il-common"},
    },
)
