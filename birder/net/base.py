import logging
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Literal
from typing import Optional
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import NotRequired

from birder.model_registry import Task
from birder.model_registry import registry

logger = logging.getLogger(__name__)

DataShapeType = TypedDict("DataShapeType", {"data_shape": list[int]})
SignatureType = TypedDict("SignatureType", {"inputs": list[DataShapeType], "outputs": list[DataShapeType]})
TokenRetentionResultType = TypedDict(
    "TokenRetentionResultType",
    {"features": NotRequired[torch.Tensor], "embedding": NotRequired[torch.Tensor]},
)


def get_signature(input_shape: tuple[int, ...], num_outputs: int) -> SignatureType:
    return {
        "inputs": [{"data_shape": [0, *input_shape[1:]]}],
        "outputs": [{"data_shape": [0, num_outputs]}],
    }


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original TensorFlow repository.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """

    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


class BaseNet(nn.Module):
    default_size: tuple[int, int] = (224, 224)
    block_group_regex: Optional[str]
    auto_register = False
    scriptable = True
    square_only = False
    task = str(Task.IMAGE_CLASSIFICATION)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.__name__ in ["PreTrainEncoder", "DetectorBackbone"]:
            # Exclude all other base classes here
            return

        if cls.auto_register is False:
            # Exclude networks with custom config (initialized only with aliases)
            return

        registry.register_model(cls.__name__.lower(), cls)

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        if hasattr(self, "net_param") is False:  # Avoid overriding aliases
            self.net_param = net_param
        if hasattr(self, "config") is False:  # Avoid overriding aliases
            self.config = config
        elif config is not None:
            assert self.config is not None
            self.config.update(config)  # Override with custom config

        if size is not None:
            self.size = size
        else:
            self.size = self.default_size

        assert isinstance(self.size, tuple)
        assert isinstance(self.size[0], int)
        assert isinstance(self.size[1], int)
        if self.square_only is True:
            assert self.size[0] == self.size[1]

        self.dynamic_size = False

        self.classifier: nn.Module
        self.embedding_size: int

    def create_classifier(self, embed_dim: Optional[int] = None) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size

        return nn.Linear(embed_dim, self.num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.classifier = self.create_classifier()

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        logger.debug(f"Setting dynamic size to: {dynamic_size}")
        self.dynamic_size = dynamic_size

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        """
        Override this when one time adjustments for different resolutions is required.
        This should run after load_state_dict.
        """
        if new_size == self.size:
            return

        if self.square_only is True:
            assert new_size[0] == new_size[1]

        logger.info(f"Adjusting model input resolution from {self.size} to {new_size}")
        self.size = new_size

    def freeze(self, freeze_classifier: bool = True, unfreeze_features: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad = True
        if unfreeze_features is True and hasattr(self, "features") is True:
            for param in self.features.parameters():
                param.requires_grad = True

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        return self.classify(x)


class PreTrainEncoder(BaseNet):  # pylint: disable=abstract-method
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
        self.max_stride: int = 32
        self.stem_stride: int
        self.stem_width: int
        self.encoding_size: int
        self.decoder_block: Callable[[int], nn.Module]


class MaskedTokenOmissionMixin:
    num_special_tokens: int

    def masked_encoding_omission(
        self, x: torch.Tensor, ids_keep: Optional[torch.Tensor] = None, return_all_features: bool = False
    ) -> torch.Tensor:
        # Returned size (N, L, D)
        raise NotImplementedError


class MaskedTokenRetentionMixin:
    def masked_encoding_retention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: Optional[torch.Tensor] = None,
        return_keys: Literal["all", "features", "embedding"] = "features",
    ) -> TokenRetentionResultType:
        # Returned features size (B, C, H, W), embedding (B, D)
        raise NotImplementedError


class DetectorBackbone(BaseNet):
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
        self.return_stages = ["stage1", "stage2", "stage3", "stage4"]
        self.return_channels: list[int]

    def transform_to_backbone(self) -> None:
        if hasattr(self, "features") is True:
            self.features = nn.Identity()  # pylint: disable=attribute-defined-outside-init

        self.classifier = nn.Identity()

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def freeze_stages(self, up_to_stage: int) -> None:
        raise NotImplementedError


def pos_embedding_sin_cos_2d(
    h: int, w: int, dim: int, num_special_tokens: int, temperature: int = 10000
) -> torch.Tensor:
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sin-cos emb"

    (y, x) = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.concat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

    if num_special_tokens > 0:
        pe = torch.concat([torch.zeros([num_special_tokens, dim]), pe], dim=0)

    return pe


def interpolate_attention_bias(
    attention_bias: torch.Tensor,
    old_resolution: tuple[int, int],
    new_resolution: tuple[int, int],
    mode: Literal["bilinear", "bicubic"] = "bicubic",
) -> torch.Tensor:
    (H, _) = attention_bias.size()

    # Interpolate
    orig_dtype = attention_bias.dtype
    attention_bias = attention_bias.float()  # Interpolate needs float32
    attention_bias = attention_bias.reshape(1, old_resolution[0], old_resolution[1], H).permute(0, 3, 1, 2)
    attention_bias = F.interpolate(attention_bias, size=new_resolution, mode=mode, antialias=True)
    attention_bias = attention_bias.permute(0, 2, 3, 1).reshape(H, new_resolution[0] * new_resolution[1])
    attention_bias = attention_bias.to(orig_dtype)

    return attention_bias


def reparameterize_available(net: nn.Module) -> bool:
    return hasattr(net, "reparameterize_model")


def get_activation_module(name: str) -> type[nn.Module]:
    activation_dict = {
        "identity": nn.Identity,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
        "celu": nn.CELU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
        "gelu_tanh": partial(nn.GELU, approximate="tanh"),
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "hard_sigmoid": nn.Hardsigmoid,
        "hard_swish": nn.Hardswish,
        "hard_tanh": nn.Hardtanh,
    }

    return activation_dict[name]  # type: ignore
