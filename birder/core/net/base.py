from collections.abc import Callable
from typing import Optional
from typing import TypedDict

import torch
from torch import nn

DataShapeType = TypedDict("DataShapeType", {"data_shape": list[int]})
SignatureType = TypedDict("SignatureType", {"inputs": list[DataShapeType], "outputs": list[DataShapeType]})


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
    default_size: int
    task = "image_classification"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.__name__ in ["PreTrainEncoder", "DetectorBackbone"]:
            # Exclude all other base classes here
            return

        _REGISTERED_NETWORKS[cls.__name__.lower()] = cls

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        if hasattr(self, "net_param") is False:  # Avoid overriding aliases
            self.net_param = net_param

        if size is not None:
            self.size = size

        else:
            self.size = self.default_size

        assert isinstance(self.size, int)
        self.classifier: nn.Module
        self.embedding_size: int

    def create_classifier(self) -> nn.Module:
        raise NotImplementedError

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.classifier = self.create_classifier()

    def adjust_size(self, new_size: int) -> None:
        """
        Override this when one time adjustments for different resolutions is required.
        This should run after load_state_dict.
        """

        self.size = new_size

    def freeze(self, freeze_classifier: bool = True) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.classifier.parameters():
                param.requires_grad = True

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        return self.classify(x)


def net_factory(
    name: str,
    input_channels: int,
    num_classes: int,
    net_param: Optional[float] = None,
    size: Optional[int] = None,
) -> BaseNet:
    return _REGISTERED_NETWORKS[name](input_channels, num_classes, net_param, size)


def create_alias(alias: str, module: type[BaseNet], net_param: float) -> None:
    _REGISTERED_NETWORKS[alias] = type(alias, (module,), {"net_param": net_param})


_REGISTERED_NETWORKS: dict[str, type[BaseNet]] = {}


class PreTrainEncoder(BaseNet):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        self.encoding_size: int
        self.decoder_block: Callable[[int], nn.Module]

    def masked_encoding(
        self, x: torch.Tensor, mask_ratio: float, mask_token: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError


class DetectorBackbone(BaseNet):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        self.return_stages = ["stage1", "stage2", "stage3", "stage4"]
        self.return_channels: list[int]

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def freeze_stages(self, up_to_stage: int) -> None:
        raise NotImplementedError


def network_names_filter(t: type) -> list[str]:
    network_names = []
    for name, network_type in _REGISTERED_NETWORKS.items():
        if issubclass(network_type, t) is True:
            network_names.append(name)

    return network_names
