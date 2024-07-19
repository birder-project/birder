import warnings
from enum import Enum
from typing import TYPE_CHECKING
from typing import Optional

from birder.model_registry import manifest

if TYPE_CHECKING is True:
    from birder.core.net.base import BaseNet  # pylint: disable=cyclic-import
    from birder.core.net.base import DetectorBackbone  # pylint: disable=cyclic-import
    from birder.core.net.base import PreTrainEncoder  # pylint: disable=cyclic-import
    from birder.core.net.detection.base import DetectionBaseNet  # pylint: disable=cyclic-import
    from birder.core.net.pretraining.base import PreTrainBaseNet  # pylint: disable=cyclic-import

    BaseNetType = type[BaseNet] | type[DetectionBaseNet] | type[PreTrainBaseNet]


class Task(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_PRETRAINING = "image_pretraining"

    __str__ = str.__str__


class ModelRegistry:
    def __init__(self) -> None:
        self.aliases: dict[str, BaseNetType] = {}
        self._nets: dict[str, type["BaseNet"]] = {}
        self._detection_nets: dict[str, type["DetectionBaseNet"]] = {}
        self._pretrain_nets: dict[str, type["PreTrainBaseNet"]] = {}
        self._pretrained_nets = manifest.REGISTRY_MANIFEST

    @property
    def all_nets(self) -> dict[str, "BaseNetType"]:
        return {**self._nets, **self._detection_nets, **self._pretrain_nets}

    def register_model(self, name: str, net_type: "BaseNetType") -> None:
        if net_type.task == Task.IMAGE_CLASSIFICATION:
            if name in self._nets:
                warnings.warn(f"Network named {name} is already registered", UserWarning)

            self._nets[name] = net_type

        elif net_type.task == Task.OBJECT_DETECTION:
            if name in self._detection_nets:
                warnings.warn(f"Detection network named {name} is already registered", UserWarning)

            self._detection_nets[name] = net_type

        elif net_type.task == Task.IMAGE_PRETRAINING:
            if name in self._pretrain_nets:
                warnings.warn(f"Pretrain network named {name} is already registered", UserWarning)

            self._pretrain_nets[name] = net_type

        else:
            raise ValueError(f"Unsupported model task: {net_type.task}")

    def register_alias(self, alias: str, net_type: "BaseNetType", net_param: float) -> None:
        """
        Just by defining the `type(alias, (net_type,), ...) the network is registered
        no further registration is needed.
        The aliases dictionary is kept only for bookkeeping.
        """

        if alias in self.aliases:
            warnings.warn(f"Alias {alias} is already registered", UserWarning)

        self.aliases[alias] = type(alias, (net_type,), {"net_param": net_param})

    def _get_model_by_name(self, name: str) -> "BaseNetType":
        if name in self._nets:
            net = self._nets[name]
        elif name in self._detection_nets:
            net = self._detection_nets[name]
        elif name in self._pretrain_nets:
            net = self._pretrain_nets[name]
        else:
            raise ValueError(f"Network with name: {name} not found")

        return net

    def _get_models_for_task(self, task: Task) -> dict[str, "BaseNetType"]:
        if task == Task.IMAGE_CLASSIFICATION:
            nets = self._nets
        elif task == Task.OBJECT_DETECTION:
            nets = self._detection_nets
        elif task == Task.IMAGE_PRETRAINING:
            nets = self._pretrain_nets
        else:
            raise ValueError(f"Unsupported model task: {task}")

        return nets

    def list_models(self, *, task: Optional[Task] = None, net_type: Optional[type] = None) -> list[str]:
        nets = self.all_nets
        if task is not None:
            nets = self._get_models_for_task(task)

        if net_type is not None:
            nets = {name: t for name, t in nets.items() if issubclass(t, net_type) is True}

        return list(nets.keys())

    def exists(self, name: str, task: Optional[Task] = None, net_type: Optional[type] = None) -> bool:
        nets = self.all_nets
        if task is not None:
            nets = self._get_models_for_task(task)

        if net_type is not None:
            nets = {name: t for name, t in nets.items() if issubclass(t, net_type) is True}

        return name in nets

    def list_pretrained_models(self) -> list[str]:
        return list(self._pretrained_nets.keys())

    def get_default_size(self, model_name: str) -> int:
        net = self._get_model_by_name(model_name)
        return net.default_size

    def get_pretrained_info(self, model_name: str) -> manifest.ModelInfoType:
        return self._pretrained_nets[model_name]

    def net_factory(
        self,
        name: str,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> "BaseNet":
        return self._nets[name](input_channels, num_classes, net_param, size)

    def detection_net_factory(
        self,
        name: str,
        num_classes: int,
        backbone: "DetectorBackbone",
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> "DetectorBackbone":
        return self._detection_nets[name](num_classes, backbone, net_param, size)

    def pretrain_net_factory(
        self,
        name: str,
        encoder: "PreTrainEncoder",
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> "PreTrainBaseNet":
        return self._pretrain_nets[name](encoder, net_param, size)


registry = ModelRegistry()
