from enum import Enum
from typing import TYPE_CHECKING
from typing import Optional
from typing import TypedDict

if TYPE_CHECKING is True:
    from birder.core.net.base import BaseNet  # pylint: disable=cyclic-import
    from birder.core.net.base import DetectorBackbone  # pylint: disable=cyclic-import
    from birder.core.net.base import PreTrainEncoder  # pylint: disable=cyclic-import
    from birder.core.net.detection.base import DetectionBaseNet  # pylint: disable=cyclic-import
    from birder.core.net.pretraining.base import PreTrainBaseNet  # pylint: disable=cyclic-import

    BaseNetType = type[BaseNet] | type[DetectionBaseNet] | type[PreTrainBaseNet]


ModelInfoType = TypedDict("ModelInfoType", {"sha256": str, "formats": list[str]})


class Task(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_PRETRAINING = "image_pretraining"

    __str__ = str.__str__


_MODEL_REGISTRY: dict[str, ModelInfoType] = {
    "mobilenet_v3_1_0": {
        "sha256": "d014717fbfef85828acdb85076b018af72bab9ac48ff367e10426259b4360d9d",
        "formats": ["pt"],
    },
    "mobilenet_v3_1_0_quantized": {
        "sha256": "3d4c9621077267f0e3e8ad7a5fb05030a3fb9eab3272c89c2caf08e5cd661697",
        "formats": ["pts"],
    },
}


class ModelRegistry:
    def __init__(self) -> None:
        self.nets: dict[str, type["BaseNet"]] = {}
        self.detection_nets: dict[str, type["DetectionBaseNet"]] = {}
        self.pretrain_nets: dict[str, type["PreTrainBaseNet"]] = {}
        self.pretrained_nets = _MODEL_REGISTRY

    @property
    def all_nets(self) -> dict[str, "BaseNetType"]:
        return {**self.nets, **self.detection_nets, **self.pretrain_nets}

    def register_model(self, name: str, net_type: "BaseNetType") -> None:
        if net_type.task == Task.IMAGE_CLASSIFICATION:
            self.nets[name] = net_type

        elif net_type.task == Task.OBJECT_DETECTION:
            self.detection_nets[name] = net_type

        elif net_type.task == Task.IMAGE_PRETRAINING:
            self.pretrain_nets[name] = net_type

        else:
            raise ValueError(f"Unsupported model task: {net_type.task}")

    def register_alias(self, alias: str, net_type: "BaseNetType", net_param: float) -> None:
        if net_type.task == Task.IMAGE_CLASSIFICATION:
            self.nets[alias] = type(alias, (net_type,), {"net_param": net_param})

        elif net_type.task == Task.OBJECT_DETECTION:
            self.detection_nets[alias] = type(alias, (net_type,), {"net_param": net_param})

        elif net_type.task == Task.IMAGE_PRETRAINING:
            self.pretrain_nets[alias] = type(alias, (net_type,), {"net_param": net_param})

        else:
            raise ValueError(f"Unsupported model task: {net_type.task}")

    def _get_model_by_name(self, name: str) -> "BaseNetType":
        if name in self.nets:
            net = self.nets[name]

        elif name in self.detection_nets:
            net = self.detection_nets[name]

        elif name in self.pretrain_nets:
            net = self.pretrain_nets[name]

        else:
            raise ValueError(f"Network with name: {name} not found")

        return net

    def _get_models_for_task(self, task: Task) -> dict[str, "BaseNetType"]:
        if task == Task.IMAGE_CLASSIFICATION:
            nets = self.nets

        elif task == Task.OBJECT_DETECTION:
            nets = self.detection_nets

        elif task == Task.IMAGE_PRETRAINING:
            nets = self.pretrain_nets

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

    def list_pretrained_models(self) -> list[str]:
        return list(self.pretrained_nets.keys())

    def get_default_size(self, model_name: str) -> int:
        net = self._get_model_by_name(model_name)
        return net.default_size

    def get_pretrained_info(self, model_name: str) -> ModelInfoType:
        return self.pretrained_nets[model_name]

    def net_factory(
        self,
        name: str,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> "BaseNet":
        return self.nets[name](input_channels, num_classes, net_param, size)

    def detection_net_factory(
        self,
        name: str,
        num_classes: int,
        backbone: "DetectorBackbone",
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> "DetectorBackbone":
        return self.detection_nets[name](num_classes, backbone, net_param, size)

    def pretrain_net_factory(
        self,
        name: str,
        encoder: "PreTrainEncoder",
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> "PreTrainBaseNet":
        return self.pretrain_nets[name](encoder, net_param, size)


registry = ModelRegistry()
