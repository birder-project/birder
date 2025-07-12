import fnmatch
import warnings
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Optional

from birder.model_registry import manifest

if TYPE_CHECKING is True:
    from birder.net.base import BaseNet  # pylint: disable=cyclic-import
    from birder.net.base import DetectorBackbone  # pylint: disable=cyclic-import
    from birder.net.base import PreTrainEncoder  # pylint: disable=cyclic-import
    from birder.net.detection.base import DetectionBaseNet  # pylint: disable=cyclic-import
    from birder.net.mim.base import MIMBaseNet  # pylint: disable=cyclic-import
    from birder.net.ssl.base import SSLBaseNet  # pylint: disable=cyclic-import

    BaseNetObjType = BaseNet | DetectionBaseNet | MIMBaseNet | SSLBaseNet
    BaseNetType = type[BaseNet] | type[DetectionBaseNet] | type[MIMBaseNet] | type[SSLBaseNet]


def group_sort(model_list: list[str]) -> list[str]:
    # Sort by model group for visibility
    index_map = {item: index for index, item in enumerate(model_list)}
    model_list = sorted(model_list, key=lambda x: (x.split("_")[0], index_map[x]))
    return model_list


class Task(StrEnum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    MASKED_IMAGE_MODELING = "masked_image_modeling"
    SELF_SUPERVISED_LEARNING = "self_supervised_learning"


class ModelRegistry:
    def __init__(self) -> None:
        self.aliases: dict[str, "BaseNetType"] = {}
        self._nets: dict[str, type["BaseNet"]] = {}
        self._detection_nets: dict[str, type["DetectionBaseNet"]] = {}
        self._mim_nets: dict[str, type["MIMBaseNet"]] = {}
        self._ssl_nets: dict[str, type["SSLBaseNet"]] = {}
        self._pretrained_nets = manifest.REGISTRY_MANIFEST

    @property
    def all_nets(self) -> dict[str, "BaseNetType"]:
        return {**self._nets, **self._detection_nets, **self._mim_nets, **self._ssl_nets}

    def register_model(self, name: str, net_type: "BaseNetType") -> None:
        if net_type.task == Task.IMAGE_CLASSIFICATION:
            if name in self._nets:
                warnings.warn(f"Network named {name} is already registered", UserWarning)

            self._nets[name] = net_type

        elif net_type.task == Task.OBJECT_DETECTION:
            if name in self._detection_nets:
                warnings.warn(f"Detection network named {name} is already registered", UserWarning)

            self._detection_nets[name] = net_type

        elif net_type.task == Task.MASKED_IMAGE_MODELING:
            if name in self._mim_nets:
                warnings.warn(f"MIM network named {name} is already registered", UserWarning)

            self._mim_nets[name] = net_type

        elif net_type.task == Task.SELF_SUPERVISED_LEARNING:
            if name in self._ssl_nets:
                warnings.warn(f"SSL network named {name} is already registered", UserWarning)

            self._ssl_nets[name] = net_type

        else:
            raise ValueError(f"Unsupported model task: {net_type.task}")

    def register_model_config(
        self,
        alias: str,
        net_type: "BaseNetType",
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        When auto_register is active, just by defining the `type(alias, (net_type,), ...) the network is registered
        no further registration is needed.
        """

        if net_type.auto_register is False:
            # Register the model manually, as the base class doesn't take care of that for us
            registry.register_model(alias, type(alias, (net_type,), {"net_param": net_param, "config": config}))

        if alias in self.aliases:
            warnings.warn(f"Alias {alias} is already registered", UserWarning)

        self.aliases[alias] = type(alias, (net_type,), {"net_param": net_param, "config": config})

    def register_weights(self, name: str, weights_info: manifest.ModelMetadataType) -> None:
        if name in self._pretrained_nets:
            warnings.warn(f"Weights {name} is already registered", UserWarning)

        if "task" not in weights_info:
            weights_info["task"] = self.all_nets[weights_info["net"]["network"]].task

        manifest.REGISTRY_MANIFEST[name] = weights_info
        self._pretrained_nets[name] = weights_info

    def _get_model_by_name(self, name: str) -> "BaseNetType":
        if name in self._nets:
            net = self._nets[name]
        elif name in self._detection_nets:
            net = self._detection_nets[name]
        elif name in self._mim_nets:
            net = self._mim_nets[name]
        elif name in self._ssl_nets:
            net = self._ssl_nets[name]
        else:
            raise ValueError(f"Network with name: {name} not found")

        return net

    def _get_models_for_task(self, task: Task) -> dict[str, "BaseNetType"]:
        if task == Task.IMAGE_CLASSIFICATION:
            nets = self._nets
        elif task == Task.OBJECT_DETECTION:
            nets = self._detection_nets
        elif task == Task.MASKED_IMAGE_MODELING:
            nets = self._mim_nets
        elif task == Task.SELF_SUPERVISED_LEARNING:
            nets = self._ssl_nets
        else:
            raise ValueError(f"Unsupported model task: {task}")

        return nets

    def list_models(
        self,
        *,
        include_filter: Optional[str] = None,
        task: Optional[Task] = None,
        net_type: Optional[type | tuple[type, ...]] = None,
        net_type_op: Literal["AND", "OR"] = "AND",
    ) -> list[str]:
        nets = self.all_nets
        if task is not None:
            nets = self._get_models_for_task(task)

        if net_type is not None:
            if not isinstance(net_type, tuple):
                net_type = (net_type,)

            if net_type_op == "OR":
                nets = {name: t for name, t in nets.items() if issubclass(t, net_type) is True}
            elif net_type_op == "AND":
                nets = {name: t for name, t in nets.items() if all(issubclass(t, nt) for nt in net_type)}
            else:
                raise ValueError(f"Unknown op {net_type_op}")

        model_list = list(nets.keys())
        if include_filter is not None:
            model_list = fnmatch.filter(model_list, include_filter)

        return model_list

    def exists(self, name: str, task: Optional[Task] = None, net_type: Optional[type] = None) -> bool:
        nets = self.all_nets
        if task is not None:
            nets = self._get_models_for_task(task)

        if net_type is not None:
            nets = {name: t for name, t in nets.items() if issubclass(t, net_type) is True}

        return name in nets

    def get_model_base_name(self, model: "BaseNetObjType") -> str:
        type_name = model.__class__.__name__.lower()
        if type_name in self.aliases:
            type_name = model.__class__.__bases__[0].__name__.lower()

        return type_name

    def get_model_alias(self, model: "BaseNetObjType") -> Optional[str]:
        type_name = model.__class__.__name__.lower()
        if type_name in self.aliases:
            return type_name

        return None

    def list_pretrained_models(self, include_filter: Optional[str] = None, task: Optional[Task] = None) -> list[str]:
        """
        Parameters
        ----------
        include_filter
            Filter string that goes into fnmatch
        task
            Filter by network task

        Returns
        -------
        Sorted models list (by model group) of pretrained networks.
        """

        model_list = list(self._pretrained_nets.keys())

        if include_filter is not None:
            model_list = fnmatch.filter(model_list, include_filter)

        if task is not None:
            model_list = [model_name for model_name in model_list if self._pretrained_nets[model_name]["task"] == task]

        return group_sort(model_list)

    def pretrained_exists(self, model_name: str) -> bool:
        return model_name in self._pretrained_nets

    def get_default_size(self, model_name: str) -> tuple[int, int]:
        net = self._get_model_by_name(model_name)
        return net.default_size

    def get_pretrained_metadata(self, model_name: str) -> manifest.ModelMetadataType:
        metadata = self._pretrained_nets[model_name]
        if "task" not in metadata:
            metadata["task"] = self.all_nets[metadata["net"]["network"]].task

        return metadata

    def net_factory(
        self,
        name: str,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> "BaseNet":
        return self._nets[name](input_channels, num_classes, net_param=net_param, config=config, size=size)

    def detection_net_factory(
        self,
        name: str,
        num_classes: int,
        backbone: "DetectorBackbone",
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> "DetectionBaseNet":
        return self._detection_nets[name](num_classes, backbone, net_param=net_param, config=config, size=size)

    def mim_net_factory(
        self,
        name: str,
        encoder: "PreTrainEncoder",
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> "MIMBaseNet":
        return self._mim_nets[name](encoder, net_param=net_param, config=config, size=size)


registry = ModelRegistry()
list_pretrained_models = registry.list_pretrained_models
