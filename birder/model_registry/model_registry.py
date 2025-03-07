import fnmatch
import warnings
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional

from birder.model_registry import manifest

if TYPE_CHECKING is True:
    from birder.net.base import BaseNet  # pylint: disable=cyclic-import
    from birder.net.base import DetectorBackbone  # pylint: disable=cyclic-import
    from birder.net.base import PreTrainEncoder  # pylint: disable=cyclic-import
    from birder.net.detection.base import DetectionBaseNet  # pylint: disable=cyclic-import
    from birder.net.mim.base import MIMBaseNet  # pylint: disable=cyclic-import

    BaseNetObjType = BaseNet | DetectionBaseNet | MIMBaseNet
    BaseNetType = type[BaseNet] | type[DetectionBaseNet] | type[MIMBaseNet]


def group_sort(model_list: list[str]) -> list[str]:
    # Sort by model group for visibility
    index_map = {item: index for index, item in enumerate(model_list)}
    model_list = sorted(model_list, key=lambda x: (x.split("_")[0], index_map[x]))
    return model_list


class Task(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    MASKED_IMAGE_MODELING = "masked_image_modeling"
    SELF_SUPERVISED_LEARNING = "self_supervised_learning"

    __str__ = str.__str__


class ModelRegistry:
    def __init__(self) -> None:
        self.aliases: dict[str, "BaseNetType"] = {}
        self._nets: dict[str, type["BaseNet"]] = {}
        self._detection_nets: dict[str, type["DetectionBaseNet"]] = {}
        self._mim_nets: dict[str, type["MIMBaseNet"]] = {}
        self._pretrained_nets = manifest.REGISTRY_MANIFEST

    @property
    def all_nets(self) -> dict[str, "BaseNetType"]:
        return {**self._nets, **self._detection_nets, **self._mim_nets}

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

        else:
            raise ValueError(f"Unsupported model task: {net_type.task}")

    def register_alias(
        self,
        alias: str,
        net_type: "BaseNetType",
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Just by defining the `type(alias, (net_type,), ...) the network is registered
        no further registration is needed.
        The aliases dictionary is kept only for bookkeeping.
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

        manifest.REGISTRY_MANIFEST[name] = weights_info
        self._pretrained_nets[name] = weights_info

    def _get_model_by_name(self, name: str) -> "BaseNetType":
        if name in self._nets:
            net = self._nets[name]
        elif name in self._detection_nets:
            net = self._detection_nets[name]
        elif name in self._mim_nets:
            net = self._mim_nets[name]
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
        else:
            raise ValueError(f"Unsupported model task: {task}")

        return nets

    def list_models(
        self, *, include_filter: Optional[str] = None, task: Optional[Task] = None, net_type: Optional[type] = None
    ) -> list[str]:
        nets = self.all_nets
        if task is not None:
            nets = self._get_models_for_task(task)

        if net_type is not None:
            nets = {name: t for name, t in nets.items() if issubclass(t, net_type) is True}

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

    def list_pretrained_models(self, include_filter: Optional[str] = None) -> list[str]:
        """
        Parameters
        ----------
        include_filter
            Filter string that goes into fnmatch

        Returns
        -------
        Sorted models list (by model group) of pretrained networks.
        """

        model_list = list(self._pretrained_nets.keys())

        if include_filter is not None:
            model_list = fnmatch.filter(model_list, include_filter)

        return group_sort(model_list)

    def pretrained_exists(self, model_name: str) -> bool:
        return model_name in self._pretrained_nets

    def get_default_size(self, model_name: str) -> tuple[int, int]:
        net = self._get_model_by_name(model_name)
        return net.default_size

    def get_pretrained_metadata(self, model_name: str) -> manifest.ModelMetadataType:
        return self._pretrained_nets[model_name]

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
