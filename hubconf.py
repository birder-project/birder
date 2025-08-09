# pylint: disable=wrong-import-position
dependencies = ["torch"]

from functools import partial as _partial  # noqa: E402
from typing import Optional  # noqa: E402

import torch  # noqa: E402

import birder  # noqa: E402
import birder.model_registry  # noqa: E402
from birder.model_registry import Task  # noqa: E402


def _load(
    model: str,
    pretrained: bool = True,
    input_channels: int = 3,
    num_classes: int = 0,
    size: Optional[tuple[int, int]] = None,
) -> torch.nn.Module:
    dst = torch.hub.get_dir() + f"/{model}.pt"
    if pretrained is True:
        return birder.load_pretrained_model(model, dst=dst)[0]

    model_metadata = birder.model_registry.registry.get_pretrained_metadata(model)
    if model_metadata["task"] == Task.IMAGE_CLASSIFICATION:
        network = model_metadata["net"]["network"]
        net_param = model_metadata["net"].get("net_param", None)
        reparameterized = model_metadata["net"].get("reparameterized", False)
        net = birder.model_registry.registry.net_factory(
            network, input_channels, num_classes, net_param=net_param, size=size
        )
        if reparameterized is True:
            net.reparameterize_model()

        return net

    if model_metadata["task"] == Task.OBJECT_DETECTION:
        network = model_metadata["net"]["network"]
        net_param = model_metadata["net"].get("net_param", None)
        reparameterized = model_metadata["net"].get("reparameterized", False)
        backbone = model_metadata["backbone"]["network"]
        backbone_param = model_metadata["backbone"].get("net_param", None)
        backbone_reparameterized = model_metadata["backbone"].get("reparameterized", False)

        net_backbone = birder.model_registry.registry.net_factory(
            backbone, input_channels, num_classes, net_param=backbone_param, size=size
        )
        if backbone_reparameterized is True:
            net_backbone.reparameterize_model()

        net = birder.model_registry.registry.detection_net_factory(
            network, num_classes, net_backbone, net_param=net_param, size=size
        )
        if reparameterized is True:
            net.reparameterize_model()

        return net

    raise ValueError(f"Unknown model type: {model_metadata['task']}")


_model_dict = {}
for m in birder.list_pretrained_models():
    _model_dict[m.replace("-", "_")] = _partial(_load, m)

globals().update(_model_dict)
