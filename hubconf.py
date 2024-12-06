# pylint: disable=wrong-import-position
dependencies = ["torch"]

import torch  # noqa: E402

import birder  # noqa: E402
import birder.model_registry  # noqa: E402


def _load(model: str, pretrained: bool = True, input_channels: int = 3, num_classes: int = 0) -> torch.nn.Module:
    dst = torch.hub.get_dir() + f"/{model}.pt"
    if pretrained is True:
        return birder.load_pretrained_model(model, dst=dst)

    model_info = birder.model_registry.registry.get_pretrained_info(model)
    network = model_info["net"]["network"]
    net_param = model_info["net"].get("net_param", None)
    reparameterized = model_info["net"].get("reparameterized", False)
    net = birder.model_registry.registry.net_factory(network, input_channels, num_classes, net_param=net_param)
    if reparameterized is True:
        net.reparameterize_model()

    return net


_model_dict = {}
for m in birder.list_pretrained_models():
    _model_dict[m.replace("-", "_")] = _load

globals().update(_model_dict)
