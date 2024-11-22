import os
from typing import Any
from typing import Optional

from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.base import SignatureType
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.base import DetectionSignatureType
from birder.transforms.classification import RGBType


def get_size_from_signature(signature: SignatureType | DetectionSignatureType) -> tuple[int, int]:
    return tuple(signature["inputs"][0]["data_shape"][2:4])  # type: ignore[return-value]


def get_channels_from_signature(signature: SignatureType | DetectionSignatureType) -> int:
    return signature["inputs"][0]["data_shape"][1]


def get_num_labels_from_signature(signature: SignatureType | DetectionSignatureType) -> int:
    if "num_labels" in signature:
        return signature["num_labels"]  # type: ignore[typeddict-item]

    return signature["outputs"][0]["data_shape"][1]


def get_label_from_path(path: str) -> str:
    """
    Returns the last directory from the path.

    For data/validation/Alpine swift/000001.npy return value will be 'Alpine swift'.
    """

    return os.path.basename(os.path.dirname(path))


def get_network_name(network: str, net_param: Optional[float], tag: Optional[str] = None) -> str:
    if net_param is not None:
        if int(net_param) == net_param:
            net_param = int(net_param)

        network_name = f"{network}_{net_param}"

    else:
        network_name = network

    if tag is not None:
        network_name = f"{network_name}_{tag}"

    return network_name


def get_mim_network_name(
    network: str,
    net_param: Optional[float],
    encoder: str,
    encoder_param: Optional[float],
    tag: Optional[str] = None,
) -> str:
    prefix = get_network_name(network, net_param)
    suffix = get_network_name(encoder, encoder_param, tag)
    return f"{prefix}_{suffix}"


def get_detection_network_name(
    network: str,
    net_param: Optional[float],
    tag: Optional[str],
    backbone: str,
    backbone_param: Optional[float],
    backbone_tag: Optional[str],
) -> str:
    prefix = get_network_name(network, net_param, tag)
    suffix = get_network_name(backbone, backbone_param, backbone_tag)
    return f"{prefix}_{suffix}"


def detection_class_to_idx(class_to_idx: dict[str, int]) -> dict[str, int]:
    # Give place to "background" class (always index 0)
    for key in class_to_idx:
        class_to_idx[key] += 1

    return class_to_idx


def get_network_config(
    net: BaseNet | DetectionBaseNet, signature: SignatureType | DetectionSignatureType, rgb_stats: RGBType
) -> dict[str, Any]:
    model_name = registry.get_model_base_name(net)
    alias = registry.get_model_alias(net)
    net_param = None
    model_config = None
    if net.config is not None:
        model_config = net.config
    if net.net_param is not None:
        net_param = net.net_param

    backbone_config: dict[str, Any] = {}
    if isinstance(net, DetectionBaseNet):
        backbone_config["backbone"] = registry.get_model_base_name(net.backbone)
        backbone_config["backbone_alias"] = registry.get_model_alias(net.backbone)
        if net.backbone.config is not None:
            backbone_config["backbone_config"] = net.backbone.config
        if net.backbone.net_param is not None:
            backbone_config["backbone_net_param"] = net.backbone.net_param

    net_config = {
        "name": model_name,
        "alias": alias,
        "task": net.task,
        "net_param": net_param,
        "model_config": model_config,
        **backbone_config,
        "signature": signature,
        "rgb_stats": rgb_stats,
    }

    return net_config
