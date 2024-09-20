import os
from typing import Optional

from birder.net.base import SignatureType
from birder.net.detection.base import DetectionSignatureType


def get_size_from_signature(signature: SignatureType | DetectionSignatureType) -> tuple[int, int]:
    return tuple(signature["inputs"][0]["data_shape"][2:4])  # type: ignore[return-value]


def get_channels_from_signature(signature: SignatureType | DetectionSignatureType) -> int:
    return signature["inputs"][0]["data_shape"][1]


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
    prefix = get_network_name(network, net_param, backbone_tag)
    suffix = get_network_name(backbone, backbone_param, tag)
    return f"{prefix}_{suffix}"


def detection_class_to_idx(class_to_idx: dict[str, int]) -> dict[str, int]:
    # Give place to "background" class (always index 0)
    for key in class_to_idx:
        class_to_idx[key] += 1

    return class_to_idx
