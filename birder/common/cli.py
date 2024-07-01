import argparse
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

import torch
from torchvision.datasets.folder import IMG_EXTENSIONS

from birder.common import lib
from birder.common.lib import get_detection_network_name
from birder.common.lib import get_network_name
from birder.common.lib import get_pretrain_network_name
from birder.conf import settings
from birder.core.net.base import BaseNet
from birder.core.net.base import SignatureType
from birder.core.net.base import net_factory
from birder.core.net.detection.base import DetectionBaseNet
from birder.core.net.detection.base import DetectionSignatureType
from birder.core.net.detection.base import detection_net_factory
from birder.core.net.pretraining.base import PreTrainBaseNet
from birder.core.net.pretraining.base import pretrain_net_factory
from birder.core.transforms.classification import RGBType


class ArgumentHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def write_signature(network_name: str, signature: Union[SignatureType, DetectionSignatureType]) -> None:
    signature_file = settings.MODELS_DIR.joinpath(f"{network_name}.json")
    logging.info(f"Writing {signature_file}")
    with open(signature_file, "w", encoding="utf-8") as handle:
        json.dump(signature, handle, indent=2)


def read_signature(network_name: str) -> Union[SignatureType, DetectionSignatureType]:
    signature_file = settings.MODELS_DIR.joinpath(f"{network_name}.json")
    logging.info(f"Reading {signature_file}")
    with open(signature_file, "r", encoding="utf-8") as handle:
        signature: Union[SignatureType, DetectionSignatureType] = json.load(handle)

    return signature


def read_class_file(path: Union[Path, str]) -> dict[str, int]:
    if Path(path).exists() is False:
        logging.warning(f"Class file '{path}' not found... class_to_idx returns empty")
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        class_list = handle.read().splitlines()

    class_to_idx = {k: v for v, k in enumerate(class_list)}

    return class_to_idx


def model_path(
    network_name: str,
    *,
    epoch: Optional[int] = None,
    quantized: bool = False,
    script: bool = False,
    lite: bool = False,
    pt2: bool = False,
    onnx: bool = False,
    states: bool = False,
) -> Path:
    """
    Return the file path of a model
    """

    if epoch is not None:
        file_name = f"{network_name}_{epoch}"

    else:
        file_name = network_name

    if quantized is True:
        file_name = f"{file_name}_quantized"

    if states is True:
        file_name = f"{file_name}_states"

    elif lite is True:
        file_name = f"{file_name}.ptl"

    elif pt2 is True:
        file_name = f"{file_name}.pt2"

    elif onnx is True:
        file_name = f"{file_name}.onnx"

    elif script is True:
        file_name = f"{file_name}.pts"

    else:
        file_name = f"{file_name}.pt"

    return settings.MODELS_DIR.joinpath(file_name)


def _checkpoint_states(
    states_path: Path,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.grad_scaler.GradScaler],
) -> None:
    if optimizer is None or scheduler is None:
        return

    if scaler is not None:
        scaler_state = scaler.state_dict()

    else:
        scaler_state = None

    torch.save(
        {
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler_state,
        },
        states_path,
    )


def checkpoint_model(
    network_name: str,
    epoch: int,
    net: torch.nn.Module,
    signature: Union[SignatureType, DetectionSignatureType],
    class_to_idx: dict[str, int],
    rgb_values: RGBType,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.grad_scaler.GradScaler],
) -> None:
    path = model_path(network_name, epoch=epoch, script=False)
    states_path = model_path(network_name, epoch=epoch, script=False, states=True)
    logging.info(f"Saving model checkpoint {path}...")
    torch.save(
        {
            "state": net.state_dict(),
            "task": net.task,
            "signature": signature,
            "class_to_idx": class_to_idx,
            "rgb_values": rgb_values,
        },
        path,
    )

    _checkpoint_states(states_path, optimizer, scheduler, scaler)


def _load_states(states_path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if states_path.exists() is True:
        states_dict: dict[str, Any] = torch.load(states_path, map_location=device)
        optimizer_state = states_dict["optimizer_state"]
        scheduler_state = states_dict["scheduler_state"]
        scaler_state = states_dict["scaler_state"]

    else:
        logging.warning(f"States file '{states_path}' not found")
        optimizer_state = {}
        scheduler_state = {}
        scaler_state = {}

    return (optimizer_state, scheduler_state, scaler_state)


def load_checkpoint(
    device: torch.device,
    network: str,
    *,
    net_param: Optional[float],
    tag: Optional[str] = None,
    epoch: Optional[int] = None,
    new_size: Optional[int] = None,
) -> tuple[BaseNet, dict[str, int], dict[str, Any], dict[str, Any], dict[str, Any]]:
    network_name = get_network_name(network, net_param, tag)
    path = model_path(network_name, epoch=epoch, script=False)
    states_path = model_path(network_name, epoch=epoch, script=False, states=True)
    logging.info(f"Loading model from {path} on device {device}...")

    model_dict: dict[str, Any] = torch.load(path, map_location=device)

    signature: SignatureType = model_dict["signature"]
    input_channels = signature["inputs"][0]["data_shape"][1]
    num_classes = signature["outputs"][0]["data_shape"][1]
    size = signature["inputs"][0]["data_shape"][2]
    net = net_factory(network, input_channels, num_classes, net_param=net_param, size=size)
    net.load_state_dict(model_dict["state"])
    if new_size is not None:
        net.adjust_size(new_size)

    net.to(device)
    class_to_idx: dict[str, int] = model_dict["class_to_idx"]
    (optimizer_state, scheduler_state, scaler_state) = _load_states(states_path, device)

    return (net, class_to_idx, optimizer_state, scheduler_state, scaler_state)


def load_pretrain_checkpoint(
    device: torch.device,
    network: str,
    *,
    net_param: Optional[float],
    encoder: str,
    encoder_param: Optional[float],
    tag: Optional[str] = None,
    epoch: Optional[int] = None,
) -> tuple[PreTrainBaseNet, dict[str, Any], dict[str, Any], dict[str, Any]]:
    network_name = get_pretrain_network_name(
        network, net_param=net_param, encoder=encoder, encoder_param=encoder_param, tag=tag
    )
    path = model_path(network_name, epoch=epoch, script=False)
    states_path = model_path(network_name, epoch=epoch, script=False, states=True)
    logging.info(f"Loading model from {path} on device {device}...")

    model_dict: dict[str, Any] = torch.load(path, map_location=device)

    signature: SignatureType = model_dict["signature"]
    input_channels = signature["inputs"][0]["data_shape"][1]
    num_classes = signature["outputs"][0]["data_shape"][1]
    size = signature["inputs"][0]["data_shape"][2]
    net_encoder = net_factory(encoder, input_channels, num_classes, net_param=encoder_param, size=size)
    net = pretrain_net_factory(network, net_encoder, net_param, size)
    net.load_state_dict(model_dict["state"])
    net.to(device)

    (optimizer_state, scheduler_state, scaler_state) = _load_states(states_path, device)

    return (net, optimizer_state, scheduler_state, scaler_state)


def load_detection_checkpoint(
    device: torch.device,
    network: str,
    *,
    net_param: Optional[float],
    tag: Optional[str] = None,
    backbone: str,
    backbone_param: Optional[float],
    backbone_tag: Optional[str],
    epoch: Optional[int] = None,
) -> tuple[DetectionBaseNet, dict[str, int], dict[str, Any], dict[str, Any], dict[str, Any]]:
    network_name = get_detection_network_name(
        network,
        net_param=net_param,
        tag=tag,
        backbone=backbone,
        backbone_param=backbone_param,
        backbone_tag=backbone_tag,
    )
    path = model_path(network_name, epoch=epoch, script=False)
    states_path = model_path(network_name, epoch=epoch, script=False, states=True)
    logging.info(f"Loading model from {path} on device {device}...")

    model_dict: dict[str, Any] = torch.load(path, map_location=device)

    signature: DetectionSignatureType = model_dict["signature"]
    input_channels = signature["inputs"][0]["data_shape"][1]
    num_classes = signature["num_labels"]
    size = signature["inputs"][0]["data_shape"][2]
    net_backbone = net_factory(backbone, input_channels, num_classes, net_param=backbone_param, size=size)
    net = detection_net_factory(network, num_classes, net_backbone, net_param, size)
    net.load_state_dict(model_dict["state"])
    net.to(device)

    class_to_idx: dict[str, int] = model_dict["class_to_idx"]
    (optimizer_state, scheduler_state, scaler_state) = _load_states(states_path, device)

    return (net, class_to_idx, optimizer_state, scheduler_state, scaler_state)


def load_model(
    device: torch.device,
    network: str,
    *,
    net_param: Optional[float] = None,
    tag: Optional[str] = None,
    epoch: Optional[int] = None,
    new_size: Optional[int] = None,
    quantized: bool = False,
    inference: bool,
    script: bool = False,
    pt2: bool = False,
) -> tuple[Union[torch.nn.Module, torch.ScriptModule], dict[str, int], SignatureType, RGBType]:
    network_name = get_network_name(network, net_param, tag)
    path = model_path(network_name, epoch=epoch, quantized=quantized, script=script, pt2=pt2)
    logging.info(f"Loading model from {path} on device {device}...")

    if script is True:
        extra_files = {"task": "", "class_to_idx": "", "signature": "", "rgb_values": ""}
        net = torch.jit.load(path, map_location=device, _extra_files=extra_files)
        _ = extra_files["task"]
        class_to_idx: dict[str, int] = json.loads(extra_files["class_to_idx"])
        signature: SignatureType = json.loads(extra_files["signature"])
        rgb_values: RGBType = json.loads(extra_files["rgb_values"])

    elif pt2 is True:
        extra_files = {"task": "", "class_to_idx": "", "signature": "", "rgb_values": ""}
        net = torch.export.load(path, extra_files=extra_files).module()
        net.to(device)
        _ = extra_files["task"]
        class_to_idx = json.loads(extra_files["class_to_idx"])
        signature = json.loads(extra_files["signature"])
        rgb_values = json.loads(extra_files["rgb_values"])

    else:
        model_dict: dict[str, Any] = torch.load(path, map_location=device)
        signature = model_dict["signature"]
        input_channels = signature["inputs"][0]["data_shape"][1]
        num_classes = signature["outputs"][0]["data_shape"][1]
        size = signature["inputs"][0]["data_shape"][2]

        net = net_factory(network, input_channels, num_classes, net_param=net_param, size=size)
        net.load_state_dict(model_dict["state"])
        if new_size is not None:
            net.adjust_size(new_size)

        net.to(device)
        class_to_idx = model_dict["class_to_idx"]
        rgb_values = model_dict["rgb_values"]

    if inference is True:
        for param in net.parameters():
            param.requires_grad = False

        if pt2 is False:  # Remove when GraphModule add support for 'eval'
            net.eval()

    return (net, class_to_idx, signature, rgb_values)


# pylint:disable=too-many-locals
def load_detection_model(
    device: torch.device,
    network: str,
    *,
    net_param: Optional[float] = None,
    tag: Optional[str] = None,
    backbone: str,
    backbone_param: Optional[float],
    backbone_tag: Optional[str],
    epoch: Optional[int] = None,
    new_size: Optional[int] = None,
    quantized: bool = False,
    inference: bool,
    script: bool = False,
) -> tuple[Union[torch.nn.Module, torch.ScriptModule], dict[str, int], DetectionSignatureType, RGBType]:
    network_name = get_detection_network_name(
        network,
        net_param=net_param,
        tag=tag,
        backbone=backbone,
        backbone_param=backbone_param,
        backbone_tag=backbone_tag,
    )
    path = model_path(network_name, epoch=epoch, quantized=quantized, script=script)
    logging.info(f"Loading model from {path} on device {device}...")

    if script is True:
        extra_files = {"class_to_idx": "", "signature": "", "rgb_values": ""}
        net = torch.jit.load(path, map_location=device, _extra_files=extra_files)
        class_to_idx: dict[str, int] = json.loads(extra_files["class_to_idx"])
        signature: DetectionSignatureType = json.loads(extra_files["signature"])
        rgb_values: RGBType = json.loads(extra_files["rgb_values"])

    else:
        model_dict: dict[str, Any] = torch.load(path, map_location=device)
        signature = model_dict["signature"]
        input_channels = signature["inputs"][0]["data_shape"][1]
        num_classes = signature["num_labels"]
        size = signature["inputs"][0]["data_shape"][2]

        net_backbone = net_factory(backbone, input_channels, num_classes, net_param=backbone_param, size=size)
        net = detection_net_factory(network, num_classes, net_backbone, net_param, size)
        net.load_state_dict(model_dict["state"])
        if new_size is not None:
            net.adjust_size(new_size)

        net.to(device)
        class_to_idx = model_dict["class_to_idx"]
        rgb_values = model_dict["rgb_values"]

    if inference is True:
        for param in net.parameters():
            param.requires_grad = False

        net.eval()

    return (net, class_to_idx, signature, rgb_values)


def file_iter(data_path: str, extensions: list[str]) -> Iterator[str]:
    for path, _dirs, files in os.walk(data_path, followlinks=True):
        files = sorted(files)
        for filename in files:
            file_path = os.path.join(path, filename)
            suffix = os.path.splitext(filename)[1].lower()
            if os.path.isfile(file_path) is True and (suffix in extensions):
                yield file_path


def sample_iter(data_path: str, class_to_idx: dict[str, int]) -> Iterator[tuple[str, int]]:
    """
    Generate file paths of specified path (file path, label)

    If the data path is a directory, the function will recursively walk through the directory,
    including all subdirectories, and yield file paths of any files that have a matching file extension.
    """

    if os.path.isdir(data_path) is True:
        for file_path in file_iter(data_path, extensions=IMG_EXTENSIONS):
            label = lib.get_label_from_path(file_path)
            if label in class_to_idx:
                yield (file_path, class_to_idx[label])

            else:
                yield (file_path, -1)

    else:
        suffix = os.path.splitext(data_path)[1].lower()
        label = lib.get_label_from_path(data_path)
        if suffix in IMG_EXTENSIONS:
            if label in class_to_idx:
                yield (data_path, class_to_idx[label])

            else:
                yield (data_path, -1)


def samples_from_paths(data_paths: list[str], class_to_idx: dict[str, int]) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    for data_path in data_paths:
        samples.extend(sample_iter(data_path, class_to_idx=class_to_idx))

    return sorted(samples)


def wds_braces_from_path(wds_directory: Path) -> tuple[str, int]:
    shard_names = sorted([f.stem for f in wds_directory.glob("*.tar")])
    shard_name = shard_names[0]
    idx = len(shard_name)
    for c in shard_name[::-1]:
        if c != "0":
            break

        idx -= 1

    shard_prefix = shard_name[:idx]
    shard_num_start = shard_names[0][idx:]
    shard_num_end = shard_names[-1][idx:]
    wds_path = f"{wds_directory}/{shard_prefix}{{{shard_num_start}..{shard_num_end}}}.tar"
    num_shards = len(shard_names)

    return (wds_path, num_shards)
