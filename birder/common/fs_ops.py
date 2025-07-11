import json
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import NamedTuple
from typing import Optional
from urllib.request import Request
from urllib.request import urlopen

import torch
import torch.amp
import webdataset as wds
from torchvision.datasets.folder import IMG_EXTENSIONS

from birder.common import cli
from birder.common import lib
from birder.common.lib import get_detection_network_name
from birder.common.lib import get_mim_network_name
from birder.common.lib import get_network_name
from birder.common.lib import get_pretrained_model_url
from birder.conf import settings
from birder.data.transforms.classification import RGBType
from birder.model_registry import Task
from birder.model_registry import registry
from birder.model_registry.manifest import FileFormatType
from birder.net.base import BaseNet
from birder.net.base import SignatureType
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.base import DetectionSignatureType
from birder.net.mim.base import MIMBaseNet
from birder.net.mim.base import MIMSignatureType
from birder.net.ssl.base import SSLSignatureType
from birder.version import __version__

try:
    import safetensors
    import safetensors.torch

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False

logger = logging.getLogger(__name__)


def read_url(url: str | Request) -> str:
    with urlopen(url) as r:  # nosec # allowing all schemas (including file:)
        return r.read().decode(r.headers.get_content_charset("utf-8"))  # type: ignore[no-any-return]


def write_signature(network_name: str, signature: SignatureType | DetectionSignatureType) -> None:
    signature_file = settings.MODELS_DIR.joinpath(f"{network_name}.json")
    logger.info(f"Writing {signature_file}")
    with open(signature_file, "w", encoding="utf-8") as handle:
        json.dump(signature, handle, indent=2)


def read_signature(network_name: str) -> SignatureType | DetectionSignatureType:
    signature_file = settings.MODELS_DIR.joinpath(f"{network_name}.json")
    logger.info(f"Reading {signature_file}")
    with open(signature_file, "r", encoding="utf-8") as handle:
        signature: SignatureType | DetectionSignatureType = json.load(handle)

    return signature


def write_config(
    network_name: str, net: torch.nn.Module, signature: SignatureType | DetectionSignatureType, rgb_stats: RGBType
) -> None:
    model_config = lib.get_network_config(net, signature, rgb_stats)
    config_file = settings.MODELS_DIR.joinpath(f"{network_name}.json")
    logger.info(f"Writing {config_file}")
    with open(config_file, "w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)


def read_config(network_name: str) -> dict[str, Any]:
    config_file = settings.MODELS_DIR.joinpath(f"{network_name}.json")
    logger.info(f"Reading {config_file}")
    with open(config_file, "r", encoding="utf-8") as handle:
        model_config: dict[str, Any] = json.load(handle)

    return model_config


def read_config_from_path(path: str | Path) -> dict[str, Any]:
    logger.info(f"Reading {path}")
    with open(path, "r", encoding="utf-8") as handle:
        model_config: dict[str, Any] = json.load(handle)

    return model_config


def read_class_file(path: str | Path) -> dict[str, int]:
    if isinstance(path, str) and "://" in path:
        class_list = read_url(path).splitlines()
    else:
        with open(path, "r", encoding="utf-8") as handle:
            class_list = handle.read().splitlines()

    class_to_idx = {k: v for v, k in enumerate(class_list)}

    return class_to_idx


def read_json_class_file(path: str | Path) -> dict[str, int]:
    if isinstance(path, str) and "://" in path:
        class_dict = json.loads(read_url(path))
    else:
        with open(path, "r", encoding="utf-8") as handle:
            class_dict = json.load(handle)

    class_to_idx = {k: int(v) for v, k in class_dict.items()}

    return class_to_idx


def read_wds_info(path: str | Path) -> dict[str, Any]:
    if isinstance(path, str) and "://" in path:
        with wds.gopen(path) as handle:
            info: dict[str, Any] = json.load(handle)
    else:
        with open(path, "r", encoding="utf-8") as handle:
            info = json.load(handle)

    return info


def model_path(
    network_name: str,
    *,
    epoch: Optional[int | str] = None,
    quantized: bool = False,
    pts: bool = False,
    lite: bool = False,
    pt2: bool = False,
    st: bool = False,
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
    elif st is True:
        file_name = f"{file_name}.safetensors"
    elif onnx is True:
        file_name = f"{file_name}.onnx"
    elif pts is True:
        file_name = f"{file_name}.pts"
    else:
        file_name = f"{file_name}.pt"

    return settings.MODELS_DIR.joinpath(file_name)


def _checkpoint_states(
    states_path: Path,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.grad_scaler.GradScaler],
    model_base: Optional[torch.nn.Module],
    **extra_states: Optional[dict[str, Any]],
) -> None:
    if optimizer is None or scheduler is None:
        return

    if scaler is not None:
        scaler_state = scaler.state_dict()
    else:
        scaler_state = None

    if model_base is not None:
        model_base_state = model_base.state_dict()
    else:
        model_base_state = None

    torch.save(
        {
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler_state,
            "model_base_state": model_base_state,
            **extra_states,
        },
        states_path,
    )


def checkpoint_model(
    network_name: str,
    epoch: int,
    net: torch.nn.Module,
    signature: SignatureType | DetectionSignatureType | MIMSignatureType | SSLSignatureType,
    class_to_idx: dict[str, int],
    rgb_stats: RGBType,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.grad_scaler.GradScaler],
    model_base: Optional[torch.nn.Module],
    *,
    external_config: Optional[dict[str, Any]] = None,
    external_backbone_config: Optional[dict[str, Any]] = None,
    **extra_states: Optional[dict[str, Any]],
) -> None:
    kwargs = {}
    if external_config is not None:
        kwargs["config"] = external_config
    if external_backbone_config is not None:
        kwargs["backbone_config"] = external_backbone_config

    path = model_path(network_name, epoch=epoch)
    states_path = model_path(network_name, epoch=epoch, states=True)
    logger.info(f"Saving model checkpoint {path}...")
    torch.save(
        {
            "state": net.state_dict(),
            "birder_version": __version__,
            "task": net.task,
            "signature": signature,
            "class_to_idx": class_to_idx,
            "rgb_stats": rgb_stats,
            **kwargs,
        },
        path,
    )

    _checkpoint_states(states_path, optimizer, scheduler, scaler, model_base, **extra_states)


def clean_checkpoints(network_name: str, keep_last: int) -> None:
    epoch = "*[0-9]"
    models_glob = str(model_path(network_name, epoch=epoch))
    states_glob = str(model_path(network_name, epoch=epoch, states=True))
    model_pattern = re.compile(r".*_([1-9][0-9]*)\.pt$")
    states_pattern = re.compile(r".*_([1-9][0-9]*)_states$")

    model_paths = list(settings.BASE_DIR.glob(models_glob))
    for p in sorted(model_paths, key=lambda p: p.stat().st_mtime)[:-keep_last]:
        if model_pattern.search(str(p)) is not None:
            logger.info(f"Removing checkpoint {p}...")
            p.unlink()

    state_paths = list(settings.BASE_DIR.glob(states_glob))
    for p in sorted(state_paths, key=lambda p: p.stat().st_mtime)[:-keep_last]:
        if states_pattern.search(str(p)) is not None:
            logger.info(f"Removing checkpoint states {p}...")
            p.unlink()


def load_state_dict(device: torch.device, network_name: str, *, epoch: Optional[int] = None) -> dict[str, Any]:
    path = model_path(network_name, epoch=epoch)

    # Load state dict
    logger.info(f"Loading state dict from {path} on device {device}...")
    model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)["state"]

    return model_dict


class TrainingStates(NamedTuple):
    optimizer_state: Optional[dict[str, Any]]
    scheduler_state: Optional[dict[str, Any]]
    scaler_state: Optional[dict[str, Any]]
    model_base_state: Optional[dict[str, Any]]
    ema_model_state: Optional[dict[str, Any]] = None
    extra_states: Optional[dict[str, Any]] = None

    @classmethod
    def empty(cls) -> "TrainingStates":
        return cls(None, None, None, None, None)


def _load_states(states_path: Path, device: torch.device) -> TrainingStates:
    if states_path.exists() is True:
        states_dict: dict[str, Any] = torch.load(states_path, map_location=device, weights_only=True)
        optimizer_state = states_dict.pop("optimizer_state")
        scheduler_state = states_dict.pop("scheduler_state")
        scaler_state = states_dict.pop("scaler_state")
        model_base_state = states_dict.pop("model_base_state")
        extra_states = {}
        for state in states_dict:
            extra_states[state] = states_dict[state]

        return TrainingStates(
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            scaler_state=scaler_state,
            model_base_state=model_base_state,
            extra_states=extra_states,
        )

    return TrainingStates.empty()


class SimpleCheckpointStates(NamedTuple):
    net: torch.nn.Module
    training_states: TrainingStates


def load_simple_checkpoint(
    device: torch.device, net: torch.nn.Module, network_name: str, *, epoch: Optional[int] = None
) -> SimpleCheckpointStates:
    path = model_path(network_name, epoch=epoch)
    states_path = model_path(network_name, epoch=epoch, states=True)

    # Load model and training states
    logger.info(f"Loading model from {path} on device {device}...")
    model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    training_states = _load_states(states_path, device)

    # When a checkpoint was trained with EMA:
    #   The primary weights in the checkpoint file are the EMA weights
    #   The base_state contain the non-EMA weights
    if training_states.model_base_state is not None:
        net.load_state_dict(training_states.model_base_state)
        training_states = training_states._replace(ema_model_state=model_dict["state"])
    else:
        net.load_state_dict(model_dict["state"])

    net.to(device)

    return SimpleCheckpointStates(net, training_states)


class CheckpointStates(NamedTuple):
    net: BaseNet
    class_to_idx: dict[str, int]
    training_states: TrainingStates


def load_checkpoint(
    device: torch.device,
    network: str,
    *,
    net_param: Optional[float],
    config: Optional[dict[str, Any]] = None,
    tag: Optional[str] = None,
    epoch: Optional[int] = None,
    new_size: Optional[tuple[int, int]] = None,
) -> CheckpointStates:
    network_name = get_network_name(network, net_param, tag)
    path = model_path(network_name, epoch=epoch)
    states_path = model_path(network_name, epoch=epoch, states=True)

    # Load model and training states
    logger.info(f"Loading model from {path} on device {device}...")
    model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    training_states = _load_states(states_path, device)

    # Extract auxiliary data
    class_to_idx: dict[str, int] = model_dict["class_to_idx"]
    signature: SignatureType = model_dict["signature"]
    input_channels = lib.get_channels_from_signature(signature)
    num_classes = lib.get_num_labels_from_signature(signature)
    size = lib.get_size_from_signature(signature)

    # Initialize network and restore checkpoint state
    net = registry.net_factory(network, input_channels, num_classes, net_param=net_param, config=config, size=size)

    # When a checkpoint was trained with EMA:
    #   The primary weights in the checkpoint file are the EMA weights
    #   The base_state contain the non-EMA weights
    if training_states.model_base_state is not None:
        net.load_state_dict(training_states.model_base_state)
        training_states = training_states._replace(ema_model_state=model_dict["state"])
    else:
        net.load_state_dict(model_dict["state"])

    if new_size is not None:
        net.adjust_size(new_size)

    net.to(device)

    return CheckpointStates(net, class_to_idx, training_states)


class MIMCheckpointStates(NamedTuple):
    net: MIMBaseNet
    training_states: TrainingStates


def load_mim_checkpoint(
    device: torch.device,
    network: str,
    *,
    net_param: Optional[float] = None,
    config: Optional[dict[str, Any]] = None,
    encoder: str,
    encoder_param: Optional[float] = None,
    encoder_config: Optional[dict[str, Any]] = None,
    tag: Optional[str] = None,
    epoch: Optional[int] = None,
) -> MIMCheckpointStates:
    network_name = get_mim_network_name(
        network, net_param=net_param, encoder=encoder, encoder_param=encoder_param, tag=tag
    )
    path = model_path(network_name, epoch=epoch, pts=False)
    states_path = model_path(network_name, epoch=epoch, pts=False, states=True)

    # Load model and training states
    logger.info(f"Loading model from {path} on device {device}...")
    model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    training_states = _load_states(states_path, device)

    # Extract auxiliary data
    signature: MIMSignatureType = model_dict["signature"]
    input_channels = lib.get_channels_from_signature(signature)
    num_classes = 0
    size = lib.get_size_from_signature(signature)

    # Initialize network and restore checkpoint state
    net_encoder = registry.net_factory(
        encoder, input_channels, num_classes, net_param=encoder_param, config=encoder_config, size=size
    )
    net = registry.mim_net_factory(network, net_encoder, net_param=net_param, config=config, size=size)
    net.load_state_dict(model_dict["state"])
    net.to(device)

    return MIMCheckpointStates(net, training_states)


class DetectionCheckpointStates(NamedTuple):
    net: DetectionBaseNet
    class_to_idx: dict[str, int]
    training_states: TrainingStates


def load_detection_checkpoint(
    device: torch.device,
    network: str,
    *,
    net_param: Optional[float] = None,
    config: Optional[dict[str, Any]] = None,
    tag: Optional[str] = None,
    backbone: str,
    backbone_param: Optional[float] = None,
    backbone_config: Optional[dict[str, Any]] = None,
    backbone_tag: Optional[str] = None,
    epoch: Optional[int] = None,
    new_size: Optional[tuple[int, int]] = None,
) -> DetectionCheckpointStates:
    network_name = get_detection_network_name(
        network,
        net_param=net_param,
        tag=tag,
        backbone=backbone,
        backbone_param=backbone_param,
        backbone_tag=backbone_tag,
    )
    path = model_path(network_name, epoch=epoch, pts=False)
    states_path = model_path(network_name, epoch=epoch, pts=False, states=True)

    # Load model and training states
    logger.info(f"Loading model from {path} on device {device}...")
    model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    training_states = _load_states(states_path, device)

    # Extract auxiliary data
    class_to_idx: dict[str, int] = model_dict["class_to_idx"]
    signature: DetectionSignatureType = model_dict["signature"]
    input_channels = lib.get_channels_from_signature(signature)
    num_classes = lib.get_num_labels_from_signature(signature)
    size = lib.get_size_from_signature(signature)

    # Initialize network and restore checkpoint state
    net_backbone = registry.net_factory(
        backbone, input_channels, num_classes, net_param=backbone_param, config=backbone_config, size=size
    )
    net = registry.detection_net_factory(
        network, num_classes, net_backbone, net_param=net_param, config=config, size=size
    )

    # When a checkpoint was trained with EMA:
    #   The primary weights in the checkpoint file are the EMA weights
    #   The base_state contain the non-EMA weights
    if training_states.model_base_state is not None:
        net.load_state_dict(training_states.model_base_state)
        training_states = training_states._replace(ema_model_state=model_dict["state"])
    else:
        net.load_state_dict(model_dict["state"])

    if new_size is not None:
        net.adjust_size(new_size)

    net.to(device)

    return DetectionCheckpointStates(net, class_to_idx, training_states)


class ModelInfo(NamedTuple):
    class_to_idx: dict[str, int]
    signature: SignatureType
    rgb_stats: RGBType
    custom_config: Optional[dict[str, Any]] = None


# pylint: disable=too-many-locals,too-many-branches
def load_model(
    device: torch.device,
    network: str,
    *,
    path: Optional[str | Path] = None,
    net_param: Optional[float] = None,
    config: Optional[dict[str, Any]] = None,
    tag: Optional[str] = None,
    epoch: Optional[int] = None,
    new_size: Optional[tuple[int, int]] = None,
    quantized: bool = False,
    inference: bool,
    reparameterized: bool = False,
    pts: bool = False,
    pt2: bool = False,
    st: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.nn.Module | torch.ScriptModule, ModelInfo]:
    if path is None:
        _network_name = get_network_name(network, net_param, tag)
        path = model_path(_network_name, epoch=epoch, quantized=quantized, pts=pts, pt2=pt2, st=st)

    logger.info(f"Loading model from {path} on device {device}...")

    loaded_config = {}
    if pts is True:
        extra_files = {"task": "", "class_to_idx": "", "signature": "", "rgb_stats": ""}
        net = torch.jit.load(path, map_location=device, _extra_files=extra_files)
        net.task = extra_files["task"]
        class_to_idx: dict[str, int] = json.loads(extra_files["class_to_idx"])
        signature: SignatureType = json.loads(extra_files["signature"])
        rgb_stats: RGBType = json.loads(extra_files["rgb_stats"])

    elif pt2 is True:
        extra_files = {"task": "", "class_to_idx": "", "signature": "", "rgb_stats": ""}
        net = torch.export.load(path, extra_files=extra_files).module()
        net.to(device)
        net.task = extra_files["task"]
        class_to_idx = json.loads(extra_files["class_to_idx"])
        signature = json.loads(extra_files["signature"])
        rgb_stats = json.loads(extra_files["rgb_stats"])

    elif st is True:
        assert _HAS_SAFETENSORS, "'pip install safetensors' to use .safetensors"
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            extra_files = f.metadata()

        class_to_idx = json.loads(extra_files["class_to_idx"])
        signature = json.loads(extra_files["signature"])
        rgb_stats = json.loads(extra_files["rgb_stats"])
        input_channels = lib.get_channels_from_signature(signature)
        num_classes = lib.get_num_labels_from_signature(signature)
        size = lib.get_size_from_signature(signature)

        if "config" in extra_files and len(extra_files["config"]) > 0:
            loaded_config = json.loads(extra_files["config"])

        # Merge configs, with external config taking precedence
        merged_config = {**loaded_config}
        if config is not None:
            merged_config.update(config)
        if len(merged_config) == 0:
            merged_config = None  # type: ignore[assignment]

        model_state: dict[str, Any] = safetensors.torch.load_file(path, device=device.type)
        net = registry.net_factory(
            network, input_channels, num_classes, net_param=net_param, config=merged_config, size=size
        )
        if reparameterized is True:
            net.reparameterize_model()

        net.load_state_dict(model_state)
        if new_size is not None:
            net.adjust_size(new_size)

        net.to(device)

    else:
        model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
        signature = model_dict["signature"]
        input_channels = lib.get_channels_from_signature(signature)
        num_classes = lib.get_num_labels_from_signature(signature)
        size = lib.get_size_from_signature(signature)

        if "config" in model_dict:
            loaded_config = model_dict["config"]

        # Merge configs, with external config taking precedence
        merged_config = {**loaded_config}
        if config is not None:
            merged_config.update(config)
        if len(merged_config) == 0:
            merged_config = None

        net = registry.net_factory(
            network, input_channels, num_classes, net_param=net_param, config=merged_config, size=size
        )
        if reparameterized is True:
            net.reparameterize_model()

        net.load_state_dict(model_dict["state"])
        if new_size is not None:
            net.adjust_size(new_size)

        net.to(device)
        class_to_idx = model_dict["class_to_idx"]
        rgb_stats = model_dict["rgb_stats"]

    if dtype is not None:
        net.to(dtype)
    if inference is True:
        for param in net.parameters():
            param.requires_grad = False

        if pt2 is False:  # Remove when GraphModule add support for 'eval'
            net.eval()

    if len(loaded_config) == 0:
        custom_config = None
    else:
        custom_config = loaded_config
        logger.debug(f"Model loaded with custom config: {custom_config}")

    return (net, ModelInfo(class_to_idx, signature, rgb_stats, custom_config))


class DetectionModelInfo(NamedTuple):
    class_to_idx: dict[str, int]
    signature: DetectionSignatureType
    rgb_stats: RGBType
    custom_config: Optional[dict[str, Any]] = None
    backbone_custom_config: Optional[dict[str, Any]] = None


# pylint: disable=too-many-locals,too-many-arguments,too-many-statements
def load_detection_model(
    device: torch.device,
    network: str,
    *,
    path: Optional[str | Path] = None,
    net_param: Optional[float] = None,
    config: Optional[dict[str, Any]] = None,
    tag: Optional[str] = None,
    reparameterized: bool = False,
    backbone: str,
    backbone_param: Optional[float] = None,
    backbone_config: Optional[dict[str, Any]] = None,
    backbone_tag: Optional[str] = None,
    backbone_reparameterized: bool = False,
    epoch: Optional[int] = None,
    new_size: Optional[tuple[int, int]] = None,
    quantized: bool = False,
    inference: bool,
    pts: bool = False,
    pt2: bool = False,
    st: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.nn.Module | torch.ScriptModule, DetectionModelInfo]:
    if path is None:
        _network_name = get_detection_network_name(
            network,
            net_param=net_param,
            tag=tag,
            backbone=backbone,
            backbone_param=backbone_param,
            backbone_tag=backbone_tag,
        )
        path = model_path(_network_name, epoch=epoch, quantized=quantized, pts=pts, pt2=pt2, st=st)

    logger.info(f"Loading model from {path} on device {device}...")

    backbone_loaded_config = {}
    loaded_config = {}
    if pts is True:
        extra_files = {"task": "", "class_to_idx": "", "signature": "", "rgb_stats": ""}
        net = torch.jit.load(path, map_location=device, _extra_files=extra_files)
        net.task = extra_files["task"]
        class_to_idx: dict[str, int] = json.loads(extra_files["class_to_idx"])
        signature: DetectionSignatureType = json.loads(extra_files["signature"])
        rgb_stats: RGBType = json.loads(extra_files["rgb_stats"])

    elif pt2 is True:
        extra_files = {"task": "", "class_to_idx": "", "signature": "", "rgb_stats": ""}
        net = torch.export.load(path, extra_files=extra_files).module()
        net.to(device)
        net.task = extra_files["task"]
        class_to_idx = json.loads(extra_files["class_to_idx"])
        signature = json.loads(extra_files["signature"])
        rgb_stats = json.loads(extra_files["rgb_stats"])

    elif st is True:
        assert _HAS_SAFETENSORS, "'pip install safetensors' to use .safetensors"
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            extra_files = f.metadata()

        class_to_idx = json.loads(extra_files["class_to_idx"])
        signature = json.loads(extra_files["signature"])
        rgb_stats = json.loads(extra_files["rgb_stats"])
        input_channels = lib.get_channels_from_signature(signature)
        num_classes = lib.get_num_labels_from_signature(signature)
        size = lib.get_size_from_signature(signature)

        if "backbone_config" in extra_files and len(extra_files["backbone_config"]) > 0:
            backbone_loaded_config = json.loads(extra_files["backbone_config"])
        if "config" in extra_files and len(extra_files["config"]) > 0:
            loaded_config = json.loads(extra_files["config"])

        # Merge configs, with external config taking precedence
        backbone_merged_config = {**backbone_loaded_config}
        if backbone_config is not None:
            backbone_merged_config.update(backbone_config)
        if len(backbone_merged_config) == 0:
            backbone_merged_config = None  # type: ignore[assignment]

        merged_config = {**loaded_config}
        if config is not None:
            merged_config.update(config)
        if len(merged_config) == 0:
            merged_config = None  # type: ignore[assignment]

        model_state: dict[str, Any] = safetensors.torch.load_file(path, device=device.type)
        net_backbone = registry.net_factory(
            backbone, input_channels, num_classes, net_param=backbone_param, config=backbone_merged_config, size=size
        )
        if backbone_reparameterized is True:
            net_backbone.reparameterize_model()

        net = registry.detection_net_factory(
            network, num_classes, net_backbone, net_param=net_param, config=merged_config, size=size
        )
        if reparameterized is True:
            net.reparameterize_model()

        net.load_state_dict(model_state)
        if new_size is not None:
            net.adjust_size(new_size)

        net.to(device)

    else:
        model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
        signature = model_dict["signature"]
        input_channels = lib.get_channels_from_signature(signature)
        num_classes = lib.get_num_labels_from_signature(signature)
        size = lib.get_size_from_signature(signature)

        if "backbone_config" in model_dict:
            backbone_loaded_config = model_dict["backbone_config"]
        if "config" in model_dict:
            loaded_config = model_dict["config"]

        # Merge configs, with external config taking precedence
        backbone_merged_config = {**backbone_loaded_config}
        if backbone_config is not None:
            backbone_merged_config.update(backbone_config)
        if len(backbone_merged_config) == 0:
            backbone_merged_config = None

        merged_config = {**loaded_config}
        if config is not None:
            merged_config.update(config)
        if len(merged_config) == 0:
            merged_config = None

        net_backbone = registry.net_factory(
            backbone, input_channels, num_classes, net_param=backbone_param, config=backbone_merged_config, size=size
        )
        if backbone_reparameterized is True:
            net_backbone.reparameterize_model()

        net = registry.detection_net_factory(
            network, num_classes, net_backbone, net_param=net_param, config=merged_config, size=size
        )
        if reparameterized is True:
            net.reparameterize_model()

        net.load_state_dict(model_dict["state"])
        if new_size is not None:
            net.adjust_size(new_size)

        net.to(device)
        class_to_idx = model_dict["class_to_idx"]
        rgb_stats = model_dict["rgb_stats"]

    if dtype is not None:
        net.to(dtype)
    if inference is True:
        for param in net.parameters():
            param.requires_grad = False

        net.eval()

    if len(backbone_loaded_config) == 0:
        backbone_custom_config = None
    else:
        backbone_custom_config = backbone_loaded_config
        logger.debug(f"Backbone loaded with custom config: {backbone_custom_config}")

    if len(loaded_config) == 0:
        custom_config = None
    else:
        custom_config = loaded_config
        logger.debug(f"Model loaded with custom config: {custom_config}")

    return (net, DetectionModelInfo(class_to_idx, signature, rgb_stats, custom_config, backbone_custom_config))


def load_pretrained_model(
    weights: str,
    *,
    dst: Optional[str | Path] = None,
    file_format: FileFormatType = "pt",
    inference: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    custom_config: Optional[dict[str, Any]] = None,
    progress_bar: bool = True,
) -> tuple[BaseNet | DetectionBaseNet, ModelInfo | DetectionModelInfo]:
    """
    Loads a pre-trained model from the model registry or a specified destination.

    Parameters
    ----------
    weights
        Name of the pre-trained weights to load from the model registry.
    dst
        Destination path where the model weights will be downloaded or loaded from.
        If None, the model will be saved in the default models directory.
    file_format
        Model format (e.g. pt, pt2, safetensors, etc.)
    inference
        Flag to prepare the model for inference mode.
    device
        The device to load the model on (cpu/cuda).
    dtype
        Data type for model parameters and computations (e.g., torch.float32, torch.float16).
        Determines precision of numerical operations.
    custom_config
        Additional model configuration that overrides or extends the predefined configuration.
    progress_bar
        Whether to display a progress bar during file download.

    Returns
    -------
    A tuple containing four elements:
    - A PyTorch module (neural network model) loaded with pre-trained weights.
    - Class to index mapping.
    - A signature defining the expected input and output tensor shapes.
    - The model's RGB processing type.

    Notes
    -----
    - Creates the models directory if it doesn't exist.
    - Downloads the model weights if not already present locally.
    - When inference=True, the model is set to evaluation mode with gradient calculation disabled.
    - If device is None, it will default to CPU.

    Examples
    --------
    >>> (net, model_info) = load_pretrained_model("mobilenet_v4_l_eu-common")
    >>> (net, model_info) = load_pretrained_model(
    ...     "rdnet_s_arabian-peninsula", inference=True, device=torch.device("cuda"))
    """

    if settings.MODELS_DIR.exists() is False:
        logger.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True)

    model_metadata = registry.get_pretrained_metadata(weights)
    if file_format == "ptl":
        raise ValueError("ptl format not supported")
    if file_format not in model_metadata["formats"]:
        available_formats = ", ".join(model_metadata["formats"].keys())
        raise ValueError(f"{file_format} not available for {weights}, available formats: {available_formats}")

    (model_file, url) = get_pretrained_model_url(weights, file_format)
    if dst is None:
        dst = settings.MODELS_DIR.joinpath(model_file)

    cli.download_file(url, dst, model_metadata["formats"][file_format]["sha256"], progress_bar=progress_bar)

    format_args: dict[str, Any] = {
        "pts": file_format == "pts",
        "pt2": file_format == "pt2",
        "st": file_format == "safetensors",
    }

    if device is None:
        device = torch.device("cpu")

    if model_metadata["task"] == Task.IMAGE_CLASSIFICATION:
        return load_model(
            device,
            model_metadata["net"]["network"],
            path=dst,
            net_param=model_metadata["net"].get("net_param", None),
            config=custom_config,
            tag=model_metadata["net"].get("tag", None),
            reparameterized=model_metadata["net"].get("reparameterized", False),
            inference=inference,
            dtype=dtype,
            **format_args,
        )

    if model_metadata["task"] == Task.OBJECT_DETECTION:
        return load_detection_model(
            device,
            model_metadata["net"]["network"],
            path=dst,
            net_param=model_metadata["net"].get("net_param", None),
            config=custom_config,
            tag=model_metadata["net"].get("tag", None),
            reparameterized=model_metadata["net"].get("reparameterized", False),
            backbone=model_metadata["backbone"]["network"],
            backbone_param=model_metadata["backbone"].get("net_param", None),
            backbone_tag=model_metadata["backbone"].get("tag", None),
            backbone_reparameterized=model_metadata["backbone"].get("reparameterized", False),
            inference=inference,
            dtype=dtype,
            **format_args,
        )

    raise ValueError(f"Unknown model type: {model_metadata['task']}")


def load_model_with_cfg(
    cfg: dict[str, Any] | str | Path, weights_path: Optional[str | Path]
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Loads a neural network model based on a configuration dictionary or configuration file path and optional weights.

    Parameters
    ----------
    cfg
        A model configuration dictionary or a path to a json configuration file.
    weights_path
        Path to the model weights file. Supports .pt and .safetensors formats.
        If None, returns an untrained model.

    Returns
    -------
    A PyTorch neural network model, optionally loaded with pre-trained weights.
    """

    if not isinstance(cfg, dict):
        cfg = read_config_from_path(cfg)

    if cfg["alias"] is not None:
        name = cfg["alias"]
    else:
        name = cfg["name"]

    net_param = cfg["net_param"]
    model_config = cfg["model_config"]
    signature = cfg["signature"]

    input_channels = lib.get_channels_from_signature(signature)
    num_classes = lib.get_num_labels_from_signature(signature)
    size = lib.get_size_from_signature(signature)

    if cfg["task"] == Task.MASKED_IMAGE_MODELING:
        if cfg["encoder_alias"] is not None:
            encoder_name = cfg["encoder_alias"]
        else:
            encoder_name = cfg["encoder"]

        encoder_net_param = cfg.get("encoder_net_param", None)
        encoder_config = cfg.get("encoder_config", None)
        encoder = registry.net_factory(
            encoder_name, input_channels, num_classes=0, net_param=encoder_net_param, config=encoder_config, size=size
        )
        net = registry.mim_net_factory(name, encoder, net_param=net_param, config=model_config, size=size)

    elif cfg["task"] == Task.OBJECT_DETECTION:
        if cfg["backbone_alias"] is not None:
            backbone_name = cfg["backbone_alias"]
        else:
            backbone_name = cfg["backbone"]

        backbone_net_param = cfg.get("backbone_net_param", None)
        backbone_config = cfg.get("backbone_config", None)
        backbone = registry.net_factory(
            backbone_name, input_channels, num_classes, net_param=backbone_net_param, config=backbone_config, size=size
        )
        if cfg.get("backbone_reparameterized", False) is True:
            backbone.reparameterize_model()

        net = registry.detection_net_factory(
            name, num_classes, backbone, net_param=net_param, config=model_config, size=size
        )

    elif cfg["task"] == Task.IMAGE_CLASSIFICATION:
        net = registry.net_factory(
            name, input_channels, num_classes, net_param=net_param, config=model_config, size=size
        )

    else:
        raise ValueError(f"Configuration not supported: {cfg['task']}")

    if cfg.get("reparameterized", False) is True:
        net.reparameterize_model()

    if weights_path is None:
        return (net, cfg)

    if isinstance(weights_path, str):
        weights_path = Path(weights_path)

    device = torch.device("cpu")
    if weights_path.suffix == ".safetensors":
        assert _HAS_SAFETENSORS, "'pip install safetensors' to use .safetensors"
        model_state: dict[str, Any] = safetensors.torch.load_file(weights_path, device=device.type)
    else:
        model_dict: dict[str, Any] = torch.load(weights_path, map_location=device, weights_only=True)
        model_state = model_dict["state"]

    net.load_state_dict(model_state)

    return (net, cfg)


def save_pts(
    scripted_module: torch.ScriptModule,
    dst: str | Path,
    task: str,
    class_to_idx: dict[str, int],
    signature: SignatureType | DetectionSignatureType,
    rgb_stats: RGBType,
) -> None:
    torch.jit.save(
        scripted_module,
        str(dst),
        _extra_files={
            "birder_version": __version__,
            "task": task,
            "class_to_idx": json.dumps(class_to_idx),
            "signature": json.dumps(signature),
            "rgb_stats": json.dumps(rgb_stats),
        },
    )


def save_pt2(
    exported_net: torch.export.ExportedProgram,
    dst: str | Path,
    task: str,
    class_to_idx: dict[str, int],
    signature: SignatureType | DetectionSignatureType,
    rgb_stats: RGBType,
) -> None:
    torch.export.save(
        exported_net,
        dst,
        extra_files={
            "birder_version": __version__,
            "task": task,
            "class_to_idx": json.dumps(class_to_idx),
            "signature": json.dumps(signature),
            "rgb_stats": json.dumps(rgb_stats),
        },
    )


def save_st(
    net: torch.nn.Module,
    dst: str,
    task: str,
    class_to_idx: dict[str, int],
    signature: SignatureType | DetectionSignatureType,
    rgb_stats: RGBType,
    *,
    external_config: Optional[dict[str, Any]] = None,
    external_backbone_config: Optional[dict[str, Any]] = None,
) -> None:
    assert _HAS_SAFETENSORS, "'pip install safetensors' to use .safetensors"
    kwargs = {}
    if external_config is not None:
        kwargs["config"] = json.dumps(external_config)
    if external_backbone_config is not None:
        kwargs["backbone_config"] = json.dumps(external_backbone_config)

    safetensors.torch.save_model(
        net,
        str(dst),
        {
            "birder_version": __version__,
            "task": task,
            "class_to_idx": json.dumps(class_to_idx),
            "signature": json.dumps(signature),
            "rgb_stats": json.dumps(rgb_stats),
            **kwargs,
        },
    )


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


def wds_braces_from_path(wds_directory: Path, prefix: str = "") -> tuple[str, int]:
    shard_names = sorted([f.stem for f in wds_directory.glob(f"{prefix}*.tar")])
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


def class_to_idx_from_paths(data_paths: list[str]) -> dict[str, int]:
    class_to_idx = {}
    base = 0
    for data_path in data_paths:
        classes = sorted(entry.name for entry in os.scandir(data_path) if entry.is_dir())
        class_to_idx.update({cls_name: i + base for i, cls_name in enumerate(classes)})
        base += len(classes)

    return class_to_idx
