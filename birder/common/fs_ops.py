import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import Optional

import torch
import torch.amp
from torchvision.datasets.folder import IMG_EXTENSIONS

from birder.common import cli
from birder.common import lib
from birder.common.lib import get_detection_network_name
from birder.common.lib import get_mim_network_name
from birder.common.lib import get_network_name
from birder.conf import settings
from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.base import SignatureType
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.base import DetectionSignatureType
from birder.net.mim.base import MIMBaseNet
from birder.net.mim.base import MIMSignatureType
from birder.transforms.classification import RGBType
from birder.version import __version__

try:
    import safetensors
    import safetensors.torch

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


def write_signature(network_name: str, signature: SignatureType | DetectionSignatureType) -> None:
    signature_file = settings.MODELS_DIR.joinpath(f"{network_name}.json")
    logging.info(f"Writing {signature_file}")
    with open(signature_file, "w", encoding="utf-8") as handle:
        json.dump(signature, handle, indent=2)


def read_signature(network_name: str) -> SignatureType | DetectionSignatureType:
    signature_file = settings.MODELS_DIR.joinpath(f"{network_name}.json")
    logging.info(f"Reading {signature_file}")
    with open(signature_file, "r", encoding="utf-8") as handle:
        signature: SignatureType | DetectionSignatureType = json.load(handle)

    return signature


def write_config(
    network_name: str, net: torch.nn.Module, signature: SignatureType | DetectionSignatureType, rgb_stats: RGBType
) -> None:
    model_config = lib.get_network_config(net, signature, rgb_stats)
    config_file = settings.MODELS_DIR.joinpath(f"{network_name}.json")
    logging.info(f"Writing {config_file}")
    with open(config_file, "w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)


def read_config(network_name: str) -> dict[str, Any]:
    config_file = settings.MODELS_DIR.joinpath(f"{network_name}.json")
    logging.info(f"Reading {config_file}")
    with open(config_file, "r", encoding="utf-8") as handle:
        model_config: dict[str, Any] = json.load(handle)

    return model_config


def read_config_from_path(path: str | Path) -> dict[str, Any]:
    logging.info(f"Reading {path}")
    with open(path, "r", encoding="utf-8") as handle:
        model_config: dict[str, Any] = json.load(handle)

    return model_config


def read_class_file(path: Path | str) -> dict[str, int]:
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
    signature: SignatureType | DetectionSignatureType | MIMSignatureType,
    class_to_idx: dict[str, int],
    rgb_stats: RGBType,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.grad_scaler.GradScaler],
) -> None:
    path = model_path(network_name, epoch=epoch)
    states_path = model_path(network_name, epoch=epoch, states=True)
    logging.info(f"Saving model checkpoint {path}...")
    torch.save(
        {
            "state": net.state_dict(),
            "birder_version": __version__,
            "task": net.task,
            "signature": signature,
            "class_to_idx": class_to_idx,
            "rgb_stats": rgb_stats,
        },
        path,
    )

    _checkpoint_states(states_path, optimizer, scheduler, scaler)


def _load_states(states_path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if states_path.exists() is True:
        states_dict: dict[str, Any] = torch.load(states_path, map_location=device, weights_only=True)
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
    config: Optional[dict[str, Any]] = None,
    tag: Optional[str] = None,
    epoch: Optional[int] = None,
    new_size: Optional[int] = None,
) -> tuple[BaseNet, dict[str, int], dict[str, Any], dict[str, Any], dict[str, Any]]:
    network_name = get_network_name(network, net_param, tag)
    path = model_path(network_name, epoch=epoch)
    states_path = model_path(network_name, epoch=epoch, states=True)
    logging.info(f"Loading model from {path} on device {device}...")

    model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)

    signature: SignatureType = model_dict["signature"]
    input_channels = lib.get_channels_from_signature(signature)
    num_classes = signature["outputs"][0]["data_shape"][1]
    size = lib.get_size_from_signature(signature)[0]
    net = registry.net_factory(network, input_channels, num_classes, net_param=net_param, config=config, size=size)
    net.load_state_dict(model_dict["state"])
    if new_size is not None:
        net.adjust_size(new_size)

    net.to(device)
    class_to_idx: dict[str, int] = model_dict["class_to_idx"]
    (optimizer_state, scheduler_state, scaler_state) = _load_states(states_path, device)

    return (net, class_to_idx, optimizer_state, scheduler_state, scaler_state)


def load_mim_checkpoint(
    device: torch.device,
    network: str,
    *,
    net_param: Optional[float],
    config: Optional[dict[str, Any]] = None,
    encoder: str,
    encoder_param: Optional[float],
    encoder_config: Optional[dict[str, Any]] = None,
    tag: Optional[str] = None,
    epoch: Optional[int] = None,
) -> tuple[MIMBaseNet, dict[str, Any], dict[str, Any], dict[str, Any]]:
    network_name = get_mim_network_name(
        network, net_param=net_param, encoder=encoder, encoder_param=encoder_param, tag=tag
    )
    path = model_path(network_name, epoch=epoch, pts=False)
    states_path = model_path(network_name, epoch=epoch, pts=False, states=True)
    logging.info(f"Loading model from {path} on device {device}...")

    model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)

    signature: MIMSignatureType = model_dict["signature"]
    input_channels = lib.get_channels_from_signature(signature)
    num_classes = 0
    size = lib.get_size_from_signature(signature)[0]
    net_encoder = registry.net_factory(
        encoder, input_channels, num_classes, net_param=encoder_param, config=encoder_config, size=size
    )
    net = registry.mim_net_factory(network, net_encoder, net_param=net_param, config=config, size=size)
    net.load_state_dict(model_dict["state"])
    net.to(device)

    (optimizer_state, scheduler_state, scaler_state) = _load_states(states_path, device)

    return (net, optimizer_state, scheduler_state, scaler_state)


def load_detection_checkpoint(
    device: torch.device,
    network: str,
    *,
    net_param: Optional[float],
    config: Optional[dict[str, Any]] = None,
    tag: Optional[str] = None,
    backbone: str,
    backbone_param: Optional[float],
    backbone_config: Optional[dict[str, Any]] = None,
    backbone_tag: Optional[str],
    epoch: Optional[int] = None,
    new_size: Optional[int] = None,
) -> tuple[DetectionBaseNet, dict[str, int], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    logging.info(f"Loading model from {path} on device {device}...")

    model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)

    signature: DetectionSignatureType = model_dict["signature"]
    input_channels = lib.get_channels_from_signature(signature)
    num_classes = signature["num_labels"]
    size = lib.get_size_from_signature(signature)[0]
    net_backbone = registry.net_factory(
        backbone, input_channels, num_classes, net_param=backbone_param, config=backbone_config, size=size
    )
    net = registry.detection_net_factory(
        network, num_classes, net_backbone, net_param=net_param, config=config, size=size
    )
    net.load_state_dict(model_dict["state"])
    if new_size is not None:
        net.adjust_size(new_size)

    net.to(device)

    class_to_idx: dict[str, int] = model_dict["class_to_idx"]
    (optimizer_state, scheduler_state, scaler_state) = _load_states(states_path, device)

    return (net, class_to_idx, optimizer_state, scheduler_state, scaler_state)


# pylint: disable=too-many-locals
def load_model(
    device: torch.device,
    network: str,
    *,
    path: Optional[str | Path] = None,
    net_param: Optional[float] = None,
    config: Optional[dict[str, Any]] = None,
    tag: Optional[str] = None,
    epoch: Optional[int] = None,
    new_size: Optional[int] = None,
    quantized: bool = False,
    inference: bool,
    reparameterized: bool = False,
    pts: bool = False,
    pt2: bool = False,
    st: bool = False,
) -> tuple[torch.nn.Module | torch.ScriptModule, dict[str, int], SignatureType, RGBType]:
    if path is None:
        _network_name = get_network_name(network, net_param, tag)
        path = model_path(_network_name, epoch=epoch, quantized=quantized, pts=pts, pt2=pt2, st=st)

    logging.info(f"Loading model from {path} on device {device}...")

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
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:  # type: ignore[attr-defined]
            extra_files = f.metadata()

        class_to_idx = json.loads(extra_files["class_to_idx"])
        signature = json.loads(extra_files["signature"])
        rgb_stats = json.loads(extra_files["rgb_stats"])
        input_channels = lib.get_channels_from_signature(signature)
        num_classes = lib.get_num_labels_from_signature(signature)
        size = lib.get_size_from_signature(signature)[0]

        model_state: dict[str, Any] = safetensors.torch.load_file(path, device=device.type)
        net = registry.net_factory(network, input_channels, num_classes, net_param=net_param, config=config, size=size)
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
        size = lib.get_size_from_signature(signature)[0]

        net = registry.net_factory(network, input_channels, num_classes, net_param=net_param, config=config, size=size)
        if reparameterized is True:
            net.reparameterize_model()

        net.load_state_dict(model_dict["state"])
        if new_size is not None:
            net.adjust_size(new_size)

        net.to(device)
        class_to_idx = model_dict["class_to_idx"]
        rgb_stats = model_dict["rgb_stats"]

    if inference is True:
        for param in net.parameters():
            param.requires_grad = False

        if pt2 is False:  # Remove when GraphModule add support for 'eval'
            net.eval()

    return (net, class_to_idx, signature, rgb_stats)


# pylint: disable=too-many-locals,too-many-arguments
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
    backbone_param: Optional[float],
    backbone_config: Optional[dict[str, Any]] = None,
    backbone_tag: Optional[str] = None,
    backbone_reparameterized: bool = False,
    epoch: Optional[int] = None,
    new_size: Optional[int] = None,
    quantized: bool = False,
    inference: bool,
    pts: bool = False,
    pt2: bool = False,
    st: bool = False,
) -> tuple[torch.nn.Module | torch.ScriptModule, dict[str, int], DetectionSignatureType, RGBType]:
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

    logging.info(f"Loading model from {path} on device {device}...")

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
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:  # type: ignore[attr-defined]
            extra_files = f.metadata()

        class_to_idx = json.loads(extra_files["class_to_idx"])
        signature = json.loads(extra_files["signature"])
        rgb_stats = json.loads(extra_files["rgb_stats"])
        input_channels = lib.get_channels_from_signature(signature)
        num_classes = lib.get_num_labels_from_signature(signature)
        size = lib.get_size_from_signature(signature)[0]

        model_state: dict[str, Any] = safetensors.torch.load_file(path, device=device.type)
        net_backbone = registry.net_factory(
            backbone, input_channels, num_classes, net_param=backbone_param, config=backbone_config, size=size
        )
        if backbone_reparameterized is True:
            net_backbone.reparameterize_model()

        net = registry.detection_net_factory(
            network, num_classes, net_backbone, net_param=net_param, config=config, size=size
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
        size = lib.get_size_from_signature(signature)[0]

        net_backbone = registry.net_factory(
            backbone, input_channels, num_classes, net_param=backbone_param, config=backbone_config, size=size
        )
        if backbone_reparameterized is True:
            net_backbone.reparameterize_model()

        net = registry.detection_net_factory(
            network, num_classes, net_backbone, net_param=net_param, config=config, size=size
        )
        if reparameterized is True:
            net.reparameterize_model()

        net.load_state_dict(model_dict["state"])
        if new_size is not None:
            net.adjust_size(new_size)

        net.to(device)
        class_to_idx = model_dict["class_to_idx"]
        rgb_stats = model_dict["rgb_stats"]

    if inference is True:
        for param in net.parameters():
            param.requires_grad = False

        net.eval()

    return (net, class_to_idx, signature, rgb_stats)


def load_pretrained_model(
    weights: str,
    *,
    dst: Optional[str | Path] = None,
    inference: bool = False,
    device: Optional[torch.device] = None,
    progress_bar: bool = True,
) -> tuple[torch.nn.Module | torch.ScriptModule, dict[str, int], SignatureType, RGBType]:
    """
    Loads a pre-trained model from the model registry or a specified destination.

    Parameters
    ----------
    weights
        Name of the pre-trained weights to load from the model registry.
    dst
        Destination path where the model weights will be downloaded or loaded from.
        If None, the model will be saved in the default models directory.
    inference
        Flag to prepare the model for inference mode.
    device
        The device to load the model on (cpu/cuda).
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
    """

    if settings.MODELS_DIR.exists() is False:
        logging.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True)

    model_info = registry.get_pretrained_info(weights)
    assert "pt" in model_info["formats"], "Can only load pt type files"

    model_file = f"{weights}.pt"
    if dst is None:
        dst = settings.MODELS_DIR.joinpath(model_file)

    if "url" in model_info:
        url = model_info["url"]
    else:
        url = f"{settings.REGISTRY_BASE_UTL}/{model_file}"

    cli.download_file(url, dst, model_info["formats"]["pt"]["sha256"], progress_bar=progress_bar)

    if device is None:
        device = torch.device("cpu")

    return load_model(
        device,
        model_info["net"]["network"],
        path=dst,
        net_param=model_info["net"].get("net_param", None),
        tag=model_info["net"].get("tag", None),
        reparameterized=model_info["net"].get("reparameterized", False),
        inference=inference,
    )


def load_model_with_cfg(cfg: dict[str, Any] | str | Path, weights_path: Optional[str | Path]) -> torch.nn.Module:
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
    size = lib.get_size_from_signature(signature)[0]

    if "backbone" in cfg:
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
    else:
        net = registry.net_factory(
            name, input_channels, num_classes, net_param=net_param, config=model_config, size=size
        )

    if cfg.get("reparameterized", False) is True:
        net.reparameterize_model()

    if weights_path is None:
        return net

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

    return net


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
) -> None:
    assert _HAS_SAFETENSORS, "'pip install safetensors' to use .safetensors"
    safetensors.torch.save_model(
        net,
        str(dst),
        {
            "birder_version": __version__,
            "task": task,
            "class_to_idx": json.dumps(class_to_idx),
            "signature": json.dumps(signature),
            "rgb_stats": json.dumps(rgb_stats),
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
