import argparse
import json
import logging
import os
import time
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Tuple

import mxnet as mx

from birder.conf import settings
from birder.core.net.base import get_signature


class ArgumentHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def list_images(
    data_path: str, write_synset: bool, skip_aug: bool = False, skip_adv: bool = False
) -> Generator[Tuple[str, int], None, None]:
    """
    List all images in subdirectories and assign labels

    Returns a tuple (full path, label)
    """

    exts = [".jpg", ".jpeg", ".png"]

    if write_synset is True:
        classes: Dict[str, int] = {}

    else:
        classes = read_synset()

    for (path, dirs, _files) in os.walk(data_path, followlinks=True):
        dirs = sorted(dirs)

        for directory in dirs:
            for fname in sorted(os.listdir(os.path.join(path, directory))):
                if skip_aug is True and fname.startswith("aug_"):
                    continue

                if skip_adv is True and fname.startswith("adv_"):
                    continue

                fpath = os.path.join(path, directory, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) is True and (suffix in exts):
                    if directory not in classes and write_synset is True:
                        classes[directory] = len(classes)

                    elif directory not in classes and write_synset is False:
                        raise ValueError(f"Unknown image class found {directory}")

                    yield (fpath, classes[directory])

    if write_synset is True:
        logging.info(f"Writing {settings.SYNSET_FILENAME} ({len(classes)} classes)")
        with open(settings.SYNSET_FILENAME, "w") as handle:
            handle.write(os.linesep.join(list(classes.keys())))
            handle.write(os.linesep)


def get_model_path(network_name: str) -> str:
    return os.path.join(settings.MODELS_DIR, network_name)


def get_model_signature_path(network_name: str) -> str:
    return os.path.join(settings.MODELS_DIR, f"{network_name}-signature.json")


def get_label_from_path(path: str) -> str:
    """
    Returns the last directory from the path.

    For val_data/Barn owl/000001.jpeg return value will be 'Barn owl'.
    """

    return os.path.basename(os.path.dirname(path))


def read_synset() -> Dict[str, int]:
    with open(settings.SYNSET_FILENAME, "r") as handle:
        classes: Dict[str, int] = {}
        for idx, line in enumerate(handle.read().splitlines()):
            classes[line] = idx

    return classes


def read_synset_reverse() -> Dict[int, str]:
    classes = read_synset()
    inverse_classes = dict(zip(classes.values(), classes.keys()))

    return inverse_classes


def read_synset_as_list() -> List[str]:
    classes = read_synset()
    return list(classes.keys())


def write_signature(network_name: str, size: int, rgb_values: Dict[str, float]) -> None:
    signature = get_signature(size)
    signature.update(rgb_values)
    signature_file = get_model_signature_path(network_name)
    logging.info(f"Writing {signature_file}")
    with open(signature_file, "w") as handle:
        json.dump(signature, handle, indent=2)


def read_signature(network_name: str) -> Dict[str, Any]:
    signature_file = get_model_signature_path(network_name)
    with open(signature_file, "r") as handle:
        signature: Dict[str, Any] = json.load(handle)

    return signature


def get_signature_size(signature: Dict[str, Any]) -> int:
    """
    Return input size (last index)
    """

    input_shape: List[int] = signature["inputs"][0]["data_shape"]

    return input_shape[-1]


def get_signature_rgb(signature: Dict[str, Any]) -> Dict[str, float]:
    return {
        "mean_r": signature["mean_r"],
        "mean_g": signature["mean_g"],
        "mean_b": signature["mean_b"],
        "std_r": signature["std_r"],
        "std_g": signature["std_g"],
        "std_b": signature["std_b"],
        "scale": signature["scale"],
    }


def get_rgb_mean(signature: Dict[str, Any]) -> mx.nd.NDArray:
    return mx.nd.array([signature["mean_r"], signature["mean_g"], signature["mean_b"]], dtype="float32")


def get_rgb_std(signature: Dict[str, Any]) -> mx.nd.NDArray:
    return mx.nd.array([signature["std_r"], signature["std_g"], signature["std_b"]], dtype="float32")


def read_rgb() -> Dict[str, float]:
    with open(settings.RGB_VALUES_FILENAME, "r") as handle:
        rgb_values: Dict[str, float] = json.load(handle)

    return rgb_values


def write_logfile(prefix: str, log_name: str, data: Dict[str, Any]) -> None:
    log_file = f"{prefix}{log_name}-{int(time.time())}.json"
    log_path = os.path.join(settings.TRAINING_LOGS_DIR, log_file)

    logging.info(f"Writing log '{log_file}'")
    with open(log_path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
