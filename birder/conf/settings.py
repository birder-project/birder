import logging.config
import os
import tomllib
from pathlib import Path
from typing import Any

_DERIVED_PATHS = {
    "BASE_DIR": {
        "DATA_DIR": "data",
        "MODELS_DIR": "models",
        "TRAINING_RUNS_PATH": "runs",
        "RESULTS_DIR": "results",
    },
    "DATA_DIR": {
        "TRAINING_DATA_PATH": "training",
        "VALIDATION_DATA_PATH": "validation",
        "TESTING_DATA_PATH": "testing",
        "DETECTION_DATA_PATH": "detection_data",
        "WEAKLY_LABELED_DATA_PATH": "raw_data",
        "WEAKLY_VAL_LABELED_DATA_PATH": "raw_data_validation",
    },
    "DETECTION_DATA_PATH": {
        "TRAINING_DETECTION_PATH": "training",
        "VALIDATION_DETECTION_PATH": "validation",
        "TRAINING_DETECTION_ANNOTATIONS_PATH": "training_annotations",
        "VALIDATION_DETECTION_ANNOTATIONS_PATH": "validation_annotations",
    },
}


def _refresh_derived_paths(explicit: set[str]) -> None:
    changed = set(explicit)
    while len(changed) > 0:
        root = changed.pop()
        for name, suffix in _DERIVED_PATHS.get(root, {}).items():
            if name in explicit:
                continue

            path = globals()[root].joinpath(suffix)
            if globals().get(name) != path:
                globals()[name] = path
                changed.add(name)


# Data paths
BASE_DIR = Path(".")
DATA_DIR: Path
TRAINING_DATA_PATH: Path
VALIDATION_DATA_PATH: Path
TESTING_DATA_PATH: Path
DETECTION_DATA_PATH: Path
TRAINING_DETECTION_PATH: Path
VALIDATION_DETECTION_PATH: Path
TRAINING_DETECTION_ANNOTATIONS_PATH: Path
VALIDATION_DETECTION_ANNOTATIONS_PATH: Path
WEAKLY_LABELED_DATA_PATH: Path
WEAKLY_VAL_LABELED_DATA_PATH: Path
MODELS_DIR: Path
TRAINING_RUNS_PATH: Path
RESULTS_DIR: Path
_refresh_derived_paths({"BASE_DIR"})

CLASS_LIST_NAME = "classes.txt"
PACK_PATH_SUFFIX = "packed"

# Labels
NO_LABEL = -1

# Inputs
DEFAULT_NUM_CHANNELS = 3

# Results
TOP_K = 3
MAX_DETECTIONS = [1, 10, 100]

# Model registry
REGISTRY_BASE_URL = "https://f000.backblazeb2.com/file/birder/models"

# Logging
# https://docs.python.org/3/library/logging.config.html
LOG_LEVEL = "INFO"


def _load_config() -> dict[str, Any]:
    config_path = Path("birder.toml")
    if config_path.exists() is False:
        return {}

    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _coerce_config_value(value: Any, current: Any) -> Any:
    if isinstance(current, Path):
        return Path(value)

    return value


def _apply_config(config: dict[str, Any]) -> None:
    for name, value in config.items():
        if name.startswith("_") is True or name.isupper() is False:
            raise KeyError(name)

        globals()[name] = _coerce_config_value(value, globals()[name])


def _apply_env(explicit: set[str]) -> None:
    for name in _DERIVED_PATHS["BASE_DIR"]:
        if name in os.environ:
            globals()[name] = Path(os.environ[name])
            explicit.add(name)

    if "LOG_LEVEL" in os.environ:
        globals()["LOG_LEVEL"] = os.environ["LOG_LEVEL"]
        explicit.add("LOG_LEVEL")


_CONFIG = _load_config()
_EXPLICIT = set(_CONFIG)
_apply_config(_CONFIG)
_apply_env(_EXPLICIT)
_refresh_derived_paths(_EXPLICIT)

_LOGGING: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{asctime}.{msecs:04.0f} {levelname} {filename}:{lineno:<4d}] {message}",
            "style": "{",
            "datefmt": "%d/%b/%Y %H:%M:%S",
        },
        "simple": {"format": "[{asctime} {levelname}] {message}", "style": "{"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "level": "DEBUG", "formatter": "verbose"},
    },
    "loggers": {
        "birder": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
    },
}

logging.config.dictConfig(_LOGGING)
