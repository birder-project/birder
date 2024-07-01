import logging.config
import os
from pathlib import Path
from typing import Any

# Data paths
BASE_DIR = Path(os.path.relpath(__file__)).parent.parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR.joinpath("data")))

TRAINING_DATA_PATH = DATA_DIR.joinpath("training")
VALIDATION_DATA_PATH = DATA_DIR.joinpath("validation")
TESTING_DATA_PATH = DATA_DIR.joinpath("testing")
DETECTION_DATA_PATH = DATA_DIR.joinpath("detection_data")
TRAINING_DETECTION_PATH = DETECTION_DATA_PATH.joinpath("testing")
VALIDATION_DETECTION_PATH = DETECTION_DATA_PATH.joinpath("validation")
TRAINING_DETECTION_ANNOTATIONS_PATH = DETECTION_DATA_PATH.joinpath("training_annotations")
VALIDATION_DETECTION_ANNOTATIONS_PATH = DETECTION_DATA_PATH.joinpath("validation_annotations")
WEAKLY_LABELED_DATA_PATH = DATA_DIR.joinpath("raw_data")
CLASS_LIST_PATH = DETECTION_DATA_PATH.joinpath("classes.txt")
PACK_PATH_SUFFIX = "packed"

MODELS_DIR = BASE_DIR.joinpath("models")
TRAINING_RUNS_PATH = BASE_DIR.joinpath("runs")
RESULTS_DIR = BASE_DIR.joinpath("results")

# Results
TOP_K = 3

# Logging
# https://docs.python.org/3/library/logging.config.html
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOGGING: dict[str, Any] = {
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
    "loggers": {},
    "root": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": True},
}

logging.config.dictConfig(LOGGING)
