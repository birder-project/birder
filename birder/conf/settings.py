"""
Birder project settings.

For more information on this file, see
TODO: link to settings document
"""

import logging
import logging.config
import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Absolute path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.relpath(__file__))))  # Relative path

# Paths and files
DATA_DIR = os.path.join(BASE_DIR, "data")
VAL_DATA_DIR = os.path.join(BASE_DIR, "val_data")
DETECTION_DATA_DIR = os.path.join(BASE_DIR, "detection_data")
DETECTION_VAL_DATA_DIR = os.path.join(BASE_DIR, "detection_val_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAINING_LOGS_DIR = os.path.join(BASE_DIR, "training_logs")
MODEL_STAGING_DIR = os.path.join(BASE_DIR, "model-archive-staging")
SYNSET_FILENAME = os.path.join(MODELS_DIR, "synset.txt")
RGB_VALUES_FILENAME = os.path.join(MODELS_DIR, "rgb_values.json")
REC_FILENAME = "data.rec"
IDX_FILENAME = "data.idx"
DATA_PATH = os.path.join(DATA_DIR, REC_FILENAME)
VAL_PATH = os.path.join(VAL_DATA_DIR, REC_FILENAME)

PREPROCESS_PY_FILE = os.path.join(BASE_DIR, "birder", "common", "preprocess.py")

# Custom network module path
NET_MODULE = "birder.net"

# Inference
TOP_K = 3

# Augmentation, pre-pack time (offline)
AUG_RATIO = 0.1
AUG_EXCLUDE = [""]
POST_AUG_EXCLUDE = ["Rotate"]

# Training (defaults, can override in command line)
NUM_EPOCHS = 120
SAVE_FREQUENCY = 20
RESTART_EPOCHS = [NUM_EPOCHS + 40, NUM_EPOCHS + 80]

# Logging configuration
# https://docs.python.org/3/library/logging.config.html
LOG_LEVEL = logging.INFO

LOGGING = {
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
    "handlers": {"console": {"class": "logging.StreamHandler", "level": LOG_LEVEL, "formatter": "verbose"}},
    "loggers": {},
    "root": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": True},
}

logging.config.dictConfig(LOGGING)
logging.captureWarnings(True)
logging.debug("Settings loaded")
