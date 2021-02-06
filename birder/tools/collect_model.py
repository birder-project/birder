import argparse
import logging
import os
import pathlib
import shutil

from birder.common import util
from birder.conf import settings


def clean_dir(directory: str):
    logging.info(f"Removing {directory} contents")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

        else:
            logging.error(f"Cannot handle {file_path}")
            raise SystemExit(1)


def set_parser(subparsers):
    subparser = subparsers.add_parser(
        "collect-model",
        help="collect all required model files for export",
        formatter_class=util.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--network", type=str, required=True, help="the neural network to use (i.e. mobilenet_v1_1.0)"
    )
    subparser.add_argument("-e", "--epoch", type=int, default=0, help="model checkpoint to load")
    subparser.add_argument("--force", action="store_true", help="remove existing files from destination")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    dst = settings.MODEL_STAGING_DIR
    os.makedirs(dst, exist_ok=True)
    logging.info(f"Target path is: {dst}")

    if len(os.listdir(dst)) != 0:
        if args.force is True:
            clean_dir(dst)

        else:
            logging.warning(f"{dst} directory is not empty, use --force to override")
            raise SystemExit(1)

    classes = util.read_synset()

    network_name = f"{args.network}_{len(classes)}"
    model_path = util.get_model_path(network_name)

    symbol_path = f"{model_path}-symbol.json"
    params_path = f"{model_path}-{args.epoch:04d}.params"
    signature_path = util.get_model_signature_path(network_name)

    # Copy data
    logging.info(f"Copying {symbol_path}")
    shutil.copy(symbol_path, dst)
    logging.info(f"Copying {params_path} as epoch 0")
    shutil.copy(params_path, os.path.join(dst, f"{network_name}-0000.params"))
    logging.info(f"Copying {settings.SYNSET_FILENAME}")
    shutil.copy(settings.SYNSET_FILENAME, dst)
    logging.info(f"Copying {signature_path} as signature.json")
    shutil.copy(signature_path, os.path.join(dst, "signature.json"))

    if os.path.isfile(settings.RGB_VALUES_FILENAME) is True:
        logging.info(f"Copying {settings.RGB_VALUES_FILENAME}")
        shutil.copy(settings.RGB_VALUES_FILENAME, dst)

    else:
        logging.warning(f"RGB file not found ({settings.RGB_VALUES_FILENAME}), skipping")

    # Copy code
    logging.info(f"Copying {settings.PREPROCESS_PY_FILE}")
    shutil.copy(settings.PREPROCESS_PY_FILE, dst)

    for py_file in pathlib.Path("birder", "service").glob("*.py"):
        logging.info(f"Copying {py_file}")
        shutil.copy(py_file, dst)

    logging.info("Done, you can now use the model-archiver")
    logging.info(
        f"Try 'model-archiver --model-name {network_name} --model-path {dst} --handler classification:handle'"
    )
