import argparse
import hashlib
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any
from urllib.request import Request
from urllib.request import urlopen

from tqdm import tqdm

from birder.common import cli
from birder.conf import settings
from birder.model_registry import registry


def download_file(url: str, dst: Path | str, expected_sha256: str) -> None:
    # Adapted from torch.hub download_url_to_file function

    chunk_size = 128 * 1024
    file_size = None
    req = Request(url, headers={"User-Agent": "birder.datahub"})
    u = urlopen(req)  # pylint: disable=consider-using-with  # nosec
    meta = u.info()
    if hasattr(meta, "getheaders") is True:
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")

    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after download is complete.
    # This prevents a local working checkpoint being overridden by a broken download.
    tmp_dst = str(dst) + "." + uuid.uuid4().hex + ".partial"
    try:
        f = open(tmp_dst, "w+b")  # pylint: disable=consider-using-with
        sha256 = hashlib.sha256()
        with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as progress:
            while True:
                buffer = u.read(chunk_size)
                if len(buffer) == 0:
                    break

                f.write(buffer)
                sha256.update(buffer)
                progress.update(len(buffer))

        f.close()
        u.close()
        digest = sha256.hexdigest()
        if digest != expected_sha256:
            raise RuntimeError(f'invalid hash value (expected "{expected_sha256}", got "{digest}")')

        shutil.move(f.name, dst)
        logging.info(f"Finished, model saved at {dst}")

    finally:
        f.close()
        if os.path.exists(f.name) is True:
            os.remove(f.name)


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "fetch-model",
        allow_abbrev=False,
        help="download pretrained model",
        description="download pretrained model",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools fetch-model mobilenet_v3_1_0\n"
            "python -m birder.tools fetch-model convnext_v2_4_0 --force\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--format", type=str, choices=["pt", "pt2", "ptl", "pts"], default="pt", help="model serialization format"
    )
    subparser.add_argument("--force", action="store_true", help="force download even if model already exists")
    subparser.add_argument("model_name", choices=registry.list_pretrained_models(), help="the model to download")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    if settings.MODELS_DIR.exists() is False:
        logging.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True)

    model_info = registry.get_pretrained_info(args.model_name)
    if args.format not in model_info["formats"]:
        logging.warning(f"Available formats for {args.model_name} are: {model_info['formats']}")
        raise SystemExit(1)

    model_file = f"{args.model_name}.{args.format}"
    dst = settings.MODELS_DIR.joinpath(model_file)
    if dst.exists() is True and args.force is False:
        logging.warning(f"Model {args.model_name} already exists... aborting")
        raise SystemExit(1)

    url = f"{settings.REGISTRY_BASE_UTL}/{model_file}"
    download_file(url, dst, model_info["sha256"])
