import argparse
import ast
import hashlib
import json
import logging
import shutil
import ssl
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit
from urllib.request import Request
from urllib.request import urlopen

from tqdm import tqdm

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    pass


class ArgumentHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


class FlexibleDictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore
        existing_dict = getattr(namespace, self.dest, {})
        if existing_dict is None:
            existing_dict = {}

        # Try parsing as JSON first
        try:
            parsed_value = json.loads(values)
            if isinstance(parsed_value, dict):
                existing_dict.update(parsed_value)
                setattr(namespace, self.dest, existing_dict)
                return

        except (json.JSONDecodeError, TypeError):
            pass

        # Try parsing comma-separated simple key-value pairs
        try:
            pairs = [pair.strip() for pair in values.split(",")]
            new_dict = {}
            for pair in pairs:
                # Split each pair into key and value
                key, value = pair.split("=", 1)
                key = key.strip()

                # Try to safely evaluate the value (handles ints and strings mostly)
                try:
                    parsed_value = ast.literal_eval(value.strip())
                except (ValueError, SyntaxError):
                    # If literal_eval fails, keep it as a string
                    parsed_value = value.strip()

                new_dict[key] = parsed_value

            # Update the existing dictionary
            existing_dict.update(new_dict)
            setattr(namespace, self.dest, existing_dict)

        except ValueError as e:
            parser.error(f"Invalid input format for {option_string}: {e}")


def parse_size(args_size: Optional[Sequence[int] | int]) -> Optional[tuple[int, int]]:
    if args_size is None:
        return None

    if isinstance(args_size, Sequence):
        if len(args_size) == 1:
            return (args_size[0], args_size[0])
        if len(args_size) == 2:
            return (args_size[0], args_size[1])

        raise ValueError("Invalid size format. Use either a single value for square or two values for height and width")

    args_size = int(args_size)

    return (args_size, args_size)


def calc_sha256(file_path: str | Path) -> str:
    chunk_size = 64 * 4096
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as handle:
        file_buffer = handle.read(chunk_size)
        while len(file_buffer) > 0:
            sha256.update(file_buffer)
            file_buffer = handle.read(chunk_size)

    return sha256.hexdigest()


def download_file(
    url: str,
    dst: str | Path,
    expected_sha256: Optional[str] = None,
    override: bool = False,
    progress_bar: bool = True,
    ignore_ssl_errors: bool = False,
) -> None:
    # Adapted from torch.hub download_url_to_file function

    chunk_size = 128 * 1024

    if isinstance(dst, str):
        dst = Path(dst)

    # Reuse an existing file unless the caller explicitly requests an override
    if dst.exists() is True:
        if override is True:
            logger.warning(f"Overriding existing file {dst}")
        elif expected_sha256 is None:
            logger.debug("Found existing file, skipping download")
            return
        elif calc_sha256(dst) == expected_sha256:
            logger.debug("Found existing file with the same hash, skipping download")
            return
        else:
            raise FileExistsError(f"{dst} exists with a different SHA256, pass override=True to replace it")

    fname = urlsplit(url)[2].split("/")[-1]
    logger.info(f"Downloading {fname} to {dst}...")

    file_size = None
    req = Request(url, headers={"User-Agent": "birder.datahub"})
    ssl_context = None
    if ignore_ssl_errors is True:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    # We deliberately save it in a temp file and move it after download is complete
    # This prevents a local working checkpoint being overridden by a broken download
    tmp_dst = Path(f"{dst}.{uuid.uuid4().hex}.partial")
    try:
        with urlopen(req, context=ssl_context) as response:  # nosec
            meta = response.info()
            if hasattr(meta, "getheaders") is True:
                content_length = meta.getheaders("Content-Length")
            else:
                content_length = meta.get_all("Content-Length")

            if content_length is not None and len(content_length) > 0:
                file_size = int(content_length[0])

            sha256 = hashlib.sha256()
            with open(tmp_dst, "w+b") as handle:
                with tqdm(
                    total=file_size, unit="B", unit_scale=True, unit_divisor=1024, disable=not progress_bar
                ) as progress:
                    while True:
                        buffer = response.read(chunk_size)
                        if len(buffer) == 0:
                            break

                        handle.write(buffer)
                        sha256.update(buffer)
                        progress.update(len(buffer))

        digest = sha256.hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(f'invalid hash value (expected "{expected_sha256}", got "{digest}")')

        shutil.move(tmp_dst, dst)
        logger.info(f"Finished, file saved at {dst}")

    finally:
        tmp_dst.unlink(missing_ok=True)
