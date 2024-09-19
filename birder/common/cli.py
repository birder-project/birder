import argparse
import hashlib
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional
from urllib.request import Request
from urllib.request import urlopen

from tqdm import tqdm


class ArgumentHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def calc_sha256(file_path: str | Path) -> str:
    chunk_size = 64 * 4096
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as handle:
        file_buffer = handle.read(chunk_size)
        while len(file_buffer) > 0:
            sha256.update(file_buffer)
            file_buffer = handle.read(chunk_size)

    return sha256.hexdigest()


def download_file(url: str, dst: str | Path, expected_sha256: Optional[str] = None, override: bool = False) -> None:
    # Adapted from torch.hub download_url_to_file function

    chunk_size = 128 * 1024

    if isinstance(dst, str) is True:
        dst = Path(dst)

    # If file by the same name exists, check sha256 before overriding
    if dst.exists() is True:  # type: ignore[union-attr]
        if expected_sha256 is None or calc_sha256(dst) == expected_sha256:
            return

        if override is False:
            logging.warning("Found existing file with different SHA256, aborting...")

        logging.warning("Overriding existing file with different SHA256")

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

        digest = sha256.hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(f'invalid hash value (expected "{expected_sha256}", got "{digest}")')

        shutil.move(f.name, dst)
        logging.info(f"Finished, file saved at {dst}")

    finally:
        f.close()
        u.close()
        if os.path.exists(f.name) is True:
            os.remove(f.name)
