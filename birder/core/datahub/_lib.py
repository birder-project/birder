import hashlib
import logging
from pathlib import Path
from urllib.request import Request
from urllib.request import urlopen

from tqdm import tqdm


def calc_sha256(file_path: str | Path) -> str:
    chunk_size = 64 * 4096
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as handle:
        file_buffer = handle.read(chunk_size)
        while len(file_buffer) > 0:
            sha256.update(file_buffer)
            file_buffer = handle.read(chunk_size)

    return sha256.hexdigest()


def _download(url: str, filename: str | Path, chunk_size: int = 1024 * 32) -> None:
    req = Request(url, headers={"User-Agent": "birder.datahub"})
    if url.startswith("http") is False:
        raise ValueError("Only http and https url's are allowed")

    with urlopen(req) as response:  # nosec
        with (
            open(filename, "wb") as f,
            tqdm(total=response.length, unit="B", unit_scale=True, unit_divisor=1024) as progress,
        ):
            while True:
                buffer = response.read(chunk_size)
                if len(buffer) == 0:
                    break

                f.write(buffer)
                progress.update(len(buffer))


def download_url(url: str, target: str | Path, sha256: str) -> None:
    if isinstance(target, str) is True:
        target = Path(target)

    if target.exists() is True:  # type: ignore
        if calc_sha256(target) == sha256:
            logging.debug("File already downloaded and verified")

        else:
            raise RuntimeError("Downloaded file is corrupted")

    else:
        logging.info(f"Downloading {url} to {target}")
        _download(url, target)
        if calc_sha256(target) != sha256:
            raise RuntimeError("Downloaded file is corrupted")
