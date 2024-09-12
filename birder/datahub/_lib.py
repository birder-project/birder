import logging
import tarfile
from pathlib import Path

from birder.common import cli


def download_url(url: str, target: str | Path, sha256: str) -> bool:
    if isinstance(target, str) is True:
        target = Path(target)

    if target.exists() is True:  # type: ignore
        if cli.calc_sha256(target) == sha256:
            logging.debug("File already downloaded and verified")
            return False

        raise RuntimeError("Downloaded file is corrupted")

    target.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    logging.info(f"Downloading {url} to {target}")
    cli.download_file(url, target, sha256)  # type: ignore[arg-type]
    return True


def extract_archive(from_path: str | Path, to_path: str | Path) -> None:
    logging.info(f"Extracting {from_path} to {to_path}")
    with tarfile.open(from_path, "r") as tar:
        if hasattr(tarfile, "data_filter") is True:
            tar.extractall(to_path, filter="data")
        else:
            # Remove once minimum Python version is 3.12 or above
            tar.extractall(to_path)  # nosec - tarfile_unsafe_members
