# Copyright 2023 AlphaBetter Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Basic implementation of download function
"""
import hashlib
import logging
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from tqdm import tqdm

__all__ = [
    "DownloadInfo", "DownloadProgressBar", "download_and_extract",
]

logger = logging.getLogger(__name__)

__all__ = [
    "DownloadInfo", "DownloadProgressBar", "download_and_extract",
]

UNSAFE_PATTERNS = ["/etc/", "/root/"]


@dataclass
class DownloadInfo:
    r"""A download method similar to class"""
    name: str
    url: str
    hash: str
    filename: str | None = None


class DownloadProgressBar(tqdm):
    r"""Create progress bar for urlretrieve. Subclasses `tqdm`.

        References:
            For information about the parameters in constructor, refer to `tqdm`'s documentation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total: int | float | None

    def update_to(self, chunk_number: int = 1, max_chunk_size: int = 1, total_size: int | float = None) -> None:
        r"""Progress bar hook for tqdm.

        Args:
            chunk_number (int, optional): The current chunk being processed. Defaults to 1.
            max_chunk_size (int, optional): Maximum size of each chunk. Defaults to 1.
            total_size ([type], optional): Total download size. Defaults to None.

        References:
            https://github.com/tqdm/tqdm#hooks-and-callbacks
        """
        if total_size is not None:
            self.total = total_size
        self.update(chunk_number * max_chunk_size - self.n)


def _is_file_potentially_dangerous(file_name: str) -> bool:
    r"""Check if a file is potentially dangerous.

    Args:
        file_name (str): Filename.

    Returns:
        bool: True if the member is potentially dangerous, False otherwise.
    """
    return any(re.search(pattern, file_name) for pattern in UNSAFE_PATTERNS)


def _is_within_directory(directory: Path, target: Path) -> bool:
    r"""Checks if a target path is located within a given directory.

    Args:
        directory (Path): path of the parent directory
        target (Path): path of the target
    Returns:
        (bool): True if the target is within the directory, False otherwise
    """
    try:
        target.relative_to(directory)
        return True
    except ValueError:
        return False


def _calculate_md5(file_path: Path) -> str:
    r"""Calculate the MD5 hash of a file.

    Args:
        file_path (Path): Path to file.

    Returns:
        str: The MD5 hash of the file.
    """
    hash_md5 = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _hash_check(file_path: Path, expected_hash: str) -> None:
    r"""Raise assert error if hash does not match the calculated hash of the file.

    Args:
        file_path (Path): Path to file.
        expected_hash (str): Expected hash of the file.
    """
    assert (_calculate_md5(file_path) == expected_hash), f"Downloaded file {file_path} does not match the required hash."


def _extract_zip(file_name: Path, root: Path) -> None:
    """Extract a zip file.

    Args:
        file_name (Path): Path of the file to be extracted.
        root (Path): Root directory where the dataset will be stored.
    """
    with ZipFile(file_name, "r") as zip_file:
        for file_info in zip_file.infolist():
            if not _is_file_potentially_dangerous(file_info.filename):
                zip_file.extract(file_info, root)


def _extract_tar(file_name: Path, root: Path) -> None:
    r"""Extract a tar file.

    Args:
        file_name (Path): Path of the file to be extracted.
        root (Path): Root directory where the dataset will be stored.
    """
    with tarfile.open(file_name) as tar_file:
        members = tar_file.getmembers()
        safe_members = [member for member in members if not _is_file_potentially_dangerous(member.name)]
        for safe_member in safe_members:
            tar_file.extract(safe_member, root)


def _extract(file_name: Path, root: Path) -> None:
    r"""Extract a dataset

    Args:
        file_name (Path): Path of the file to be extracted.
        root (Path): Root directory where the dataset will be stored.
    """
    logger.info("Extracting dataset into root folder.")

    if file_name.suffix == ".zip":
        _extract_zip(file_name, root)
    elif file_name.suffix in (".tar", ".gz", ".xz", ".tgz"):
        _extract_tar(file_name, root)
    else:
        raise ValueError(f"Unrecognized file format: {file_name}")

    logger.info("Cleaning up files.")
    file_name.unlink()


def download_and_extract(root: Path, info: DownloadInfo) -> None:
    r"""Download and extract a dataset.

    Args:
        root (Path): Root directory where the dataset will be stored.
        info (DownloadInfo): Info needed to download the dataset.
    """
    root.mkdir(parents=True, exist_ok=True)

    # save the compressed file in the specified root directory, using the same file name as on the server
    if info.filename:
        downloaded_file_path = root / info.filename
    else:
        downloaded_file_path = root / info.url.split("/")[-1]

    if downloaded_file_path.exists():
        logger.info("Existing dataset archive found. Skipping download stage.")
    else:
        logger.info(f"Downloading the {info.name} dataset.")
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=info.name) as progress_bar:
            urlretrieve(
                url=info.url,
                filename=downloaded_file_path,
                reporthook=progress_bar.update_to,
            )
        logger.info("Checking the hash of the downloaded file.")
        _hash_check(downloaded_file_path, info.hash)

    _extract(downloaded_file_path, root)
