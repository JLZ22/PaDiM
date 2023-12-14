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
Common utilities.
"""
import logging

import torch

__all__ = [
    "select_device", "get_image_height_and_width"
]

logger = logging.getLogger(__name__)


def select_device(device: str = "cpu") -> torch.device:
    r"""Automatically select device (CPU or GPU).

    Args:
        device (str, optional): device name. Defaults to "cpu".

    Raises:
        ValueError: device not supported.

    Examples:
        >>> select_device("cpu")
        Use CPU.
        >>> select_device("cuda")
        Use CUDA.
        >>> select_device("gpu")
        Use CUDA.
        >>> select_device("tpu")
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in select_device
        ValueError: Device 'tpu' not supported. Choices: ['cpu', 'cuda', 'gpu']

    Returns:
        torch.device: device.
    """
    supported_devices = ["cpu", "cuda", "gpu"]
    if device not in supported_devices:
        raise ValueError(f"Device '{device}' not supported. Choices: {supported_devices}")

    if device == "cpu":
        logger.info("Use CPU.")
        device = torch.device("cpu")
        if torch.cuda.is_available():
            logger.info("You have a CUDA device, enabling CUDA will give you a large boost in performance.")
    elif device in ["cuda", "gpu"]:
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, switching to CPU.")
            device = torch.device("cpu")
        else:
            logger.info("Use CUDA.")
            device = torch.device("cuda")

    return device


def get_image_height_and_width(image_size: int | tuple | None = None) -> tuple[int | None, int | None]:
    """Get image height and width from ``image_size`` variable.

    Args:
        image_size (int | tuple | None, optional): Input image size.

    Raises:
        ValueError: Image size must be either int or tuple[int, int].

    Examples:
        >>> get_image_height_and_width(image_size=256)
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256))
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256, 3))
        (256, 256)

        >>> get_image_height_and_width(image_size=256.)
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in get_image_height_and_width
        ValueError: ``image_size`` could be either int or tuple[int, int]

    Returns:
        tuple[int | None, int | None]: A tuple containing image height and width values.
    """
    height_and_width: tuple[int | None, int | None]
    if isinstance(image_size, int):
        height_and_width = (image_size, image_size)
    elif isinstance(image_size, tuple):
        height_and_width = int(image_size[0]), int(image_size[1])
    else:
        raise ValueError("``image_size`` could be either int or tuple[int, int]")

    return height_and_width
