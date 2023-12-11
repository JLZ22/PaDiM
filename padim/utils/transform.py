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
import logging

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

__all__ = [
    "get_data_transforms",
]


def get_data_transforms(config: DictConfig, mask_mode: bool = False) -> A.Compose:
    """Get transforms from config or image size.

    Args:
        config (str | A.Compose | None, optional): Albumentations transforms.
            Either config or albumentations ``Compose`` object. Defaults to None.
        mask_mode (bool, optional): If True, return transforms for mask. Defaults to False.

    Returns:
        A.Compose: Albumentation ``Compose`` object containing the image transforms.

    Examples:
        >>> config = OmegaConf.load("/tmp/transforms.yaml")
        >>> config = OmegaConf.create(config)
        >>> transforms = get_data_transforms(config)
    """
    transforms: A.Compose
    transforms_list = []

    if config is None:
        raise ValueError("Not found transform in config.")

    if config.RESIZE is not None:
        transforms_list.append(A.Resize(config.RESIZE.HEIGHT, config.RESIZE.WIDTH))
    if config.NORMALIZE is not None and not mask_mode:
        transforms_list.append(A.Normalize(config.NORMALIZE.MEAN, config.NORMALIZE.STD))

    if len(transforms_list) == 0:
        raise ValueError("Not found transform in config.")

    transforms_list.append(ToTensorV2())
    transforms = A.Compose(transforms_list)

    return transforms
