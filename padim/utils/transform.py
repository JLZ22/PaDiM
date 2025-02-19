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
Based on 'albumentations' formulation of data transformation method
"""
import logging

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_data_transform(transform_dict: DictConfig) -> list:
    r"""Get data transform.

    Args:
        transform_dict (DictConfig): transform config.

    Returns:
        A.Compose: data transform.
    """
    transform: A.Compose
    transform_list = []

    resize_transform = transform_dict.get("RESIZE", {})
    shift_scale_rotate_transform = transform_dict.get("SHIFT_SCALE_ROTATE", {})
    affine = transform_dict.get("AFFINE", {})
    center_crop_transform = transform_dict.get("CENTER_CROP", {})
    normalize_transform = transform_dict.get("NORMALIZE", {})

    if resize_transform:
        transform_list.append(A.Resize(resize_transform.get("HEIGHT"), resize_transform.get("WIDTH"), cv2.INTER_NEAREST))

    if affine:
        transform_list.append(A.Affine(
            translate_percent=affine.get("TRANSLATE_PERCENT"),
            p=affine.get("P"),
        ))

    if center_crop_transform:
        transform_list.append(A.CenterCrop(center_crop_transform.get("HEIGHT"), center_crop_transform.get("WIDTH")))

    if normalize_transform:
        transform_list.append(A.Normalize(normalize_transform.get("MEAN"), normalize_transform.get("STD"), 1.0))

    transform_list.append(ToTensorV2())
    logger.info(f"transform_list: {transform_list}")

    return transform_list
