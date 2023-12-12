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

from torchvision import transforms

logger = logging.getLogger(__name__)

__all__ = [
    "get_data_transforms",
]


def get_data_transforms(
        image_size: tuple[int, int],
        center_crop: tuple[int, int],
        norm_mean: tuple[float, float, float],
        norm_std: tuple[float, float, float],
) -> [transforms.Compose, transforms.Compose]:
    """Get transforms from config or image size.

    Args:
        image_size (tuple[int, int]): image size.
        center_crop (tuple[int, int]): center crop size.
        norm_mean (tuple[float]): mean value for normalization.
        norm_std (tuple[float]): std value for normalization.

    Returns:
        [transforms.Compose, transforms.Compose]: image and mask transforms.
    """
    image_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
    ])
    return image_transforms, mask_transforms
