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
import os
from pathlib import Path

import torch
import torch.utils.data
from PIL import Image
from omegaconf import DictConfig

from padim.utils.download import DownloadInfo
from padim.utils.transform import get_data_transforms
from torchvision import transforms as T

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="mvtec",
    url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
    hash="eefca59f2cede9c3fc5b6befbfec275e",
)

CLASS_NAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTecDataset(torch.utils.data.Dataset):
    r"""MVTec Dataset

    Args:
        root (str | Path): root directory of dataset where directory ``mvtec_anomaly_detection`` exists.
        category (str): category name of dataset (``bottle``, ``cable``, ``capsule``, ``carpet``, ``grid``, ``hazelnut``, ``leather``, ``metal_nut``, ``pill``, ``screw``, ``tile``, ``toothbrush``, ``transistor``, ``wood``, ``zipper``).
        transforms_dict_config (DictConfig): transforms config for image and mask.
        is_train (bool, optional): if True, load train dataset, else load test dataset. Defaults to True.

    Examples:
        >>> from padim.datasets import MVTecDataset
        >>> from omegaconf import OmegaConf
        >>> transforms_dict_config = OmegaConf.load("configs/transforms.yaml")
        >>> dataset = MVTecDataset(root="./data/mvtec_anomaly_detection", category="bottle", transforms_dict_config=transforms_dict_config, is_train=True)
        >>> len(dataset)
        209
        >>> image, target, mask = dataset[0]
        >>> image.shape
        torch.Size([3, 256, 256])
        >>> target
        0
        >>> mask.shape
        torch.Size([1, 256, 256])
    """

    def __init__(
            self,
            root: str | Path,
            category: str,
            image_size: tuple[int, int],
            center_crop: tuple[int, int],
            normalize_mean: tuple[float, float, float],
            normalize_std: tuple[float, float, float],
            is_train: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.category = category
        self.image_size = image_size
        self.center_crop = center_crop
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.is_train = is_train

        # set transforms
        self.image_transforms, self.mask_transforms = get_data_transforms(image_size,
                                                                          center_crop,
                                                                          normalize_mean,
                                                                          normalize_std)

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.image_transforms(x)

        if y == 0:
            mask = torch.zeros([1, 224, 224])
        else:
            mask = Image.open(mask)
            mask = self.mask_transforms(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.root, self.category, phase)
        gt_dir = os.path.join(self.root, self.category, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
