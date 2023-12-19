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
MVTec Dataset `https://www.mvtec.com/company/research/datasets/mvtec-ad/`
"""
import logging
import os
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import torch
import torch.utils.data
from torch import Tensor

from padim.utils.download import DownloadInfo, download_and_extract_archive

logger = logging.getLogger(__name__)


class MVTecDataset(torch.utils.data.Dataset):
    r"""MVTec Dataset `https://www.mvtec.com/company/research/datasets/mvtec-ad/`

    Args:
        root (str | Path): root directory of dataset where directory ``mvtec_anomaly_detection`` exists.
        category (str): category name of dataset (``bottle``, ``cable``, ``capsule``, ``carpet``, ``grid``, ``hazelnut``, ``leather``, ``metal_nut``, ``pill``, ``screw``, ``tile``, ``toothbrush``, ``transistor``, ``wood``, ``zipper``).
        image_transform (A.Compose, optional): image transform. Defaults to None.
        mask_transform (A.Compose, optional): mask transform. Defaults to None.
        mask_size (tuple[int, int], optional): mask size after resizing. Defaults to (224, 224).
        train (bool): if True, load train dataset, else load test dataset.

    Examples:
        >>> from padim.datasets import MVTecDataset
        >>> from omegaconf import OmegaConf
        >>> from padim.utils import get_data_transform
        >>> config = OmegaConf.load("configs/mvtec.yaml")
        >>> config = OmegaConf.create(config)
        >>> image_transforms = A.Compose(get_data_transform(config.DATASETS.TRANSFORMS))
        >>> mask_transforms = A.Compose(get_data_transform(config.DATASETS.TRANSFORMS))
        >>> mask_size = (config.DATASETS.TRANSFORMS.CENTER_CROP.HEIGHT, config.DATASETS.TRANSFORMS.CENTER_CROP.WIDTH)
        >>> dataset = MVTecDataset("data/mvtec_anomaly_detection", "bottle", image_transforms, mask_transforms, mask_size, False)
        >>> sample = dataset[0]
        >>> image, target, mask, image_path = sample["image"], sample["target"], sample["mask"], sample["image_path"]
        >>> print(image.shape, target, mask.shape, image_path)
        torch.Size([3, 224, 224]) 0 torch.Size([1, 224, 224]) ./data/mvtec_anomaly_detection/bottle/test/broken/good_000.png
        >>> len(dataset)
        126
    """

    download_info = DownloadInfo(
        name="mvtec",
        url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
        hash="eefca59f2cede9c3fc5b6befbfec275e",
    )

    category_names = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
        "wood", "zipper",
    ]

    def __init__(
            self,
            root: str | Path,
            category: str,
            image_transform: A.Compose = None,
            mask_transform: A.Compose = None,
            mask_size: tuple[int, int] = (224, 224),
            train: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.category = category
        self.image_transforms = image_transform
        self.mask_transforms = mask_transform
        self.mask_size = mask_size
        self.train = train

        # load dataset
        self.image_path_list, self.target_type_list, self.mask_path_list = self.load()

    def __getitem__(self, index: int) -> dict[str, str | int | Tensor | Any]:
        image_path = self.image_path_list[index]
        target_type = self.target_type_list[index]
        mask_path = self.mask_path_list[index]

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        image = self.image_transforms(image=image)["image"]

        if target_type == 0:
            mask = torch.zeros([1, self.mask_size[0], self.mask_size[1]])
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = mask.astype("float32") / 255.0
            mask = self.mask_transforms(image=mask)["image"]

        return {"image": image,
                "target": target_type,
                "mask": mask,
                "image_path": image_path}

    def __len__(self):
        return len(self.image_path_list)

    def download(self) -> None:
        if self.root.is_dir():
            logger.info("Files already downloaded and verified")
            return
        download_and_extract_archive(self.root, self.download_info)

    def load(self):
        phase = "train" if self.train else "test"
        image_list, target_list, mask_list = [], [], []

        image_dir = Path(self.root, self.category, phase)
        ground_truth_dir = Path(self.root, self.category, "ground_truth")

        image_types = sorted(os.listdir(image_dir))
        for image_type in image_types:
            # load images
            image_type_dir = os.path.join(image_dir, image_type)

            if not os.path.isdir(image_type_dir):
                continue

            image_file_path_list = sorted([os.path.join(image_type_dir, f) for f in os.listdir(image_type_dir) if f.endswith(".png")])
            image_list.extend(image_file_path_list)

            # load ground_truth labels
            if image_type == "good":
                target_list.extend([0] * len(image_file_path_list))
                mask_list.extend([None] * len(image_file_path_list))
            else:
                target_list.extend([1] * len(image_file_path_list))
                ground_truth_type_dir = os.path.join(ground_truth_dir, image_type)
                image_file_name_list = [os.path.splitext(os.path.basename(f))[0] for f in image_file_path_list]
                ground_truth_file_path_list = [os.path.join(ground_truth_type_dir, img_fname + "_mask.png") for img_fname in image_file_name_list]
                mask_list.extend(ground_truth_file_path_list)

        assert len(image_list) == len(target_list), "number of x and y should be same"

        return image_list, target_list, mask_list
