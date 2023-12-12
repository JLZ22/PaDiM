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
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from padim.utils.download import DownloadInfo

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
        image_size (int): image size after resizing.
        normalize_mean (list[float]): mean values for normalization.
        normalize_std (list[float]): std values for normalization.
        is_train (bool, optional): if True, load train dataset, else load test dataset. Defaults to True.

    Examples:
        >>> from padim.datasets import MVTecDataset
        >>> from omegaconf import OmegaConf
        >>> transforms_dict_config = OmegaConf.load("configs/transforms.yaml")
        >>> dataset = MVTecDataset("./data/mvtec_anomaly_detection", "bottle", is_train=True)
        >>> len(dataset)
        209
        >>> image, target, mask = dataset[0]
        >>> image.shape
        torch.Size([3, 224, 224])
        >>> target
        0
        >>> mask.shape
        torch.Size([1, 224, 224])
    """

    def __init__(
            self,
            root: str | Path,
            category: str,
            image_size: int = 224,
            normalize_mean: [float, float, float] = None,
            normalize_std: [float, float, float] = None,
            is_train: bool = True,
    ) -> None:
        super().__init__()
        if normalize_mean is None:
            normalize_mean = [0.485, 0.456, 0.406]
        if normalize_std is None:
            normalize_std = [0.229, 0.224, 0.225]

        self.root = root
        self.category = category
        self.image_size = image_size
        self.is_train = is_train

        # set transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # load dataset
        self.images, self.targets, self.masks, self.image_fname_list = self.load_dataset_folder()

    def load_dataset_folder(self):
        phase = "train" if self.is_train else "test"
        images, targets, masks, image_fname_list = [], [], [], []

        image_dir = os.path.join(self.root, self.category, phase)
        gt_dir = os.path.join(self.root, self.category, "ground_truth")

        image_types = sorted(os.listdir(image_dir))
        for image_type in image_types:

            # load images
            image_type_dir = os.path.join(image_dir, image_type)
            if not os.path.isdir(image_type_dir):
                continue
            image_fpath_list = sorted([os.path.join(image_type_dir, f) for f in os.listdir(image_type_dir) if f.endswith(".png")])
            images.extend(image_fpath_list)

            # load gt labels
            if image_type == "good":
                targets.extend([0] * len(image_fpath_list))
                masks.extend([None] * len(image_fpath_list))
            else:
                targets.extend([1] * len(image_fpath_list))
                gt_type_dir = os.path.join(gt_dir, image_type)
                image_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in image_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, image_fname + "_mask.png") for image_fname in image_fname_list]
                masks.extend(gt_fpath_list)

        assert len(images) == len(targets), "number of x and y should be same"

        return list(images), list(targets), list(masks), image_fname_list

    def __getitem__(self, index: int):
        image, target, mask = self.images[index], self.targets[index], self.masks[index]

        image = Image.open(image).convert("RGB")
        image = self.image_transforms(image)

        if target == 0:
            mask = torch.zeros([1, self.image_size, self.image_size])
        else:
            mask = Image.open(mask)
            mask = self.mask_transforms(mask)

        return image, target, mask, self.image_fname_list[index]

    def __len__(self):
        return len(self.images)
