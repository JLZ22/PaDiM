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
import time
from abc import ABC
from copy import deepcopy
from pathlib import Path
from typing import Dict

import albumentations as A
import torch
import torch.utils.data
from omegaconf import DictConfig
from torch import nn, Tensor

from padim.datasets import MVTecDataset, FolderDataset
from padim.datasets.utils import CPUPrefetcher, CUDAPrefetcher
from padim.models import PaDiM
from padim.utils import select_device, get_data_transform
from padim.utils.logger import AverageMeter, ProgressMeter
from .base import Base
from .evaler import Evaler

logger = logging.getLogger(__name__)


class Trainer(Base, ABC):
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device = select_device(config["DEVICE"])

        self.stats: list[Tensor] = []
        self.embeddings: list[Tensor] = []

        self.cls_task = self.config.TASK == "classification"

        transforms_dict = self.config.DATASETS.TRANSFORMS
        self.image_transforms, self.mask_transforms = self.create_transform(transforms_dict)
        self.mask_size = (transforms_dict.CENTER_CROP.get("HEIGHT"), transforms_dict.CENTER_CROP.get("WIDTH"))
        self.model = self.create_model()
        self.train_loader, self.val_loader = self.get_dataloader()

        # Evaluate configure
        self.evaler = Evaler(config)

        self.save_weights_dir: Path = Path("results") / "train" / config.EXP_NAME
        self.save_weights_path: Path = Path(self.save_weights_dir) / "model.pkl"
        self.save_visuals_dir: Path = Path(self.save_weights_dir) / "visuals"
        self.save_weights_dir.mkdir(exist_ok=True, parents=True)
        self.save_visuals_dir.mkdir(exist_ok=True, parents=True)

    def create_model(self) -> nn.Module:
        """Create a model."""
        logger.info(f"Create model: {self.config.MODEL.BACKBONE}")
        model = PaDiM(self.config.MODEL.BACKBONE, self.config.MODEL.RETURN_NODES, mask_size=self.mask_size)
        model = model.to(self.device)
        return model

    def create_transform(self, transforms_list: DictConfig) -> [A.Compose, A.Compose]:
        """Get the loader for training and validation."""
        image_transforms_list = get_data_transform(transforms_list)
        mask_transforms_list = image_transforms_list.copy()
        mask_transforms_list.pop(-2)  # Remove the normalization transform
        image_transforms = A.Compose(image_transforms_list)
        mask_transforms = A.Compose(mask_transforms_list)

        return image_transforms, mask_transforms

    def create_datasets(self, train: bool) -> FolderDataset | MVTecDataset:
        if self.cls_task:
            logger.info("Load classification dataset.")
            datasets = FolderDataset(
                self.config.DATASETS.ROOT.get("TRAIN") if train else self.config.DATASETS.ROOT.get("VAL"),
                self.image_transforms,
                self.mask_size,
                train,
            )
        else:
            logger.info("Load segmentation dataset.")
            datasets = MVTecDataset(
                self.config.DATASETS.ROOT,
                self.config.DATASETS.CATEGORY,
                self.image_transforms,
                self.mask_transforms,
                self.mask_size,
                train,
            )
        return datasets

    def create_dataloader(self, datasets: FolderDataset | MVTecDataset, train: bool):
        dataloader = torch.utils.data.DataLoader(
            datasets,
            batch_size=self.config.TRAIN.HYP.get("IMGS_PER_BATCH") if train else len(datasets),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        if self.device == "cuda":
            dataloader = CUDAPrefetcher(dataloader, self.device)
        else:
            dataloader = CPUPrefetcher(dataloader)

        return dataloader

    def get_dataloader(self) -> [CPUPrefetcher | CUDAPrefetcher, CPUPrefetcher | CUDAPrefetcher]:
        train_datasets = self.create_datasets(train=True)
        val_datasets = self.create_datasets(train=False)

        train_dataloader = self.create_dataloader(train_datasets, train=True)
        val_dataloader = self.create_dataloader(val_datasets, train=False)

        return train_dataloader, val_dataloader

    def get_embeddings(self) -> None:
        """Get features from the backbone network."""
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        progress = ProgressMeter(len(self.train_loader), [batch_time, data_time], prefix="Get features ")

        end = time.time()
        for i, batch_data in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            image = batch_data["image"].to(self.device, non_blocking=True)
            embedding = self.model(image)
            self.embeddings.append(embedding)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.config.TRAIN.PRINT_FREQ == 0:
                progress.display(i + 1)

    def compute_patch_distribution(self):
        logger.info("Collecting the embeddings from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Applying Gaussian fitting to the embeddings from the training set.")
        self.stats = self.model.multi_variate_gaussian.fit(embeddings)

    def create_state_dict(self) -> Dict:
        """Create a state dictionary for saving the model."""
        state_dict = {
            "model": deepcopy(self.model),
            "image_transforms": self.image_transforms,
            "mask_transforms": self.mask_transforms,
            "mask_size": self.mask_size,
            "config": self.config,
        }
        return state_dict

    def save_checkpoint(self, state_dict: Dict) -> None:
        """Save the model checkpoint."""
        logger.info(f"Save the model to '{self.save_weights_path}'. please wait...")
        torch.save(state_dict, self.save_weights_path)
        logger.info("Save the model successfully.")

    def train(self) -> None:
        self.get_embeddings()
        self.compute_patch_distribution()

        state_dict = self.create_state_dict()
        self.save_checkpoint(state_dict)

        self.evaler.run_validation(
            self.model,
            self.val_loader,
            self.cls_task,
            self.device,
            self.save_visuals_dir,
        )
