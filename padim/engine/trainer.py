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

logger = logging.getLogger(__name__)


class Trainer(Base, ABC):
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device = select_device(config["DEVICE"])

        self.stats: list[Tensor] = []
        self.embeddings: list[Tensor] = []

        self.cls_task: bool = False
        if self.config.TASK == "classification":
            self.cls_task = True

        transforms_dict = self.config.DATASETS.TRANSFORMS
        self.image_transforms, self.mask_transforms = self.get_transform(transforms_dict)
        self.mask_size = (transforms_dict.CENTER_CROP.get("HEIGHT"), transforms_dict.CENTER_CROP.get("WIDTH"))
        self.model = self.create_model()
        self.train_loader, self.val_loader = self.get_loader()

        # Create a folder to save the model
        save_weights_dir = Path("results", "train", config.EXP_NAME)
        save_weights_dir.mkdir(exist_ok=True, parents=True)
        self.save_weights_path = Path(save_weights_dir, "model.pkl")

        # Evaluate the model
        # self.evaler = Evaler(config)
        # # Create a folder to save the visual results
        # self.save_visual_dir = Path("results", "train", config.EXP_NAME, "visual")
        # self.save_visual_dir.mkdir(exist_ok=True, parents=True)

    def create_model(self) -> nn.Module:
        """Create a model."""
        logger.info(f"Create model: {self.config.MODEL.BACKBONE}")
        model = PaDiM(self.config.MODEL.BACKBONE, self.config.MODEL.RETURN_NODES, mask_size=self.mask_size)
        model = model.to(self.device)
        return model

    def get_transform(self, transforms_list: DictConfig) -> [A.Compose, A.Compose]:
        """Get the loader for training and validation."""
        image_transforms_list = get_data_transform(transforms_list)
        mask_transforms_list = image_transforms_list.copy()
        mask_transforms_list.pop(-2)  # Remove the normalization transform
        image_transforms = A.Compose(image_transforms_list)
        mask_transforms = A.Compose(mask_transforms_list)

        return image_transforms, mask_transforms

    def get_loader(self) -> [CPUPrefetcher | CUDAPrefetcher, CPUPrefetcher | CUDAPrefetcher]:
        if self.cls_task:
            logger.info("Load classification dataset.")
            train_dataset = FolderDataset(
                self.config.DATASETS.ROOT.TRAIN,
                self.image_transforms,
                self.mask_size,
            )
            val_dataset = FolderDataset(
                self.config.DATASETS.ROOT.get("VAL"),
                self.image_transforms,
                self.mask_size,
            )
        else:
            logger.info("Load segmentation dataset.")
            train_dataset = MVTecDataset(
                self.config.DATASETS.ROOT,
                self.config.DATASETS.CATEGORY,
                self.image_transforms,
                self.mask_transforms,
                self.mask_size,
                train=True,
            )
            val_dataset = MVTecDataset(
                self.config.DATASETS.ROOT,
                self.config.DATASETS.CATEGORY,
                self.image_transforms,
                self.mask_transforms,
                self.mask_size,
                train=False,
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.TRAIN.HYP.get("IMGS_PER_BATCH"),
            shuffle=True,
            num_workers=4,
            sampler=None,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            shuffle=False,
            num_workers=4,
            sampler=None,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )

        if self.device == "cuda":
            train_loader = CUDAPrefetcher(train_loader, self.device)
            val_loader = CUDAPrefetcher(val_loader, self.device)
        else:
            train_loader = CPUPrefetcher(train_loader)
            val_loader = CPUPrefetcher(val_loader)

        return train_loader, val_loader

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

    def save_checkpoint(self) -> None:
        """Save the model checkpoint."""
        logger.info(f"Save the model to '{self.save_weights_path}'. please wait...")
        torch.save({
            "model": deepcopy(self.model),
            "stats": self.stats,
            "image_transforms": self.image_transforms,
            "mask_transforms": self.mask_transforms,
            "mask_size": self.mask_size,
            "config": self.config,
        }, self.save_weights_path)
        logger.info("Save the model successfully.")

    def train(self) -> None:
        self.get_embeddings()
        self.compute_patch_distribution()
        self.save_checkpoint()

        # self.evaler.run_validation(
        #     self.model,
        #     self.val_loader,
        #     self.cls_task,
        #     self.device,
        #     self.save_visual_dir,
        # )
