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
import time
from abc import ABC
from copy import deepcopy

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

        self.cls_task: bool = False
        if self.config.TASK == "classification":
            self.cls_task = True

        self.image_transforms, self.mask_transforms, self.mask_size = self.get_transform(self.config.DATASETS.TRANSFORMS)
        self.model = self.create_model()
        self.train_dataloader, self.val_dataloader = self.get_dataloader()

        self.stats: list[Tensor] = []
        self.embeddings: list[Tensor] = []

        # self.evaler = Evaler(config)

        # Create a folder to save the model
        save_weights_dir = os.path.join("results", "train", config.EXP_NAME)
        self.save_weights_path = os.path.join(save_weights_dir, "model.pkl")
        os.makedirs(save_weights_dir, exist_ok=True)
        # Create a folder to save the visual results
        self.save_visual_dir = os.path.join("results", "train", config.EXP_NAME, "visual")
        os.makedirs(self.save_visual_dir, exist_ok=True)

    def create_model(self) -> nn.Module:
        """Create a model."""
        logger.info(f"Create model: {self.config.MODEL.BACKBONE}")
        model = PaDiM(self.config.MODEL.BACKBONE, self.mask_size, self.config.MODEL.RETURN_NODES)
        model = model.to(self.device)
        return model

    def get_transform(self, transforms_list: DictConfig) -> [A.Compose, A.Compose, tuple[int, int]]:
        """Get the dataloader for training and validation."""
        image_transforms_list = get_data_transform(transforms_list)
        mask_transforms_list = image_transforms_list.copy()
        image_transforms = A.Compose(image_transforms_list)
        mask_transforms = A.Compose(mask_transforms_list)
        mask_transforms_list.pop(-2)  # Remove the normalization transform
        if transforms_list.get("RESIZE") is not None:
            mask_size = (transforms_list.RESIZE.get("HEIGHT"), transforms_list.RESIZE.get("WIDTH"))
        elif transforms_list.get("CENTER_CROP") is not None:
            mask_size = (transforms_list.CENTER_CROP.get("HEIGHT"), transforms_list.CENTER_CROP.get("WIDTH"))
        else:
            logger.error("Please specify the size of the mask.")
            return

        return image_transforms, mask_transforms, mask_size

    def get_dataloader(self) -> [torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        if self.cls_task:
            logger.info("Load classification dataset.")
            train_dataset = FolderDataset(
                self.config.DATASETS.ROOT.TRAIN,
                self.image_transforms,
                self.mask_size,
            )
            val_dataset = FolderDataset(
                self.config.DATASETS.ROOT.TEST,
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

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
            shuffle=True,
            num_workers=4,
            sampler=None,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config["VAL"]["HYP"]["IMGS_PER_BATCH"],
            shuffle=False,
            num_workers=4,
            sampler=None,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )

        if self.device == "cuda":
            train_dataloader = CUDAPrefetcher(train_dataloader, self.device)
            val_dataloader = CUDAPrefetcher(val_dataloader, self.device)
        else:
            train_dataloader = CPUPrefetcher(train_dataloader)
            val_dataloader = CPUPrefetcher(val_dataloader)

        return train_dataloader, val_dataloader

    def get_features(self) -> None:
        """Get features from the backbone network."""
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        progress = ProgressMeter(len(self.train_dataloader), [batch_time, data_time], prefix="Get features ")

        end = time.time()
        for i, data in enumerate(self.train_dataloader):
            # measure data loading time
            data_time.update(time.time() - end)
            image = data["image"].to(self.device, non_blocking=True)

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
        torch.save({
            "model": deepcopy(self.model).half(),
            "stats": self.stats,
            "config": self.config,
        }, self.save_weights_path)
        logger.info(f"Save the model to {self.save_weights_path}.")

    def train(self) -> None:
        self.get_features()
        self.compute_patch_distribution()
        self.save_checkpoint()

        # if self.task == 0:
        #     category = ""
        # else:
        #     category = self.config.DATASETS.CATEGORY
        #
        # self.evaler.run_validation(
        #     self.model,
        #     category,
        #     self.val_dataloader,
        #     OmegaConf.to_container(self.config.MODEL.RETURN_NODES),
        #     self.index,
        #     train_features,
        #     transforms_list.RESIZE,
        #     self.task,
        #     self.device,
        #     self.save_visual_dir,
        # )
