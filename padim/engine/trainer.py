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
import pickle
import random
from collections import OrderedDict
from typing import Any

import torch
import torch.utils.data
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from padim.datasets import MVTecDataset, FolderDataset
from padim.models import PaDiM, MODEL_NUM_FEATURES, MODEL_MAX_FEATURES
from padim.utils import select_device, cal_multivariate_gaussian_distribution, generate_embedding
from .evaler import Evaler

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device = select_device(config["DEVICE"])

        if self.config.TASK == "classification":
            self.task = 0
        elif self.config.TASK == "segmentation":
            self.task = 1
        else:
            raise ValueError(f"Task '{self.config.TASK}' is not supported.")

        self.model = self.create_model()
        self.train_dataloader, self.val_dataloader = self.get_dataloader()

        max_features = MODEL_MAX_FEATURES[self.config.MODEL.BACKBONE]
        num_features = MODEL_NUM_FEATURES[self.config.MODEL.BACKBONE]
        self.index = torch.tensor(random.sample(range(0, max_features), num_features))

        self.train_features = OrderedDict((layer, []) for layer in self.config.MODEL.RETURN_NODES)
        self.eval_features = OrderedDict((layer, []) for layer in self.config.MODEL.RETURN_NODES)

        self.evaler = Evaler(config)

        # Create a folder to save the model
        save_weights_dir = os.path.join("results", "train", config.EXP_NAME)
        self.save_weights_path = os.path.join(save_weights_dir, "model.pkl")
        os.makedirs(save_weights_dir, exist_ok=True)
        # Create a folder to save the visual results
        self.save_visual_dir = os.path.join("results", "train", config.EXP_NAME, "visual")
        os.makedirs(self.save_visual_dir, exist_ok=True)

    def create_model(self) -> nn.Module:
        model = PaDiM(self.config.MODEL.BACKBONE, OmegaConf.to_container(self.config.MODEL.RETURN_NODES))
        model = model.to(self.device)
        return model

    def get_dataloader(self) -> [torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        if self.task == 0:
            train_dataset = FolderDataset(
                self.config.DATASETS.ROOT.TRAIN,
                self.config.DATASETS.TRANSFORMS.RESIZE,
            )
            val_dataset = FolderDataset(
                self.config.DATASETS.ROOT.TEST,
                self.config.DATASETS.TRANSFORMS.RESIZE,
            )
        else:
            train_dataset = MVTecDataset(
                self.config.DATASETS.ROOT,
                self.config.DATASETS.CATEGORY,
                self.config.DATASETS.TRANSFORMS.RESIZE,
                is_train=True,
            )
            val_dataset = MVTecDataset(
                self.config.DATASETS.ROOT,
                self.config.DATASETS.CATEGORY,
                self.config.DATASETS.TRANSFORMS.RESIZE,
                is_train=False,
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
            shuffle=True,
            num_workers=4,
            sampler=None,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        return train_dataloader, val_dataloader

    def get_features(self) -> Any:
        features = OrderedDict((layer, []) for layer in self.config.MODEL.RETURN_NODES)
        for (images, _, _, _) in tqdm(self.train_dataloader, f"train model"):
            if self.device.type == "cuda" and torch.cuda.is_available():
                images = images.to(self.device, non_blocking=True)
            feature = self.model(images)
            # get intermediate layer outputs
            for k, v in feature.items():
                features[k].append(v)

        return features

    def save_checkpoint(self, state_dict: Any) -> None:
        with open(self.save_weights_path, "wb") as f:
            pickle.dump(state_dict, f)

    def train(self) -> None:
        train_features = self.get_features()
        self.index = self.index.to(self.device)
        embedding = generate_embedding(train_features, self.config.MODEL.RETURN_NODES, self.index)
        mean, inv_covariance = cal_multivariate_gaussian_distribution(embedding)
        train_features = [mean, inv_covariance]

        self.save_checkpoint(train_features)

        if self.task == 0:
            category = ""
        else:
            category = self.config.DATASETS.CATEGORY

        self.evaler.run_validation(
            self.model,
            category,
            self.val_dataloader,
            OmegaConf.to_container(self.config.MODEL.RETURN_NODES),
            self.index,
            train_features,
            self.config.DATASETS.TRANSFORMS.RESIZE,
            self.task,
            self.device,
            self.save_visual_dir,
        )
