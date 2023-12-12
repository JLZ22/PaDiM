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
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch import nn
from torch.cuda import amp
from tqdm import tqdm

from padim.datasets import MVTecDataset
from padim.models import PaDiM, MODEL_NUM_FEATURES, MODEL_MAX_FEATURES
from padim.utils import select_device, cal_multivariate_gaussian_distribution, embedding_concat

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: DictConfig) -> None:
        self.config = config

        self.device = select_device(config["DEVICE"])
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            self.scaler = amp.GradScaler(enabled=True)
        else:
            self.scaler = amp.GradScaler(enabled=False)

        self.model = self.create_model()
        self.train_dataloader = self.get_dataloader()

        max_features = MODEL_MAX_FEATURES[self.config.MODEL.BACKBONE]
        num_features = MODEL_NUM_FEATURES[self.config.MODEL.BACKBONE]
        self.idx = torch.tensor(random.sample(range(0, max_features), num_features))

        # Create a folder to save the model
        save_weights_dir = os.path.join("results", "train", config.EXP_NAME)
        self.save_weights_path = os.path.join(save_weights_dir, f"{self.config.DATASETS.CATEGORY}.pkl")
        os.makedirs(save_weights_dir, exist_ok=True)

        # Create a folder to save the visual results
        self.save_visual_dir = os.path.join("results", "train", config.EXP_NAME, "visual")
        os.makedirs(self.save_visual_dir, exist_ok=True)

    def create_model(self) -> nn.Module:
        model = PaDiM(self.config.MODEL.BACKBONE, OmegaConf.to_container(self.config.MODEL.RETURN_NODES))
        model = model.to(self.device)
        return model

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        train_dataset = MVTecDataset(
            self.config.DATASETS.ROOT,
            self.config.DATASETS.CATEGORY,
            self.config.DATASETS.TRANSFORMS,
            is_train=True,
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
        return train_dataloader

    def save_checkpoint(self, state_dict: Any) -> None:
        with open(self.save_weights_path, "wb") as f:
            pickle.dump(state_dict, f)

    def train(self) -> None:
        train_features_output = OrderedDict((layer, []) for layer in self.config.MODEL.RETURN_NODES)
        for (images, _, _) in tqdm(self.train_dataloader, f"train `{self.config.DATASETS.CATEGORY}`"):
            if self.device.type == "cuda" and torch.cuda.is_available():
                images = images.to(self.device, non_blocking=True)
            features = self.model(images)
            # get intermediate layer outputs
            for k, v in features.items():
                train_features_output[k].append(v)

        # Concatenate the features
        for k, v in train_features_output.items():
            train_features_output[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = train_features_output[self.config.MODEL.RETURN_NODES[0]]
        for layer_name in self.config.MODEL.RETURN_NODES[1:]:
            embedding_vectors = embedding_concat(embedding_vectors, train_features_output[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
        # calculate multivariate Gaussian distribution
        mean, inv_covariance = cal_multivariate_gaussian_distribution(embedding_vectors)

        self.save_checkpoint([mean, inv_covariance])
