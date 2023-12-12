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

import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from torch import nn, Tensor
from tqdm import tqdm

from padim.datasets import MVTecDataset
from padim.models import PaDiM, MODEL_NUM_FEATURES, MODEL_MAX_FEATURES
from padim.utils import plot_fig, calculate_distance_matrix, generate_embedding, get_abnormal_score, plot_score_map
from padim.utils import select_device

logger = logging.getLogger(__name__)


class Evaler:
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def create_model(self, device: torch.device) -> nn.Module:
        model = PaDiM(self.config.MODEL.BACKBONE, OmegaConf.to_container(self.config.MODEL.RETURN_NODES))
        model = model.to(device)
        return model

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        val_dataset = MVTecDataset(
            self.config.DATASETS.ROOT,
            self.config.DATASETS.CATEGORY,
            self.config.DATASETS.TRANSFORMS.RESIZE,
            self.config.DATASETS.TRANSFORMS.CENTER_CROP,
            is_train=False,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
            shuffle=True,
            num_workers=4,
            sampler=None,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        return val_dataloader

    def get_features(self) -> Any:
        weights_path = self.config.VAL.WEIGHTS_PATH
        if not weights_path.endswith(".pkl"):
            raise ValueError(f"Only support '.pkl' file, but got {weights_path}")

        with open(weights_path, "rb") as f:
            features = pickle.load(f)

        return features

    @staticmethod
    def run_validation(
            model: nn.Module,
            category: str,
            val_dataloader: torch.utils.data.DataLoader,
            return_nodes: list,
            index: Tensor,
            train_features: list,
            image_size: int,
            task: int,
            device: torch.device = torch.device("cpu"),
            save_visual_dir: str = "results/eval/visual",
    ) -> None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig_image_roc_auc = ax[0]
        fig_pixel_roc_auc = ax[1]
        total_roc_auc = []
        total_pixel_roc_auc = []
        test_images = []
        gt_list = []
        gt_mask_list = []
        test_image_names = []

        eval_features = OrderedDict((layer, []) for layer in return_nodes)
        # get intermediate layer outputs
        for (images, targets, masks, image_names) in tqdm(val_dataloader, f"eval"):
            test_images.extend(images.cpu().detach().numpy())
            gt_list.extend(targets.cpu().detach().numpy())
            gt_mask_list.extend(masks.cpu().detach().numpy())
            test_image_names.append(image_names)
            # model prediction
            if device.type == "cuda" and torch.cuda.is_available():
                images = images.to(device, non_blocking=True)
            feature = model(images)
            # get intermediate layer outputs
            for k, v in feature.items():
                eval_features[k].append(v)

        # Embedding concat
        index = index.to(device)
        embedding = generate_embedding(eval_features, return_nodes, index)

        # calculate distance matrix
        distances = calculate_distance_matrix(embedding, train_features)

        # up-sample
        scores = get_abnormal_score(distances, image_size)

        if task == 0:
            num_images = len(test_image_names[0])
            for i in range(num_images):
                save_file_name = os.path.join(save_visual_dir, f"{test_image_names[0][i]}.png")
                plot_score_map(test_images[i], scores[i], save_file_name)
        else:
            # calculate image-level ROC AUC score
            image_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
            gt_list = np.asarray(gt_list)
            fpr, tpr, _ = roc_curve(gt_list, image_scores)
            image_roc_auc = roc_auc_score(gt_list, image_scores)
            total_roc_auc.append(image_roc_auc)
            print(f"image ROCAUC: {image_roc_auc:.3f}")
            fig_image_roc_auc.plot(fpr, tpr, label=f"{category} image_ROCAUC: {image_roc_auc:.3f}")

            # get optimal threshold
            gt_mask = np.asarray(gt_mask_list)
            precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
            a = 2 * precision * recall
            b = precision + recall
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            threshold = thresholds[np.argmax(f1)]

            # calculate per-pixel level ROCAUC
            fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
            per_pixel_roc_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())
            total_pixel_roc_auc.append(per_pixel_roc_auc)
            print(f"pixel ROCAUC: {per_pixel_roc_auc:.3f}")

            fig_pixel_roc_auc.plot(fpr, tpr, label=f"{category} ROCAUC: {per_pixel_roc_auc:.3f}")
            plot_fig(test_images, scores, gt_mask_list, threshold, save_visual_dir, category)

            fig.tight_layout()
            fig.savefig(os.path.join(save_visual_dir, "roc_curve.png"), dpi=100)

    def validation(self) -> None:
        device = select_device(self.config["DEVICE"])

        if self.config.TASK == "classification":
            task = 0
        elif self.config.TASK == "segmentation":
            task = 1
        else:
            raise ValueError(f"Task '{self.config.TASK}' is not supported.")

        model = self.create_model(device)
        val_dataloader = self.get_dataloader()
        train_features = self.get_features()

        max_features = MODEL_MAX_FEATURES[self.config.MODEL.BACKBONE]
        num_features = MODEL_NUM_FEATURES[self.config.MODEL.BACKBONE]
        index = torch.tensor(random.sample(range(0, max_features), num_features))

        # Create a folder to save the visual results
        save_visual_dir = os.path.join("results", "eval", self.config.EXP_NAME, "visual")
        os.makedirs(save_visual_dir, exist_ok=True)

        if task == 0:
            category = ""
        else:
            category = self.config.DATASETS.CATEGORY

        self.run_validation(
            model,
            category,
            val_dataloader,
            OmegaConf.to_container(self.config.MODEL.RETURN_NODES),
            index,
            train_features,
            self.config.DATASETS.TRANSFORMS.CENTER_CROP,
            task,
            device,
            save_visual_dir,
        )
