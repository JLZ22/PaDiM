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
from abc import ABC
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from torch import nn

from padim.datasets import FolderDataset, MVTecDataset
from padim.datasets.utils import CPUPrefetcher, CUDAPrefetcher
from padim.utils import plot_score_map, select_device, plot_fig
from .base import Base

logger = logging.getLogger(__name__)


class Evaler(Base, ABC):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    @staticmethod
    def create_model(checkpoint: Any, device: torch.device) -> nn.Module:
        """Create a model from checkpoint."""
        logger.info(f"load model from checkpoint")
        model = checkpoint["model"]
        model.eval()
        model = model.to(device)
        return model

    @staticmethod
    def create_transform(checkpoint: Any) -> tuple[A.Compose, A.Compose]:
        """Get image and mask transforms."""
        return checkpoint["image_transforms"], checkpoint["mask_transforms"]

    @staticmethod
    def get_dataloader(
            root: str | Path,
            category: str,
            image_transforms: A.Compose,
            mask_transforms: A.Compose,
            mask_size: tuple[int, int],
            cls_task: bool,
            device: torch.device = torch.device("cpu"),
    ) -> CPUPrefetcher | CUDAPrefetcher:
        if cls_task:
            logger.info("Load classification dataset.")
            datasets = FolderDataset(root, image_transforms, mask_size, False)
        else:
            logger.info("Load segmentation dataset.")
            datasets = MVTecDataset(root, category, image_transforms, mask_transforms, mask_size, train=False)

        dataloader = torch.utils.data.DataLoader(
            datasets,
            batch_size=len(datasets),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        if device == "cuda":
            dataloader = CUDAPrefetcher(dataloader, device)
        else:
            dataloader = CPUPrefetcher(dataloader)
        return dataloader

    @staticmethod
    def run_validation(
            model: nn.Module,
            val_loader: CPUPrefetcher | CUDAPrefetcher,
            cls_task: bool,
            device: torch.device = torch.device("cpu"),
            save_visuals_dir: str | Path = "results/eval/visual",
    ) -> None:
        model.eval()

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig_image_roc_auc = ax[0]
        fig_pixel_roc_auc = ax[1]
        image_data_list = []
        target_data_list = []
        mask_data_list = []
        image_path_list = []

        # Load data
        batch_data = next(iter(val_loader))
        image = batch_data["image"].to(device, non_blocking=True)
        target = batch_data["target"].to(device, non_blocking=True)
        mask = batch_data["mask"].to(device, non_blocking=True)
        image_path = batch_data["image_path"]

        # get all images anomaly map
        anomaly_map = model(image)

        image_data_list.extend(image.cpu().detach().numpy())
        target_data_list.extend(target.cpu().detach().numpy())
        mask_data_list.extend(mask.cpu().detach().numpy())
        image_path_list.extend(image_path)

        # Normalization
        anomaly_map = anomaly_map.detach().cpu().numpy()
        max_score = anomaly_map.max()
        min_score = anomaly_map.min()
        scores = (anomaly_map - min_score) / (max_score - min_score)

        if cls_task:
            num_images = len(scores)
            for i in range(num_images):
                save_visuals_path = Path(save_visuals_dir) / os.path.basename(image_path_list[i])
                plot_score_map(image_data_list[i], scores[i], 0, 255, save_visuals_path)
        else:
            # calculate image-level ROC AUC score
            image_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
            gt_list = np.asarray(target_data_list)
            fpr, tpr, _ = roc_curve(gt_list, image_scores)
            image_roc_auc = roc_auc_score(gt_list, image_scores)
            print(f"image ROCAUC: {image_roc_auc:.3f}")
            fig_image_roc_auc.plot(fpr, tpr, label=f"image_ROCAUC: {image_roc_auc:.3f}")

            # get optimal threshold
            gt_mask = np.asarray(mask_data_list)
            precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
            a = 2 * precision * recall
            b = precision + recall
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            threshold = thresholds[np.argmax(f1)]

            # calculate per-pixel level ROCAUC
            fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
            per_pixel_roc_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())
            print(f"pixel ROCAUC: {per_pixel_roc_auc:.3f}")

            fig_pixel_roc_auc.plot(fpr, tpr, label=f"pixel_ROCAUC: {per_pixel_roc_auc:.3f}")
            plot_fig(image_data_list, scores, mask_data_list, threshold, save_visuals_dir)

            fig.tight_layout()
            save_fig_path = Path(save_visuals_dir) / "roc_curve.png"
            fig.savefig(save_fig_path, dpi=100)

    def validation(self) -> None:
        device = select_device(self.config["DEVICE"])

        cls_task: bool = False
        if self.config.TASK == "classification":
            cls_task = True
            category = ""
        else:
            category = self.config.DATASETS.CATEGORY

        # Create a folder to save the visual results
        save_visual_dir = Path("results") / "eval" / self.config.EXP_NAME / "visual"
        os.makedirs(save_visual_dir, exist_ok=True)

        checkpoint = torch.load(self.config.VAL.WEIGHTS_PATH, map_location=device)
        model = self.create_model(checkpoint, device)
        image_transforms, mask_transforms = self.create_transform(checkpoint)
        mask_size = checkpoint["mask_size"]
        val_loader = self.get_dataloader(
            self.config.DATASETS.ROOT.get("VAL'") if cls_task else self.config.DATASETS.ROOT,
            category,
            image_transforms,
            mask_transforms,
            mask_size,
            cls_task,
            device)

        self.run_validation(
            model,
            val_loader,
            cls_task,
            device,
            save_visual_dir,
        )
