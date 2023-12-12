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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from omegaconf import DictConfig
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F_torch
from tqdm import tqdm

from padim.datasets import MVTecDataset
from padim.models import PaDiM, MODEL_NUM_FEATURES, MODEL_MAX_FEATURES
from padim.utils import select_device, cal_multivariate_gaussian_distribution, embedding_concat, plot_fig, calculate_distance_matrix

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
        self.train_dataloader, self.val_dataloader = self.get_dataloader()

        max_features = MODEL_MAX_FEATURES[self.config.MODEL.BACKBONE]
        num_features = MODEL_NUM_FEATURES[self.config.MODEL.BACKBONE]
        self.idx = torch.tensor(random.sample(range(0, max_features), num_features))
        self.train_features_output = OrderedDict((layer, []) for layer in self.config.MODEL.RETURN_NODES)
        self.eval_features_output = OrderedDict((layer, []) for layer in self.config.MODEL.RETURN_NODES)

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

    def get_dataloader(self) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = MVTecDataset(
            self.config.DATASETS.ROOT,
            self.config.DATASETS.CATEGORY,
            self.config.DATASETS.TRANSFORMS,
            is_train=True,
        )
        val_dataset = MVTecDataset(
            self.config.DATASETS.ROOT,
            self.config.DATASETS.CATEGORY,
            self.config.DATASETS.TRANSFORMS,
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
            shuffle=False,
            num_workers=4,
            sampler=None,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        return train_dataloader, val_dataloader

    def load_checkpoint(self, path: str | Path) -> None:
        if not path.endswith(".pkl"):
            raise ValueError(f"Invalid weights path: {path}")

        with open(path, "rb") as f:
            self.train_features_output = pickle.load(f)

    def save_checkpoint(self) -> None:
        with open(self.save_weights_path, "wb") as f:
            pickle.dump(self.train_features_output, f)

    def train(self) -> None:
        weights_path = self.config.TRAIN.WEIGHTS_PATH
        if os.path.exists(weights_path):
            self.load_checkpoint(weights_path)

            test_images = []
            gt_list = []
            gt_mask_list = []

            # extract test set features
            for (images, targets, masks) in tqdm(self.val_dataloader, f"eval `{self.config.DATASETS.CATEGORY}`"):
                images = images["image"]
                masks = masks["image"]

                test_images.extend(images.cpu().detach().numpy())
                gt_list.extend(targets.cpu().detach().numpy())
                gt_mask_list.extend(masks.cpu().detach().numpy())
                # model prediction
                if self.device.type == "cuda" and torch.cuda.is_available():
                    images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                # get intermediate layer outputs
                for k, v in features.items():
                    self.eval_features_output[k].append(v)

            # Concatenate the features
            for k, v in self.eval_features_output.items():
                self.eval_features_output[k] = torch.cat(v, 0)

            # Embedding concat
            embedding_vectors = self.eval_features_output[self.config.MODEL.RETURN_NODES[0]]
            for layer_name in self.config.MODEL.RETURN_NODES[1:]:
                embedding_vectors = embedding_concat(embedding_vectors, self.eval_features_output[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

            distance = calculate_distance_matrix(embedding_vectors, self.train_features_output)

            # upsample
            dist_list = torch.tensor(distance)
            score_map = F_torch.interpolate(dist_list.unsqueeze(1), size=images.size(2), mode='bilinear',
                                            align_corners=False).squeeze().numpy()

            # apply gaussian smoothing on the score map
            for i in range(score_map.shape[0]):
                score_map[i] = gaussian_filter(score_map[i], sigma=4)

            # Normalization
            max_score = score_map.max()
            min_score = score_map.min()
            scores = (score_map - min_score) / (max_score - min_score)

            # calculate image-level ROC AUC score
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            fig_img_rocauc = ax[0]
            fig_pixel_rocauc = ax[1]

            total_roc_auc = []
            total_pixel_roc_auc = []
            img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
            gt_list = np.asarray(gt_list)
            fpr, tpr, _ = roc_curve(gt_list, img_scores)
            img_roc_auc = roc_auc_score(gt_list, img_scores)
            total_roc_auc.append(img_roc_auc)
            print('image ROCAUC: %.3f' % (img_roc_auc))
            fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (self.config.DATASETS.CATEGORY, img_roc_auc))

            # get optimal threshold
            gt_mask = np.asarray(gt_mask_list)
            precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
            a = 2 * precision * recall
            b = precision + recall
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            threshold = thresholds[np.argmax(f1)]

            # calculate per-pixel level ROCAUC
            fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
            per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
            total_pixel_roc_auc.append(per_pixel_rocauc)
            print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

            fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (self.config.DATASETS.CATEGORY, per_pixel_rocauc))
            plot_fig(test_images, scores, gt_mask_list, threshold, self.save_visual_dir, self.config.DATASETS.CATEGORY)

            fig.tight_layout()
            fig.savefig(os.path.join(self.save_visual_dir, 'roc_curve.png'), dpi=100)
        else:
            self.train_features_output = OrderedDict((layer, []) for layer in self.config.MODEL.RETURN_NODES)
            for (images, _, _) in tqdm(self.train_dataloader, f"train `{self.config.DATASETS.CATEGORY}`"):
                images = images["image"]
                if self.device.type == "cuda" and torch.cuda.is_available():
                    images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                # get intermediate layer outputs
                for k, v in features.items():
                    self.train_features_output[k].append(v)

            # Concatenate the features
            for k, v in self.train_features_output.items():
                self.train_features_output[k] = torch.cat(v, 0)

            # Embedding concat
            embedding_vectors = self.train_features_output[self.config.MODEL.RETURN_NODES[0]]
            for layer_name in self.config.MODEL.RETURN_NODES[1:]:
                embedding_vectors = embedding_concat(embedding_vectors, self.train_features_output[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
            # calculate multivariate Gaussian distribution
            mean, inv_covariance = cal_multivariate_gaussian_distribution(embedding_vectors)
            self.train_features_output = [mean, inv_covariance]

            self.save_checkpoint()
