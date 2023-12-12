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
import os
import pickle
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from torch import nn
from torch.nn import functional as F_torch
from tqdm import tqdm

from padim.datasets import MVTecDataset
from padim.models import PaDiM, MODEL_NUM_FEATURES, MODEL_MAX_FEATURES
from padim.utils import plot_fig
from padim.utils import select_device, embedding_concat


class Evaler:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device = select_device(config["DEVICE"])

        self.model = self.create_model()
        self.val_dataloader = self.get_dataloader()
        self.features = self._get_features(config.VAL.WEIGHTS_PATH)

        max_features = MODEL_MAX_FEATURES[self.config.MODEL.BACKBONE]
        num_features = MODEL_NUM_FEATURES[self.config.MODEL.BACKBONE]
        self.idx = torch.tensor(random.sample(range(0, max_features), num_features))

        # Create a folder to save the visual results
        self.save_visual_dir = os.path.join("results", "train", config.EXP_NAME, "visual")
        os.makedirs(self.save_visual_dir, exist_ok=True)

    def create_model(self) -> nn.Module:
        model = PaDiM(self.config.MODEL.BACKBONE, OmegaConf.to_container(self.config.MODEL.RETURN_NODES))
        model = model.to(self.device)
        return model

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        val_dataset = MVTecDataset(
            self.config.DATASETS.ROOT,
            self.config.DATASETS.CATEGORY,
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

    @staticmethod
    def _get_features(path: str | Path) -> Any:
        if not path.endswith(".pkl"):
            raise ValueError(f"Invalid weights path: {path}")

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        return checkpoint

    def validation(self) -> None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig_img_rocauc = ax[0]
        fig_pixel_rocauc = ax[1]

        total_roc_auc = []
        total_pixel_roc_auc = []

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        eval_features_output = OrderedDict((layer, []) for layer in self.config.MODEL.RETURN_NODES)
        for (x, y, mask) in tqdm(self.val_dataloader, '| feature extraction | test | %s |' % self.config.DATASETS.CATEGORY):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            if self.device.type == "cuda" and torch.cuda.is_available():
                x = x.to(self.device, non_blocking=True)
            features = self.model(x)
            # get intermediate layer outputs
            for k, v in features.items():
                eval_features_output[k].append(v)

        # Concatenate the features
        for k, v in eval_features_output.items():
            eval_features_output[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = eval_features_output[self.config.MODEL.RETURN_NODES[0]]
        for layer_name in self.config.MODEL.RETURN_NODES[1:]:
            embedding_vectors = embedding_concat(embedding_vectors, eval_features_output[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = self.features[0][:, i]
            conv_inv = np.linalg.inv(self.features[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F_torch.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                        align_corners=False).squeeze().numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        # calculate image-level ROC AUC score
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
        plot_fig(test_imgs, scores, gt_mask_list, threshold, self.save_visual_dir, self.config.DATASETS.CATEGORY)
