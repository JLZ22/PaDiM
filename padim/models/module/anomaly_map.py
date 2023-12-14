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
"""
Modified from 'https://github.com/openvinotoolkit/anomalib/blob/main/src/anomalib/models/padim/anomaly_map.py'
"""
import torch
from omegaconf import ListConfig
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision.transforms import GaussianBlur

__all__ = [
    "AnomalyMap",
]


class AnomalyMap(nn.Module):
    def __init__(self, image_size: ListConfig | tuple, sigma: float = 4.0):
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=(sigma, sigma))

    @staticmethod
    def compute_distance(embedding: Tensor, stats: list[Tensor]) -> Tensor:
        r"""Compute anomaly score to the patch in position(i,j) of a test image.

        Args:
            embedding (Tensor): Embedding Vector
            stats (list[Tensor]): Mean and Covariance Matrix of the multivariate Gaussian distribution

        Returns:
            Anomaly score of a test image via mahalanobis distance.
        """
        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        # calculate mahalanobis distances
        mean, inv_covariance = stats
        delta = (embedding - mean).permute(2, 0, 1)
        distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
        distances = distances.reshape(batch, 1, height, width)
        distances = distances.clamp(0).sqrt()

        return distances

    def forward(self, embedding: Tensor, mean: Tensor, inv_covariance: Tensor) -> Tensor:
        mean = mean.to(embedding.device)
        inv_covariance = inv_covariance.to(embedding.device)

        anomaly_map = self.compute_distance(embedding, [mean, inv_covariance])
        anomaly_map = F_torch.interpolate(
            anomaly_map,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        anomaly_map = self.blur(anomaly_map)

        return anomaly_map
