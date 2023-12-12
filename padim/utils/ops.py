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
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from torch import Tensor
from torch.nn import functional as F_torch

__all__ = [
    "calculate_distance_matrix", "cal_multivariate_gaussian_distribution", "generate_embedding", "get_anomaly_map", "de_normalization",
    "embedding_concat",
]


def calculate_distance_matrix(embedding: Tensor, stats: list[Tensor]) -> np.ndarray:
    r"""Calculate the distance matrix of the input tensor.

    Args:
        embedding (Tensor): The input tensor of shape (batch_size, channels, height, width).
        stats (list[Tensor]): The statistics of the training data.

    Examples:
        >>> import torch
        >>> from padim.utils import calculate_distance_matrix
        >>> embedding = torch.rand((32, 3, 256, 256))
        >>> stats = [torch.rand((32, 3, 256, 256)), torch.rand((32, 3, 3, 256, 256))]
        >>> out = calculate_distance_matrix(embedding, stats)
        >>> out.shape
            torch.Size([32, 196608, 196608])

    Returns:
        distances (np.ndarray): The distance matrix of the input tensor.
    """
    batch_size, channels, height, width = embedding.size()
    embedding_vectors = embedding.cpu().view(batch_size, channels, height * width).numpy()
    distances = []
    for i in range(height * width):
        mean = stats[0][:, i]
        conv_inv = np.linalg.inv(stats[1][:, :, i])
        distance = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        distances.append(distance)

    distances = np.array(distances).transpose(1, 0).reshape(batch_size, height, width)
    return distances


def cal_multivariate_gaussian_distribution(x: Tensor) -> [np.ndarray, np.ndarray]:
    r"""Calculate the mean and inverse covariance of a multivariate Gaussian distribution.

    Args:
        x (Tensor): The input tensor of shape (batch_size, channels, height, width).

    Examples:
        >>> import torch
        >>> from padim.utils import cal_multivariate_gaussian_distribution
        >>> x = torch.rand((32, 3, 256, 256))
        >>> mean, inv_covariance = cal_multivariate_gaussian_distribution(x)
        >>> mean.shape
            (3, 196608)
        >>> inv_covariance.shape
            (32, 3, 196608)

    Returns:
        mean (np.ndarray): The mean of the multivariate Gaussian distribution.
        cov_inv (np.ndarray): The inverse covariance of the multivariate Gaussian distribution.
    """
    batch_size, channels, height, width = x.size()
    embedding_vectors = x.view(batch_size, channels, height * width).cpu()
    mean = torch.mean(embedding_vectors, dim=0).numpy()
    inv_covariance = torch.zeros(channels, channels, height * width).numpy()
    I = np.identity(channels)
    for i in range(height * width):
        inv_covariance[:, :, i] = np.cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * I

    return mean, inv_covariance


def generate_embedding(features: Tensor | Any, return_nodes: list, index: Tensor) -> Any:
    # Concatenate the features
    for k, v in features.items():
        features[k] = torch.cat(v, 0)

    # Embedding concat
    embedding = features[return_nodes[0]]
    for layer_name in return_nodes[1:]:
        embedding = embedding_concat(embedding, features[layer_name])

    # randomly select d dimension
    embedding = torch.index_select(embedding, 1, index)

    return embedding


def get_anomaly_map(distances: np.ndarray, image_size: int) -> np.ndarray:
    # up-sample
    distances = torch.tensor(distances)
    anomaly_map = F_torch.interpolate(distances.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze().numpy()

    # apply gaussian smoothing on the score map
    for i in range(anomaly_map.shape[0]):
        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

    return anomaly_map


def de_normalization(
        x: np.ndarray,
        mean: tuple[float] = None,
        std: tuple[float] = None,
) -> np.ndarray:
    r"""De-normalize the input tensor.

    Args:
        x (np.ndarray): The input tensor of shape (batch_size, channels, height, width).
        mean (tuple[float]): The mean of the data.
        std (tuple[float]): The standard deviation of the data.

    Examples:
        >>> import torch
        >>> from padim.utils import de_normalization
        >>> x = np.array([0.485, 0.456, 0.406])
        >>> out = de_normalization(x)
        >>> out
            array([123.675, 116.28 , 103.53 ], dtype=float32)

    Returns:
        x (np.ndarray): The de-normalized tensor.
    """
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])

    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def embedding_concat(x: Tensor, y: Tensor) -> Tensor:
    r"""Concatenate the embedding vectors of two images.

    Args:
        x (Tensor): The input tensor of shape (batch_size, channels, height, width).
        y (Tensor): The input tensor of shape (batch_size, channels, height, width).

    Examples:
        >>> import torch
        >>> from padim.utils import embedding_concat
        >>> x = torch.rand((32, 3, 256, 256))
        >>> y = torch.rand((32, 3, 256, 256))
        >>> out = embedding_concat(x, y)
        >>> out.shape
            torch.Size([32, 6, 256, 256])

    Returns:
        out (Tensor): The concatenated embedding vectors.
    """
    batch_size, channels_x, height_x, width_x = x.size()
    _, channels_y, height_y, width_y = y.size()

    scale_ratio = int(height_x / height_y)
    x = F_torch.unfold(x, kernel_size=scale_ratio, dilation=1, stride=scale_ratio)
    x = x.view(batch_size, channels_x, -1, height_y, width_y)
    out = torch.zeros(batch_size, channels_x + channels_y, x.size(2), height_y, width_y, device=x.device)

    # Concatenate x and y
    for i in range(x.size(2)):
        out[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    out = out.view(batch_size, -1, height_y * width_y)
    out = F_torch.fold(out, kernel_size=scale_ratio, output_size=(height_x, width_x), stride=scale_ratio)

    return out
