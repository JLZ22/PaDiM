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
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F_torch

__all__ = [
    "cal_multivariate_gaussian_distribution", "de_normalization", "embedding_concat",
]


def cal_multivariate_gaussian_distribution(x: Tensor) -> [np.ndarray, np.ndarray]:
    r"""Calculate the mean and inverse covariance of a multivariate Gaussian distribution.

    Args:
        x (Tensor): The input tensor of shape (batch_size, channels, height, width).

    Examples:
        >>> import torch
        >>> from padim.utils import cal_multivariate_gaussian_distribution
        >>> x = torch.rand((32, 3, 256, 256))
        >>> mean, cov_inv = cal_multivariate_gaussian_distribution(x)
        >>> mean.shape
            (3, 196608)
        >>> cov_inv.shape
            (32, 3, 196608)

    Returns:
        mean (np.ndarray): The mean of the multivariate Gaussian distribution.
        cov_inv (np.ndarray): The inverse covariance of the multivariate Gaussian distribution.
    """
    batch_size, channels, height, width = x.size()
    device = x.device
    embedding_vectors = x.view(batch_size, channels, height * width)
    mean = torch.mean(embedding_vectors, dim=0).numpy()
    _cov = torch.zeros(channels, channels).numpy()
    cov_inv = torch.zeros(batch_size, channels, height * width).numpy()
    identity_channels = np.identity(channels)
    for i in range(height * width):
        _cov = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * identity_channels
        _cov = torch.Tensor(_cov).to(device)
        cov_inv[:, :, i] = torch.linalg.inv(_cov).cpu().numpy()

    return mean, cov_inv


def de_normalization(x: Tensor) -> np.ndarray:
    r"""De-normalize the input tensor.

    Args:
        x (Tensor): The input tensor of shape (batch_size, channels, height, width).

    Examples:
        >>> import torch
        >>> from padim.utils import de_normalization
        >>> x = torch.rand((32, 3, 256, 256))
        >>> out = de_normalization(x)
        >>> out.shape
            torch.Size([32, 3, 256, 256])

    Returns:
        x (np.ndarray): The de-normalized tensor.
    """
    mean = np.array([0.485, 0.456, 0.406])
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
    out = torch.zeros(batch_size, channels_x + channels_y, x.size(2), height_y, width_y)

    # Concatenate x and y
    for i in range(x.size(2)):
        out[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    out = out.view(batch_size, -1, height_y * width_y)
    out = F_torch.fold(out, kernel_size=scale_ratio, output_size=(height_x, width_x), stride=scale_ratio)

    return out
