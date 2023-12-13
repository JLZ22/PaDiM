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
import random

import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch

from padim.models.module import AnomalyMap, FeatureExtractor, MultiVariateGaussian


class PaDiM(nn.Module):
    r"""Pytorch implementation of PaDiM.

    Args:
        backbone (str): The backbone of the feature extractor.
        image_size (tuple[int, int]): The input image size.
        return_nodes (list[str]): The nodes to return from the feature extractor.
        pretrained (bool): Whether to use pretrained weights for the feature extractor.

    Raises:
        ValueError: If the backbone is not supported.

    Examples:
        >>> import torch
        >>> from padim.models import PaDiM
        >>> model = PaDiM("wide_resnet50_2", (224, 224), ["layer1.2.relu_2", "layer2.3.relu_2", "layer3.3.relu_2"])
        >>> x = torch.rand((32, 3, 224, 224))
        >>> out = model(x)
        >>> out.shape
            torch.Size([32, 550, 56, 56])

    """
    num_features_dict = {
        "resnet18": 100,
        "wide_resnet50_2": 550,
    }
    max_features_dict = {
        "resnet18": 448,
        "wide_resnet50_2": 1792,
    }

    def __init__(
            self,
            backbone: str,
            image_size: tuple[int, int],
            return_nodes: list[str],
            pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.return_nodes = return_nodes
        self.feature_extractor = FeatureExtractor(backbone, return_nodes, pretrained)
        self.anomaly_map = AnomalyMap(image_size)

        self.index: Tensor
        max_features = self.max_features_dict[backbone]
        num_features = self.num_features_dict[backbone]
        self.register_buffer("index", torch.tensor(random.sample(range(0, max_features), num_features)))

        self.multi_variate_gaussian = MultiVariateGaussian(num_features, max_features)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.feature_extractor(x)
            embeddings = self.generate_embedding(features)

        if self.training:
            return embeddings
        else:
            mean, inv_covariance = self.multi_variate_gaussian(embeddings)
            return self.anomaly_map(embeddings, mean, inv_covariance)

    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features (dict[str, Tensor]): Hierarchical feature map from a CNN (ResNet18 or WideResnet)

        Returns:
            Embedding vector
        """

        embeddings = features[self.return_nodes[0]]
        for layer in self.return_nodes[1:]:
            layer_embedding = features[layer]
            layer_embedding = F_torch.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        index = self.index.to(embeddings.device)
        embeddings = torch.index_select(embeddings, 1, index)
        return embeddings
