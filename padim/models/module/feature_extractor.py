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
import torch
from torch import nn, Tensor
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "FeatureExtractor",
]

BACKBONE_WEIGHTS_DICT = {
    "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    "wide_resnet50_2": models.Wide_ResNet50_2_Weights.IMAGENET1K_V2,
}


class FeatureExtractor(nn.Module):
    def __init__(
            self,
            backbone: str,
            return_nodes: list[str],
            pre_trained: bool = True,
            requires_grad: bool = False,
    ) -> None:
        r"""Extract features from a CNN.

        Args:
            backbone (str): The backbone to which the feature extraction hooks are attached.
            return_nodes (list[str]): List of node names of the backbone to which the hooks are attached.
            pre_trained (bool): Whether to use a pre-trained backbone. Defaults to True.
            requires_grad (bool): Whether to require gradients for the backbone. Defaults to False.
                Models like ``stfpm`` use the feature extractor model as a trainable network. In such cases gradient
                computation is required.

        Examples:
            >>> import torch
            >>> from padim.models import FeatureExtractor

            >>> model = FeatureExtractor(backbone="resnet18", return_nodes=["layer1.1.relu_1", "layer2.1.relu_1", "layer3.1.relu_1"])
            >>> x = torch.rand((32, 3, 256, 256))
            >>> features = model(x)

            >>> [layer for layer in features.keys()]
                ['layer1.1.relu_1', 'layer2.1.relu_1', 'layer3.1.relu_1']
            >>> [feature.shape for feature in features.values()]
                [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.Size([32, 256, 16, 16])]
        """
        super().__init__()
        if backbone not in BACKBONE_WEIGHTS_DICT.keys():
            raise ValueError(f"Backbone {backbone} not supported. Supported backbones are {list(BACKBONE_WEIGHTS_DICT.keys())}")

        model = models.__dict__[backbone](weights=BACKBONE_WEIGHTS_DICT[backbone] if pre_trained else None)
        self.feature_extractor = create_feature_extractor(model, return_nodes)

        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = requires_grad

    def forward(self, x: Tensor) -> Tensor:
        return self.feature_extractor(x)
