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

from .module import FeatureExtractor

MODEL_NUM_FEATURES = {
    "resnet18": 100,
    "wide_resnet50_2": 550,
}

MODEL_MAX_FEATURES = {
    "resnet18": 448,
    "wide_resnet50_2": 1792,
}


class PaDiM(nn.Module):
    def __init__(
            self,
            backbone: str,
            return_nodes: list[str],
            pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone, return_nodes, pretrained)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.feature_extractor(x)
            return features
