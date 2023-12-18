# PaDiM

## Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [Local Install](#local-install)
- [MVTec AD](#mvtec-ad)
    - [Results](#results)
        - [Image-Level AUC](#image-level-auc)
        - [Pixel-Level AUC](#pixel-level-auc)
        - [Image F1 Score](#image-f1-score)
    - [Train (e.g bottle)](#train-eg-bottle)
    - [Test (e.g bottle)](#test-eg-bottle)
- [Folder](#folder)
- [Train](#train)
- [Test](#test)
- [Contributing](#contributing)
- [Credit](#credit)
    - [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](#padim-a-patch-distribution-modeling-framework-for-anomaly-detection-and-localization)

## Introduction

PyTorch unofficial implements `PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization` paper.

## Getting Started

### Requirements

- Python 3.10+
- PyTorch 2.0.0+
- CUDA 11.8+
- Ubuntu 22.04+

### Local Install

```bash
git clone https://github.com/Lornatang/PaDiM.git
cd PaDiM
pip install -r requirements.txt
pip install -e .
```

## MVTec AD

### Results

#### Image-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
|----------------|:-----:|:------:|:-----:|:-------:|:-----:|:-----:|:------:|:-----:|:-------:|:--------:|:---------:|:-----:|:-----:|:----------:|:----------:|:------:|
| ResNet-18      | 0.891 | 0.945  | 0.857 |  0.982  | 0.950 | 0.976 | 0.994  | 0.844 |  0.901  |  0.750   |   0.961   | 0.863 | 0.759 |   0.889    |   0.920    | 0.780  |
| Wide ResNet-50 | 0.950 | 0.995  | 0.942 |   1.0   | 0.974 | 0.993 | 0.999  | 0.878 |  0.927  |  0.964   |   0.989   | 0.939 | 0.845 |   0.942    |   0.976    | 0.882  |

#### Pixel-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
|----------------|:-----:|:------:|:-----:|:-------:|:-----:|:-----:|:------:|:-----:|:-------:|:--------:|:---------:|:-----:|:-----:|:----------:|:----------:|:------:|
| ResNet-18      | 0.968 | 0.984  | 0.918 |  0.994  | 0.934 | 0.947 | 0.983  | 0.965 |  0.984  |  0.978   |   0.970   | 0.957 | 0.978 |   0.988    |   0.968    | 0.979  |
| Wide ResNet-50 | 0.979 | 0.991  | 0.970 |  0.993  | 0.955 | 0.957 | 0.985  | 0.970 |  0.988  |  0.985   |   0.982   | 0.966 | 0.988 |   0.991    |   0.976    | 0.986  |

#### Image F1 Score

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
|----------------|:-----:|:------:|:-----:|:-------:|:-----:|:-----:|:------:|:-----:|:-------:|:--------:|:---------:|:-----:|:-----:|:----------:|:----------:|:------:|
| ResNet-18      | 0.916 | 0.930  | 0.893 |  0.984  | 0.934 | 0.952 | 0.976  | 0.858 |  0.960  |  0.836   |   0.974   | 0.932 | 0.879 |   0.923    |   0.796    | 0.915  |
| Wide ResNet-50 | 0.951 | 0.989  | 0.930 |   1.0   | 0.960 | 0.983 | 0.992  | 0.856 |  0.982  |  0.937   |   0.978   | 0.946 | 0.895 |   0.952    |   0.914    | 0.947  |

### Train (e.g bottle)

Prepare the dataset, dataset structure see [here](data/README.md#mvtec_anomaly_detection-eg-bottle)

```shell
python tools/train.py ./configs/mvtec.yaml
```

### Test (e.g bottle)

```shell
python tools/eval.py ./configs/mvtec.yaml
```

<div align="center">
<img src="figure/0.png" width="1200" height="300">
</div>

more visualization results see `results/eval/mvtec_bottle/visual`

## Folder

### Train

Prepare the dataset, dataset structure see [here](data/README.md#folder)

```shell
python tools/train.py ./configs/folder.yaml
```

### Test

```shell
python tools/eval.py ./configs/folder.yaml
```

<div align="center">
<img src="figure/1.png" width="400" height="300">
</div>

more visualization results see `results/eval/folder/visual`

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization

_Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier_ <br>

**Abstract** <br>
We present a new framework for Patch Distribution Modeling, PaDiM, to concurrently detect and localize anomalies in images in a one-class learning
setting. PaDiM makes use of a pretrained convolutional neural network (CNN) for patch embedding, and of multivariate Gaussian distributions to get a
probabilistic representation of the normal class. It also exploits correlations between the different semantic levels of CNN to better localize
anomalies. PaDiM outperforms current state-of-the-art approaches for both anomaly detection and localization on the MVTec AD and STC datasets. To
match real-world visual industrial inspection, we extend the evaluation protocol to assess performance of anomaly localization algorithms on
non-aligned dataset. The state-of-the-art performance and low complexity of PaDiM make it a good candidate for many industrial applications.

[[Paper]](https://arxiv.org/pdf/2011.08785.pdf)

```bibtex
@article{DBLP:journals/corr/abs-2011-08785,
  author       = {Thomas Defard and
                  Aleksandr Setkov and
                  Angelique Loesch and
                  Romaric Audigier},
  title        = {PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection
                  and Localization},
  journal      = {CoRR},
  volume       = {abs/2011.08785},
  year         = {2020},
  url          = {https://arxiv.org/abs/2011.08785},
  eprinttype    = {arXiv},
  eprint       = {2011.08785},
  timestamp    = {Wed, 18 Nov 2020 16:48:35 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2011-08785.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```