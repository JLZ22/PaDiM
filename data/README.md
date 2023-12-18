# Dataset structure

## mvtec_anomaly_detection (e.g bottle)

```text

├── mvtec_anomaly_detection
    ├── bottle
        ├── ground_truth
            ├── broken_large
                ├── 000_mask.png
                ├── 001_mask.png
                ├── ...
            ├── broken_small
                ├── 000_mask.png
                ├── 001_mask.png
                ├── ...
            ├── contamination
                ├── 000_mask.png
                ├── 001_mask.png
                ├── ...
        ├── test
            ├── broken_large
                ├── 000.png
                ├── 001.png
                ├── ...
            ├── broken_small
                ├── 000.png
                ├── 001.png
                ├── ...
            ├── contamination
                ├── 000.png
                ├── 001.png
                ├── ...
            ├── good
                ├── 000.png
                ├── 001.png
                ├── ...
        ├── train
            ├── good
                ├── 000.png
                ├── 001.png
                ├── ...
```

## folder

```text
├── folder
    ├── train
        ├── good
            ├── 000.png
            ├── 001.png
            ├── ...
    ├── val
        ├── good
            ├── 000.png
            ├── 001.png
            ├── ...
        ├── bad
            ├── 000.png
            ├── 001.png
            ├── ...
```