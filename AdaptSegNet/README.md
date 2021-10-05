# Semantic segmentation with AdaptSeg

## Environment
Python 3.6.9, Pytorch 1.6.0, Torch Vision 0.7.0.
To track the training details, we also used [neptune](https://docs.neptune.ai/getting-started/installation), but this is optional configuration.

## Preparation
For dataset preparation, please follow [adaptseg.md](adaptseg.md) to setup.

## Train

```
sh train.sh
```
This script runs training on different hyper-parameters.

## Acknowledgement
The large proportion of this directory is borrowed from [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet).








