# Image classification with CDAN and MCC

## Environment
Python 3.6.9, Pytorch 1.6.0, Torch Vision 0.7.0.
To track the training details, we also used [neptune](https://docs.neptune.ai/getting-started/installation), but this is optional configuration.
You also need sklearn (0.23.2 is used).


## Preparation
For dataset preparation, please follow [nc_ps/README.md](https://github.com/VisionLearningGroup/SND/blob/main/nc_ps/README.md)

We are using the same datasets. Do not forget to put a dataset and txt file link in this directory (./data, ./txt).

## Train

Scripts are stored in scripts directory.
Please change the random seed when testing on different source validation samples.

MCC
```
sh scripts/train_a2d_mcc.sh
```
CDAN
```
sh scripts/train_a2d.sh
```

Note that only in the experiments of MCC on visda, we employ ResNet101 following their paper.

## Acknowledgement
The large proportion of this directory is borrowed from [CDAN](https://github.com/thuml/CDAN) and [MCC](https://github.com/thuml/Versatile-Domain-Adaptation).




