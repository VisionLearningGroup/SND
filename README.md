# [Tune it the Right Way: Unsupervised Validation of Domain Adaptationvia Soft Neighborhood Densit (ICCV 2021)](https://arxiv.org/pdf/2108.10860.pdf)

This repository provides code for the paper.
Please go to our project page to quickly understand the content of the paper or read our paper.
### [Project Page (Coming soon)]()  [Paper](https://arxiv.org/pdf/2108.10860.pdf)

## Introduction
This repository contains codes used for experiments of image classification, semantic segmentation, and toy datasets.
We split the codes into four directories, base, cdan, adaptseg, and advent since each method employs different structures.
[base](nc_ps/README.md) contains code for image classification with pseudo-labeling (PS) and neighborhood clustering (NC), and toy experiments.
[cdan](cdan/README.md) contains code for image classification with CDAN and MCC (borrowed from [CDAN](https://github.com/thuml/CDAN) and [MCC](https://github.com/thuml/Versatile-Domain-Adaptation)).
adaptseg is from [AdaptSeg](https://github.com/wasidennis/AdaptSegNet). advent is from [ADVENT](https://github.com/valeoai/ADVENT).


## Environment
The code in the repository should work with Python 3.6.9, Pytorch 1.6.0, Torch Vision 0.7.0, [Apex](https://github.com/NVIDIA/apex).
In some experiments, we used the nvidia apex library for memory efficient high-speed training.
To track the training details, we also used [neptune](https://docs.neptune.ai/getting-started/installation), but this is optional configuration.
You also need sklearn (0.23.2 is used), which is required in image classification.
Please follow the instructions on each directory for other requirements.

### Reference
This repository is contributed by [Kuniaki Saito](http://cs-people.bu.edu/keisaito/).
If you consider using this code or its derivatives, please consider citing:

```
@article{saito2021tune,
  title={Tune it the Right Way: Unsupervised Validation of Domain Adaptation via Soft Neighborhood Density},
  author={Saito, Kuniaki and Kim, Donghyun and Teterwak, Piotr and Sclaroff, Stan and Darrell, Trevor and Saenko, Kate},
  journal={arXiv preprint arXiv:2108.10860},
  year={2021}
}
```

