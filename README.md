# [Tune it the Right Way: Unsupervised Validation of Domain Adaptationvia Soft Neighborhood Density (ICCV 2021)](https://arxiv.org/pdf/2108.10860.pdf)
#### [[Project Page]](https://cs-people.bu.edu/keisaito/research/SND.html)  [[Paper]](https://arxiv.org/pdf/2108.10860.pdf)
![Overview](imgs/mainfig_git_snd.png)

Kuniaki Saito, Donghyun Kim, Piotr Teterwak, Stan Sclaroff, Trevor Darrell, and Kate Saenko

## Introduction
In this work, we aim to build a criterion that can tune the hyper-parameters of unsupervised domain adaptation model in an unsupervised way.
This repository contains codes used for experiments of image classification, semantic segmentation, and toy datasets.

## Directories
We split the codes into four directories, base, cdan, adaptseg, and advent since each method employs different structures.

[nc_ps](nc_ps): image classification with pseudo-labeling (PS) and neighborhood clustering (NC), and toy experiments. <br>
[cdan](cdan): image classification with CDAN and MCC (borrowed from [CDAN](https://github.com/thuml/CDAN) and [MCC](https://github.com/thuml/Versatile-Domain-Adaptation)). <br>
[AdaptSegNet](AdaptSegNet): semantic segmentation with adaptsegnet. <br>
Advent(coming soon) will be from [ADVENT](https://github.com/valeoai/ADVENT).


## Requirement
Python 3.6.9, Pytorch 1.6.0, Torch Vision 0.7.0, [Apex](https://github.com/NVIDIA/apex), and sklearn (0.23.2). <br>

In some experiments, we used the nvidia apex library for memory efficient high-speed training. <br>

To track the training details, we also used [neptune](https://docs.neptune.ai/getting-started/installation), but this is optional configuration.
Please follow the instructions on each directory for other requirements.

## Reference
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

