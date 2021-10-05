# Image classification with Neighborhood Clustering and Pseudo-Labeling

## Requirements
Python 3.6.9, Pytorch 1.6.0, Torch Vision 0.7.0, [Apex](https://github.com/NVIDIA/apex).
We used the nvidia apex library for memory efficient high-speed training. You also need sklearn (0.23.2 is used).

## Dataset Preparation

[Office Dataset](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
[OfficeHome Dataset](http://hemanthdv.org/OfficeHome-Dataset/) [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)
[DomainNet](http://ai.bu.edu/M3SDA/)

Prepare dataset in data directory.
```
./data/amazon/images/ ## Office
./data/dslr/images/ ## Office
./data/webcam/images/ ## Office
./data/Real ## OfficeHome
./data/Clipart ## OfficeHome
./data/Art ## OfficeHome
./data/Product ## OfficeHome
./data/DomainNet/real ## DomainNet real
./data/DomainNet/clipart ## DomainNet clipart
./data/visda/train ## VisDA synthetic images
./data/visda/validation ## VisDA real images
```

File list is stored in ./txt.

## Training and evaluation

All training script is stored in scripts_exp directory.

```
sh scripts_exp/run_a2d_nc.sh $gpu-id
```

The script defines the search space of the hyper-parameter.

