#!/bin/bash
#SBATCH --partition=gpu-L --gres=gpu:1 --constraint="gpu6" --cpus-per-gpu=10

d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/neighbor_density
source activate pytorch
python $2  --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Real_pada.txt --target ./txt/target_Art_pada.txt --gpu $1 --hp 1.0 2.0 1.5 0.5 --use_neptune  --random_seed $2
