CUDA_VISIBLE_DEVICES=$1 python evaluate_cityscapes.py --restore-from $2 --gpu $1
python compute_iou.py ./data/Cityscapes/gtFine/val result/cityscapes
