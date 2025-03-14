#!/bin/bash
# conda activate <your_env>
# cd <path_to_Vim>/vim;

# 0. Prepare SeTa path
export PYTHONPATH=.

type=${1:-SeTa} # Static, InfoBatch, SeTa
model=${2:-resnet18}

ratio=${3:-0.0}
scale=${4:-0.6}
num_group=${5:-5}

dataset=IMNET

name="./results/$model/$dataset/pr$ratio-ng$num_group-ws$scale"

CUDA_VISIBLE_DEVICES=4 \
python -m torch.distributed.launch --nproc_per_node=1 \
    --use_env examples/imagenet/main.py \
    --model $model \
    --batch-size 128 \
    --drop-path 0.0 \
    --weight-decay 0.1 \
    --num_workers 8 \
    --data-path /data/zhou/datasets/imagenet/ \
    --no_amp \
    --epochs 300 \
    \
    --data-set $dataset \
    --output_dir $name \
    \
    --prune_type $type \
    --prune_ratio $ratio \
    --num_group $num_group \
    --window_scale $scale