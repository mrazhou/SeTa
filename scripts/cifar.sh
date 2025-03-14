# 0. Prepare SeTa path
export PYTHONPATH=.

# 1. Get arguments
type=${1:-SeTa} # Static, InfoBatch, SeTa
ratio=${2:-0.1}
num_group=${3:-5}
window_scale=${4:-0.9}

# 2. Concatenate arguments
name=$type/pr$ratio-ng$num_group-ws$window_scale
data=cifar100
model=resnet18
exp="results/$model/$data/${name}"

# 3. Run
CUDA_VISIBLE_DEVICES=0 \
python3 examples/cifar/main.py \
    --optimizer lars --max-lr 5.2 \
    --model $model \
    --save_path $exp \
    --data $data \
    --use_info_batch \
    --delta 0.875 \
    \
    --prune_type $type \
    --ratio $ratio \
    --num_group $num_group \
    --window_scale $window_scale
