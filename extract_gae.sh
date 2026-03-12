#!/bin/bash

CONFIG=${1:-"configs/train_gae_dit_xl.yaml"}
VAE_CONFIG=${2:-"configs/vae_configs/gae.yaml"}


if [ -z "$GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

if [ "$MASTER_PORT" == "" ] || [ "$MASTER_PORT" == "1236" ]; then
    MASTER_PORT=$(shuf -i 10000-20000 -n 1)
fi

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

echo "==========================================="
echo "RANK: $NODE_RANK, WORLD_SIZE: $NNODES"
echo "GPU PER NODE: $GPUS_PER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR:$MASTER_PORT"
echo "CONFIG: $CONFIG"
echo "VAE_CONFIG: $VAE_CONFIG"
echo "==========================================="


torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    extract_gae.py \
    --config "$CONFIG" \
    --vae_config "$VAE_CONFIG"