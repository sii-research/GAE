#!/bin/bash
CONFIG=${1:-"configs/train_gae_dit_xl_800ep.yaml"}
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



accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision bf16 \
    train_gae.py \
    --config "$CONFIG" \
    --vae_config "$VAE_CONFIG"
