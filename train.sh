#!/bin/bash

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=auto \
    text_diffusion/train.py \
    --batch_size 128 \
    --update_freq 1 \
    --lr 0.0001 \
    --epochs 1000 \
    --eval_every 2 \
    --check_every 20 \
    --diffusion_steps 1000 \
    --transformer_depth 12 \
    --transformer_heads 16 \
    --transformer_local_heads 8 \
    --gamma 0.99 \
    --parallel ddp
