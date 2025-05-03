#!/usr/bin/env bash

set -e

# Explicitly disable IPv6 for NCCL
export NCCL_SOCKET_IFNAME=em2  # Use Ethernet instead of InfiniBand
export GLOO_SOCKET_IFNAME=em2
export NCCL_IB_DISABLE=1
export MASTER_ADDR=$(ip -4 addr show em2 | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1)  # Use first IP address
export MASTER_PORT=29500
export NCCL_SOCKET_FAMILY=AF_INET  # Force IPv4 only
export NCCL_DEBUG=INFO

# Timeout and debug settings
export TORCH_NCCL_BLOCKING_WAIT=1              # Updated from deprecated NCCL_BLOCKING_WAIT
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1       # Updated from deprecated NCCL_ASYNC_ERROR_HANDLING
export NCCL_DEBUG=WARN                         # Reduce verbosity
export TORCH_DISTRIBUTED_DEBUG=INFO            # Less detailed than DETAIL

# Set world size
export WORLD_SIZE=2

# Rank 0 → GPU 0
RANK=0 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 \
  python main.py \
    --mode train --rank 0 --world-size 2 \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --batch-size 32 --epochs 3 &

# Rank 1 → GPU 1
RANK=1 LOCAL_RANK=1 CUDA_VISIBLE_DEVICES=1 \
  python main.py \
    --mode train --rank 1 --world-size 2 \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --batch-size 32 --epochs 3 &

wait