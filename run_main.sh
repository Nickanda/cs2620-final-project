#!/usr/bin/env bash
# run_main.sh  –  start two independent Python processes, one per GPU,
#                 on a single machine, no torchrun.

set -e                                  # stop on the first error

# ───────── rendez-vous TCP store (must be reachable by both ranks) ────
export NCCL_SOCKET_IFNAME=ib0  # Or eth0 on Linux
export GLOO_SOCKET_IFNAME=ib0  # Same as above
export NCCL_IB_DISABLE=1
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=12345                # pick an unused port
# Add IPv6 disabling flag
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO
# Force PyTorch to prefer IPv4
export NCCL_SOCKET_FAMILY=AF_INET

# ───────── common variables (identical for *all* ranks) ───────────────
export WORLD_SIZE=2                     # total number of ranks

# (optional safety) tell NCCL to stay on TCP and avoid IB until things work
# export NCCL_IB_DISABLE=1

# ───────── Rank 0  →  GPU 0 ───────────────────────────────────────────
RANK=0 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN \
  python main.py \
    --mode train --rank 0 --world-size 2 \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --batch-size 32 --epochs 3 &

# ───────── Rank 1  →  GPU 1 ───────────────────────────────────────────
RANK=1 LOCAL_RANK=1 CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN \
  python main.py \
    --mode train --rank 1 --world-size 2 \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --batch-size 32 --epochs 3 &

wait   # suspends this shell until both background jobs finish

# # (in bash, after setting MASTER_ADDR/PORT and the IFNAME vars)
# # Rank 0 → GPU 0
# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --mode train --rank 0 --world-size 2 \
#   --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
#   --batch-size 32 --epochs 10 &

# # Rank 1 → GPU 1
# CUDA_VISIBLE_DEVICES=1 python main.py \
#   --mode train --rank 1 --world-size 2 \
#   --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
#   --batch-size 32 --epochs 10 &

# waitR=10.31.180.213
# export MASTER_PORT=29502

# # (in bash, after setting MASTER_ADDR/PORT and the IFNAME vars)
# # Rank 0 → GPU 0
# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --mode train --rank 0 --world-size 2 \
#   --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
#   --batch-size 32 --epochs 10 &

# # Rank 1 → GPU 1
# CUDA_VISIBLE_DEVICES=1 python main.py \
#   --mode train --rank 1 --world-size 2 \
#   --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
#   --batch-size 32 --epochs 10 &

# wait
