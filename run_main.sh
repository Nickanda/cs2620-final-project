#!/usr/bin/env bash

# Print all commands as they're executed for better debugging
# set -x

# Don't exit immediately on error for better debugging
set -e

# Kill any processes using port 29500 or 29501 (more gracefully)
echo "Checking for processes using ports 29500 or 29501..."
if command -v lsof &> /dev/null; then
  procs=$(lsof -i:29500,29501 | grep LISTEN | awk '{print $2}' 2>/dev/null || echo "")
  if [ ! -z "$procs" ]; then
    echo "Found processes using ports 29500/29501: $procs"
    echo "Killing processes: $procs"
    for pid in $procs; do
      kill -9 $pid 2>/dev/null || echo "Could not kill process $pid"
    done
  else
    echo "No processes found on port 29500/29501"
  fi
else
  echo "lsof not found, skipping port check"
fi

# Wait a moment to ensure ports are freed
sleep 2

# Use localhost for testing to avoid network issues
export MASTER_ADDR="127.0.0.1"
# Select a random port in the range 20000-30000 to avoid conflicts
export MASTER_PORT=$((20000 + RANDOM % 10000))
echo "Using port: $MASTER_PORT"
echo "MASTER_ADDR set to: $MASTER_ADDR"

# For Apple Silicon - enable CPU fallback for MPS operations that aren't implemented
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Debug mode for distributed 
export TORCH_DISTRIBUTED_DEBUG=INFO

# Set world size to 2 for distributed training
export WORLD_SIZE=2

# Check if MPS is available on macOS
if python -c "import torch; print(torch.backends.mps.is_available())" | grep -q "True"; then
  echo "MPS is available - using Apple Silicon GPU"
  export USE_MPS=1
else
  echo "MPS is not available - checking for CUDA"
  if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA is available - using NVIDIA GPU"
    export USE_CUDA=1
  else
    echo "Neither MPS nor CUDA is available - using CPU"
  fi
fi

# For Apple Silicon Macs with MPS
if [ "$USE_MPS" == "1" ]; then
  echo "Running with MPS (Apple Silicon GPU)"
  
  # Rank 0
  RANK=0 LOCAL_RANK=0 \
    python main.py \
      --mode train --rank 0 --world-size 2 \
      --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
      --batch-size 32 --epochs 25 &

  # Rank 1
  RANK=1 LOCAL_RANK=1 \
    python main.py \
      --mode train --rank 1 --world-size 2 \
      --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
      --batch-size 32 --epochs 25 &

# For NVIDIA GPUs with CUDA
elif [ "$USE_CUDA" == "1" ]; then
  echo "Running with CUDA (NVIDIA GPU)"
  
  # Rank 0 → GPU 0
  RANK=0 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 \
    python main.py \
      --mode train --rank 0 --world-size 2 \
      --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
      --batch-size 32 --epochs 25 &

  # Rank 1 → GPU 1
  RANK=1 LOCAL_RANK=1 CUDA_VISIBLE_DEVICES=1 \
    python main.py \
      --mode train --rank 1 --world-size 2 \
      --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
      --batch-size 32 --epochs 25 &

# Fallback to CPU
else
  echo "Running on CPU"
  
  # Rank 0
  RANK=0 LOCAL_RANK=0 \
    python main.py \
      --mode train --rank 0 --world-size 2 \
      --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
      --batch-size 16 --epochs 25 &  # Smaller batch size for CPU

  # Rank 1
  RANK=1 LOCAL_RANK=1 \
    python main.py \
      --mode train --rank 1 --world-size 2 \
      --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
      --batch-size 16 --epochs 25 &  # Smaller batch size for CPU
fi

wait