export MASTER_ADDR=10.31.144.218
export MASTER_PORT=29502

# (in bash, after setting MASTER_ADDR/PORT and the IFNAME vars)
# Rank 0 → GPU 0
CUDA_VISIBLE_DEVICES=0 python main.py \
  --mode train --rank 0 --world-size 4 \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  --batch-size 32 --epochs 10 &

# Rank 1 → GPU 1
CUDA_VISIBLE_DEVICES=1 python main.py \
  --mode train --rank 1 --world-size 4 \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  --batch-size 32 --epochs 10 &

# Rank 2 → GPU 2
CUDA_VISIBLE_DEVICES=2 python main.py \
  --mode train --rank 2 --world-size 4 \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  --batch-size 32 --epochs 10 &

# Rank 3 → GPU 3
CUDA_VISIBLE_DEVICES=3 python main.py \
  --mode train --rank 3 --world-size 4 \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  --batch-size 32 --epochs 10 &

wait

python main.py \
  --mode train --rank 0 --world-size 1 \
  --master-addr 10.31.144.218 --master-port 29502 \
  --batch-size 64 --epochs 10