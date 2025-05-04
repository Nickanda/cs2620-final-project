# Fault-Tolerant Distributed ResNet50

A distributed implementation of ResNet50 with fault tolerance capabilities for resilient deep learning training across multiple machines.

## Project Overview

This project provides a robust implementation of a distributed ResNet50 model that can continue training even when one or more nodes fail. It distributes the ResNet architecture across multiple machines and implements failover mechanisms to ensure training continuity despite node failures.

![ex_model](images/ex_model.png)


### Key Features

- **Model Parallelism**: Splits ResNet50 across multiple nodes/devices
- **Fault Tolerance**: Provides backup nodes for each stage of computation
- **Heartbeat Monitoring**: Continuously checks node health and triggers failover when needed
- **Checkpoint Management**: Saves/loads model state for recovery
- **Leader-Backup Architecture**: Each stage has a leader and multiple backup nodes

## Architecture

The system divides ResNet50 into 5 stages:

- **Stage 0**: Initial convolution, batch normalization, and pooling
- **Stage 1-4**: ResNet block layers 1-4

Each stage has a designated leader node and optional backup nodes. If a leader fails, a backup automatically takes over. The system uses PyTorch's distributed communication to coordinate between nodes.

## Requirements

```
torch>=1.7.0
torchvision>=0.8.0
matplotlib>=3.3.0
tqdm>=4.50.0
numpy>=1.19.0
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/cs2620-final-project.git
   cd cs2620-final-project
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Single Machine (Testing)

Run a single-node experiment:

```bash
python main.py --mode train --rank 0 --world-size 1
```

### Multiple Machines (Distributed)

1. On the master node:

   ```bash
   python main.py --mode train --rank 0 --world-size N --master-addr <master-ip> --master-port 29500
   ```

2. On worker nodes (adjust rank accordingly):
   ```bash
   python main.py --mode train --rank 1 --world-size N --master-addr <master-ip> --master-port 29500
   ```

### Command Line Arguments

- `--mode`: Operation mode (`train` or `eval`)
- `--rank`: Node rank in the distributed setup
- `--world-size`: Total number of nodes in the cluster
- `--master-addr`: IP address of the master node
- `--master-port`: Port number for communication
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--checkpoint-dir`: Directory for saving checkpoints
- `--simulate-failure`: Flag to simulate node failures for testing

## Testing Fault Tolerance

To test the fault tolerance capabilities:

```bash
python main.py --mode train --rank 0 --world-size 2 --simulate-failure
```

This will simulate a failure of rank 1 during training.

## How Fault Tolerance Works

1. **Heartbeat Monitoring**: Each node regularly broadcasts a heartbeat
2. **Failure Detection**: Missing heartbeats for a certain period trigger failover
3. **Leader Promotion**: A backup node is promoted to leader
4. **Checkpoint Recovery**: The new leader loads the latest checkpoint
5. **Training Continuation**: Training resumes from the most recent consistent state

## Project Structure

- `main.py`: Entry point for training and evaluation
- `model_parallel_resnet.py`: Core implementation of the fault-tolerant ResNet
- `fault_tolerant_distributed.py`: Distributed setup and configuration
- `layers.py`: Custom layer implementations
- `utils.py`: Helper functions

## Acknowledgements

This project was developed as part of CS2620.
