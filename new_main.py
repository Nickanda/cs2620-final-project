#!/usr/bin/env python
# Main script for fault-tolerant ResNet50

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Import local modules
from fault_tolerant_distributed import (
    init_fault_tolerant_distributed,
    get_fault_tolerant_stage_config,
)
from utils import resnet50_fault_tolerant, evaluate_fault_tolerance

# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fault_tolerant_resnet")

# Example usage
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Fault-Tolerant ResNet50 Training/Evaluation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Operation mode: train or evaluate",
    )
    parser.add_argument("--rank", type=int, default=0, help="Rank of this process")
    parser.add_argument(
        "--world-size", type=int, default=1, help="Total number of processes"
    )
    parser.add_argument(
        "--master-addr", type=str, default="localhost", help="Master node address"
    )
    parser.add_argument(
        "--master-port", type=int, default=29500, help="Master node port"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Input batch size")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--simulate-failure",
        action="store_true",
        help="Simulate a node failure during training/evaluation",
    )

    args = parser.parse_args()

    # Set up distributed environment
    if args.world_size > 1:
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)

        # Initialize distributed process group
        init_fault_tolerant_distributed(
            master_addr=args.master_addr, master_port=args.master_port
        )

    # Set device and create model with fault tolerance
    # Check for MPS (Metal Performance Shaders) on Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Create stage configuration for fault tolerance
    stage_config = get_fault_tolerant_stage_config(args.world_size)

    # Create fault-tolerant model
    model = resnet50_fault_tolerant(
        num_classes=10,  # For CIFAR-10
        include_top=True,
        stage_config=stage_config,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Print the stage configuration
    if args.rank == 0:
        logger.info("Stage configuration:")
        for i, config in enumerate(stage_config):
            leader = config["leader"]
            backups = config["backups"]
            backup_ranks = [b["rank"] for b in backups]
            logger.info(
                f"Stage {i}: Leader = {leader['rank']}, Backups = {backup_ranks}"
            )

    # Define transforms and dataset
    transform = transforms.Compose(
        [
            transforms.Resize(224),  # ResNet expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train or evaluate
    if args.mode == "train":
        # Only train if we're a leader for at least one stage
        if model.is_leader:
            # Set up optimizer and criterion
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss().to(device)
            # Ensure model is on the correct device
            model = model.to(device)

            # Training loop
            for epoch in range(args.epochs):
                model.train()
                train_loss = 0
                correct = 0
                total = 0

                progress_bar = tqdm(
                    train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True
                )
                for batch_idx, (data, target) in enumerate(progress_bar):
                    # Simulate failure if requested
                    if (
                        args.simulate_failure
                        and epoch == args.epochs // 2
                        and batch_idx == 10
                    ):
                        if args.rank == 0:
                            logger.info(
                                "Simulating leader failure by exiting rank 1..."
                            )
                            if args.world_size > 1:
                                # Send a message to rank 1 to simulate failure
                                kill_signal = torch.tensor([1.0])
                                dist.send(kill_signal, dst=1)

                        # If this is rank 1, simulate a crash
                        if args.rank == 1:
                            logger.info("Rank 1: Simulating crash...")
                            # Wait a moment for things to sync
                            time.sleep(5)
                            # This will cause a "crash" for this process
                            kill_signal = torch.tensor([0.0])
                            dist.recv(kill_signal, src=0)
                            if kill_signal[0] > 0:
                                logger.info("Received kill signal, exiting process...")
                                # In a real application, you would handle this more gracefully
                                os._exit(0)

                    # Skip batches that this node is not responsible for
                    # In a real application, you'd use a distributed sampler
                    if batch_idx % args.world_size != args.rank:
                        continue

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    # Backward pass and optimize
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    # Update statistics
                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

                    # Save checkpoints periodically
                    if batch_idx % 100 == 0:
                        model.save_checkpoints()

                # Print epoch statistics
                accuracy = 100.0 * correct / total if total > 0 else 0
                logger.info(
                    f"Rank {args.rank}, Epoch {epoch}: Loss = {train_loss / len(train_loader):.3f}, Accuracy = {accuracy:.2f}%"
                )

    elif args.mode == "eval":
        # Only the node responsible for the final stage should evaluate
        if 4 in model.my_stages and model.current_stage_leaders[4] == args.rank:
            # Evaluate the model
            evaluate_fault_tolerance(model, test_loader, device)

    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()


