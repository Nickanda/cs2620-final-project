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
import ssl

# Disable SSL certificate verification for dataset downloads (macOS workaround)
ssl._create_default_https_context = ssl._create_unverified_context

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

    # If simulate-failure flag is on, exit the process after 20 seconds
    if args.simulate_failure:
        logger.info("Simulate-failure flag is on. Process will exit after 20 seconds...")
        import threading
        
        def exit_process():
            logger.info("Simulated failure: Exiting process now...")
            os._exit(0)
            
        # Schedule process exit after 20 seconds
        threading.Timer(20.0, exit_process).start()

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
                batches_processed = 0
                recovery_mode = False

                progress_bar = tqdm(
                    train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True
                )
                
                # Wrap in try-except to handle potential communication errors during node failure
                try:
                    for batch_idx, (data, target) in enumerate(progress_bar):
                        # Skip batches that this node is not responsible for
                        # In a real application, you'd use a distributed sampler
                        if batch_idx % args.world_size != args.rank:
                            continue

                        # Zero the parameter gradients
                        optimizer.zero_grad()

                        # Forward pass with error handling
                        try:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            
                            # Check if we're in recovery mode from a node failure
                            if hasattr(model, '_in_recovery') and model._in_recovery:
                                if not recovery_mode:
                                    logger.info(f"Rank {args.rank}: Training in recovery mode after node failure")
                                    recovery_mode = True
                            
                            # Ensure output and target have compatible shapes for cross entropy loss
                            if output.dim() != 2 or output.size(1) != 10:
                                logger.warning(f"Rank {args.rank}: Output has incorrect shape {output.shape}, expected [batch_size, 10]")
                                # Skip the batch but don't fail the epoch
                                continue
                            
                            if target.dim() != 1:
                                logger.warning(f"Rank {args.rank}: Target has incorrect shape {target.shape}, expected 1D tensor")
                                if target.dim() > 1:
                                    target = target.reshape(-1)
                                else:
                                    continue
                                    
                            # Make sure batch sizes match
                            if output.size(0) != target.size(0):
                                logger.warning(f"Rank {args.rank}: Batch size mismatch: output {output.size(0)}, target {target.size(0)}")
                                # Skip this batch
                                continue

                            # Backward pass and optimize
                            try:
                                loss = criterion(output, target)
                                loss.backward()
                                optimizer.step()

                                # Update statistics
                                train_loss += loss.item()
                                _, predicted = output.max(1)
                                total += target.size(0)
                                correct += predicted.eq(target).sum().item()
                                batches_processed += 1
                                
                                # If we were in recovery mode but completed a successful backward pass,
                                # we can exit recovery mode
                                if recovery_mode and hasattr(model, '_in_recovery'):
                                    model._in_recovery = False
                                    recovery_mode = False
                                    logger.info(f"Rank {args.rank}: Exiting recovery mode, training resumed successfully")
                                    
                            except Exception as e:
                                logger.error(f"Rank {args.rank}: Error in backward pass: {str(e)}")
                                # Enable recovery mode for next iteration
                                if not recovery_mode:
                                    logger.info(f"Rank {args.rank}: Entering recovery mode after error")
                                    model._in_recovery = True
                                    recovery_mode = True
                                continue

                            # Save checkpoints more frequently during recovery or regularly otherwise
                            checkpoint_interval = 10 if recovery_mode else 100
                            if batch_idx % checkpoint_interval == 0:
                                model.save_checkpoints()
                                
                        except Exception as e:
                            logger.error(f"Rank {args.rank}: Error in forward pass: {str(e)}")
                            # Enable recovery mode for next iteration
                            if not recovery_mode:
                                logger.info(f"Rank {args.rank}: Entering recovery mode after error")
                                model._in_recovery = True
                                recovery_mode = True
                            continue
                            
                except Exception as e:
                    logger.error(f"Rank {args.rank}: Error during training loop: {str(e)}")
                    # Don't abort the epoch - try to continue with the next one
                    
                # Print epoch statistics
                if batches_processed > 0:
                    accuracy = 100.0 * correct / total if total > 0 else 0
                    avg_loss = train_loss / batches_processed
                    logger.info(
                        f"Rank {args.rank}, Epoch {epoch}: Loss = {avg_loss:.3f}, Accuracy = {accuracy:.2f}%, Batches = {batches_processed}"
                    )
                else:
                    logger.warning(f"Rank {args.rank}, Epoch {epoch}: No batches processed successfully")
                
                # Save checkpoints at the end of each epoch
                model.save_checkpoints()

    elif args.mode == "eval":
        # Only the node responsible for the final stage should evaluate
        if 4 in model.my_stages and model.current_stage_leaders[4] == args.rank:
            # Evaluate the model
            evaluate_fault_tolerance(model, test_loader, device)

    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()
