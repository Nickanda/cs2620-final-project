"""
fault_tolerant_distributed.py

Utilities for fault-tolerant, multi-node training with PyTorch.

This module provides:

1. **init_fault_tolerant_distributed(...)**
   Sets up a `torch.distributed` process group with sensible defaults, extended
   timeouts, and detailed logging.  It auto-selects an appropriate backend
   (`nccl` for GPUs, `gloo` otherwise) and highlights common mis-configuration
   issues.

2. **get_fault_tolerant_stage_config(world_size)**
   Produces a five-stage pipeline configuration (ResNet’s stem + four blocks)
   that assigns a *leader* rank and up to two *backup* ranks per stage.  The
   mapping adapts to the available `world_size`, supports minimal single-machine
   runs, and prioritizes GPU / MPS devices when present.

Together, these helpers make it easier to launch ResNet training
jobs that stay alive—even when individual nodes drop—by redistributing work and
maintaining redundant copies of each stage.
"""

import os
import datetime
import torch
import torch.distributed as dist
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fault_tolerant_resnet")


def init_fault_tolerant_distributed(
    master_addr=None, master_port=None, backend=None, timeout_minutes=10
):
    """
    Initialize fault-tolerant distributed environment for PyTorch distributed training.

    Args:
        master_addr (str, optional): IP address of the master node. If None, uses localhost
            or the MASTER_ADDR environment variable. Default is None.
        master_port (int, optional): Port for the master node. If None, uses 29500 or the
            MASTER_PORT environment variable. Default is None.
        backend (str, optional): The distributed backend to use ('nccl', 'gloo', etc.).
            If None, selects 'nccl' for CUDA devices and 'gloo' otherwise. Default is None.
        timeout_minutes (int, optional): Timeout in minutes for distributed operations.
            Default is 10.

    Raises:
        Exception: If distributed initialization fails due to network, firewall, or
            configuration issues.
    """
    # Set environment variables for distributed training
    if master_addr:
        os.environ["MASTER_ADDR"] = master_addr
    else:
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")

    if master_port:
        os.environ["MASTER_PORT"] = str(master_port)
    else:
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    logger.info(
        f"Attempting to initialize distributed with MASTER_ADDR: {os.environ['MASTER_ADDR']}, "
        f"MASTER_PORT: {os.environ['MASTER_PORT']}, "
        f"RANK: {os.environ.get('RANK', 'unknown')}, "
        f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'unknown')}"
    )

    # Remove interface specification that may be causing issues
    # if "GLOO_SOCKET_IFNAME" in os.environ:
    #     del os.environ["GLOO_SOCKET_IFNAME"]

    # # Set Gloo buffer allocation timeout to a higher value
    # if "GLOO_DEVICE_TRANSPORT" not in os.environ:
    #     os.environ["GLOO_DEVICE_TRANSPORT"] = "tcp"

    # Choose appropriate backend based on available hardware
    if backend is None:
        if torch.cuda.is_available():
            backend = "nccl"  # Use NCCL for NVIDIA GPUs
        else:
            backend = "gloo"  # Use Gloo for CPU and MPS (Apple Silicon)

    logger.info(f"Using '{backend}' backend for distributed training")

    # Initialize process group with extended timeout for fault detection
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend=backend,
                timeout=datetime.timedelta(minutes=timeout_minutes),
            )
            logger.info(
                f"Initialized distributed process group: rank {dist.get_rank()}/{dist.get_world_size()}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize distributed process group: {str(e)}")
            logger.error(
                "This could be due to firewall issues, incorrect network configuration, or unreachable nodes."
            )
            raise


def get_fault_tolerant_stage_config(world_size):
    """
    Generate a stage configuration for fault tolerance based on world size.

    Creates a configuration that assigns leaders and backup nodes for each stage
    of the ResNet model, optimizing for fault tolerance within the available nodes.

    Args:
        world_size (int): Total number of machines/nodes available in the distributed system.

    Returns:
        list: A stage configuration list with leader and backup assignments for each stage.
            Each element is a dict with 'leader' (containing 'rank' and 'device') and
            'backups' (list of dicts with 'rank' and 'device') for one stage.
    """
    # Need at least 5 machines for full redundancy (one per stage)
    if world_size < 5:
        logger.warning(
            f"World size {world_size} is less than ideal for full fault tolerance (need 5+ nodes)"
        )

    # For a minimal setup, we need at least 2 machines
    if world_size < 2:
        # Default device based on available hardware
        if torch.cuda.is_available():
            default_device = "cuda:0"
        elif torch.backends.mps.is_available():
            default_device = "mps"
        else:
            default_device = "cpu"

        return [
            {"leader": {"rank": 0, "device": default_device}, "backups": []},
            {"leader": {"rank": 0, "device": default_device}, "backups": []},
            {"leader": {"rank": 0, "device": default_device}, "backups": []},
            {"leader": {"rank": 0, "device": default_device}, "backups": []},
            {"leader": {"rank": 0, "device": default_device}, "backups": []},
        ]

    # Default device to use per machine - prioritize MPS on Apple Silicon
    if torch.cuda.is_available():
        default_device = "cuda:0"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

    # Set up the new configuration where rank 0 handles stages 0, 1, 2 and rank 1 handles stages 3, 4
    config = []

    # For each stage (5 total: initial + layer1-4)
    for stage_idx in range(5):
        # Assign leader ranks: 0 for first three stages, 1 for next two
        if stage_idx < 3:
            leader_rank = 0
        else:
            leader_rank = 1

        # Create backups - use all machines except the leader
        backups = []
        backup_count = min(2, world_size - 1)  # Use at most 2 backups

        # Select machines as backups, skipping the leader
        backup_candidates = [rank for rank in range(world_size) if rank != leader_rank]
        for i in range(min(backup_count, len(backup_candidates))):
            backup_rank = backup_candidates[i]
            backups.append({"rank": backup_rank, "device": default_device})

        # Add stage configuration
        config.append(
            {
                "leader": {"rank": leader_rank, "device": default_device},
                "backups": backups,
            }
        )

    return config
