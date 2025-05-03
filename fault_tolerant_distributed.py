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
    master_addr=None, master_port=None, backend="nccl", timeout_minutes=5
):
    """Initialize fault-tolerant distributed environment"""
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

    # Choose backend based on available hardware
    if backend == "nccl" and not torch.cuda.is_available():
        logger.warning(
            "NCCL backend requested but CUDA not available. Falling back to gloo backend."
        )
        backend = "gloo"

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

    Args:
        world_size: Total number of machines/nodes available

    Returns:
        A stage configuration list with leaders and backups
    """
    # Need at least 5 machines for full redundancy (one per stage)
    if world_size < 5:
        logger.warning(
            f"World size {world_size} is less than ideal for full fault tolerance (need 5+ nodes)"
        )

    # For a minimal setup, we need at least 2 machines
    if world_size < 2:
        return [
            {"leader": {"rank": 0, "device": "cuda:0"}, "backups": []},
            {"leader": {"rank": 0, "device": "cuda:0"}, "backups": []},
            {"leader": {"rank": 0, "device": "cuda:0"}, "backups": []},
            {"leader": {"rank": 0, "device": "cuda:0"}, "backups": []},
            {"leader": {"rank": 0, "device": "cuda:0"}, "backups": []},
        ]

    # Default GPU device to use per machine (can be customized)
    default_device = "cuda:0"

    # Distribute the stages with redundancy
    config = []

    # For each stage (5 total: initial + layer1-4)
    for stage_idx in range(5):
        leader_rank = stage_idx % world_size

        # Create backups - use all machines except the leader
        backups = []
        backup_count = min(2, world_size - 1)  # Use at most 2 backups

        # Select next machines in order as backups
        for i in range(backup_count):
            backup_rank = (leader_rank + i + 1) % world_size
            backups.append({"rank": backup_rank, "device": default_device})

        # Add stage configuration
        config.append(
            {
                "leader": {"rank": leader_rank, "device": default_device},
                "backups": backups,
            }
        )

    return config
