"""
utils.py

Convenience helpers for the fault-tolerant, stage-parallel ResNet stack
----------------------------------------------------------------------

This file contains two high-level utilities that make it easy to **build**
and **validate** the model-parallel ResNet defined in
`model_parallel_resnet.py`:

1. **`resnet50_fault_tolerant(...)`**
   • Instantiates a `FaultTolerantModelParallelResNet`
     pre-configured as a ResNet-50 (layers [3, 4, 6, 3]).  
   • Accepts options for ImageNet pre-training, custom class counts,
     inclusion/exclusion of the top FC head, alternate pooling strategies,
     an explicit `stage_config`, and a checkpoint directory.

2. **`evaluate_fault_tolerance(model, test_loader, device)`**
   • Runs a forward-only evaluation loop while **artificially killing the
     stage-2 leader at batch 10** to exercise the heartbeat / fail-over
     machinery.  
   • After the induced failure it reports overall test accuracy and logs
     recovery progress.

Both functions emit detailed messages through the module-level logger
(`fault_tolerant_resnet`) to enable watching checkpointing, promotion of
backups, and other events unfold in real time.
"""

import os
import time
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fault_tolerant_resnet")


def resnet50_fault_tolerant(
    pretrained=False,
    num_classes=1000,
    include_top=True,
    pooling=None,
    stage_config=None,
    checkpoint_dir=None,
):
    """
    Constructs a fault-tolerant model-parallel ResNet-50 model.

    Args:
        pretrained (bool): If True, load pretrained ImageNet weights (if available).
            Default is False.
        num_classes (int): Number of classes for the final classification layer.
            Default is 1000.
        include_top (bool): Whether to include the final fully-connected layer.
            Default is True.
        pooling (str, optional): Pooling mode for feature extraction when include_top is False.
            Options are 'avg' or 'max'. Default is None.
        stage_config (list, optional): Configuration for each stage's leader and backup nodes.
            Format is list of dicts with 'leader' and 'backups' keys. Default is None.
        checkpoint_dir (str, optional): Directory for storing checkpoints for fault recovery.
            Default is None.

    Returns:
        FaultTolerantModelParallelResNet: An instance of the fault-tolerant ResNet50 model.
    """
    # Deferred import to avoid circular dependencies
    from model_parallel_resnet import FaultTolerantModelParallelResNet
    from layers import Bottleneck

    model = FaultTolerantModelParallelResNet(
        Bottleneck,
        [3, 4, 6, 3],  # ResNet50 configuration
        num_classes=num_classes,
        include_top=include_top,
        pooling=pooling,
        stage_config=stage_config,
        checkpoint_dir=checkpoint_dir,
    )

    # Load pretrained weights if requested
    if pretrained:
        # Code to load pretrained weights would go here.
        pass

    return model


def evaluate_fault_tolerance(model, test_loader, device):
    """
    Evaluate the model with simulated failures during the testing phase.
    Simulates a failure of the stage 2 leader at batch 10 to test failover mechanisms.

    Args:
        model (FaultTolerantModelParallelResNet): The fault-tolerant ResNet model to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for test data, providing batches
            of input images and target labels.
        device (torch.device): Device to use for evaluation (cuda, mps, or cpu).

    Returns:
        float: The test accuracy as a percentage (0-100).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            # Simulate a failure if this is batch 10
            if batch_idx == 10:
                logger.info("Simulating failure of stage 2 leader...")

                # Get the current leader for stage 2
                current_leader = model.current_stage_leaders[2]

                # Force failover by marking leader as failed
                del model.last_heartbeats[current_leader]

                # Set heartbeat time to be very old
                model.last_heartbeats[current_leader] = time.time() - 1000

                # Manually trigger failover
                model._handle_leader_failure(2)

            # Forward pass
            output = model(data)

            # Compute accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    logger.info(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy
