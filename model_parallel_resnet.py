import os
import time
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging
import datetime

# Import local modules
from layers import Scale

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fault_tolerant_resnet")


class FaultTolerantModelParallelResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        include_top=True,
        pooling=None,
        stage_config=None,
        checkpoint_dir=None,
    ):
        """
        Args:
            block: Block type (here, Bottleneck).
            layers: List with the number of blocks in each of the 4 layers.
            num_classes: Number of output classes.
            include_top: Whether to include the final FC layer.
            pooling: Optional pooling mode when include_top is False (avg or max).
            stage_config: Configuration for fault tolerance with leader and backup machines.
                Format: [
                    {'leader': {'rank': 0, 'device': 'mps'}, 'backups': [{'rank': 1, 'device': 'mps'}]},
                    {'leader': {'rank': 2, 'device': 'mps'}, 'backups': [{'rank': 3, 'device': 'mps'}]},
                    ...
                ]
            checkpoint_dir: Directory to save/load model checkpoints for fault recovery.
        """
        super(FaultTolerantModelParallelResNet, self).__init__()
        self.inplanes = 64
        self.include_top = include_top
        self.pooling = pooling
        self.checkpoint_dir = checkpoint_dir or os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Create hearthbeat monitoring system
        self.heartbeat_interval = 5  # seconds
        self.last_heartbeats = {}
        self.heartbeat_thread = None
        self.is_leader = False
        self.current_stage_leaders = {}

        # Default epsilon for batch norm
        eps = 1.1e-5

        # Handle fault-tolerant stage configuration
        self.stage_config = stage_config

        # If no stage config provided, create a basic one with current device
        if self.stage_config is None:
            # Check for MPS (Metal Performance Shaders) on Apple Silicon first
            if torch.backends.mps.is_available():
                current_device = torch.device("mps")
            elif torch.cuda.is_available():
                current_device = torch.device("cuda")
            else:
                current_device = torch.device("cpu")
                
            # Create default config with single machine for all stages
            self.stage_config = [
                {"leader": {"rank": 0, "device": current_device}, "backups": []}
                for _ in range(5)  # Input stage + 4 ResNet stages
            ]

        # Initialize distributed coordination if not already done
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # Determine which stages this machine is responsible for
        self.my_stages = []
        self.is_leader_for_stages = []

        for stage_idx, stage in enumerate(self.stage_config):
            # Check if this rank is the leader for this stage
            if stage["leader"]["rank"] == self.rank:
                self.my_stages.append(stage_idx)
                self.is_leader_for_stages.append(True)
                self.is_leader = True

            # Check if this rank is a backup for this stage
            for backup in stage["backups"]:
                if backup["rank"] == self.rank:
                    self.my_stages.append(stage_idx)
                    self.is_leader_for_stages.append(False)

        logger.info(f"Rank {self.rank} is responsible for stages: {self.my_stages}")
        logger.info(
            f"Rank {self.rank} is leader for stages: {[stage for i, stage in enumerate(self.my_stages) if self.is_leader_for_stages[i]]}"
        )

        # Map stage indices to devices
        self.stage_devices = {}
        for stage_idx, stage in enumerate(self.stage_config):
            if stage_idx in self.my_stages:
                leader_idx = self.is_leader_for_stages[self.my_stages.index(stage_idx)]
                if leader_idx:
                    self.stage_devices[stage_idx] = torch.device(
                        stage["leader"]["device"]
                    )
                else:
                    # Find which backup we are
                    for backup in stage["backups"]:
                        if backup["rank"] == self.rank:
                            self.stage_devices[stage_idx] = torch.device(
                                backup["device"]
                            )

        # Start with all stages using their leader
        for stage_idx, stage in enumerate(self.stage_config):
            self.current_stage_leaders[stage_idx] = stage["leader"]["rank"]

        # Initial convolutional layer (stage 0)
        # Only create if this node is responsible for stage 0
        if 0 in self.my_stages:
            device = self.stage_devices[0]
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            ).to(device)
            self.bn1 = nn.BatchNorm2d(64, eps=eps).to(device)
            self.scale1 = Scale(64).to(device)
            self.relu = nn.ReLU(inplace=True).to(device)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)

        # Build ResNet layers (stages 1-4)
        # Only create layers for stages this node is responsible for
        if 1 in self.my_stages:
            self.layer1 = self._make_layer(
                block, 64, layers[0], stride=1, device=self.stage_devices[1]
            )
        if 2 in self.my_stages:
            self.layer2 = self._make_layer(
                block, 128, layers[1], stride=2, device=self.stage_devices[2]
            )
        if 3 in self.my_stages:
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=2, device=self.stage_devices[3]
            )
        if 4 in self.my_stages:
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=2, device=self.stage_devices[4]
            )

        # Final classification layers (only create if this node handles stage 4)
        if 4 in self.my_stages:
            device = self.stage_devices[4]
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
            if include_top:
                self.fc = nn.Linear(512 * block.expansion, num_classes).to(device)
            else:
                self.fc = None

        # Initialize weights (optional but common)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Scale):
                nn.init.constant_(m.gamma, 1)
                nn.init.constant_(m.beta, 0)

        # Start heartbeat thread if using distributed
        if dist.is_initialized() and self.world_size > 1:
            self._start_heartbeat_monitoring()

    def _make_layer(self, block, planes, blocks, stride=1, device=None):
        """
        Build one stage of the ResNet model.

        Args:
            block: Block type.
            planes: Number of channels for the intermediate layers.
            blocks: Number of blocks to stack.
            stride: Stride for the first block.
            device: Device to place this layer on.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, eps=1.1e-5),
                Scale(planes * block.expansion),
            )

            # Move downsample to device if specified
            if device is not None:
                downsample = downsample.to(device)

        layers = []
        # First block may have a different stride
        layers.append(block(self.inplanes, planes, stride, downsample, device=device))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, device=device))

        return nn.Sequential(*layers)

    def _start_heartbeat_monitoring(self):
        """Start heartbeat mechanism for fault detection"""
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()
        logger.info(f"Rank {self.rank}: Started heartbeat monitoring thread")

    def _heartbeat_loop(self):
        """Loop to send and monitor heartbeats with improved error handling for Apple Silicon"""
        error_count = 0
        max_errors = 10
        
        # Use CPU tensors for heartbeats to avoid MPS/CUDA issues
        device = "cpu"

        # Keep track of the last time we saw a heartbeat from each rank
        # Initialize with current time to avoid immediate failure detection
        for i in range(self.world_size):
            self.last_heartbeats[i] = time.time()

        logger.info(f"Rank {self.rank}: Initialized heartbeat monitoring for {self.world_size} ranks")

        while True:
            try:
                # Send heartbeat
                if dist.is_initialized():
                    # All ranks send heartbeats to ensure we can detect when any rank fails
                    # Not just leaders, as backups need to respond too
                    heartbeat = torch.tensor(
                        [time.time(), float(self.rank)],
                        dtype=torch.float32,
                        device=device,
                    )
                    
                    # Use non-blocking communication with explicit timeout
                    try:
                        # Broadcast my heartbeat
                        work = dist.broadcast(heartbeat, src=self.rank, async_op=True)
                        work.wait(timeout=datetime.timedelta(seconds=10))
                    except Exception as e:
                        logger.warning(f"Rank {self.rank}: Heartbeat broadcast error (non-critical): {str(e)}")
                        # Continue anyway - this is non-critical

                    # Receive heartbeats from all other ranks, not just leaders
                    for other_rank in range(self.world_size):
                        if other_rank != self.rank:  # Don't need to record own heartbeat
                            try:
                                heartbeat = torch.zeros(2, dtype=torch.float32, device=device)
                                # Use non-blocking communication with timeout
                                work = dist.broadcast(heartbeat, src=other_rank, async_op=True)
                                
                                # Lower timeout to faster detect failures
                                success = work.wait(timeout=datetime.timedelta(seconds=5))
                                
                                if success:
                                    self.last_heartbeats[other_rank] = time.time()  # Use current time instead of heartbeat time
                                    logger.debug(f"Rank {self.rank}: Received heartbeat from rank {other_rank}")
                            except Exception as e:
                                # Mark potential failure - increment missed heartbeat counter
                                # Non-critical error
                                logger.warning(f"Rank {self.rank}: Error receiving heartbeat from rank {other_rank}: {str(e)}")
                                # Consider this a potential failure
                                if other_rank in self.last_heartbeats:
                                    # If this is a persistent missed heartbeat, we'll mark for potential takeover
                                    # Move heartbeat timestamp further in the past to accelerate detection
                                    if time.time() - self.last_heartbeats[other_rank] > 2 * self.heartbeat_interval:
                                        self.last_heartbeats[other_rank] -= self.heartbeat_interval  # Age the heartbeat faster

                    # Check for failed ranks and initiate takeover if needed
                    logger.debug(f"Rank {self.rank}: Checking heartbeats: {self.last_heartbeats}")
                    
                    for stage_idx, stage in enumerate(self.stage_config):
                        # Only worry about stages we're a backup for
                        if stage_idx not in self.my_stages:
                            continue
                            
                        # Get the current leader for this stage
                        current_leader = self.current_stage_leaders[stage_idx]
                        
                        # Skip checking our own heartbeat
                        if current_leader == self.rank:
                            continue
                        
                        # Check if we're a backup for this stage
                        am_backup = False
                        for backup in stage["backups"]:
                            if backup["rank"] == self.rank:
                                am_backup = True
                                break
                        
                        if am_backup:
                            # Check if leader for this stage is alive
                            if current_leader in self.last_heartbeats:
                                last_beat_time = self.last_heartbeats[current_leader]
                                time_since_last_beat = time.time() - last_beat_time
                                
                                # Log the time since last heartbeat for debugging
                                logger.debug(f"Rank {self.rank}: Time since last heartbeat from rank {current_leader}: {time_since_last_beat:.2f}s")
                                
                                # If we haven't seen a heartbeat in a while, mark as failure
                                if time_since_last_beat > 3 * self.heartbeat_interval:
                                    logger.warning(
                                        f"Rank {self.rank}: Detected failure of rank {current_leader} for stage {stage_idx}! "
                                        f"Last heartbeat: {time_since_last_beat:.2f}s ago"
                                    )
                                    self._handle_leader_failure(stage_idx)

                # Reset error count after successful execution
                error_count = 0
                time.sleep(self.heartbeat_interval)

            except Exception as e:
                error_count += 1
                logger.error(
                    f"Rank {self.rank}: Error in heartbeat loop: {str(e)} (Error count: {error_count}/{max_errors})"
                )

                # Exit loop if error threshold is reached
                if error_count >= max_errors:
                    logger.critical(
                        f"Rank {self.rank}: Heartbeat loop exiting after {max_errors} consecutive errors"
                    )
                    break

                time.sleep(self.heartbeat_interval)

    def _handle_leader_failure(self, stage_idx):
        """Handle a leader failure by promoting a backup and taking over all stages from failed rank"""
        # Get the stage configuration
        stage_config = self.stage_config[stage_idx]
        failed_leader_rank = self.current_stage_leaders[stage_idx]
        
        logger.warning(f"Rank {self.rank}: Detected failure of leader (rank {failed_leader_rank}) for stage {stage_idx}!")
        
        # Find the first backup - prioritize by order in the backup list
        for i, backup in enumerate(stage_config["backups"]):
            backup_rank = backup["rank"]

            # Check if this backup is alive (not necessary for self)
            is_alive = (backup_rank == self.rank) or (
                backup_rank in self.last_heartbeats
                and time.time() - self.last_heartbeats[backup_rank]
                < 3 * self.heartbeat_interval
            )

            if is_alive:
                # Promote this backup to leader for this stage
                self.current_stage_leaders[stage_idx] = backup_rank

                logger.info(
                    f"Rank {self.rank}: Promoting backup (rank {backup_rank}) to leader for stage {stage_idx}"
                )

                # If I'm the new leader, load the latest checkpoint
                if backup_rank == self.rank:
                    # Update my status
                    idx = self.my_stages.index(stage_idx)
                    self.is_leader_for_stages[idx] = True
                    self.is_leader = True

                    # Load checkpoint if available
                    self._load_stage_checkpoint(stage_idx)
                    
                    # Take over all stages that the failed rank was leading
                    self._take_over_failed_rank_stages(failed_leader_rank)
                    
                    # Reset all activations to ensure clean gradient computation
                    self._reset_activations()

                    # Broadcast to other ranks that I'm now the leader (skip this if all other ranks have failed)
                    if dist.is_initialized():
                        try:
                            status_update = torch.tensor(
                                [float(stage_idx), float(backup_rank)], dtype=torch.float32, 
                                device="cpu"  # Always use CPU for communication
                            )
                            for r in range(self.world_size):
                                if r != self.rank:
                                    try:
                                        dist.send(status_update, dst=r)
                                    except Exception as e:
                                        logger.warning(f"Rank {self.rank}: Failed to send status update to rank {r}: {str(e)}")
                        except Exception as e:
                            logger.warning(f"Rank {self.rank}: Error broadcasting leadership change: {str(e)}")

                    logger.info(f"Rank {self.rank}: Now leader for stage {stage_idx}")

                # Reshufffle backups (remove promoted one and add failed leader as last backup)
                new_backups = [
                    b for b in stage_config["backups"] if b["rank"] != backup_rank
                ]

                # Update the stage configuration
                self.stage_config[stage_idx] = {
                    "leader": {"rank": backup_rank, "device": backup["device"]},
                    "backups": new_backups,
                }

                return

        logger.error(f"Rank {self.rank}: No available backup for stage {stage_idx}!")
        
    def _take_over_failed_rank_stages(self, failed_rank):
        """Take over all stages that were led by a failed rank"""
        logger.info(f"Rank {self.rank}: Taking over all stages led by failed rank {failed_rank}")
        
        # Track all newly acquired stages
        newly_acquired_stages = []
        
        # Identify all stages where the failed rank was the leader
        for stage_idx, stage in enumerate(self.stage_config):
            # Skip if we've already handled this stage or if I'm already the leader
            if self.current_stage_leaders[stage_idx] != failed_rank or self.current_stage_leaders[stage_idx] == self.rank:
                continue
                
            # Check if I'm a backup for this stage
            am_backup = False
            for backup in stage["backups"]:
                if backup["rank"] == self.rank:
                    am_backup = True
                    break
                    
            if am_backup:
                # If I'm a backup, promote myself to leader for this stage
                logger.info(f"Rank {self.rank}: Taking over leadership of stage {stage_idx} from failed rank {failed_rank}")
                newly_acquired_stages.append(stage_idx)
                
                # Update leader status
                self.current_stage_leaders[stage_idx] = self.rank
                
                # Update my status if stage in my_stages
                if stage_idx in self.my_stages:
                    idx = self.my_stages.index(stage_idx)
                    self.is_leader_for_stages[idx] = True
                    self.is_leader = True
                else:
                    # Add this stage to my responsibilities
                    self.my_stages.append(stage_idx)
                    self.is_leader_for_stages.append(True)
                    
                    # Determine device for this stage
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                    elif torch.backends.mps.is_available():
                        device = torch.device("mps")
                    else:
                        device = torch.device("cpu")
                    
                    self.stage_devices[stage_idx] = device
                    
                    # Initialize the stage-specific layers if not already initialized
                    self._initialize_stage_layers(stage_idx)
                
                # Load checkpoint
                self._load_stage_checkpoint(stage_idx)
                
                # Update stage configuration
                new_backups = [b for b in stage["backups"] if b["rank"] != self.rank]
                self.stage_config[stage_idx] = {
                    "leader": {"rank": self.rank, "device": self.stage_devices[stage_idx]},
                    "backups": new_backups,
                }
                
        # Log updated responsibilities
        logger.info(f"Rank {self.rank}: Now responsible for stages: {self.my_stages}")
        logger.info(f"Rank {self.rank}: Leader for stages: {[stage for i, stage in enumerate(self.my_stages) if self.is_leader_for_stages[i]]}")
        
        # If we took over any stages, wait a moment to stabilize before continuing
        if newly_acquired_stages:
            time.sleep(1)
            
    def _reset_activations(self):
        """Reset activations to ensure clean gradient computation after takeover"""
        # This is critical to avoid the "modified by an inplace operation" error
        # when a node takes over stages from a failed node
        logger.info(f"Rank {self.rank}: Resetting activations to ensure clean gradient computation")
        
        # Reset any stored tensors
        if hasattr(self, '_last_input'):
            del self._last_input
        
        # Reset any intermediate activations that might be stored
        # Force garbage collection to release any tensors
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Clear any gradients
        self.zero_grad()
        
    def zero_grad(self):
        """Clear gradients of all parameters"""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def _initialize_stage_layers(self, stage_idx):
        """Initialize the layers for a specific stage if they don't exist already"""
        device = self.stage_devices[stage_idx]
        
        if stage_idx == 0 and not hasattr(self, 'conv1'):
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
            self.bn1 = nn.BatchNorm2d(64, eps=1.1e-5).to(device)
            self.scale1 = Scale(64).to(device)
            self.relu = nn.ReLU(inplace=True).to(device)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
            
        elif stage_idx == 1 and not hasattr(self, 'layer1'):
            # We need to know how many layers to create - use default ResNet50 config
            block = Bottleneck  # This should be defined elsewhere in your code
            layers = [3, 4, 6, 3]  # Default ResNet50 configuration
            self.layer1 = self._make_layer(block, 64, layers[0], stride=1, device=device)
            
        elif stage_idx == 2 and not hasattr(self, 'layer2'):
            block = Bottleneck
            layers = [3, 4, 6, 3]
            # Need to set inplanes to the correct value
            self.inplanes = 256  # After layer1 for ResNet50
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, device=device)
            
        elif stage_idx == 3 and not hasattr(self, 'layer3'):
            block = Bottleneck
            layers = [3, 4, 6, 3]
            # Need to set inplanes to the correct value
            self.inplanes = 512  # After layer2 for ResNet50
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, device=device)
            
        elif stage_idx == 4 and not hasattr(self, 'layer4'):
            block = Bottleneck
            layers = [3, 4, 6, 3]
            # Need to set inplanes to the correct value
            self.inplanes = 1024  # After layer3 for ResNet50
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, device=device)
            
            # Final layers
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
            if self.include_top:
                self.fc = nn.Linear(512 * block.expansion, 10).to(device)  # 10 classes for CIFAR-10

    def _save_stage_checkpoint(self, stage_idx):
        """Save checkpoint for a specific stage"""
        if stage_idx not in self.my_stages:
            return

        # Only save if we're the leader for this stage
        idx = self.my_stages.index(stage_idx)
        if not self.is_leader_for_stages[idx]:
            return

        checkpoint = {}

        # Save different components based on stage
        if stage_idx == 0:
            checkpoint = {
                "conv1": self.conv1.state_dict(),
                "bn1": self.bn1.state_dict(),
                "scale1": self.scale1.state_dict(),
            }
        elif stage_idx == 1:
            checkpoint = {"layer1": self.layer1.state_dict()}
        elif stage_idx == 2:
            checkpoint = {"layer2": self.layer2.state_dict()}
        elif stage_idx == 3:
            checkpoint = {"layer3": self.layer3.state_dict()}
        elif stage_idx == 4:
            checkpoint = {
                "layer4": self.layer4.state_dict(),
                "avgpool": self.avgpool.state_dict(),
            }
            if self.fc is not None:
                checkpoint["fc"] = self.fc.state_dict()

        # Save to file
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"stage_{stage_idx}_checkpoint.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        # logger.info(f"Rank {self.rank}: Saved checkpoint for stage {stage_idx}")

    def _load_stage_checkpoint(self, stage_idx):
        """Load checkpoint for a specific stage"""
        if stage_idx not in self.my_stages:
            return False

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"stage_{stage_idx}_checkpoint.pt"
        )

        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Rank {self.rank}: No checkpoint found for stage {stage_idx}"
            )
            return False

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Load different components based on stage
        if stage_idx == 0:
            if hasattr(self, "conv1"):
                self.conv1.load_state_dict(checkpoint["conv1"])
                self.bn1.load_state_dict(checkpoint["bn1"])
                self.scale1.load_state_dict(checkpoint["scale1"])
        elif stage_idx == 1 and hasattr(self, "layer1"):
            self.layer1.load_state_dict(checkpoint["layer1"])
        elif stage_idx == 2 and hasattr(self, "layer2"):
            self.layer2.load_state_dict(checkpoint["layer2"])
        elif stage_idx == 3 and hasattr(self, "layer3"):
            self.layer3.load_state_dict(checkpoint["layer3"])
        elif stage_idx == 4 and hasattr(self, "layer4"):
            self.layer4.load_state_dict(checkpoint["layer4"])
            self.avgpool.load_state_dict(checkpoint["avgpool"])
            if self.fc is not None and "fc" in checkpoint:
                self.fc.load_state_dict(checkpoint["fc"])

        logger.info(f"Rank {self.rank}: Loaded checkpoint for stage {stage_idx}")
        return True

    def save_checkpoints(self):
        """Save checkpoints for all stages this node is responsible for"""
        for stage_idx in self.my_stages:
            self._save_stage_checkpoint(stage_idx)

    def forward(self, x):
        """Forward pass including stage distribution and fault tolerance"""
        # Track tensors at each stage (for failure recovery)
        stage_outputs = {}
        original_batch_size = x.size(0)
        
        # Track if we're in a recovery operation after takeover
        in_recovery = hasattr(self, '_in_recovery') and self._in_recovery
        
        # Make sure the input tensor tracks gradients
        if x.requires_grad == False:
            x.requires_grad = True

        # Stage 0: Initial convolution layer
        if 0 in self.my_stages and self.current_stage_leaders[0] == self.rank:
            try:
                # Move input to appropriate device
                x = x.to(self.stage_devices[0])
                
                # Ensure gradients are tracked after device transfer
                if not x.requires_grad:
                    x.requires_grad = True

                # Process initial layers - use non-inplace operations when in recovery
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.scale1(x)
                
                # Use clone to avoid inplace operation issues during takeover
                if in_recovery:
                    x = F.relu(x)
                else:
                    x = self.relu(x)
                
                x = self.maxpool(x)

                # Store output for this stage
                stage_outputs[0] = x.detach().clone() if in_recovery else x.clone()

                # Send output to stage 1 leader if it's a different rank
                if dist.is_initialized() and self.current_stage_leaders[1] != self.rank:
                    # Always use CPU for communication to avoid MPS issues
                    x_cpu = x.detach().cpu()  # Detach to avoid gradient tracking issues during communication
                    
                    # Use simpler approach - first send a small tensor with metadata
                    # This helps establish a connection before sending the large tensor
                    metadata = torch.tensor([
                        original_batch_size,  # Item 0: Batch size
                        x_cpu.size(1),        # Item 1: Channels
                        x_cpu.size(2),        # Item 2: Height
                        x_cpu.size(3),        # Item 3: Width
                    ], dtype=torch.long, device="cpu")
                                        
                    try:
                        dist.send(metadata, dst=self.current_stage_leaders[1])
                        
                        # Now send the actual tensor with blocking call (simpler and more reliable)
                        dist.send(x_cpu, dst=self.current_stage_leaders[1])
                    except Exception as e:
                        logger.error(f"Rank {self.rank}: Error in stage 0 sending: {str(e)}")
                        # Return a proper dummy tensor for loss calculation
                        return torch.zeros((original_batch_size, 10), device=self.stage_devices.get(0, "cpu"), requires_grad=True)
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error in stage 0 processing: {str(e)}")
                return torch.zeros((original_batch_size, 10), device=self.stage_devices.get(0, "cpu"), requires_grad=True)
                
        # Stage 1: Layer 1
        if 1 in self.my_stages and self.current_stage_leaders[1] == self.rank:
            try:
                # If we're not the leader of the previous stage, receive input
                if 0 not in self.my_stages or self.current_stage_leaders[0] != self.rank:
                    # Receive metadata first
                    metadata = torch.zeros(4, dtype=torch.long, device="cpu")
                    
                    try:
                        dist.recv(metadata, src=self.current_stage_leaders[0])
                        
                        # Create tensor with the right size
                        tensor_shape = (metadata[0].item(), metadata[1].item(), 
                                        metadata[2].item(), metadata[3].item())
                        
                        # Create a tensor on CPU first
                        x_cpu = torch.zeros(tensor_shape, dtype=torch.float32, device="cpu")
                        
                        # Receive the actual tensor data
                        dist.recv(x_cpu, src=self.current_stage_leaders[0])
                        
                        # Move to the appropriate device
                        x = x_cpu.to(self.stage_devices[1])
                        
                        # CRITICAL: Ensure gradients are tracked for backward pass
                        x = x.detach().requires_grad_(True)  # Create a fresh tensor with new version
                    except Exception as e:
                        logger.error(f"Rank {self.rank}: Error in stage 1 receiving: {str(e)}")
                        return torch.zeros((original_batch_size, 10), device=self.stage_devices.get(1, "cpu"), requires_grad=True)
                
                # Process layer 1 - during recovery, we need to be careful with tensor versioning
                x = self.layer1(x)
                
                # Store output for this stage, detach if in recovery mode
                stage_outputs[1] = x.detach().clone() if in_recovery else x.clone()
                
                # Send output to stage 2 leader
                if dist.is_initialized() and self.current_stage_leaders[2] != self.rank:
                    # Use the same approach as for stage 0
                    x_cpu = x.detach().cpu()  # Detach to avoid gradient tracking issues
                    
                    # Prepare metadata
                    metadata = torch.tensor([
                        x_cpu.size(0),
                        x_cpu.size(1),
                        x_cpu.size(2),
                        x_cpu.size(3),
                    ], dtype=torch.long, device="cpu")
                    
                    try:
                        # Send metadata
                        dist.send(metadata, dst=self.current_stage_leaders[2])
                        
                        # Send tensor data
                        dist.send(x_cpu, dst=self.current_stage_leaders[2])
                    except Exception as e:
                        logger.error(f"Rank {self.rank}: Error in stage 1 sending: {str(e)}")
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error in stage 1 processing: {str(e)}")
                return torch.zeros((original_batch_size, 10), device=self.stage_devices.get(1, "cpu"), requires_grad=True)
                
        # Stage 2: Layer 2
        if 2 in self.my_stages and self.current_stage_leaders[2] == self.rank:
            try:
                # If we're not the leader of the previous stage, receive input
                if 1 not in self.my_stages or self.current_stage_leaders[1] != self.rank:
                    # Receive metadata first
                    metadata = torch.zeros(4, dtype=torch.long, device="cpu")
                    
                    try:
                        dist.recv(metadata, src=self.current_stage_leaders[1])
                        
                        # Create tensor with the right size
                        tensor_shape = (metadata[0].item(), metadata[1].item(), 
                                        metadata[2].item(), metadata[3].item())
                        
                        # Create a tensor on CPU first
                        x_cpu = torch.zeros(tensor_shape, dtype=torch.float32, device="cpu")
                        
                        # Receive the actual tensor data
                        dist.recv(x_cpu, src=self.current_stage_leaders[1])
                        
                        # Move to the appropriate device
                        x = x_cpu.to(self.stage_devices[2])
                        
                        # CRITICAL: Create a fresh tensor with gradient tracking
                        x = x.detach().requires_grad_(True)
                    except Exception as e:
                        logger.error(f"Rank {self.rank}: Error in stage 2 receiving: {str(e)}")
                        return torch.zeros((original_batch_size, 10), device=self.stage_devices.get(2, "cpu"), requires_grad=True)
                
                # Process layer 2
                x = self.layer2(x)
                
                # Store output for this stage, detach if in recovery mode
                stage_outputs[2] = x.detach().clone() if in_recovery else x.clone()
                
                # Send output to stage 3 leader
                if dist.is_initialized() and self.current_stage_leaders[3] != self.rank:
                    # Use the same approach as before
                    x_cpu = x.detach().cpu()  # Detach to avoid gradient tracking issues
                    
                    # Prepare metadata
                    metadata = torch.tensor([
                        x_cpu.size(0),
                        x_cpu.size(1),
                        x_cpu.size(2),
                        x_cpu.size(3),
                    ], dtype=torch.long, device="cpu")
                    
                    try:
                        # Send metadata
                        dist.send(metadata, dst=self.current_stage_leaders[3])
                        
                        # Send tensor data
                        dist.send(x_cpu, dst=self.current_stage_leaders[3])
                    except Exception as e:
                        logger.error(f"Rank {self.rank}: Error in stage 2 sending: {str(e)}")
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error in stage 2 processing: {str(e)}")
                return torch.zeros((original_batch_size, 10), device=self.stage_devices.get(2, "cpu"), requires_grad=True)
        
        # Stage 3: Layer 3
        if 3 in self.my_stages and self.current_stage_leaders[3] == self.rank:
            try:
                # If we're not the leader of the previous stage, receive input
                if 2 not in self.my_stages or self.current_stage_leaders[2] != self.rank:
                    # Receive metadata first
                    metadata = torch.zeros(4, dtype=torch.long, device="cpu")
                    
                    try:
                        dist.recv(metadata, src=self.current_stage_leaders[2])
                        
                        # Create tensor with the right size
                        tensor_shape = (metadata[0].item(), metadata[1].item(), 
                                        metadata[2].item(), metadata[3].item())
                        
                        # Create a tensor on CPU first
                        x_cpu = torch.zeros(tensor_shape, dtype=torch.float32, device="cpu")
                        
                        # Receive the actual tensor data
                        dist.recv(x_cpu, src=self.current_stage_leaders[2])
                        
                        # Move to the appropriate device
                        x = x_cpu.to(self.stage_devices[3])
                        
                        # CRITICAL: Create a fresh tensor with gradient tracking
                        x = x.detach().requires_grad_(True)
                    except Exception as e:
                        logger.error(f"Rank {self.rank}: Error in stage 3 receiving: {str(e)}")
                        return torch.zeros((original_batch_size, 10), device=self.stage_devices.get(3, "cpu"), requires_grad=True)
                
                # Process layer 3
                x = self.layer3(x)
                
                # Store output for this stage
                stage_outputs[3] = x.detach().clone() if in_recovery else x.clone()
                
                # Send output to stage 4 leader
                if dist.is_initialized() and self.current_stage_leaders[4] != self.rank:
                    # Use the same approach as before
                    x_cpu = x.detach().cpu()  # Detach to avoid gradient tracking issues
                    
                    # Prepare metadata
                    metadata = torch.tensor([
                        x_cpu.size(0),
                        x_cpu.size(1),
                        x_cpu.size(2),
                        x_cpu.size(3),
                    ], dtype=torch.long, device="cpu")
                    
                    try:
                        # Send metadata
                        dist.send(metadata, dst=self.current_stage_leaders[4])
                        
                        # Send tensor data
                        dist.send(x_cpu, dst=self.current_stage_leaders[4])
                    except Exception as e:
                        logger.error(f"Rank {self.rank}: Error in stage 3 sending: {str(e)}")
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error in stage 3 processing: {str(e)}")
                return torch.zeros((original_batch_size, 10), device=self.stage_devices.get(3, "cpu"), requires_grad=True)
        
        # Stage 4: Layer 4 + final layers
        if 4 in self.my_stages and self.current_stage_leaders[4] == self.rank:
            try:
                # If we're not the leader of the previous stage, receive input
                if 3 not in self.my_stages or self.current_stage_leaders[3] != self.rank:
                    # Receive metadata first
                    metadata = torch.zeros(4, dtype=torch.long, device="cpu")
                    
                    try:
                        dist.recv(metadata, src=self.current_stage_leaders[3])
                        
                        # Create tensor with the right size
                        tensor_shape = (metadata[0].item(), metadata[1].item(), 
                                        metadata[2].item(), metadata[3].item())
                        
                        # Create a tensor on CPU first
                        x_cpu = torch.zeros(tensor_shape, dtype=torch.float32, device="cpu")
                        
                        # Receive the actual tensor data
                        dist.recv(x_cpu, src=self.current_stage_leaders[3])
                        
                        # Move to the appropriate device
                        x = x_cpu.to(self.stage_devices[4])
                        
                        # CRITICAL: Create a fresh tensor with gradient tracking
                        x = x.detach().requires_grad_(True)
                    except Exception as e:
                        logger.error(f"Rank {self.rank}: Error in stage 4 receiving: {str(e)}")
                        return torch.zeros((original_batch_size, 10), device=self.stage_devices.get(4, "cpu"), requires_grad=True)
                
                # Process layer 4 - be careful with inplace operations
                x = self.layer4(x)
                
                # Process final layers - classification
                # Use functional calls instead of inplace methods when in recovery to avoid version issues
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                
                if self.include_top and hasattr(self, 'fc'):
                    x = self.fc(x)
                
                # Store output for this stage
                stage_outputs[4] = x.detach().clone() if in_recovery else x.clone()
                
                # Clear recovery mode flag after successful forward pass
                if in_recovery:
                    self._in_recovery = False
                    logger.info(f"Rank {self.rank}: Recovery complete, resuming normal processing")
                
                # Final stage - return the output
                return x
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error in stage 4 processing: {str(e)}")
                return torch.zeros((original_batch_size, 10), device=self.stage_devices.get(4, "cpu"), requires_grad=True)
        
        # If this rank is not responsible for the final output, return a dummy tensor
        # Make sure the dummy tensor has requires_grad=True to avoid backward errors
        dummy_output = torch.zeros((original_batch_size, 10), device=self.stage_devices.get(0, "cpu") if 0 in self.my_stages else "cpu", requires_grad=True)
        return dummy_output
