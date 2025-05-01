import os
import time
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging

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
                    {'leader': {'rank': 0, 'device': 'cuda:0'}, 'backups': [{'rank': 1, 'device': 'cuda:0'}]},
                    {'leader': {'rank': 2, 'device': 'cuda:0'}, 'backups': [{'rank': 3, 'device': 'cuda:0'}]},
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
            current_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
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
        """Loop to send and monitor heartbeats"""
        while True:
            try:
                # Send heartbeat
                if dist.is_initialized():
                    # Using a distributed tensor as heartbeat
                    if self.is_leader:
                        heartbeat = torch.tensor(
                            [time.time(), self.rank], dtype=torch.float
                        )
                        dist.broadcast(heartbeat, src=self.rank)

                    # Process leader failures and trigger failover if needed
                    for stage_idx, stage in enumerate(self.stage_config):
                        current_leader = self.current_stage_leaders[stage_idx]

                        # Skip checking stages where we're not involved
                        if stage_idx not in self.my_stages:
                            continue

                        # Check if we're a backup for this stage
                        if not self.is_leader_for_stages[
                            self.my_stages.index(stage_idx)
                        ]:
                            # Check if leader for this stage is alive
                            if current_leader in self.last_heartbeats:
                                last_beat_time = self.last_heartbeats[current_leader]
                                if (
                                    time.time() - last_beat_time
                                    > 3 * self.heartbeat_interval
                                ):
                                    logger.warning(
                                        f"Rank {self.rank}: Detected failure of leader (rank {current_leader}) for stage {stage_idx}!"
                                    )
                                    self._handle_leader_failure(stage_idx)

                # Record heartbeats from leaders
                for stage_idx, stage in enumerate(self.stage_config):
                    leader_rank = stage["leader"]["rank"]
                    if leader_rank != self.rank:  # Don't need to record own heartbeat
                        heartbeat = torch.zeros(2, dtype=torch.float)
                        dist.broadcast(heartbeat, src=leader_rank)
                        self.last_heartbeats[leader_rank] = heartbeat[0].item()

                time.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Rank {self.rank}: Error in heartbeat loop: {str(e)}")
                time.sleep(self.heartbeat_interval)

    def _handle_leader_failure(self, stage_idx):
        """Handle a leader failure by promoting a backup"""
        # Find the current leader
        current_leader = self.current_stage_leaders[stage_idx]

        # Get the stage configuration
        stage_config = self.stage_config[stage_idx]

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
                # Promote this backup to leader
                old_leader = self.current_stage_leaders[stage_idx]
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

                    # Broadcast to other ranks that I'm now the leader
                    if dist.is_initialized():
                        status_update = torch.tensor(
                            [stage_idx, backup_rank], dtype=torch.float
                        )
                        for r in range(self.world_size):
                            if r != self.rank:
                                dist.send(status_update, dst=r)

                    logger.info(f"Rank {self.rank}: Now leader for stage {stage_idx}")

                # Reshufffle backups (remove promoted one and add failed leader as last backup)
                new_backups = [
                    b for b in stage_config["backups"] if b["rank"] != backup_rank
                ]

                # Add the failed leader to end of backups if we want to recover it later
                # Note: We only do this if we know it will come back online
                # failed_leader_device = stage_config['leader']['device']
                # new_backups.append({'rank': old_leader, 'device': failed_leader_device})

                # Update the stage configuration
                self.stage_config[stage_idx] = {
                    "leader": {"rank": backup_rank, "device": backup["device"]},
                    "backups": new_backups,
                }

                return

        logger.error(f"Rank {self.rank}: No available backup for stage {stage_idx}!")

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
        logger.info(f"Rank {self.rank}: Saved checkpoint for stage {stage_idx}")

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

        # Stage 0: Initial convolution layer
        if 0 in self.my_stages and self.current_stage_leaders[0] == self.rank:
            # Move input to appropriate device
            x = x.to(self.stage_devices[0])

            # Process initial layers
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.scale1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            # Store output for this stage
            stage_outputs[0] = x

            # Send output to stage 1 leader
            if dist.is_initialized() and self.current_stage_leaders[1] != self.rank:
                # Send tensor to next stage's leader
                dist.send(x, dst=self.current_stage_leaders[1])

        # Stage 1: Layer 1
        if 1 in self.my_stages:
            if self.current_stage_leaders[1] == self.rank:
                # If we're not the leader of the previous stage, receive input
                if (
                    0 not in self.my_stages
                    or self.current_stage_leaders[0] != self.rank
                ):
                    x = torch.zeros_like(stage_outputs.get(0, x))
                    dist.recv(x, src=self.current_stage_leaders[0])

                # Process layer 1
                x = x.to(self.stage_devices[1])
                x = self.layer1(x)

                # Store output for this stage
                stage_outputs[1] = x

                # Send output to stage 2 leader
                if dist.is_initialized() and self.current_stage_leaders[2] != self.rank:
                    dist.send(x, dst=self.current_stage_leaders[2])

        # Stage 2: Layer 2
        if 2 in self.my_stages:
            if self.current_stage_leaders[2] == self.rank:
                # If we're not the leader of the previous stage, receive input
                if (
                    1 not in self.my_stages
                    or self.current_stage_leaders[1] != self.rank
                ):
                    # We don't know the exact shape, so we'll have to use Tensor.new_zeros
                    # with proper shape after receiving from previous stage
                    if dist.is_initialized():
                        # Receive tensor size first
                        tensor_sizes = torch.zeros(4, dtype=torch.long)
                        dist.recv(tensor_sizes, src=self.current_stage_leaders[1])

                        # Create empty tensor with right size
                        x = torch.zeros(tensor_sizes.tolist(), dtype=torch.float).to(
                            self.stage_devices[2]
                        )

                        # Receive actual tensor
                        dist.recv(x, src=self.current_stage_leaders[1])

                # Process layer 2
                x = x.to(self.stage_devices[2])
                x = self.layer2(x)

                # Store output for this stage
                stage_outputs[2] = x

                # Send output to stage 3 leader
                if dist.is_initialized() and self.current_stage_leaders[3] != self.rank:
                    # Send tensor size first
                    tensor_sizes = torch.tensor(list(x.shape), dtype=torch.long)
                    dist.send(tensor_sizes, dst=self.current_stage_leaders[3])

                    # Send actual tensor
                    dist.send(x, dst=self.current_stage_leaders[3])

        # Stage 3: Layer 3
        if 3 in self.my_stages:
            if self.current_stage_leaders[3] == self.rank:
                # If we're not the leader of the previous stage, receive input
                if (
                    2 not in self.my_stages
                    or self.current_stage_leaders[2] != self.rank
                ):
                    if dist.is_initialized():
                        # Receive tensor size first
                        tensor_sizes = torch.zeros(4, dtype=torch.long)
                        dist.recv(tensor_sizes, src=self.current_stage_leaders[2])

                        # Create empty tensor with right size
                        x = torch.zeros(tensor_sizes.tolist(), dtype=torch.float).to(
                            self.stage_devices[3]
                        )

                        # Receive actual tensor
                        dist.recv(x, src=self.current_stage_leaders[2])

                # Process layer 3
                x = x.to(self.stage_devices[3])
                x = self.layer3(x)

                # Store output for this stage
                stage_outputs[3] = x

                # Send output to stage 4 leader
                if dist.is_initialized() and self.current_stage_leaders[4] != self.rank:
                    # Send tensor size first
                    tensor_sizes = torch.tensor(list(x.shape), dtype=torch.long)
                    dist.send(tensor_sizes, dst=self.current_stage_leaders[4])

                    # Send actual tensor
                    dist.send(x, dst=self.current_stage_leaders[4])

        # Stage 4: Layer 4 + final layers
        if 4 in self.my_stages:
            if self.current_stage_leaders[4] == self.rank:
                # If we're not the leader of the previous stage, receive input
                if (
                    3 not in self.my_stages
                    or self.current_stage_leaders[3] != self.rank
                ):
                    if dist.is_initialized():
                        # Receive tensor size first
                        tensor_sizes = torch.zeros(4, dtype=torch.long)
                        dist.recv(tensor_sizes, src=self.current_stage_leaders[3])

                        # Create empty tensor with right size
                        x = torch.zeros(tensor_sizes.tolist(), dtype=torch.float).to(
                            self.stage_devices[4]
                        )

                        # Receive actual tensor
                        dist.recv(x, src=self.current_stage_leaders[3])

                # Process layer 4
                x = x.to(self.stage_devices[4])
                x = self.layer4(x)

                # Process final layers
                if self.include_top:
                    x = self.avgpool(x)
                    x = torch.flatten(x, 1)
                    x = self.fc(x)
                else:
                    # If not including the top FC layer, you can optionally apply
                    # global pooling.
                    if self.pooling == "avg":
                        x = F.adaptive_avg_pool2d(x, (1, 1))
                        x = torch.flatten(x, 1)
                    elif self.pooling == "max":
                        x = F.adaptive_max_pool2d(x, (1, 1))
                        x = torch.flatten(x, 1)

                # Store output for this stage
                stage_outputs[4] = x

        # Only return a value if we're the leader of the final stage
        if 4 in self.my_stages and self.current_stage_leaders[4] == self.rank:
            # Save checkpoint after successful forward pass
            self._save_stage_checkpoint(4)
            return x

        # If we get here, we're not responsible for the final output
        # Return a dummy tensor - in a real system, we'd need proper coordination
        dummy_output = torch.zeros((1, 1000)).to(self.stage_devices.get(4, "cpu"))
        return dummy_output
