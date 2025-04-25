import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class Scale(nn.Module):
    """
    A layer that learns a set of scale (gamma) and shift (beta) parameters.
    This mimics the Keras Scale layer after BatchNormalization.
    """

    def __init__(self, num_features):
        super(Scale, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x is assumed to be in (batch, channels, height, width) order.
        return x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


class Bottleneck(nn.Module):
    expansion = 4  # Output channels are `planes * expansion`

    def __init__(self, inplanes, planes, stride=1, downsample=None, device=None):
        """
        Args:
            inplanes: Number of input channels.
            planes: Number of channels for the intermediate conv layers.
            stride: Stride for the 3x3 convolution.
            downsample: A downsampling layer for the shortcut branch, if needed.
            device: The device to place this module on.
        """
        super(Bottleneck, self).__init__()
        eps = 1.1e-5  # epsilon used in BatchNorm layers

        # Specify device if provided
        self.device = device

        # Move module to specified device if provided
        if device is not None:
            self.to(device)

        # 1x1 conv branch
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=eps)
        self.scale1 = Scale(planes)

        # 3x3 conv branch
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, eps=eps)
        self.scale2 = Scale(planes)

        # 1x1 conv branch
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, eps=eps)
        self.scale3 = Scale(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # Move input to this module's device if needed
        if self.device is not None and x.device != self.device:
            x = x.to(self.device)

        identity = x

        # First conv -> BN -> Scale -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.scale1(out)
        out = self.relu(out)

        # Second conv -> BN -> Scale -> ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.scale2(out)
        out = self.relu(out)

        # Third conv -> BN -> Scale
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.scale3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Add shortcut and apply final ReLU
        out += identity
        out = self.relu(out)

        return out


class ModelParallelResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        include_top=True,
        pooling=None,
        devices=None,
    ):
        """
        Args:
            block: Block type (here, Bottleneck).
            layers: List with the number of blocks in each of the 4 layers.
                    For ResNet-50, this should be [3, 4, 6, 3].
            num_classes: Number of output classes.
            include_top: Whether to include the final FC layer.
            pooling: Optional pooling mode when include_top is False.
                     Options are 'avg' or 'max'.
            devices: List of devices to distribute the model across.
                    If None, the model will use a single device (the current device).
        """
        super(ModelParallelResNet, self).__init__()
        self.inplanes = 64
        self.include_top = include_top
        self.pooling = pooling
        eps = 1.1e-5

        # Handle device setup
        if devices is None:
            # Default to current device (which should be set by distributed init)
            current_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.devices = [current_device]
        else:
            self.devices = [torch.device(d) for d in devices]

        # Use auto-detection only if we're in a single-machine context
        # (Otherwise, distributed setup should handle device assignment)
        if not devices and not dist.is_initialized() and torch.cuda.device_count() > 1:
            self.devices = [
                torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
            ]
            print(
                f"Automatically using {len(self.devices)} CUDA devices for model parallelism"
            )

        # Assign devices to different parts of the network
        self.input_device = self.devices[0]  # First device for input layers

        # Initial convolutional layer (always on the first device)
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        ).to(self.input_device)
        self.bn1 = nn.BatchNorm2d(64, eps=eps).to(self.input_device)
        self.scale1 = Scale(64).to(self.input_device)
        self.relu = nn.ReLU(inplace=True).to(self.input_device)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(
            self.input_device
        )

        # Build ResNet layers (stages) on different devices
        num_layers = len(layers)
        devices_per_stage = min(len(self.devices), num_layers)

        # Distribute devices evenly across the 4 stages
        stage_devices = []
        if len(self.devices) == 1:
            # If only one device, use it for all stages
            stage_devices = [self.devices[0]] * num_layers
        else:
            # Distribute devices according to computational load
            # Layer 3 is the deepest (most blocks), so give it more devices if available
            if len(self.devices) >= 4:
                # If we have enough devices, assign one to each stage
                stage_devices = [
                    self.devices[i % len(self.devices)] for i in range(num_layers)
                ]
            elif len(self.devices) == 3:
                # With 3 devices, assign the third device to both layer2 and layer3
                stage_devices = [
                    self.devices[0],
                    self.devices[1],
                    self.devices[2],
                    self.devices[1],
                ]
            elif len(self.devices) == 2:
                # With 2 devices, split the model in half
                stage_devices = [
                    self.devices[0],
                    self.devices[0],
                    self.devices[1],
                    self.devices[1],
                ]

        # Build each layer with appropriate device assignment
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=1, device=stage_devices[0]
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, device=stage_devices[1]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, device=stage_devices[2]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, device=stage_devices[3]
        )

        # Final classification layers on the last device
        self.output_device = self.devices[-1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(self.output_device)

        if include_top:
            self.fc = nn.Linear(512 * block.expansion, num_classes).to(
                self.output_device
            )
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

    def forward(self, x):
        # Move input to the first device
        x = x.to(self.input_device)

        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.scale1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks (these handle device transitions internally)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Move to output device for final processing
        x = x.to(self.output_device)

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

        return x


def init_distributed_mode(args=None):
    """
    Initialize distributed training environment

    Args:
        args: Optional arguments that may contain:
            dist_url: URL used to set up distributed training
            dist_backend: Backend to use for distributed training
            world_size: Number of distributed processes
            rank: Rank of the current process
    """
    # Default settings
    if args is None:
        args = {
            "dist_url": "env://",
            "dist_backend": "nccl",
            "world_size": None,
            "rank": None,
        }

    # Check if already initialized
    if dist.is_initialized():
        print("Distributed already initialized, skipping init")
        return

    # Read environment variables set by the launcher
    if args["world_size"] is None:
        args["world_size"] = int(os.environ.get("WORLD_SIZE", 1))
    if args["rank"] is None:
        args["rank"] = int(os.environ.get("RANK", 0))

    # Handle local rank (GPU to use on this node)
    if "SLURM_PROCID" in os.environ:
        # SLURM setup (for supercomputing clusters)
        args["rank"] = int(os.environ["SLURM_PROCID"])
        args["gpu"] = args["rank"] % torch.cuda.device_count()
    elif "LOCAL_RANK" in os.environ:
        # Typical PyTorch launcher setup
        args["gpu"] = int(os.environ["LOCAL_RANK"])
    else:
        # Default to first GPU
        args["gpu"] = 0

    # Initialize distributed process group
    if args["world_size"] > 1:
        print(f"Initializing distributed with rank {args['rank']}/{args['world_size']}")

        # Set the device
        if torch.cuda.is_available():
            torch.cuda.set_device(args["gpu"])

        # Initialize the process group
        dist.init_process_group(
            backend=args["dist_backend"],
            init_method=args["dist_url"],
            world_size=args["world_size"],
            rank=args["rank"],
        )

        # Synchronize all processes
        dist.barrier()

        print(
            f"Process group initialized: rank {dist.get_rank()}/{dist.get_world_size()}"
        )
    else:
        print("Not using distributed mode")

    # Return local GPU device
    return args["gpu"]


def resnet50_distributed(
    pretrained=False,
    num_classes=1000,
    include_top=True,
    pooling=None,
    devices=None,
    ddp=False,
):
    """
    Constructs a model-parallel and distributed ResNet-50 model.

    Args:
        pretrained: If True, load pretrained ImageNet weights (if available).
        num_classes: Number of classes for the final classification.
        include_top: Whether to include the final fully-connected layer.
        pooling: Optional pooling mode for feature extraction when include_top is False.
                 Options are 'avg' or 'max'.
        devices: List of devices to distribute the model across locally.
                Examples: ['cuda:0', 'cuda:1'] or ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
                If None, automatically use all available CUDA devices.
        ddp: Whether to wrap the model with DistributedDataParallel for multi-machine training.

    Returns:
        An instance of the model-parallel and distributed ResNet50 model.
    """
    # Create base model with model parallelism
    model = ModelParallelResNet(
        Bottleneck,
        [3, 4, 6, 3],  # ResNet50 configuration
        num_classes=num_classes,
        include_top=include_top,
        pooling=pooling,
        devices=devices,
    )

    # Load pretrained weights if requested
    if pretrained:
        # Code to load pretrained weights would go here.
        pass

    # Wrap with DistributedDataParallel for multi-machine training if requested
    if ddp and dist.is_initialized():
        # Get device from current process
        local_rank = dist.get_rank() % torch.cuda.device_count()
        local_device = torch.device(f"cuda:{local_rank}")

        # Move model to the appropriate device for DDP
        # Note: With model parallelism, the model is already distributed across devices,
        # so we need to be careful with DDP. Here we're just ensuring the model's initial
        # parameters are on the correct device for this process.
        model.input_device = local_device

        # Wrap model with DDP
        # Note: find_unused_parameters=True is needed because with model parallelism,
        # not all parameters are used in every forward pass on every device
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        print(f"Model wrapped with DistributedDataParallel on rank {dist.get_rank()}")

    return model


def launch_distributed_workers(
    main_fn, world_size, num_gpus_per_machine, machine_rank, dist_url, args=()
):
    """
    Launch multiple distributed workers for multi-machine training.

    Args:
        main_fn: The main function to run on each worker.
        world_size: Total number of processes across all machines.
        num_gpus_per_machine: Number of GPUs per machine.
        machine_rank: The rank of this machine.
        dist_url: URL for distributed coordination (e.g., 'tcp://master.ip:port')
        args: Arguments to pass to main_fn.
    """
    import torch.multiprocessing as mp

    assert world_size >= 1

    if world_size > 1:
        # Calculate total processes per machine
        processes_per_machine = min(num_gpus_per_machine, world_size)

        print(
            f"Launching {processes_per_machine} processes on machine with rank {machine_rank}"
        )

        # Spawn processes
        mp.spawn(
            _distributed_worker,
            nprocs=processes_per_machine,
            args=(
                main_fn,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                processes_per_machine,
                dist_url,
                args,
            ),
            daemon=False,
        )
    else:
        # Single process mode
        main_fn(*args)


def _distributed_worker(
    local_rank,
    main_fn,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    processes_per_machine,
    dist_url,
    args,
):
    """Helper function for launch_distributed_workers."""
    # Calculate global rank
    global_rank = machine_rank * processes_per_machine + local_rank

    # Initialize distributed environment
    try:
        # Set environment variables for distributed training
        os.environ["RANK"] = str(global_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

        dist_args = {
            "dist_url": dist_url,
            "world_size": world_size,
            "rank": global_rank,
        }

        # Initialize process group
        gpu = init_distributed_mode(dist_args)

        # Call the main function with the current process's GPU
        main_fn(gpu, *args)
    except Exception as e:
        print(f"Error in worker {global_rank}: {e}")
        raise e


# Example usage (single machine, multiple GPUs)
def train_single_machine(gpu, batch_size=32, epochs=10):
    """Example for single machine, multiple GPUs"""
    # Set device for this process
    torch.cuda.set_device(gpu)

    # Create model using only local devices (model parallelism)
    if torch.cuda.device_count() > 1:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        model = resnet50_distributed(pretrained=False, devices=devices)
        print(
            f"Created model parallel ResNet50 distributed across {len(devices)} devices"
        )
    else:
        model = resnet50_distributed(pretrained=False, devices=[f"cuda:{gpu}"])
        print(f"Created model on single GPU {gpu}")

    # Create dummy dataset and train (in practice, replace with real data)
    dummy_input = torch.randn(batch_size, 3, 224, 224).cuda(gpu)
    dummy_target = torch.randint(0, 1000, (batch_size,)).cuda(gpu)

    # Simple training loop
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Forward pass
        output = model(dummy_input)
        loss = criterion(output, dummy_target)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Example usage (multiple machines)
def train_multi_machine(gpu, batch_size=32, epochs=10):
    """Example for multi-machine training"""
    # Initialize distributed mode (if not already done)
    if not dist.is_initialized():
        dist_args = {
            "dist_url": "env://",
            "dist_backend": "nccl",
            "world_size": int(os.environ.get("WORLD_SIZE", 1)),
            "rank": int(os.environ.get("RANK", 0)),
        }
        init_distributed_mode(dist_args)

    # Set device for this process
    torch.cuda.set_device(gpu)

    # With distributed training, each process will handle a portion of the data
    # Model parallelism is used locally within each machine
    if torch.cuda.device_count() > 1:
        # Use model parallelism locally within the machine
        local_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        model = resnet50_distributed(pretrained=False, devices=local_devices, ddp=True)
        print(
            f"Created model with both model parallelism ({len(local_devices)} GPUs) and DDP"
        )
    else:
        # Just use distributed data parallelism
        model = resnet50_distributed(
            pretrained=False, devices=[f"cuda:{gpu}"], ddp=True
        )
        print(f"Created model with DDP on GPU {gpu}")

    # In distributed training, create distributed sampler for your dataset
    # For this example, we'll use dummy data
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    batch_size_per_gpu = batch_size // world_size
    print(f"Rank {rank}/{world_size} using batch size {batch_size_per_gpu}")

    # Create dummy dataset for this example
    dummy_input = torch.randn(batch_size_per_gpu, 3, 224, 224).cuda(gpu)
    dummy_target = torch.randint(0, 1000, (batch_size_per_gpu,)).cuda(gpu)

    # Create optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Simple training loop
    for epoch in range(epochs):
        # Synchronize all processes at the beginning of each epoch
        if dist.is_initialized():
            dist.barrier()

        # Forward pass
        output = model(dummy_input)
        loss = criterion(output, dummy_target)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print only from the main process to avoid cluttering the output
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distributed ResNet50 Training")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="Training mode: single machine or multi-machine",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of total processes/GPUs to use",
    )
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="Rank of this machine (0-indexed)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs per machine",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="tcp://localhost:23456",
        help="URL used to set up distributed training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Total batch size across all GPUs/processes",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )

    args = parser.parse_args()

    if args.mode == "single":
        # Single machine mode
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                # Multiple GPUs on a single machine
                print(f"Using {torch.cuda.device_count()} GPUs on a single machine")
                launch_distributed_workers(
                    train_single_machine,
                    args.num_gpus,  # world_size = number of GPUs on this machine
                    args.num_gpus,
                    0,  # machine_rank is 0 for single machine
                    args.dist_url,
                    args=(args.batch_size, args.epochs),
                )
            else:
                # Single GPU
                print("Using a single GPU")
                train_single_machine(0, args.batch_size, args.epochs)
        else:
            # No GPU available
            print("No GPU available, falling back to CPU")
            train_single_machine(0, args.batch_size, args.epochs)

    elif args.mode == "multi":
        # Multi-machine mode
        print(
            f"Launching as machine {args.machine_rank} in a {args.world_size}-machine job"
        )
        print(
            f"Using {args.num_gpus} GPUs per machine, {args.world_size * args.num_gpus} GPUs total"
        )

        # For multi-machine training, launch a separate process for each GPU
        launch_distributed_workers(
            train_multi_machine,
            args.world_size * args.num_gpus,  # Total processes
            args.num_gpus,  # GPUs per machine
            args.machine_rank,  # Current machine's rank
            args.dist_url,  # Coordination URL
            args=(args.batch_size, args.epochs),
        )
