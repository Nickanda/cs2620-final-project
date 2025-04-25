import torch
import torch.nn as nn
import torch.nn.functional as F


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
                    If None, the model will use a single device (CPU or first available GPU).
        """
        super(ModelParallelResNet, self).__init__()
        self.inplanes = 64
        self.include_top = include_top
        self.pooling = pooling
        eps = 1.1e-5

        # Handle device setup
        if devices is None:
            # Default to first available CUDA device or CPU if CUDA not available
            self.devices = [
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            ]
        else:
            self.devices = [torch.device(d) for d in devices]

        # Automatically detect available devices if none are provided
        if not devices and torch.cuda.device_count() > 1:
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


def resnet50_parallel(
    pretrained=False, num_classes=1000, include_top=True, pooling=None, devices=None
):
    """
    Constructs a model-parallel ResNet-50 model.

    Args:
        pretrained: If True, load pretrained ImageNet weights (if available).
        num_classes: Number of classes for the final classification.
        include_top: Whether to include the final fully-connected layer.
        pooling: Optional pooling mode for feature extraction when include_top is False.
                 Options are 'avg' or 'max'.
        devices: List of devices to distribute the model across.
                Examples: ['cuda:0', 'cuda:1'] or ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
                If None, automatically use all available CUDA devices.

    Returns:
        An instance of the model-parallel ResNet50 model.
    """
    model = ModelParallelResNet(
        Bottleneck,
        [3, 4, 6, 3],  # ResNet50 configuration
        num_classes=num_classes,
        include_top=include_top,
        pooling=pooling,
        devices=devices,
    )
    if pretrained:
        # Code to load pretrained weights would go here.
        pass

    return model


# Example usage:
if __name__ == "__main__":
    # Check available devices
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA devices")

        if num_gpus > 1:
            # Use all available GPUs
            devices = [f"cuda:{i}" for i in range(num_gpus)]
            model = resnet50_parallel(pretrained=False, devices=devices)
            print(
                f"Created model parallel ResNet50 distributed across {len(devices)} devices"
            )
        else:
            # Only one GPU available, still use model parallel code but with single device
            devices = ["cuda:0"]
            model = resnet50_parallel(pretrained=False, devices=devices)
            print("Created model parallel ResNet50 on single GPU")
    else:
        # Fallback to CPU
        print("No CUDA devices found, using CPU")
        model = resnet50_parallel(pretrained=False, devices=["cpu"])

    # Create an example input tensor
    x = torch.randn(1, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        logits = model(x)

    print(f"Output shape: {logits.shape}")  # Should be [1, 1000] if include_top is True

    # Print the devices used for different parts of the model
    print("\nDevices used by model parts:")
    print(f"Input processing: {model.input_device}")
    print(f"Layer 1: {next(model.layer1.parameters()).device}")
    print(f"Layer 2: {next(model.layer2.parameters()).device}")
    print(f"Layer 3: {next(model.layer3.parameters()).device}")
    print(f"Layer 4: {next(model.layer4.parameters()).device}")
    print(f"Output processing: {model.output_device}")
