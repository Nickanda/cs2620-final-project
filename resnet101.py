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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Args:
            inplanes: Number of input channels.
            planes: Number of channels for the intermediate conv layers.
            stride: Stride for the 3x3 convolution.
            downsample: A downsampling layer for the shortcut branch, if needed.
        """
        super(Bottleneck, self).__init__()
        eps = 1.1e-5  # epsilon used in BatchNorm layers

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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, include_top=True, pooling=None):
        """
        Args:
            block: Block type (here, Bottleneck).
            layers: List with the number of blocks in each of the 4 layers.
                    For ResNet-101, this should be [3, 4, 23, 3].
            num_classes: Number of output classes.
            include_top: Whether to include the final FC layer.
            pooling: Optional pooling mode when include_top is False.
                     Options are 'avg' or 'max'.
        """
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.include_top = include_top
        self.pooling = pooling
        eps = 1.1e-5

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=eps)
        self.scale1 = Scale(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build ResNet layers (stages)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling (adaptable to any input size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if include_top:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
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

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Build one stage of the ResNet model.

        Args:
            block: Block type.
            planes: Number of channels for the intermediate layers.
            blocks: Number of blocks to stack.
            stride: Stride for the first block.
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

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.scale1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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


def resnet101(pretrained=False, num_classes=1000, include_top=True, pooling=None):
    """
    Constructs a ResNet-101 model.

    Args:
        pretrained: If True, load pretrained ImageNet weights (if available).
        num_classes: Number of classes for the final classification.
        include_top: Whether to include the final fully-connected layer.
        pooling: Optional pooling mode for feature extraction when include_top is False.
                 Options are 'avg' or 'max'.

    Returns:
        An instance of the ResNet101 model.
    """
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        include_top=include_top,
        pooling=pooling,
    )
    if pretrained:
        # Code to load pretrained weights would go here.
        # (You could adapt the torchvision.models.utils.load_state_dict_from_url API.)
        pass

    return model


# Example usage:
if __name__ == "__main__":
    model = resnet101(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    logits = model(x)
    print(
        logits.shape
    )  # Expected output: torch.Size([1, 1000]) if include_top is True.
