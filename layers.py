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

        # Move module to specified device if provided
        if device is not None:
            self.to(device)

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
