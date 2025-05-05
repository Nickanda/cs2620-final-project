"""
layers.py

Standalone PyTorch implementation of two ResNet-style building blocks:

1. **Scale**
   • Learns per-channel affine parameters (γ, β) and applies
     *x ↦ γ·x + β* after a preceding BatchNorm layer—mirroring Keras’s
     post-BatchNorm *Scale* layer.
   • Accepts an integer `num_features`, registers the learnable
     parameters, and reshapes them internally so the forward pass
     works with `(N, C, H, W)` tensors on CPU, CUDA, or Apple M-series
     (MPS).

2. **Bottleneck**
   • Classical three-layer ResNet bottleneck (1×1 → 3×3 → 1×1) with
     `expansion = 4`.
   • Each convolution is followed by BatchNorm **and** Scale, then
     ReLU (except the final 1×1, which omits ReLU before adding shortcut).
   • Supports `stride`, an optional `downsample` module for dimension
     matching, and an optional `device` argument to place the entire
     block (and inputs) on a specific device.
   • Carries out residual addition plus a tail ReLU, returning an
     output tensor whose channel dimension is `planes × expansion`.

These components are drop-in replacements for the corresponding parts of
the original ResNet paper.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class Scale(nn.Module):
    """
    A layer that learns a set of scale (gamma) and shift (beta) parameters.
    This mimics the Keras Scale layer after BatchNormalization.

    Args:
        num_features (int): Number of input features/channels to scale.
    """

    def __init__(self, num_features):
        super(Scale, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        """
        Apply scaling and shifting to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Scaled and shifted tensor of the same shape as input.
        """
        # x is assumed to be in (batch, channels, height, width) order.
        return x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


class Bottleneck(nn.Module):
    expansion = 4  # Output channels are `planes * expansion`

    def __init__(self, inplanes, planes, stride=1, downsample=None, device=None):
        """
        Initialize a bottleneck block for ResNet.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of channels for the intermediate convolutional layers.
            stride (int, optional): Stride for the 3x3 convolution. Default is 1.
            downsample (nn.Module, optional): A downsampling layer for the shortcut branch,
                used when changing dimensions or stride. Default is None.
            device (torch.device, optional): The device to place this module on (cuda, mps, or cpu).
                Default is None.
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
        """
        Forward pass through the bottleneck block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the bottleneck block.
        """
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
