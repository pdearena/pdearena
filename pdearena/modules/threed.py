import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable
from .activations import ACTIVATION_REGISTRY
from .fourier import SpectralConv3d


class FourierBasicBlock3D(nn.Module):
    """Basic 3d FNO ResNet building block consisting of two 3d convolutional layers,
    two 3d SpectralConv layers and skip connections."""

    expansion: int = 1

    def __init__(
            self, 
            in_planes: int, 
            planes: int,
            modes1: int,
            modes2: int,
            modes3: int, 
            stride: int = 1, 
            activation: str = "gelu"
            ):
        """Initialize basic 3d FNO ResNet building block
        Args:
            in_planes (int): input channels
            planes (int): output channels
            stride (int, optional): stride of 2d convolution. Defaults to 1.
        """
        super().__init__()

        self.fourier1 = SpectralConv3d(
            in_planes, planes, modes1=modes1, modes2=modes2, modes3=modes3
        )
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros")
        self.fourier2 = SpectralConv3d(
            planes, planes, modes1=modes1, modes2=modes2, modes3=modes3
        )
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=1, padding=0, padding_mode="zeros")

        # Shortcut connection, batchnorm removed
        # So far shortcut connections are not helping
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1)
            )
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of basic 3d Fourier ResNet building block.
        Args:
            x (torch.Tensor): input of shape [batch, in_planes, x, y, z]
        Returns:
            torch.Tensor: output of shape [batch, planes, x, y, z]
        """
        x1 = self.fourier1(x)
        x2 = self.conv1(x)
        out = self.activation(x1 + x2)

        x1 = self.fourier2(out)
        x2 = self.conv2(out)
        out = x1 + x2
        # out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet3D(nn.Module):
    """3d ResNet model, difference to default ResNet architectures is that
    spatial resolution and channels (in_planes) stay constant throughout the network."""

    padding = 2  # no periodic

    def __init__(
        self,
        block: nn.Module,
        num_blocks: list,
        time_history: int,
        time_future: int,
        hidden_channels: int = 64,
        activation: str = "gelu",
        diffmode: bool = False,
    ):
        """Initialize 3d ResNet model

        Args:
            block (nn.Module): basic 3d ResNet building block
            num_blocks (list): list of basic building blocks per layer
            time_history (int): input timesteps
            time_future (int): prediction timesteps
            hidden_channels (int): hidden channels in the ResNet blocks
        """
        super(ResNet3D, self).__init__()

        self.diffmode = diffmode
        self.in_planes = hidden_channels
        self.conv_in1 = nn.Conv3d(
            time_history * 6,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv3d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = nn.Conv3d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv3d(
            self.in_planes,
            time_future * 6,
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    block,
                    self.in_planes,
                    num_blocks[i],
                    stride=1,
                    activation=activation,
                )
                for i in range(len(num_blocks))
            ]
        )
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def _make_layer(
        self, 
        block: Callable, 
        planes: int, 
        num_blocks: int, 
        stride: int, 
        activation: str,
    ) -> nn.Sequential:
        """Build 3d ResNet layers out of basic building blocks.

        Args:
            block (nn.Module): basic 3d ResNet building block
            planes (int): input channels
            num_blocks (int): number of basic 3d ResNet building blocks in one layer
            stride (int): stride

        Returns:
            nn.Sequential: 3d ResNet layer as nn.Sequential
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, activation=activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def __repr__(self):
        return "ResNet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResNet in 3 spatial dimensions
        consisting of embedding layer(s), ResNet building blogs and output layer.
        Args:
            x (torch.Tensor): input tensor of shape [b, time_history, 6, x, y, z]
        Returns:
            torch.Tensor: output tensor of shape [b, time_future, 6, x, y, z]
        """
        assert x.dim() == 6
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])
        prev = x
        x = self.activation(self.conv_in1(x.float()))
        x = self.activation(self.conv_in2(x.float()))

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        for layer in self.layers:
            x = layer(x)
        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.conv_out1(x)
        x = self.conv_out2(x)

        if self.diffmode:
            x = x + prev[:, -1:, ...].detach()
        return x.reshape(orig_shape[0], -1, 6, *orig_shape[3:])