from typing import Callable, Union

import torch
from cliffordlayers.models.basic.custom_layers import (
    CliffordConv3dMaxwellDecoder,
    CliffordConv3dMaxwellEncoder,
)
from torch import nn
from torch.nn import functional as F

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
        activation: str = "gelu",
        norm: bool = False,
    ):
        """Initialize basic 3d FNO ResNet building block
        Args:
            in_planes (int): Input channels
            planes (int): Output channels
            modes1 (int): Fourier modes for x direction.
            modes2 (int): Fourier modes for y direction.
            modes3 (int): Fourier modes for z direction.
            stride (int, optional): stride of 2d convolution. Defaults to 1.
            norm (bool): Wether to use normalization. Defaults to False.
        """
        super().__init__()

        self.fourier1 = SpectralConv3d(in_planes, planes, modes1=modes1, modes2=modes2, modes3=modes3)
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros")
        self.fourier2 = SpectralConv3d(planes, planes, modes1=modes1, modes2=modes2, modes3=modes3)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=1, padding=0, padding_mode="zeros")

        # Shortcut connection, batchnorm removed
        # So far shortcut connections are not helping
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1))
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        if norm:
            raise NotImplementedError(f"Normalization for FourierBasicBlock3D not implemented")

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


class MaxwellResNet3D(nn.Module):
    """3d ResNet model for Maxwell equations, difference to default ResNet architectures is that
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
        super().__init__()

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


class CliffordMaxwellResNet3D(nn.Module):
    """3D building block for Clifford architectures with ResNet backbone network.
    The backbone networks follows these three steps:
        1. Clifford vector+bivector encoding.
        2. Basic blocks as provided.
        3. Clifford vector+bivector decoding.

    Args:
        g (Union[tuple, list, torch.Tensor]): Signature of Clifford algebra.
        block (nn.Module): Choice of basic blocks.
        num_blocks (list): List of basic blocks in each residual block.
        time_history (int): Number of input timesteps.
        time_future (int): Number of output timesteps.
        hidden_channels (int): Number of hidden channels.
        activation (Callable, optional): Activation function. Defaults to F.gelu.
        norm (bool, optional): Whether to use Clifford (group) normalization. Defaults to False.
        num_groups (int, optional): Number of groups when using Clifford (group) normalization. Defaults to 1.
    """

    # For periodic boundary conditions, set padding = 0.
    padding = 2

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        block: nn.Module,
        num_blocks: list,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        num_groups: int = 1,
        diffmode: bool = False,
    ):
        super().__init__()

        # Encoding and decoding layers.
        self.encoder = CliffordConv3dMaxwellEncoder(
            g,
            in_channels=time_history,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
        )
        self.decoder = CliffordConv3dMaxwellDecoder(
            g,
            in_channels=hidden_channels,
            out_channels=time_future,
            kernel_size=1,
            padding=0,
        )

        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        # Residual blocks.
        self.layers = nn.ModuleList(
            [
                self._make_basic_block(
                    g,
                    block,
                    hidden_channels,
                    num_blocks[i],
                    activation=self.activation,
                    norm=norm,
                    num_groups=num_groups,
                )
                for i in range(len(num_blocks))
            ]
        )

    def _make_basic_block(
        self,
        g,
        block: nn.Module,
        hidden_channels: int,
        num_blocks: int,
        activation: Callable,
        norm: bool,
        num_groups: int,
    ) -> nn.Sequential:
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                block(
                    g,
                    hidden_channels,
                    hidden_channels,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 6

        # Get data into shape where I dimension is last.
        B_dim, C_dim, I_dim, *D_dims = range(len(x.shape))
        x = x.permute(B_dim, C_dim, *D_dims, I_dim)

        # Encoding layer.
        x = self.encoder(self.activation(x))

        # Embed for non-periodic boundaries.
        if self.padding > 0:
            B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
            x = x.permute(B_dim, I_dim, C_dim, *D_dims)
            x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])
            B_dim, I_dim, C_dim, *D_dims = range(len(x.shape))
            x = x.permute(B_dim, C_dim, *D_dims, I_dim)

        # Apply residual layers.
        for layer in self.layers:
            x = layer(x)

        # Decoding layer.
        if self.padding > 0:
            B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
            x = x.permute(B_dim, I_dim, C_dim, *D_dims)
            x = x[..., : -self.padding, : -self.padding, : -self.padding]
            B_dim, I_dim, C_dim, *D_dims = range(len(x.shape))
            x = x.permute(B_dim, C_dim, *D_dims, I_dim)

        # Output layer.
        x = self.decoder(x)

        # Get data back to normal shape.
        B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
        x = x.permute(B_dim, C_dim, I_dim, *D_dims)

        return x
