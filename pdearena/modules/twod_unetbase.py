# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
from torch import nn

from .activations import ACTIVATION_REGISTRY


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu") -> None:
        super().__init__()
        self.activation = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if norm:
            # Original used BatchNorm2d
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.activation(self.norm1(self.conv1(x)))
        h = self.activation(self.norm2(self.conv2(h)))
        return h


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu") -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, num_groups, norm, activation)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        h = self.pool(x)
        h = self.conv(h)
        return h


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, num_groups, norm, activation)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h = self.up(x1)
        h = torch.cat([x2, h], dim=1)
        h = self.conv(h)
        return h


class Unetbase(nn.Module):
    """Our interpretation of the original U-Net architecture.

    Uses [torch.nn.GroupNorm][] instead of [torch.nn.BatchNorm2d][]. Also there is no `BottleNeck` block.

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input.
        time_future (int): Number of time steps in the output.
        hidden_channels (int): Number of channels in the hidden layers.
        activation (str): Activation function to use. One of ["gelu", "relu", "silu"].
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        activation="gelu",
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels
        self.activation = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 2)
        n_channels = hidden_channels
        self.image_proj = ConvBlock(insize, n_channels, activation=activation)

        self.down = nn.ModuleList(
            [
                Down(n_channels, n_channels * 2, activation=activation),
                Down(n_channels * 2, n_channels * 4, activation=activation),
                Down(n_channels * 4, n_channels * 8, activation=activation),
                Down(n_channels * 8, n_channels * 16, activation=activation),
            ]
        )
        self.up = nn.ModuleList(
            [
                Up(n_channels * 16, n_channels * 8, activation=activation),
                Up(n_channels * 8, n_channels * 4, activation=activation),
                Up(n_channels * 4, n_channels * 2, activation=activation),
                Up(n_channels * 2, n_channels, activation=activation),
            ]
        )
        out_channels = time_future * (self.n_output_scalar_components + self.n_output_vector_components * 2)
        # should there be a final norm too? but we aren't doing "prenorm" in the original
        self.final = nn.Conv2d(n_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])
        h = self.image_proj(x)

        x1 = self.down[0](h)
        x2 = self.down[1](x1)
        x3 = self.down[2](x2)
        x4 = self.down[3](x3)
        x = self.up[0](x4, x3)
        x = self.up[1](x, x2)
        x = self.up[2](x, x1)
        x = self.up[3](x, h)

        x = self.final(x)
        return x.reshape(
            orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
        )
