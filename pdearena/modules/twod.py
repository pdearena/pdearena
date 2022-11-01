# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch

from torch import nn
from torch.nn import functional as F
from .fourier import (
    SpectralConv2d,
)


#######################################################################
#######################################################################
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.bn1 = nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(num_groups, num_channels=planes)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, self.expansion * planes) if norm else nn.Identity(),
            )
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out = self.activation(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = out + self.shortcut(x)
        # out = self.activation(out)
        return out


class DilatedBasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()

        self.dilation = [1, 2, 4, 8, 4, 2, 1]
        dilation_layers = []
        for dil in self.dilation:
            dilation_layers.append(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    dilation=dil,
                    padding=dil,
                    bias=True,
                )
            )
        self.dilation_layers = nn.ModuleList(dilation_layers)
        self.norm_layers = nn.ModuleList(
            nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity()
            for dil in self.dilation
        )

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, norm in zip(self.dilation_layers, self.norm_layers):
            out = self.activation(layer(norm(out)))
        return out + x


class FourierBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        modes1: int = 16,
        modes2: int = 16,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        assert not norm
        self.fourier1 = SpectralConv2d(in_planes, planes, modes1=self.modes1, modes2=self.modes2)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)
        self.fourier2 = SpectralConv2d(planes, planes, modes1=self.modes1, modes2=self.modes2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)

        # So far shortcut connections are not helping
        """
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1)
            )
        """
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fourier1(x)
        x2 = self.conv1(x)
        out = self.activation(x1 + x2)

        x1 = self.fourier2(out)
        x2 = self.conv2(out)
        out = x1 + x2
        # out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    padding = 9

    def __init__(
        self,
        n_scalar_components: int,
        n_vector_components: int,
        block,
        num_blocks: list,
        time_history: int,
        time_future: int,
        hidden_channels: int = 64,
        activation: str = "gelu",
        norm: bool = True,
        diffmode: bool = False,
        usegrid: bool = False,
    ):
        super().__init__()
        self.n_scalar_components = n_scalar_components
        self.n_vector_components = n_vector_components
        self.diffmode = diffmode
        self.usegrid = usegrid
        self.in_planes = hidden_channels
        insize = time_history * (self.n_scalar_components + self.n_vector_components * 2)
        if self.usegrid:
            insize += 2
        self.conv_in1 = nn.Conv2d(
            insize,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            self.in_planes,
            time_future * (self.n_scalar_components + self.n_vector_components * 2),
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
                    norm=norm,
                )
                for i in range(len(num_blocks))
            ]
        )

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        num_blocks: int,
        stride: int,
        activation: str,
        norm: bool = True,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    activation=activation,
                    norm=norm,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def __repr__(self):
        return "ResNet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
        # prev = x.float()
        x = self.activation(self.conv_in1(x.float()))
        x = self.activation(self.conv_in2(x.float()))

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.layers:
            x = layer(x)

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        if self.diffmode:
            raise NotImplementedError("diffmode")
            # x = x + prev[:, -1:, ...].detach()
        return x.reshape(orig_shape[0], -1, *orig_shape[2:])


#######################################################################
#######################################################################
