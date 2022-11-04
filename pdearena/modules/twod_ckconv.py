# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Dict

import torch
from ckconv.nn import CKConv
from torch import nn

from pdearena.pde import PDEConfig


class CKBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        planes: int,
        kernel_cfg: Dict,
        conv_cfg: Dict,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = CKConv(
            in_channels,
            planes,
            data_dim=2,
            kernel_cfg=kernel_cfg,
            conv_cfg=conv_cfg,
        )
        self.norm1 = nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity()
        self.conv2 = CKConv(
            planes,
            planes,
            data_dim=2,
            kernel_cfg=kernel_cfg,
            conv_cfg=conv_cfg,
        )
        self.norm2 = nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != self.expansion * planes:
            self.shortcut = nn.Sequential(
                CKConv(
                    in_channels,
                    self.expansion * planes,
                    data_dim=2,
                    kernel_cfg=kernel_cfg,
                    conv_cfg=conv_cfg,
                ),
                nn.GroupNorm(num_groups, self.expansion * planes) if norm else nn.Identity(),
            )

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation(self.norm1(x)))
        out = self.conv2(self.activation(self.norm2(out)))
        out = out + self.shortcut(x)
        return out


class CKResNet(nn.Module):
    padding = 0

    def __init__(
        self,
        pde: PDEConfig,
        block: nn.Module,
        num_blocks: list,
        time_history: int,
        time_future: int,
        kernel_cfg,
        conv_cfg,
        hidden_channels: int = 64,
        activation: str = "gelu",
        norm: bool = True,
        diffmode: bool = False,
        usegrid: bool = False,
    ):
        super().__init__()
        self.pde = pde
        self.diffmode = diffmode
        self.usegrid = usegrid
        self.in_planes = hidden_channels
        insize = time_history * (self.pde.n_scalar_components + self.pde.n_vector_components * 2)
        if self.usegrid:
            insize += 2
        self.conv_in1 = CKConv(
            insize,
            self.in_planes,
            data_dim=2,
            kernel_cfg=kernel_cfg,
            conv_cfg=conv_cfg,
        )
        self.conv_in2 = CKConv(
            self.in_planes,
            self.in_planes,
            data_dim=2,
            kernel_cfg=kernel_cfg,
            conv_cfg=conv_cfg,
        )
        self.conv_out1 = CKConv(
            self.in_planes,
            self.in_planes,
            data_dim=2,
            kernel_cfg=kernel_cfg,
            conv_cfg=conv_cfg,
        )
        self.conv_out2 = CKConv(
            self.in_planes,
            time_future * (self.pde.n_scalar_components + self.pde.n_vector_components * 2),
            data_dim=2,
            kernel_cfg=kernel_cfg,
            conv_cfg=conv_cfg,
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    block,
                    self.in_planes,
                    num_blocks[i],
                    stride=1,
                    activation=activation,
                    kernel_cfg=kernel_cfg,
                    conv_cfg=conv_cfg,
                    norm=norm,
                )
                for i in range(len(num_blocks))
            ]
        )

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        num_blocks: int,
        stride: int,
        activation: str,
        kernel_cfg,
        conv_cfg,
        norm: bool = True,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    activation=activation,
                    norm=norm,
                    kernel_cfg=kernel_cfg,
                    conv_cfg=conv_cfg,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def __repr__(self):
        return "CKResNet"

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
