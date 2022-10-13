import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu"
    ) -> None:
        super().__init__()
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
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
    def __init__(
        self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu"
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, num_groups, norm, activation)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        h = self.pool(x)
        h = self.conv(h)
        return h


class Up(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu"
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, num_groups, norm, activation)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h = self.up(x1)
        h = torch.cat([x2, h], dim=1)
        h = self.conv(h)
        return h


class OldUnet(nn.Module):
    def __init__(self, pde, time_history, time_future, hidden_channels, activation="gelu") -> None:
        super().__init__()
        self.pde = pde
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        insize = time_history * (self.pde.n_scalar_components + self.pde.n_vector_components * 2)
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
        out_channels = time_future * (
            self.pde.n_scalar_components + self.pde.n_vector_components * 2
        )
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
        return x.reshape(orig_shape[0], -1, *orig_shape[2:])
