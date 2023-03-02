# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
from torch import nn


# Complex multiplication 1d
def batchmul1d(input, weights):
    # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", input, weights)

# Complex multiplication 2d
def batchmul2d(input, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", input, weights)

# Complex multiplication 3d
def batchmul3d(input, weights):
    # (batch, in_channel, x,y,z ), (in_channel, out_channel, x,y,z) -> (batch, out_channel, x,y,z)
    return torch.einsum("bixyz,ioxyz->boxyz", input, weights)


################################################################
# fourier layer
################################################################


class SpectralConv1d(nn.Module):
    """1D Fourier layer. Does FFT, linear transform, and Inverse FFT.
    Implemented in a way to allow multi-gpu training.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        modes (int): Number of Fourier modes 
    [paper](https://arxiv.org/abs/2010.08895)
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, 2, dtype=torch.float32)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes] = batchmul1d(
            x_ft[:, :, : self.modes], torch.view_as_complex(self.weights)
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=(x.size(-1),))
        return x


class SpectralConv2d(nn.Module):
    """2D Fourier layer. Does FFT, linear transform, and Inverse FFT.
    Implemented in a way to allow multi-gpu training.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        modes1 (int): Number of Fourier modes to keep in the first spatial direction
        modes2 (int): Number of Fourier modes to keep in the second spatial direction
    [paper](https://arxiv.org/abs/2010.08895)
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32)
        )

    def forward(self, x, x_dim=None, y_dim=None):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = batchmul2d(
            x_ft[:, :, : self.modes1, : self.modes2], torch.view_as_complex(self.weights1)
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = batchmul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], torch.view_as_complex(self.weights2)
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    
class SpectralConv3d(nn.Module):
    """3D Fourier layer. Does FFT, linear transform, and Inverse FFT.
    Implemented in a way to allow multi-gpu training.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        modes1 (int): Number of Fourier modes to keep in the first spatial direction
        modes2 (int): Number of Fourier modes to keep in the second spatial direction
        modes3 (int): Number of Fourier modes to keep in the third spatial direction
    [paper](https://arxiv.org/abs/2010.08895)
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = batchmul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], torch.view_as_complex(self.weights1)
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = batchmul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], torch.view_as_complex(self.weights2)
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = batchmul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], torch.view_as_complex(self.weights3)
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = batchmul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], torch.view_as_complex(self.weights4)
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x
