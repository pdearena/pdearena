# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn.functional as F
from torch import nn

from .activations import ACTIVATION_REGISTRY

# Based on https://github.com/ashiq24/UNO
#
# BSD 2-Clause License

# Copyright (c) 2022, Md Ashiqur rahman
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


class SpectralConv2d_Uno(nn.Module):
    """2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Modified to support multi-gpu training.

    Args:
        in_codim (int): Input co-domian dimension
        out_codim (int): output co-domain dimension

        dim1 (int): Default output grid size along x (or 1st dimension of output domain)
        dim2 (int): Default output grid size along y ( or 2nd dimension of output domain)
                    Ratio of grid size of the input and the output implecitely
                    set the expansion or contraction farctor along each dimension.
        modes1 (int), modes2 (int):  Number of fourier modes to consider for the ontegral operator
                    Number of modes must be compatibale with the input grid size
                    and desired output grid size.
                    i.e., modes1 <= min( dim1/2, input_dim1/2).
                    Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
                    Other modes also the have same constrain.
    """

    def __init__(self, in_codim, out_codim, dim1, dim2, modes1=None, modes2=None):
        super().__init__()

        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1
            self.modes2 = modes2
        else:
            self.modes1 = dim1 // 2 - 1
            self.modes2 = dim2 // 2
        self.scale = (1 / (2 * in_codim)) ** (1.0 / 2.0)
        self.weights1 = nn.Parameter(
            self.scale * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, 2, dtype=torch.float32))
        )
        self.weights2 = nn.Parameter(
            self.scale * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, 2, dtype=torch.float32))
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):

        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1=None, dim2=None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm="forward")

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            self.dim1,
            self.dim2 // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], torch.view_as_complex(self.weights1)
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], torch.view_as_complex(self.weights2)
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2), norm="forward")
        return x


class Pointwise_op_2D(nn.Module):
    """

    Args:
        in_codim (int): Input co-domian dimension
        out_codim (int): output co-domain dimension

        dim1 (int):  Default output grid size along x (or 1st dimension)
        dim2 (int): Default output grid size along y ( or 2nd dimension)
    """

    def __init__(self, in_codim: int, out_codim: int, dim1: int, dim2: int):
        super().__init__()
        self.conv = nn.Conv2d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self, x, dim1=None, dim2=None):
        #
        # input shape = (batch, in_codim, input_dim1,input_dim2)
        # output shape = (batch, out_codim, dim1,dim2)

        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)

        x_out = F.interpolate(x_out, size=(dim1, dim2), mode="bicubic", align_corners=True, antialias=True)
        return x_out


class OperatorBlock_2D(nn.Module):
    """

    Args:
        in_codim (int): Input co-domian dimension
        out_codim (int): output co-domain dimension
        dim1 (int):  Default output grid size along x (or 1st dimension)
        dim2 (int): Default output grid size along y ( or 2nd dimension)
        modes1 (int): Number of fourier modes to consider along 1st dimension
        modes2 (int): Number of fourier modes to consider along 2nd dimension
        norm (bool): Whether to use normalization ([torch.nn.InstanceNorm2d][])
        nonlin (bool): Whether to use non-linearity ([torch.nn.GELU][])


    All variables are consistent with the [`SpectralConv2d_Uno`][pdearena.modules.twod_uno.SpectralConv2d_Uno].
    """

    def __init__(self, in_codim, out_codim, dim1, dim2, modes1, modes2, norm=True, nonlin=True):
        super().__init__()
        self.conv = SpectralConv2d_Uno(in_codim, out_codim, dim1, dim2, modes1, modes2)
        self.w = Pointwise_op_2D(in_codim, out_codim, dim1, dim2)
        self.norm = norm
        self.non_lin = nonlin
        if norm:
            self.normalize_layer = nn.InstanceNorm2d(int(out_codim), affine=True)

    def forward(self, x, dim1=None, dim2=None):
        #
        # input shape = (batch, in_codim, input_dim1,input_dim2)
        # output shape = (batch, out_codim, dim1,dim2)
        x1_out = self.conv(x, dim1, dim2)
        x2_out = self.w(x, dim1, dim2)
        x_out = x1_out + x2_out
        if self.norm:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


class UNO(nn.Module):
    """UNO model

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps to include in the model
        time_future (int): Number of time steps to predict in the model
        hidden_channels (int): Number of hidden channels in the model
        pad (int): Padding to use in the model
        factor (int): Scaling factor to use in the model
        activation (str): Activation function to use in the model
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
        pad=0,
        factor=3 / 4,
        activation="gelu",
    ):
        super().__init__()

        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components

        self.width = hidden_channels
        self.factor = factor
        self.padding = pad
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        in_width = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 2)
        out_width = time_future * (self.n_output_scalar_components + self.n_output_vector_components * 2)
        self.fc = nn.Linear(in_width, self.width // 2)

        self.fc0 = nn.Linear(self.width // 2, self.width)  # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2 * factor * self.width, 48, 48, 18, 18)

        self.L1 = OperatorBlock_2D(2 * factor * self.width, 4 * factor * self.width, 32, 32, 14, 14)

        self.L2 = OperatorBlock_2D(4 * factor * self.width, 8 * factor * self.width, 16, 16, 6, 6)

        self.L3 = OperatorBlock_2D(8 * factor * self.width, 8 * factor * self.width, 16, 16, 6, 6)

        self.L4 = OperatorBlock_2D(8 * factor * self.width, 4 * factor * self.width, 32, 32, 6, 6)

        self.L5 = OperatorBlock_2D(8 * factor * self.width, 2 * factor * self.width, 48, 48, 14, 14)

        self.L6 = OperatorBlock_2D(4 * factor * self.width, self.width, 64, 64, 18, 18)  # will be reshaped

        self.fc1 = nn.Linear(2 * self.width, 4 * self.width)
        self.fc2 = nn.Linear(4 * self.width, out_width)

    def forward(self, x):
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C

        x = x.permute(0, 2, 3, 1)
        x_fc = self.fc(x)
        x_fc = self.activation(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = self.activation(x_fc0)

        x_fc0 = x_fc0.permute(0, 3, 1, 2)

        x_fc0 = F.pad(x_fc0, [self.padding, self.padding, self.padding, self.padding])

        D1, D2 = x_fc0.shape[-2], x_fc0.shape[-1]

        x_c0 = self.L0(x_fc0, int(D1 * self.factor), int(D2 * self.factor))
        x_c1 = self.L1(x_c0, D1 // 2, D2 // 2)

        x_c2 = self.L2(x_c1, D1 // 4, D2 // 4)
        x_c3 = self.L3(x_c2, D1 // 4, D2 // 4)
        x_c4 = self.L4(x_c3, D1 // 2, D2 // 2)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)
        x_c5 = self.L5(x_c4, int(D1 * self.factor), int(D2 * self.factor))
        x_c5 = torch.cat([x_c5, x_c0], dim=1)
        x_c6 = self.L6(x_c5, D1, D2)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding != 0:
            x_c6 = x_c6[..., : -self.padding, : -self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c6)
        x_fc1 = self.activation(x_fc1)

        x_out = self.fc2(x_fc1)
        x_out = x_out.permute(0, 3, 1, 2)

        return x_out.reshape(
            orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
        )
