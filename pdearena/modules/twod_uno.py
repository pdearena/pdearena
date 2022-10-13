import torch
from torch import nn
import torch.nn.functional as F

# Based on https://github.com/ashiq24/UNO


class SpectralConv2d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2, modes1=None, modes2=None):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        dim1 = Default output grid size along x (or 1st dimension of output domain)
        dim2 = Default output grid size along y ( or 2nd dimension of output domain)
        Ratio of grid size of the input and the output implecitely
        set the expansion or contraction farctor along each dimension.
        modes1, modes2 = Number of fourier modes to consider for the ontegral operator
                        Number of modes must be compatibale with the input grid size
                        and desired output grid size.
                        i.e., modes1 <= min( dim1/2, input_dim1/2).
                        Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
                        Other modes also the have same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension
        """

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
            self.scale
            * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, 2, dtype=torch.float32))
        )
        self.weights2 = nn.Parameter(
            self.scale
            * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, 2, dtype=torch.float32))
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
    dim1 = Default output grid size along x (or 1st dimension)
    dim2 = Default output grid size along y ( or 2nd dimension)
    in_codim = Input co-domian dimension
    out_codim = output co-domain dimension
    """

    def __init__(self, in_codim, out_codim, dim1, dim2):
        super().__init__()
        self.conv = nn.Conv2d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self, x, dim1=None, dim2=None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)

        x_out = F.interpolate(
            x_out, size=(dim1, dim2), mode="bicubic", align_corners=True, antialias=True
        )
        return x_out


class OperatorBlock_2D(nn.Module):
    """
    Normalize = if true performs InstanceNorm2d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv2d_Uno class.
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
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        x1_out = self.conv(x, dim1, dim2)
        x2_out = self.w(x, dim1, dim2)
        x_out = x1_out + x2_out
        if self.norm:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


class UNO(nn.Module):
    def __init__(
        self,
        pde,
        time_history,
        time_future,
        hidden_channels,
        pad=0,
        factor=3 / 4,
        activation="gelu",
    ):
        super(UNO, self).__init__()

        self.pde = pde

        self.width = hidden_channels
        self.factor = factor
        self.padding = pad

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        in_width = time_history * (self.pde.n_scalar_components + self.pde.n_vector_components * 2)
        out_width = time_future * (self.pde.n_scalar_components + self.pde.n_vector_components * 2)
        self.fc = nn.Linear(in_width, self.width // 2)

        self.fc0 = nn.Linear(self.width // 2, self.width)  # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2 * factor * self.width, 48, 48, 18, 18)

        self.L1 = OperatorBlock_2D(2 * factor * self.width, 4 * factor * self.width, 32, 32, 14, 14)

        self.L2 = OperatorBlock_2D(4 * factor * self.width, 8 * factor * self.width, 16, 16, 6, 6)

        self.L3 = OperatorBlock_2D(8 * factor * self.width, 8 * factor * self.width, 16, 16, 6, 6)

        self.L4 = OperatorBlock_2D(8 * factor * self.width, 4 * factor * self.width, 32, 32, 6, 6)

        self.L5 = OperatorBlock_2D(8 * factor * self.width, 2 * factor * self.width, 48, 48, 14, 14)

        self.L6 = OperatorBlock_2D(
            4 * factor * self.width, self.width, 64, 64, 18, 18
        )  # will be reshaped

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

        return x_out.reshape(orig_shape[0], -1, *orig_shape[2:])
