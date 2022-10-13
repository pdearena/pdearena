import torch
from torch import nn
import torch.nn.functional as F

from .fourier import SpectralConv2d
from .condition_utils import fourier_embedding, ConditionedBlock



class FourierBasicBlock(ConditionedBlock):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        cond_channels: int,
        stride: int = 1,
        modes1: int = 16,
        modes2: int = 16,
        activation: str = "gelu",
        norm: bool = False,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")
        assert not norm
        self.fourier1 = SpectralConv2d(in_planes, planes, modes1=self.modes1, modes2=self.modes2)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True
        )
        self.fourier2 = SpectralConv2d(planes, planes, modes1=self.modes1, modes2=self.modes2)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True
        )
        self.cond_emb = nn.Linear(cond_channels, planes)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x1 = self.fourier1(x)
        x2 = self.conv1(x)
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(x2.shape):
            emb_out = emb_out[..., None]

        out = self.activation(x1 + x2 + emb_out)
        x1 = self.fourier2(out)
        x2 = self.conv2(out)
        out = x1 + x2
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    padding = 9

    def __init__(
        self,
        pde,
        block: nn.Module,
        num_blocks: list,
        time_history: int,
        time_future: int,
        hidden_channels: int = 64,
        activation: str = "gelu",
        norm: bool = False,
        diffmode: bool = False,
        usegrid: bool = False,
        mul_pdes: bool = False,
    ):
        super(ResNet, self).__init__()
        self.pde = pde
        self.diffmode = diffmode
        self.usegrid = usegrid
        self.in_planes = hidden_channels
        self.mul_pdes = mul_pdes
        insize = time_history * (self.pde.n_scalar_components + self.pde.n_vector_components * 2)
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")
        time_embed_dim = hidden_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_channels, time_embed_dim),
            self.activation,
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        if self.mul_pdes:
            self.pde_emb = nn.Sequential(
                nn.Linear(hidden_channels, time_embed_dim),
                self.activation,
                nn.Linear(time_embed_dim, time_embed_dim),
            )

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
            time_future * (self.pde.n_scalar_components + self.pde.n_vector_components * 2),
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    block,
                    self.in_planes,
                    time_embed_dim,
                    num_blocks[i],
                    stride=1,
                    activation=activation,
                    norm=norm,
                )
                for i in range(len(num_blocks))
            ]
        )

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        cond_channels: int,
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
                    cond_channels=cond_channels,
                    stride=stride,
                    activation=activation,
                    norm=norm,
                )
            )
            self.in_planes = planes * block.expansion
        return EmbedSequential(*layers)

    def __repr__(self):
        return "ResNet"

    def forward(self, x: torch.Tensor, time, z=None) -> torch.Tensor:
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C

        emb = self.time_embed(fourier_embedding(time, self.in_planes))
        if z is not None:
            emb = emb + self.pde_emb(fourier_embedding(z, self.in_planes))
        # prev = x.float()
        x = self.activation(self.conv_in1(x.float()))
        x = self.activation(self.conv_in2(x.float()))

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.layers:
            x = layer(x, emb)

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        if self.diffmode:
            raise NotImplementedError("diffmode")
            # x = x + prev[:, -1:, ...].detach()
        return x.reshape(orig_shape[0], -1, *orig_shape[2:])
