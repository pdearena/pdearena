# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from pdearena.modules.activations import ACTIVATION_REGISTRY

from .condition_utils import ConditionedBlock, fourier_embedding, zero_module

# Largely based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License


def conv_layer(
    c_in: int, c_out: int, kernel_size: int, stride: int = 1, dilation: int = 1, padding: int = -1, n_dims: int = 1
):
    if padding < 0:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)
    if n_dims == 1:
        return nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    elif n_dims == 2:
        return nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    else:
        raise NotImplementedError(f"n_dims {n_dims} not implemented")


class ResidualBlock(ConditionedBlock):
    """Wide Residual Blocks used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        cond_channels (int): Number of channels in the conditioning vector.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
        n_dims (int): Number of spatial dimensions. Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        use_scale_shift_norm: bool = False,
        n_dims: int = 1,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.conv1 = conv_layer(in_channels, out_channels, kernel_size=3, n_dims=n_dims)
        self.conv2 = zero_module(conv_layer(out_channels, out_channels, kernel_size=3, n_dims=n_dims))
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = conv_layer(in_channels, out_channels, kernel_size=1, n_dims=n_dims)
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.cond_emb = nn.Linear(cond_channels, 2 * out_channels if use_scale_shift_norm else out_channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(x)))
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift  # where we do -1 or +1 doesn't matter
            h = self.conv2(self.activation(h))
        else:
            h = h + emb_out
            # Second convolution layer
            h = self.conv2(self.activation(self.norm2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class ConditionalResidualBlock(ConditionedBlock):
    """Wide Residual Blocks used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        cond_channels (int): Number of channels in the conditioning vector.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
        n_dims (int): Number of spatial dimensions. Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels_main: int,
        cond_channels_emb: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        use_scale_shift_norm: bool = False,
        n_dims: int = 1,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.conv1 = conv_layer(in_channels + cond_channels_main, out_channels, kernel_size=3, n_dims=n_dims)
        self.conv2 = zero_module(conv_layer(out_channels, out_channels, kernel_size=3, n_dims=n_dims))
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = conv_layer(in_channels, out_channels, kernel_size=1, n_dims=n_dims)
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels + cond_channels_main)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.cond_emb = nn.Linear(cond_channels_emb, 2 * out_channels if use_scale_shift_norm else out_channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, cond: torch.Tensor):
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(torch.cat([x, cond], dim=1))))
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift  # where we do -1 or +1 doesn't matter
            h = self.conv2(self.activation(h))
        else:
            h = h + emb_out
            # Second convolution layer
            h = self.conv2(self.activation(self.norm2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Attention block This is similar to [transformer multi-head
    attention](https://arxiv.org/abs/1706.03762).

    Args:
        n_channels: the number of channels in the input
        n_heads:  the number of heads in multi-head attention
        d_k: the number of dimensions in each head
        n_groups: the number of groups for [group normalization][torch.nn.GroupNorm]

    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 1):
        """ """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Get shape
        orig_shape = x.shape
        if x.ndim == 3:
            x = x.unsqueeze(2)  # Pretend we have a height of 1 for 1D inputs
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(*orig_shape)
        return res


class DownBlock(ConditionedBlock):
    """Down block This combines `ResidualBlock` and `AttentionBlock`.

    These are used in the first half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        cond_channels (int): Number of channels in the conditioning vector.
        has_attn (bool): Whether to use attention block
        activation (nn.Module): Activation function
        norm (bool): Whether to use normalization
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
        n_dims (int): Number of spatial dimensions. Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        use_scale_shift_norm: bool = False,
        n_dims: int = 1,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels,
            out_channels,
            cond_channels,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
            n_dims=n_dims,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class UpBlock(ConditionedBlock):
    """Up block This combines `ResidualBlock` and `AttentionBlock`.

    These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        cond_channels (int): Number of channels in the conditioning vector.
        has_attn (bool): Whether to use attention block
        activation (str): Activation function
        norm (bool): Whether to use normalization
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
        n_dims (int): Number of spatial dimensions. Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        use_scale_shift_norm: bool = False,
        n_dims: int = 1,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels,
            out_channels,
            cond_channels,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
            n_dims=n_dims,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class MiddleBlock(ConditionedBlock):
    """Middle block It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        cond_channels (int): Number of channels in the conditioning vector.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
        use_scale_shift_norm (bool, optional): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
        n_dims (int): Number of spatial dimensions. Defaults to 1. Defaults to False.
    """

    def __init__(
        self,
        n_channels: int,
        cond_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        use_scale_shift_norm: bool = False,
        n_dims: int = 1,
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels,
            n_channels,
            cond_channels,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
            n_dims=n_dims,
        )
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(
            n_channels,
            n_channels,
            cond_channels,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
            n_dims=n_dims,
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x


class Upsample(nn.Module):
    r"""Scale up the feature map by $2 \times$"""

    def __init__(self, n_channels: int, n_dims: int = 1):
        super().__init__()
        if n_dims == 1:
            self.conv = nn.ConvTranspose1d(n_channels, n_channels, 4, 2, 1)
        elif n_dims == 2:
            self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))
        elif n_dims == 3:
            self.conv = nn.ConvTranspose3d(n_channels, n_channels, (4, 4, 4), (2, 2, 2), (1, 1, 1))
        else:
            raise ValueError(f"n_dims must be 1, 2, or 3, got {n_dims}")

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$"""

    def __init__(self, n_channels: int, n_dims: int = 1):
        super().__init__()
        if n_dims == 1:
            self.conv = nn.Conv1d(n_channels, n_channels, 3, 2, 1)
        elif n_dims == 2:
            self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))
        elif n_dims == 3:
            self.conv = nn.Conv3d(n_channels, n_channels, (3, 3, 3), (2, 2, 2), (1, 1, 1))
        else:
            raise ValueError(f"n_dims must be 1, 2, or 3, got {n_dims}")

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Unet(nn.Module):
    """Modern U-Net architecture

    This is a modern U-Net architecture with wide-residual blocks and spatial attention blocks

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input
        time_future (int): Number of time steps in the output
        hidden_channels (int): Number of channels in the hidden layers
        activation (str): Activation function to use
        norm (bool): Whether to use normalization
        ch_mults (list): List of channel multipliers for each resolution
        is_attn (list): List of booleans indicating whether to use attention blocks
        mid_attn (bool): Whether to use attention block in the middle block
        n_blocks (int): Number of residual blocks in each resolution
        param_conditioning (Optional[str]): Type of conditioning to use. Defaults to None.
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`). Defaults to False.
        use1x1 (bool): Whether to use 1x1 convolutions in the initial and final layers.
        n_dims (int): Number of spatial dimensions. Defaults to 1.

    Note:
        Currently, only `scalar` parameter conditioning is supported.
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history,
        time_future,
        hidden_channels,
        activation,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        param_conditioning: Optional[str] = None,
        use_scale_shift_norm: bool = False,
        use1x1: bool = False,
        n_dims: int = 1,
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels
        self.activation = activation
        self.n_dims = n_dims
        self.param_conditioning = param_conditioning
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 2)
        n_channels = hidden_channels
        time_embed_dim = hidden_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_channels, time_embed_dim),
            self.activation,
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        if self.param_conditioning is not None:
            if self.param_conditioning.startswith("scalar"):
                num_params = 1 if "_" not in self.param_conditioning else int(self.param_conditioning.split("_")[1])
                self.pde_emb = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(hidden_channels, time_embed_dim),
                            self.activation,
                            nn.Linear(time_embed_dim, time_embed_dim),
                        )
                        for _ in range(num_params)
                    ]
                )
            else:
                raise NotImplementedError(f"Param conditioning {self.param_conditioning} not implemented")
        self.param_use_time = False
        self.param_use_cond = False

        # Project image into feature map
        if use1x1:
            self.image_proj = conv_layer(insize, n_channels, kernel_size=1, n_dims=n_dims)
        else:
            self.image_proj = conv_layer(insize, n_channels, kernel_size=3, n_dims=n_dims)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        time_embed_dim,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        use_scale_shift_norm=use_scale_shift_norm,
                        n_dims=n_dims,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels, n_dims=n_dims))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(
            out_channels,
            time_embed_dim,
            has_attn=mid_attn,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
            n_dims=n_dims,
        )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        time_embed_dim,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        use_scale_shift_norm=use_scale_shift_norm,
                        n_dims=n_dims,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    time_embed_dim,
                    has_attn=is_attn[i],
                    activation=activation,
                    norm=norm,
                    use_scale_shift_norm=use_scale_shift_norm,
                    n_dims=n_dims,
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels, n_dims=n_dims))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = time_future * (self.n_output_scalar_components + self.n_output_vector_components * 2)
        if use1x1:
            self.final = zero_module(conv_layer(in_channels, out_channels, kernel_size=1, n_dims=n_dims))
        else:
            self.final = zero_module(conv_layer(in_channels, out_channels, kernel_size=3, n_dims=n_dims))

    def forward(self, x: torch.Tensor, time: torch.Tensor = None, z: torch.Tensor = None):
        assert x.dim() == 3 + self.n_dims
        assert not (time is None and z is None)
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C

        emb = 0
        if time is not None:
            emb = emb + self.time_embed(fourier_embedding(time, self.hidden_channels))
            self.param_use_time = True
        else:
            assert not self.param_use_time, "Cannot pass time=None after using it in a previous forward pass"
        if z is not None:
            if self.param_conditioning.startswith("scalar"):
                if z.ndim == 1:
                    z = z[:, None]
                for i in range(z.shape[-1]):
                    emb = emb + self.pde_emb[i](fourier_embedding(z[..., i], self.hidden_channels))
            else:
                raise NotImplementedError(f"Param conditioning {self.param_conditioning} not implemented")
            self.param_use_cond = True
        else:
            assert not self.param_use_cond, "Cannot pass z=None after using it in a previous forward pass"

        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            if isinstance(m, Downsample):
                x = m(x)
            else:
                x = m(x, emb)
            h.append(x)

        x = self.middle(x, emb)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, emb)

        x = self.final(self.activation(self.norm(x)))
        return x.reshape(
            orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
        )
