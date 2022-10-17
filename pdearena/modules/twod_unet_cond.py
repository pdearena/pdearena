# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Tuple, List, Union
import torch
from torch import nn

from .condition_utils import ConditionedBlock, fourier_embedding, zero_module
from .fourier import SpectralConv2d




# Largely based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License




class ResidualBlock(ConditionedBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        use_scale_shift_norm: bool = False,
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

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = zero_module(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        )
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.cond_emb = nn.Linear(
            cond_channels, 2 * out_channels if use_scale_shift_norm else out_channels
        )

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
            h += emb_out
            # Second convolution layer
            h = self.conv2(self.activation(self.norm2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class FourierResidualBlock(ConditionedBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        use_scale_shift_norm: bool = False,
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

        self.modes1 = modes1
        self.modes2 = modes2

        self.fourier1 = SpectralConv2d(
            in_channels, out_channels, modes1=self.modes1, modes2=self.modes2
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0, padding_mode="zeros"
        )
        self.fourier2 = SpectralConv2d(
            out_channels, out_channels, modes1=self.modes1, modes2=self.modes2
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, padding=0, padding_mode="zeros"
        )
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.cond_emb = nn.Linear(
            cond_channels, 2 * out_channels if use_scale_shift_norm else out_channels
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        # # First convolution layer
        # h = self.conv1(self.activation(self.norm1(x)))
        # # Second convolution layer
        # h = self.conv2(self.activation(self.norm2(h)))
        # # Add the shortcut connection and return
        # return h + self.shortcut(x)
        # TODO: might need to figure out the order of norm/activations
        # h = self.norm1(x)
        # x1 = self.fourier1(h)
        # x2 = self.conv1(h)
        # out = self.activation(x1 + x2)

        # out = self.norm2(out)
        # x1 = self.fourier2(out)
        # x2 = self.conv2(out)
        # out = x1 + x2
        # # out += self.shortcut(x)
        # out = self.activation(out)
        # out = out + self.shortcut(x)
        h = self.activation(self.norm1(x))
        x1 = self.fourier1(h)
        x2 = self.conv1(h)
        out = x1 + x2
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.norm2(out) * (1 + scale) + shift  # where we do -1 or +1 doesn't matter
            h = self.activation(h)
            x1 = self.fourier2(h)
            x2 = self.conv2(h)
        else:
            out = out + emb_out
            out = self.activation(self.norm2(out))
            x1 = self.fourier2(out)
            x2 = self.conv2(out)

        out = x1 + x2 + self.shortcut(x)
        return out


class AttentionBlock(nn.Module):
    """
    ### Attention block
    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 1):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
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
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class DownBlock(ConditionedBlock):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
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
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels,
            out_channels,
            cond_channels,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class FourierDownBlock(ConditionedBlock):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.res = FourierResidualBlock(
            in_channels,
            out_channels,
            cond_channels,
            modes1=modes1,
            modes2=modes2,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
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
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
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
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class FourierUpBlock(ConditionedBlock):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = FourierResidualBlock(
            in_channels + out_channels,
            out_channels,
            cond_channels,
            modes1=modes1,
            modes2=modes2,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class MiddleBlock(ConditionedBlock):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(
        self,
        n_channels: int,
        cond_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels,
            n_channels,
            cond_channels,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(
            n_channels,
            n_channels,
            cond_channels,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        pde,
        time_history,
        time_future,
        hidden_channels,
        activation,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        mul_pdes: bool = False,
        use_scale_shift_norm: bool = False,
    ) -> None:
        super().__init__()
        self.pde = pde
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels
        self.activation = activation
        self.mul_pdes = mul_pdes
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = time_history * (self.pde.n_scalar_components + self.pde.n_vector_components * 2)
        n_channels = hidden_channels
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

        # Project image into feature map
        self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=(3, 3), padding=(1, 1))

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
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

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
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = time_future * (
            self.pde.n_scalar_components + self.pde.n_vector_components * 2
        )
        self.final = zero_module(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, x: torch.Tensor, time, z=None):
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C

        emb = self.time_embed(fourier_embedding(time, self.hidden_channels))
        if self.mul_pdes:
            emb = emb + self.pde_emb(fourier_embedding(z, self.hidden_channels))

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
        return x.reshape(orig_shape[0], -1, *orig_shape[2:])


class FourierUnet(nn.Module):
    def __init__(
        self,
        pde,
        time_history,
        time_future,
        hidden_channels,
        activation,
        modes1=12,
        modes2=12,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        n_fourier_layers: int = 2,
        mode_scaling: bool = True,
        mul_pdes: bool = False,
        use_scale_shift_norm: bool = False,
    ) -> None:
        super().__init__()
        self.pde = pde
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels
        self.activation = activation
        self.mul_pdes = mul_pdes
        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = time_history * (self.pde.n_scalar_components + self.pde.n_vector_components * 2)
        n_channels = hidden_channels
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
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
        # Project image into feature map
        self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            if i < n_fourier_layers:
                for _ in range(n_blocks):
                    down.append(
                        FourierDownBlock(
                            in_channels,
                            out_channels,
                            time_embed_dim,
                            modes1=max(modes1 // 2**i, 4) if mode_scaling else modes1,
                            modes2=max(modes2 // 2**i, 4) if mode_scaling else modes2,
                            has_attn=is_attn[i],
                            activation=activation,
                            norm=norm,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    )
                    in_channels = out_channels
            else:
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
                        )
                    )
                    in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(
            out_channels, time_embed_dim, has_attn=mid_attn, activation=activation, norm=norm
        )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            if i < n_fourier_layers:
                for _ in range(n_blocks):
                    up.append(
                        FourierUpBlock(
                            in_channels,
                            out_channels,
                            time_embed_dim,
                            modes1=max(modes1 // 2**i, 4) if mode_scaling else modes1,
                            modes2=max(modes2 // 2**i, 4) if mode_scaling else modes2,
                            has_attn=is_attn[i],
                            activation=activation,
                            norm=norm,
                        )
                    )
            else:
                for _ in range(n_blocks):
                    up.append(
                        UpBlock(
                            in_channels,
                            out_channels,
                            time_embed_dim,
                            has_attn=is_attn[i],
                            activation=activation,
                            norm=norm,
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
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = time_future * (
            self.pde.n_scalar_components + self.pde.n_vector_components * 2
        )
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, time, z=None):
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C

        emb = self.time_embed(fourier_embedding(time, self.hidden_channels))
        if z is not None:
            emb = emb + self.pde_emb(fourier_embedding(z, self.hidden_channels))

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
        return x.reshape(orig_shape[0], -1, *orig_shape[2:])
