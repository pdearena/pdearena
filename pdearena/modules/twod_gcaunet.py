# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from cliffordlayers.models.gca.twod import CliffordG3UNet2d, CliffordUpsample


class GCAFluidNet2d(CliffordG3UNet2d):
    """2D GCA-UNet as introduced in the paper "Geometric Clifford Algebra Networks". https://arxiv.org/abs/2302.06594."""

    def forward(self, x: torch.Tensor):
        assert x.dim() == 5

        # Putting multivector shape in third dimension.
        x = x.permute(0, 1, 3, 4, 2)

        x = self.conv1(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, CliffordUpsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        x = self.activation(self.norm(x))
        x = self.conv2(x)

        # Putting multivector dimension back in last position.
        x = x.permute(0, 1, 4, 2, 3)
        return x
