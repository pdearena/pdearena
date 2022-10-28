# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Optional

import random
import h5py
import torch

from torch.utils import data

from pdearena.pde import PDEConfig
import pdearena.data.utils as datautils


class PDEDataset(data.Dataset):
    def __init__(self, path, mode, limit_trajectories: Optional[int] = None, usegrid: bool = False) -> None:
        super().__init__()
        self.path = path
        self.mode = mode
        self.limit_trajectories = limit_trajectories
        self.usegrid = usegrid
        if self.limit_trajectories is None or self.limit_trajectories == -1:
            self.num = self.get_num_trajectories()
        else:
            self.num = self.limit_trajectories

    def __len__(self):
        return self.num

    def get_num_trajectories(self):
        with h5py.File(self.path, "r") as f:
            num = f[self.mode]["u"].shape[0]
        return num

    def __open_file(self):
        self.data = h5py.File(self.path, "r")[self.mode]

    def __getitem__(self, idx):
        if not hasattr(self, "data"):
            self.__open_file()

        u = torch.tensor(self.data["u"][idx])
        vx = torch.tensor(self.data["vx"][idx])
        vy = torch.tensor(self.data["vy"][idx])

        # u = u / u.amax(dim=(-2, -1))[..., None, None]
        # v_norm = torch.sqrt(torch.square(vx) + torch.square(vy))
        # vx = vx / v_norm.amax(dim=(-2, -1))[..., None, None]
        # vy = vy / v_norm.amax(dim=(-2, -1))[..., None, None]

        # concatenate the velocity components such that [vx(t0), vy(t0), vx(t1), vy(t1), ...]
        v = torch.cat((vx[:, None], vy[:, None]), dim=1)
        # v = v.view(-1, *v.shape[-2:])
        if self.usegrid:
            gridx = torch.linspace(0, 1, self.data["x"][idx].shape[0])
            gridy = torch.linspace(0, 1, self.data["y"][idx].shape[0])
            gridx = gridx.reshape(1, gridx.size(0), 1,).repeat(
                1,
                1,
                gridy.size(0),
            )
            gridy = gridy.reshape(1, 1, gridy.size(0),).repeat(
                1,
                gridx.size(1),
                1,
            )
            grid = torch.cat((gridx[:, None], gridy[:, None]), dim=1)
            return u.unsqueeze(1).float(), v.float(), grid.float()
        else:
            return u.unsqueeze(1).float(), v.float(), None


class RandomizedPDETimeStepDataset(data.Dataset):
    def __init__(self, pdedataset, pde: PDEConfig, time_history: int, time_future: int, time_gap: int) -> None:
        super().__init__()
        self.pdedataset = pdedataset
        self.pde = pde
        self.time_history = time_history
        self.time_future = time_future
        self.time_gap = time_gap

    def __len__(self):
        return len(self.pdedataset) * self.pde.trajlen  # alignment with datapipe

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        idx = idx % len(self.pdedataset)  # get back to actual index
        # Length of trajectory
        time_resolution = self.pde.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.time_history
        # Number of future points to predict
        max_start_time = reduced_time_resolution - self.time_future - self.time_gap

        u, v, grid = self.pdedataset[idx]
        start_time = random.choices([t for t in range(self.pde.skip_nt, max_start_time + 1)], k=1)

        return datautils.create_data(
            self.pde,
            # u.squeeze(1),
            # v.view(-1, *v.shape[-2:]),
            u,
            v,
            grid,
            start_time[0],
            self.time_history,
            self.time_future,
            self.time_gap,
        )
