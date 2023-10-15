# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Optional

import torch

from pdearena.data.utils import PDEDataConfig

# TODO: these need to generalized further


def rollout2d(
    model: torch.nn.Module,
    initial_u: torch.Tensor,
    initial_v: torch.Tensor,
    grid: Optional[torch.Tensor],
    pde: PDEDataConfig,
    time_history: int,
    num_steps: int,
):
    traj_ls = []
    pred = torch.Tensor()
    data_vector = torch.Tensor().to(device=initial_u.device)
    for i in range(num_steps):
        if i == 0:
            if pde.n_scalar_components > 0:
                data_scalar = initial_u[:, :time_history]
            if pde.n_vector_components > 0:
                data_vector = initial_v[
                    :,
                    :time_history,
                ]

            data = torch.cat((data_scalar, data_vector), dim=2)

        else:
            data = torch.cat((data, pred), dim=1)
            data = data[
                :,
                -time_history:,
            ]

        # if model.usegrid:
        #     data = torch.cat((data, grid), dim=1)

        pred = model(data)
        traj_ls.append(pred)

    traj = torch.cat(traj_ls, dim=1)
    return traj


def cond_rollout2d(
    model: torch.nn.Module,
    initial_u: torch.Tensor,
    initial_v: torch.Tensor,
    delta_t: Optional[torch.Tensor],
    cond: Optional[torch.Tensor],
    grid: Optional[torch.Tensor],
    pde: PDEDataConfig,
    time_history: int,
    num_steps: int,
):
    traj_ls = []
    pred = torch.Tensor().to(device=initial_u.device)
    data_vector = torch.Tensor().to(device=initial_u.device)
    for i in range(num_steps):
        if i == 0:
            if pde.n_scalar_components > 0:
                data_scalar = initial_u[:, :time_history]
            if pde.n_vector_components > 0:
                data_vector = initial_v[
                    :,
                    :time_history,
                ]

            data = torch.cat((data_scalar, data_vector), dim=2)

        else:
            data = torch.cat((data, pred), dim=1)
            data = data[
                :,
                -time_history:,
            ]

        if grid is not None:
            data = torch.cat((data, grid), dim=1)

        if delta_t is not None:
            pred = model(data, delta_t, cond)
        else:
            pred = model(data, cond)
        traj_ls.append(pred)

    traj = torch.cat(traj_ls, dim=1)
    return traj


def rollout3d_maxwell(
    model: torch.nn.Module,
    initial_d: torch.Tensor,
    initial_h: torch.Tensor,
    time_history: int,
    num_steps: int,
):
    traj_ls = []
    pred = torch.Tensor()
    for i in range(num_steps):
        if i == 0:
            data = torch.cat((initial_d, initial_h), dim=2)
        else:
            data = torch.cat((data, pred), dim=1)  # along time
            data = data[
                :,
                -time_history:,
            ]

        pred = model(data)
        traj_ls.append(pred)

    traj = torch.cat(traj_ls, dim=1)
    return traj
