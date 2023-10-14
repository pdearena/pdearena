# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import functools
from typing import Optional

import h5py
import torch
import torchdata.datapipes as dp

from pdearena.data.datapipes_common import build_datapipes


class NavierStokesDatasetOpener(dp.iter.IterDataPipe):
    """DataPipe to load Navier-Stokes dataset.

    Args:
        dp (dp.iter.IterDataPipe): List of `hdf5` files containing Navier-Stokes data.
        mode (str): Mode to load data from. Can be one of `train`, `val`, `test`.
        limit_trajectories (int, optional): Limit the number of trajectories to load from individual `hdf5` file. Defaults to None.
        usegrid (bool, optional): Whether to output spatial grid or not. Defaults to False.

    Yields:
        (Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]): Tuple containing particle scalar field, velocity vector field, and optionally buoyancy force parameter value  and spatial grid.
    """

    def __init__(self, dp, mode: str, limit_trajectories: Optional[int] = None, usegrid: bool = False) -> None:
        super().__init__()
        self.dp = dp
        self.mode = mode
        self.limit_trajectories = limit_trajectories
        self.usegrid = usegrid

    def __iter__(self):
        for path in self.dp:
            with h5py.File(path, "r") as f:
                data = f[self.mode]
                if self.limit_trajectories is None or self.limit_trajectories == -1:
                    num = data["u"].shape[0]
                else:
                    num = self.limit_trajectories

                iter_start = 0
                iter_end = num

                for idx in range(iter_start, iter_end):
                    u = torch.tensor(data["u"][idx])
                    vx = torch.tensor(data["vx"][idx])
                    vy = torch.tensor(data["vy"][idx])
                    if "buo_y" in data:
                        cond = torch.tensor(data["buo_y"][idx]).unsqueeze(0).float()
                    else:
                        cond = None

                    v = torch.cat((vx[:, None], vy[:, None]), dim=1)

                    if self.usegrid:
                        gridx = torch.linspace(0, 1, data["x"][idx].shape[0])
                        gridy = torch.linspace(0, 1, data["y"][idx].shape[0])
                        gridx = gridx.reshape(
                            1,
                            gridx.size(0),
                            1,
                        ).repeat(
                            1,
                            1,
                            gridy.size(0),
                        )
                        gridy = gridy.reshape(
                            1,
                            1,
                            gridy.size(0),
                        ).repeat(
                            1,
                            gridx.size(1),
                            1,
                        )
                        grid = torch.cat((gridx[:, None], gridy[:, None]), dim=1)
                    else:
                        grid = None
                    yield u.unsqueeze(1).float(), v.float(), cond, grid


def _train_filter(fname):
    return "train" in fname and "h5" in fname


def _valid_filter(fname):
    return "valid" in fname and "h5" in fname


def _test_filter(fname):
    return "test" in fname and "h5" in fname


train_datapipe_ns = functools.partial(
    build_datapipes,
    dataset_opener=NavierStokesDatasetOpener,
    filter_fn=_train_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="train",
)
onestep_valid_datapipe_ns = functools.partial(
    build_datapipes,
    dataset_opener=NavierStokesDatasetOpener,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_ns = functools.partial(
    build_datapipes,
    dataset_opener=NavierStokesDatasetOpener,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
    onestep=False,
)

onestep_test_datapipe_ns = functools.partial(
    build_datapipes,
    dataset_opener=NavierStokesDatasetOpener,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=True,
)

trajectory_test_datapipe_ns = functools.partial(
    build_datapipes,
    dataset_opener=NavierStokesDatasetOpener,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=False,
)

train_datapipe_ns_cond = functools.partial(
    build_datapipes,
    dataset_opener=NavierStokesDatasetOpener,
    filter_fn=_train_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="train",
    onestep=True,
    conditioned=True,
)

onestep_valid_datapipe_ns_cond = functools.partial(
    build_datapipes,
    dataset_opener=NavierStokesDatasetOpener,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
    onestep=True,
    conditioned=True,
)

onestep_test_datapipe_ns_cond = functools.partial(
    build_datapipes,
    dataset_opener=NavierStokesDatasetOpener,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=True,
    conditioned=True,
)

trajectory_test_datapipe_ns_cond = functools.partial(
    build_datapipes,
    dataset_opener=NavierStokesDatasetOpener,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=False,
    conditioned=True,
)
