# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import math
import os
from typing import Optional

import torch
import torch.distributed as dist
import torchdata.datapipes as dp
import xarray as xr

from .common import ZarrLister, build_datapipes


class ShallowWaterDatasetOpener(dp.iter.IterDataPipe):
    """DataPipe for loading the shallow water dataset

    Args:
        dp: datapipe with paths to load the dataset from.
        mode (str): "train" or "valid" or "test"
        limit_trajectories: number of trajectories to load from the dataset
        usevort (bool): whether to use vorticity or velocity. If False, velocity is returned.
        usegrid (bool): whether to use grid or not. If False, no grid is returned.
        sample_rate: sample rate for the data. Default is 1, which means no sub-sampling.

    Note:
        We manually manage the data distribution across workers and processes. So make sure not to use `torchdata`'s [dp.iter.Sharder][torchdata.datapipes.iter.ShardingFilter] with this data pipe.
    """

    def __init__(
        self,
        dp: dp.iter.IterDataPipe,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usevort: bool = False,
        usegrid: bool = False,
        sample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.dp = dp
        self.mode = mode
        self.limit_trajectories = limit_trajectories
        self.usevort = usevort
        self.usegrid = usegrid
        self.sample_rate = sample_rate

    def __iter__(self):
        for path in self.dp:
            if "zarr" in path:
                data = xr.open_zarr(path)
            else:
                # Note that this is much slower
                data = xr.open_mfdataset(
                    os.path.join(path, "seed=*", "run*", "output.nc"),
                    concat_dim="b",
                    combine="nested",
                    parallel=True,
                )

            normstat = torch.load(os.path.join(path, "..", "normstats.pt"))
            if self.limit_trajectories is None or self.limit_trajectories == -1:
                num = data["u"].shape[0]
            else:
                num = self.limit_trajectories

            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1

            # Different workers should be using different trajectory batches
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                num_workers_per_dist = min(worker_info.num_workers, num)
                num_shards = num_workers_per_dist * world_size
                per_worker = int(math.floor(num / float(num_shards)))
                wid = rank * num_workers_per_dist + worker_info.id
                iter_start = wid * per_worker
                iter_end = iter_start + per_worker
            else:
                per_dist = int(math.floor(num / float(world_size)))
                iter_start = rank * per_dist
                iter_end = iter_start + per_dist

            for idx in range(iter_start, iter_end):
                if self.usevort:
                    vort = torch.tensor(data["vor"][idx].to_numpy())
                    vort = (vort - normstat["vor"]["mean"]) / normstat["vor"]["std"]
                else:
                    u = torch.tensor(data["u"][idx].to_numpy())
                    v = torch.tensor(data["v"][idx].to_numpy())
                    vecf = torch.cat((u, v), dim=1)

                pres = torch.tensor(data["pres"][idx].to_numpy())

                pres = (pres - normstat["pres"]["mean"]) / normstat["pres"]["std"]
                pres = pres.unsqueeze(1)

                if self.sample_rate > 1:
                    # TODO: hardocded skip_nt=4
                    pres = pres[4 :: self.sample_rate]
                    if self.usevort:
                        vort = vort[4 :: self.sample_rate]
                    else:
                        vecf = vecf[4 :: self.sample_rate]
                if self.usegrid:
                    raise NotImplementedError("Grid not implemented for weather data")
                else:
                    if self.usevort:
                        yield torch.cat((pres, vort), dim=1).float(), None, None, None
                    else:
                        yield pres.float(), vecf.float(), None, None


class VortShallowWaterDatasetOpener(ShallowWaterDatasetOpener):
    def __init__(self, dp, mode: str, limit_trajectories: Optional[int] = None, usegrid: bool = False) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=True, usegrid=usegrid)


class ShallowWaterDatasetOpener2Day(ShallowWaterDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=False, usegrid=usegrid, sample_rate=8)


class VortShallowWaterDatasetOpener2Day(ShallowWaterDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=True, usegrid=usegrid, sample_rate=8)


class ShallowWaterDatasetOpener1Day(ShallowWaterDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=False, usegrid=usegrid, sample_rate=4)


class VortShallowWaterDatasetOpener1Day(ShallowWaterDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=True, usegrid=usegrid, sample_rate=4)


def _sharder(x):
    return x


def _weathertrain_filter(fname):
    return "train.zarr" in fname


def _weathervalid_filter(fname):
    return "valid.zarr" in fname


def _weathertest_filter(fname):
    return "test.zarr" in fname


train_datapipe_2day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathertrain_filter,
    sharder=_sharder,
    mode="train",
)
onestep_valid_datapipe_2day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_2day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=False,
)
onestep_test_datapipe_2day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=True,
)
trajectory_test_datapipe_2day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=False,
)

train_datapipe_2day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathertrain_filter,
    sharder=_sharder,
    mode="train",
)
onestep_valid_datapipe_2day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_2day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=False,
)
onestep_test_datapipe_2day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=True,
)
trajectory_test_datapipe_2day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=False,
)

train_datapipe_1day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener1Day,
    lister=ZarrLister,
    filter_fn=_weathertrain_filter,
    sharder=_sharder,
    mode="train",
)
onestep_valid_datapipe_1day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener1Day,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_1day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener1Day,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=False,
)
onestep_test_datapipe_1day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener1Day,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=True,
)
trajectory_test_datapipe_1day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener1Day,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=False,
)

train_datapipe_1day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener1Day,
    lister=ZarrLister,
    filter_fn=_weathertrain_filter,
    sharder=_sharder,
    mode="train",
)
onestep_valid_datapipe_1day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener1Day,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_1day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener1Day,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=False,
)
onestep_test_datapipe_1day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener1Day,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=True,
)
trajectory_test_datapipe_1day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener1Day,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=False,
)
