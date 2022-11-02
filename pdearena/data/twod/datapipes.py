# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
import os.path
import random
from typing import Optional

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torchdata.datapipes as dp
import xarray as xr

import pdearena.data.utils as datautils
from pdearena.pde import PDEConfig


class NavierStokesDatasetOpener(dp.iter.IterDataPipe):
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
                    else:
                        grid = None
                    yield u.unsqueeze(1).float(), v.float(), cond, grid


class WeatherDatasetOpener(dp.iter.IterDataPipe):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usevort=False,
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


class VortWeatherDatasetOpener(WeatherDatasetOpener):
    def __init__(self, dp, mode: str, limit_trajectories: Optional[int] = None, usegrid: bool = False) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=True, usegrid=usegrid)


class WeatherDatasetOpener2Day(WeatherDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=False, usegrid=usegrid, sample_rate=8)


class VortWeatherDatasetOpener2Day(WeatherDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=True, usegrid=usegrid, sample_rate=8)


class WeatherDatasetOpener1Day(WeatherDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=False, usegrid=usegrid, sample_rate=4)


class VortWeatherDatasetOpener1Day(WeatherDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=True, usegrid=usegrid, sample_rate=4)


class RandomTimeStepPDETrainData(dp.iter.IterDataPipe):
    def __init__(self, dp, pde: PDEConfig, reweigh=True) -> None:
        super().__init__()
        self.dp = dp
        self.pde = pde
        self.reweigh = reweigh

    def __iter__(self):
        time_resolution = self.pde.trajlen

        for (u, v, cond, grid) in self.dp:
            if self.reweigh:
                end_time = random.choices(range(1, time_resolution), k=1)[0]
                start_time = random.choices(range(0, end_time), weights=1 / np.arange(1, end_time + 1), k=1)[0]
            else:
                end_time = torch.randint(low=1, high=time_resolution, size=(1,), dtype=torch.long).item()
                start_time = torch.randint(low=0, high=end_time.item(), size=(1,), dtype=torch.long).item()

            delta_t = end_time - start_time
            yield (
                *datautils.create_time_conditioned_data(
                    self.pde, u, v, grid, start_time, end_time, torch.tensor([delta_t])
                ),
                cond,
            )


class TimestepPDEEvalData(dp.iter.IterDataPipe):
    def __init__(self, dp, pde: PDEConfig, delta_t: int) -> None:
        super().__init__()
        self.dp = dp
        self.pde = pde
        assert 2 * delta_t < self.pde.trajlen
        self.delta_t = delta_t

    def __iter__(self):

        for begin in range(self.pde.trajlen - self.delta_t):
            for (u, v, cond, grid) in self.dp:
                newu = u[begin :: self.delta_t, ...]
                newv = v[begin :: self.delta_t, ...]
                max_start_time = newu.size(0)
                for start in range(max_start_time - 1):
                    end = start + 1
                    data = torch.cat((newu[start : start + 1], newv[start : start + 1]), dim=1).unsqueeze(0)
                    if grid is not None:
                        data = torch.cat((data, grid), dim=1)
                    label = torch.cat((newu[end : end + 1], newv[end : end + 1]), dim=1).unsqueeze(0)
                    if data.size(1) == 0:
                        raise ValueError("Data is empty. Likely indexing issue.")
                    yield data, label, torch.tensor([self.delta_t]), cond


class RandomizedPDETrainData(dp.iter.IterDataPipe):
    def __init__(self, dp, pde: PDEConfig, time_history: int, time_future: int, time_gap: int) -> None:
        super().__init__()
        self.dp = dp
        self.pde = pde
        self.time_history = time_history
        self.time_future = time_future
        self.time_gap = time_gap

    def __iter__(self):
        # Length of trajectory
        time_resolution = self.pde.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.time_history
        # Number of future points to predict
        max_start_time = reduced_time_resolution - self.time_future - self.time_gap

        for batch in self.dp:
            if len(batch) == 3:
                (u, v, grid) = batch
                cond = None
            elif len(batch) == 4:
                (u, v, cond, grid) = batch
            else:
                raise ValueError(f"Unknown batch length of {len(batch)}.")

            # Choose initial random time point at the PDE solution manifold
            start_time = random.choices([t for t in range(self.pde.skip_nt, max_start_time + 1)], k=1)
            yield datautils.create_data(
                self.pde,
                u,
                v,
                grid,
                start_time[0],
                self.time_history,
                self.time_future,
                self.time_gap,
            )


class PDEEvalTimeStepData(dp.iter.IterDataPipe):
    def __init__(self, dp, pde: PDEConfig, time_history: int, time_future: int, time_gap: int) -> None:
        super().__init__()
        self.dp = dp
        self.pde = pde
        self.time_history = time_history
        self.time_future = time_future
        self.time_gap = time_gap

    def __iter__(self):
        # Length of trajectory
        time_resolution = self.pde.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.time_history
        # Number of future points to predict
        max_start_time = reduced_time_resolution - self.time_future - self.time_gap
        # We ignore these timesteps in the testing
        start_time = [t for t in range(self.pde.skip_nt, max_start_time + 1, self.time_gap + self.time_future)]
        for start in start_time:
            for (u, v, cond, grid) in self.dp:

                end_time = start + self.time_history
                target_start_time = end_time + self.time_gap
                target_end_time = target_start_time + self.time_future
                data_scalar = torch.Tensor()
                data_vector = torch.Tensor()
                labels_scalar = torch.Tensor()
                labels_vector = torch.Tensor()
                if self.pde.n_scalar_components > 0:
                    data_scalar = u[
                        start:end_time,
                        ...,
                    ]
                    labels_scalar = u[
                        target_start_time:target_end_time,
                        ...,
                    ]
                if self.pde.n_vector_components > 0:
                    data_vector = v[
                        start:end_time,
                        ...,
                    ]
                    labels_vector = v[
                        target_start_time:target_end_time,
                        ...,
                    ]

                # add batch dim
                data = torch.cat((data_scalar, data_vector), dim=1).unsqueeze(0)

                # add batch dim
                labels = torch.cat((labels_scalar, labels_vector), dim=1).unsqueeze(0)
                yield data, labels
