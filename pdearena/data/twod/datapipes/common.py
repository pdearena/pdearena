# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
from typing import Sequence, Union

import numpy as np
import torch
import torchdata.datapipes as dp

import pdearena.data.utils as datautils
from pdearena.pde import PDEConfig


def build_datapipes(
    pde: PDEConfig,
    data_path,
    limit_trajectories,
    usegrid,
    dataset_opener,
    lister,
    sharder,
    filter_fn,
    mode,
    time_history=1,
    time_future=1,
    time_gap=0,
    onestep=False,
    conditioned=False,
):
    dpipe = lister(
        data_path,
    ).filter(filter_fn=filter_fn)
    if mode == "train":
        dpipe = dpipe.shuffle()

    dpipe = dataset_opener(
        sharder(dpipe),
        mode=mode,
        limit_trajectories=limit_trajectories,
        usegrid=usegrid,
    )
    if mode == "train":
        # Make sure that in expectation we have seen all the data despite randomization
        dpipe = dpipe.cycle(pde.trajlen)

    if mode == "train":
        # Training data is randomized
        if conditioned:
            dpipe = RandomTimeStepPDETrainData(dpipe, pde)
        else:
            dpipe = RandomizedPDETrainData(dpipe, pde, time_history, time_future, time_gap)
    else:
        # Evaluation data is not randomized.
        if onestep:
            dpipe = PDEEvalTimeStepData(dpipe, pde, time_history, time_future, time_gap)

    return dpipe


class ZarrLister(dp.iter.IterDataPipe):
    def __init__(
        self,
        root: Union[str, Sequence[str], dp.iter.IterDataPipe] = ".",
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = [
                root,
            ]
        if not isinstance(root, dp.iter.IterDataPipe):
            root = dp.iter.IterableWrapper(root)

        self.datapipe: dp.iter.IterDataPipe = root

    def __iter__(self):
        for path in self.datapipe:
            for dirname in os.listdir(path):
                if dirname.endswith(".zarr"):
                    yield os.path.join(path, dirname)


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
