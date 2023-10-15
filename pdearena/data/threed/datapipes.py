# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import functools
import random
from typing import Callable, Optional

import h5py
import torch
import torchdata.datapipes as dp

import pdearena.data.utils as datautils
from pdearena.data.utils import PDEDataConfig


def build_maxwell_datapipes(
    pde: PDEDataConfig,
    data_path,
    limit_trajectories,
    usegrid: bool,
    dataset_opener: Callable[..., dp.iter.IterDataPipe],
    lister: Callable[..., dp.iter.IterDataPipe],
    sharder: Callable[..., dp.iter.IterDataPipe],
    filter_fn: Callable[..., dp.iter.IterDataPipe],
    mode: str,
    time_history=1,
    time_future=1,
    time_gap=0,
    onestep=False,
):
    """Build datapipes for training and evaluation.

    Args:
        pde (PDEDataConfig): PDE configuration.
        data_path (str): Path to the data.
        limit_trajectories (int): Number of trajectories to use.
        usegrid (bool): Whether to use spatial grid as input.
        dataset_opener (Callable[..., dp.iter.IterDataPipe]): Dataset opener.
        lister (Callable[..., dp.iter.IterDataPipe]): List files.
        sharder (Callable[..., dp.iter.IterDataPipe]): Shard files.
        filter_fn (Callable[..., dp.iter.IterDataPipe]): Filter files.
        mode (str): Mode of the data. ["train", "valid", "test"]
        time_history (int, optional): Number of time steps in the past. Defaults to 1.
        time_future (int, optional): Number of time steps in the future. Defaults to 1.
        time_gap (int, optional): Number of time steps between the past and the future to be skipped. Defaults to 0.
        onestep (bool, optional): Whether to use one-step prediction. Defaults to False.

    Returns:
        dpipe (IterDataPipe): IterDataPipe for training and evaluation.
    """
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
        dpipe = RandomizedPDETrainData3D(
            dpipe,
            pde,
            time_history,
            time_future,
            time_gap,
        )
    else:
        # Evaluation data is not randomized.
        if onestep:
            dpipe = PDEEvalTimeStepData3D(
                dpipe,
                pde,
                time_history,
                time_future,
                time_gap,
            )

    return dpipe


class PDEDatasetOpener3D(dp.iter.IterDataPipe):
    def __init__(self, dp, mode: str, limit_trajectories: Optional[int] = None, usegrid: bool = False) -> None:
        super().__init__()
        self.dp = dp
        self.mode = mode
        self.limit_trajectories = limit_trajectories
        if usegrid:
            raise NotImplementedError("3D grids")

    def __iter__(self):
        for path in self.dp:
            f = h5py.File(path, "r")
            data = f[self.mode]
            if self.limit_trajectories is None or self.limit_trajectories == -1:
                num = data["d_field"].shape[0]
            else:
                num = self.limit_trajectories

            # Different workers should be using different trajectory batches
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                num_workers = min(worker_info.num_workers, num)
                per_worker = int(num / float(num_workers))
                iter_start = worker_info.id * per_worker
                iter_end = min(iter_start + per_worker, num)
            else:
                iter_start = 0
                iter_end = num

            for idx in range(iter_start, iter_end):
                d_field = torch.tensor(data["d_field"][idx])
                h_field = torch.tensor(data["h_field"][idx])
                # to T, C, X, Y, Z
                d_field = d_field.permute(
                    0,
                    4,
                    1,
                    2,
                    3,
                )
                h_field = h_field.permute(
                    0,
                    4,
                    1,
                    2,
                    3,
                )

                yield d_field.float(), h_field.float()


class RandomizedPDETrainData3D(dp.iter.IterDataPipe):
    def __init__(self, dp, pde: PDEDataConfig, time_history: int, time_future: int, time_gap: int) -> None:
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

        for d, h in self.dp:
            # Choose initial random time point at the PDE solution manifold
            start_time = random.choices([t for t in range(max_start_time + 1)], k=1)
            yield datautils.create_maxwell_data(d, h, start_time[0], self.time_history, self.time_future, self.time_gap)


class PDEEvalTimeStepData3D(dp.iter.IterDataPipe):
    def __init__(self, dp, pde: PDEDataConfig, time_history: int, time_future: int, time_gap: int) -> None:
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
            for d, h, _ in self.dp:
                end_time = start + self.time_history
                target_start_time = end_time + self.time_gap
                target_end_time = target_start_time + self.time_future
                data_dfield = torch.Tensor()
                labels_dfield = torch.Tensor()
                data_hfield = torch.Tensor()
                labels_hfield = torch.Tensor()
                data_dfield = d[
                    start:end_time,
                    ...,
                ]
                labels_dfield = d[
                    target_start_time:target_end_time,
                    ...,
                ]
                data_hfield = h[
                    start:end_time,
                    ...,
                ]
                labels_hfield = h[
                    target_start_time:target_end_time,
                    ...,
                ]

                data = torch.cat((data_dfield, data_hfield), dim=1).unsqueeze(0)  # add batch dim
                labels = torch.cat((labels_dfield, labels_hfield), dim=1).unsqueeze(0)  # add batch dim

                yield data, labels


def _train_filter(fname):
    return "train" in fname and "h5" in fname


def _valid_filter(fname):
    return "valid" in fname and "h5" in fname


def _test_filter(fname):
    return "test" in fname and "h5" in fname


train_datapipe_maxwell = functools.partial(
    build_maxwell_datapipes,
    dataset_opener=PDEDatasetOpener3D,
    filter_fn=_train_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="train",
)
onestep_valid_datapipe_maxwell = functools.partial(
    build_maxwell_datapipes,
    dataset_opener=PDEDatasetOpener3D,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
    onestep=True,
)
rollout_valid_datapipe_maxwell = functools.partial(
    build_maxwell_datapipes,
    dataset_opener=PDEDatasetOpener3D,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
    onestep=False,
)
onestep_test_datapipe_maxwell = functools.partial(
    build_maxwell_datapipes,
    dataset_opener=PDEDatasetOpener3D,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=True,
)
rollout_test_datapipe_maxwell = functools.partial(
    build_maxwell_datapipes,
    dataset_opener=PDEDatasetOpener3D,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=False,
)
