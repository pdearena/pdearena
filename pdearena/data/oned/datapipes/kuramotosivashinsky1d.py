# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import functools

import h5py
import numpy as np
import torch
import torchdata.datapipes as dp

from pdearena.data.datapipes_common import build_datapipes


class KuramotoSivashinskyDatasetOpener(dp.iter.IterDataPipe):
    """DataPipe to load the Kuramoto-Sivashinsky dataset.

    Args:
        dp (dp.iter.IterDataPipe): List of `hdf5` files containing Kolmogorov data.
        mode (str): Mode to load data from. Can be one of `train`, `val`, `test`.
        preload (bool, optional): Whether to preload all data into memory. Defaults to True.
        allow_shuffle (bool, optional): Whether to shuffle the data, recommended when preloading data. Defaults to True.
        resolution (int, optional): Which resolution to load. Defaults to full data resolution.
        usegrid (bool, optional): Whether to output spatial grid or not. Defaults to False.

    Yields:
        (Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]): Tuple containing particle scalar field, velocity vector field, and optionally buoyancy force parameter value  and spatial grid.
    """

    def __init__(
        self,
        dp,
        mode: str,
        preload: bool = True,
        allow_shuffle: bool = True,
        resolution: int = -1,
        usegrid: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dp = dp
        self.mode = mode
        self.allow_shuffle = allow_shuffle
        self.dtype = np.float32
        self.resolution = resolution
        self.usegrid = usegrid
        print(f"Loading {mode} data from {len([p for p in dp])} files.")
        self.storage = {}
        if preload:
            for path in self.dp:
                self.storage[path] = self._load_data(path)

    def _load_data(self, path):
        if path in self.storage:
            return self.storage[path]
        else:
            with h5py.File(path, "r") as f:
                data_h5 = f[self.mode]
                data_key = [k for k in data_h5.keys() if k.startswith("pde_")][0]
                data = {
                    "u": torch.tensor(data_h5[data_key][:].astype(self.dtype)),
                    "dt": torch.tensor(data_h5["dt"][:].astype(self.dtype)),
                    "dx": torch.tensor(data_h5["dx"][:].astype(self.dtype)),
                }
                if "v" in data_h5:
                    data["v"] = torch.tensor(data_h5["v"][:].astype(self.dtype))

                data["orig_dt"] = data["dt"].clone()
                if data["u"].ndim == 3:
                    data["u"] = data["u"].unsqueeze(dim=-2)  # Add channel dimension
                # Scaling dt to be in the range [0, 10] to be visible changes in fourier embeds
                if data["dt"].min() > 0.3 and data["dt"].max() < 0.5:
                    data["dt"] = (data["dt"] - 0.3) * 50.0
                elif data["dt"].min() > 0.15 and data["dt"].max() < 0.25:
                    data["dt"] = (data["dt"] - 0.15) * 100.0
                else:
                    print(
                        f"WARNING: dt is not in the expected range (min {data['dt'].min()}, max {data['dt'].max()}, mean {data['dt'].mean()}) - scaling may be incorrect."
                    )
                # Scaling dx to be in the range [0, 10] to be visible changes in fourier embeds
                if data["dx"].min() > 0.4 and data["dx"].max() < 0.6:
                    data["dx"] = (data["dx"] - 0.4) * 50.0
                elif data["dx"].min() > 0.2 and data["dx"].max() < 0.3:
                    data["dx"] = (data["dx"] - 0.2) * 100.0
                else:
                    print(
                        f"WARNING: dt is not in the expected range (min {data['dx'].min()}, max {data['dx'].max()}, mean {data['dx'].mean()}) - scaling may be incorrect."
                    )

                if "v" in data:
                    if data["v"].min() >= 0.5 and data["v"].max() <= 1.5:
                        data["v"] = (data["v"] - 0.5) * 100.0
                    else:
                        print(
                            f"WARNING: v is not in the expected range (min {data['v'].min()}, max {data['v'].max()}, mean {data['v'].mean()}) - scaling may be incorrect."
                        )

            return data

    def __iter__(self):
        for path in self.dp:
            data = self._load_data(path)
            u = data["u"]
            if u.ndim == 3:
                u = u.unsqueeze(0)
            if self.resolution > 0 and u.shape[-1] > self.resolution:
                step_size = u.shape[-1] // self.resolution
                start_idx = 0 if not self.mode == "train" else np.random.randint(0, step_size)
                u = u[..., start_idx::step_size]
            idxs = np.arange(int(u.shape[0] * self.data_proportion))
            if self.mode == "train" and self.allow_shuffle:
                np.random.shuffle(idxs)
            for i in range(idxs.shape[0]):
                idx = idxs[i]
                cond = [data["dt"][idx], data["dx"][idx]]
                if "v" in data:
                    cond.append(data["v"][idx])
                if self.usegrid:
                    grid = np.linspace(0, 1, u.shape[-1])
                else:
                    grid = None
                yield u[idx], torch.zeros_like(u[idx, :, 0:0]), torch.tensor(cond), grid


def _train_filter(fname):
    return "_train_" in fname and "h5" in fname


def _valid_filter(fname):
    return "_valid_" in fname and "h5" in fname


def _test_filter(fname):
    return "_test_" in fname and "h5" in fname


train_datapipe_ks = functools.partial(
    build_datapipes,
    dataset_opener=KuramotoSivashinskyDatasetOpener,
    filter_fn=_train_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="train",
)

onestep_valid_datapipe_ks = functools.partial(
    build_datapipes,
    dataset_opener=KuramotoSivashinskyDatasetOpener,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_ks = functools.partial(
    build_datapipes,
    dataset_opener=KuramotoSivashinskyDatasetOpener,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
    onestep=False,
)

onestep_test_datapipe_ks = functools.partial(
    build_datapipes,
    dataset_opener=KuramotoSivashinskyDatasetOpener,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=True,
)

trajectory_test_datapipe_ks = functools.partial(
    build_datapipes,
    dataset_opener=KuramotoSivashinskyDatasetOpener,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=False,
)