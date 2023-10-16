import os

import h5py
import numpy as np
import pytest
from torch.utils.data import DataLoader

from pdearena.data.datamodule import collate_fn_cat, collate_fn_stack
from pdearena.data.registry import DATAPIPE_REGISTRY
from pdearena.data.utils import PDEDataConfig


@pytest.fixture(scope="session")
def synthetic_kuramoto_sivashinsky(tmpdir_factory):
    """Generate an artificial Navier-Stokes dataset."""
    tmpdir = tmpdir_factory.mktemp("synth_kuramoto_sivashinsky")
    modes = ["train", "valid", "test"]
    seed = 42
    num_samples = 16
    nt = 32
    nx = 128
    pde = PDEDataConfig(n_scalar_components=1, n_vector_components=0, trajlen=nt // 4, n_spatial_dim=1)
    pde.nx = nx

    filenames = {}
    for mode in modes:
        file_name = os.path.join(tmpdir, f"kuramoto_sivashinsky_{mode}_{seed}_{num_samples}.h5")
        filenames[mode] = file_name

        h5f = h5py.File(file_name, "a")
        dataset = h5f.create_group(mode)
        h5f_u = dataset.create_dataset("pde_traj", (num_samples, nt, nx), dtype=float)
        h5f_u[...] = np.random.rand(num_samples, nt, nx)
        dt = dataset.create_dataset("dt", (num_samples,), dtype=float)
        dt[...] = np.random.rand(num_samples) * 0.1 + 0.15
        dx = dataset.create_dataset("dx", (num_samples,), dtype=float)
        dx[...] = np.random.rand(num_samples) * 0.1 + 0.2
        v = dataset.create_dataset("v", (num_samples,), dtype=float)
        v[...] = np.random.rand(num_samples) + 0.5

        h5f.close()

    return modes, pde, filenames


def test_kuramoto_sivashinsky_dataloader(synthetic_kuramoto_sivashinsky):
    modes, pde, filenames = synthetic_kuramoto_sivashinsky

    batch_size = 2
    time_history = 2
    time_future = 1
    time_gap = 0
    dps = DATAPIPE_REGISTRY["KuramotoSivashinsky1D"]
    for mode in modes:
        if mode == "train":
            train_dp = dps[mode](
                pde=pde,
                data_path=filenames[mode],
                limit_trajectories=-1,
                usegrid=False,
                time_history=time_history,
                time_future=time_future,
                time_gap=time_gap,
            )

            train_dataloader = DataLoader(
                dataset=train_dp,
                num_workers=1,
                pin_memory=True,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn_cat,
            )

            for idx, (x, y, cond) in enumerate(train_dataloader):
                assert x.shape[0] == y.shape[0] == cond.shape[0] == batch_size
                assert x.shape[1] == time_history
                assert y.shape[1] == time_future
                assert x.shape[2] == y.shape[2] == 1
                assert cond.shape[1] == 3
            assert idx > 0

        elif mode == "valid" or mode == "test":
            valid_dp1 = dps[mode][0](
                pde=pde,
                data_path=filenames[mode],
                limit_trajectories=-1,
                usegrid=False,
                time_history=time_history,
                time_future=time_future,
                time_gap=time_gap,
            )
            valid_dataloader1 = DataLoader(
                dataset=valid_dp1,
                num_workers=1,
                pin_memory=True,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=collate_fn_cat,
            )
            for idx, (x, y, cond) in enumerate(valid_dataloader1):
                print(x.shape, y.shape, cond.shape)
                assert x.shape[0] == y.shape[0] == cond.shape[0] == batch_size
                assert x.shape[1] == time_history
                assert y.shape[1] == time_future
                assert x.shape[2] == y.shape[2] == 1
                assert cond.shape[1] == 3
            assert idx > 0

            valid_dp2 = dps[mode][1](
                pde=pde,
                data_path=filenames[mode],
                limit_trajectories=-1,
                usegrid=False,
                time_history=time_history,
                time_future=time_future,
                time_gap=time_gap,
            )
            valid_dataloader2 = DataLoader(
                dataset=valid_dp2,
                num_workers=1,
                pin_memory=True,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=collate_fn_stack,
            )
            for idx, x in enumerate(valid_dataloader2):
                assert x[0].shape == (batch_size, pde.trajlen, pde.n_scalar_components, pde.nx)
                assert np.prod(x[1].shape) == 0  # No vector components
                assert x[2].shape == (batch_size, 3)
            assert idx > 0
