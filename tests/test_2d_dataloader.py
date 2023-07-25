import pytest
import h5py
import numpy as np
import os
from pdedatagen.pde import NavierStokes2D
from pdearena.data.registry import DATAPIPE_REGISTRY
from torch.utils.data import DataLoader
from pdearena.data.datamodule import collate_fn_cat


@pytest.fixture(scope="session")
def synthetic_navier_stokes(tmpdir_factory):
    """Generate an artificial Navier-Stokes dataset."""
    tmpdir = tmpdir_factory.mktemp("synth_navier_stokes")
    pde = NavierStokes2D()
    mode = "train"
    seed = 42
    num_samples = 16
    nt = 14
    pde.nt = nt
    nx = 128
    ny = 128
    file_name = os.path.join(tmpdir, "_".join([str(pde), mode, str(seed), f"{pde.buoyancy_y:.5f}"]))
    file_name = file_name + "_" + str(num_samples) + ".h5"
    h5f = h5py.File(file_name, "a")
    
    dataset = h5f.create_group(mode)
    h5f_u = dataset.create_dataset("u", (num_samples, nt, nx, ny), dtype=float)
    h5f_u[...] = np.random.rand(num_samples, nt, nx, ny)
    h5f_vx = dataset.create_dataset("vx", (num_samples, nt, nx, ny), dtype=float)
    h5f_vx[...] = np.random.rand(num_samples, nt, nx, ny)
    h5f_vy = dataset.create_dataset("vy", (num_samples, nt, nx, ny), dtype=float)
    h5f_vy[...] = np.random.rand(num_samples, nt, nx, ny)
    tcoord = dataset.create_dataset("t", (num_samples, nt), dtype=float)
    tcoord[...] = np.random.rand(num_samples, nt)
    dt = dataset.create_dataset("dt", (num_samples,), dtype=float)
    dt[...] = np.random.rand(num_samples)
    xcoord = dataset.create_dataset("x", (num_samples, nx), dtype=float)
    xcoord[...] = np.random.rand(num_samples, nx)
    dx = dataset.create_dataset("dx", (num_samples,), dtype=float)
    dx[...] = np.random.rand(num_samples)
    ycoord = dataset.create_dataset("y", (num_samples, ny), dtype=float)
    ycoord[...] = np.random.rand(num_samples, ny)
    dy = dataset.create_dataset("dy", (num_samples,), dtype=float)
    dy[...] = np.random.rand(num_samples)
    buo_y = dataset.create_dataset("buo_y", (num_samples,), dtype=float)
    buo_y[...] = np.random.rand(num_samples)

    h5f.close()

    return mode, pde, file_name


def test_navier_stokes_dataloader(synthetic_navier_stokes):
    mode, pde, file_name = synthetic_navier_stokes

    batch_size = 2
    time_history = 2
    time_future = 1
    time_gap = 0
    dps = DATAPIPE_REGISTRY["NavierStokes2D"]
    test_dp = dps[mode](
        pde=pde,
        data_path=file_name,
        limit_trajectories=-1,
        usegrid=False,
        time_history=time_history,
        time_future=time_future,
        time_gap=time_gap,
        )
    
    test_dataloader = DataLoader(
        dataset=test_dp,
        num_workers=1,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_cat,
    )
    
    for idx, (x, y) in enumerate(test_dataloader):
        assert x.shape[0] == y.shape[0] == batch_size
        assert x.shape[1] == time_history
        assert y.shape[1] == time_future
        assert x.shape[2] == y.shape[2] == 3
    assert idx > 0

