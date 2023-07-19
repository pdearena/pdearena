import pytest
import h5py
import numpy as np
import os
from pdedatagen.pde import NavierStokes2D


@pytest.fixture(scope="session")
def synthetic_navier_stokes(tmpdir_factory):
    """Generate an artificial Navier-Stokes dataset."""
    tmpdir = tmpdir_factory.mktemp("synth_navier_stokes")
    pde = NavierStokes2D()
    mode = "train"
    seed = 42
    num_samples = 16
    nt = 14
    nx = 128
    ny = 128
    file_name = os.path.join(tmpdir, "_".join([str(pde), mode, str(seed), f"{pde.buoyancy_y:.5f}"]))
    if mode == "train":
        file_name = file_name + "_" + str(num_samples) + ".h5"
    h5f = h5py.File(file_name, "a")
    
    dataset = h5f.create_group(mode)
    h5f_u = dataset.create_dataset("u", (num_samples, nt, nx, ny), dtype=float)
    h5f_u[...] = np.random.rand(num_samples, nt, nx, ny)

    h5f.close()

    return mode, file_name

    # tcoord, xcoord, ycoord, dt, dx, dy = {}, {}, {}, {}, {}, {}
    # h5f_u, h5f_vx, h5f_vy = {}, {}, {}

    # nt, nx, ny = pde.grid_size[0], pde.grid_size[1], pde.grid_size[2]
    # # The scalar field u, the components of the vector field vx, vy,
    # # the coordinations (tcoord, xcoord, ycoord) and dt, dx, dt are saved
    # h5f_u = dataset.create_dataset("u", (num_samples, nt, nx, ny), dtype=float)
    # h5f_vx = dataset.create_dataset("vx", (num_samples, nt, nx, ny), dtype=float)
    # h5f_vy = dataset.create_dataset("vy", (num_samples, nt, nx, ny), dtype=float)
    # tcoord = dataset.create_dataset("t", (num_samples, nt), dtype=float)
    # dt = dataset.create_dataset("dt", (num_samples,), dtype=float)
    # xcoord = dataset.create_dataset("x", (num_samples, nx), dtype=float)
    # dx = dataset.create_dataset("dx", (num_samples,), dtype=float)
    # ycoord = dataset.create_dataset("y", (num_samples, ny), dtype=float)
    # dy = dataset.create_dataset("dy", (num_samples,), dtype=float)
    # buo_y = dataset.create_dataset("buo_y", (num_samples,), dtype=float)


def test_navier_stokes_dataloader(synthetic_navier_stokes):
    mode, file_name = synthetic_navier_stokes


