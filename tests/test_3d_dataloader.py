import pytest
import h5py
import numpy as np
import os
from pdedatagen.pde import Maxwell3D
from pdearena.data.registry import DATAPIPE_REGISTRY
from torch.utils.data import DataLoader
from pdearena.data.datamodule import collate_fn_cat


@pytest.fixture(scope="session")
def synthetic_maxwell(tmpdir_factory):
    """Generate an artificial Maxwell dataset."""
    tmpdir = tmpdir_factory.mktemp("synth_maxwell")
    pde = Maxwell3D()
    mode = "train"
    seed = 42
    num_samples = 16
    nt = 8
    pde.nt = nt
    nx = 32
    ny = 32
    nz = 32
    file_name = os.path.join(tmpdir, "_".join([str(pde), mode, str(seed)]))
    file_name = file_name + "_" + str(num_samples) + ".h5"
    h5f = h5py.File(file_name, "a")
    
    dataset = h5f.create_group(mode)
    d_field = dataset.create_dataset(
        "d_field",
        (num_samples, nt, nx, ny, nz, 3),
        dtype=float,
    )
    d_field[...] = np.random.rand(num_samples, nt, nx, ny, nz, 3)
    h_field = dataset.create_dataset(
        "h_field",
        (num_samples, nt, nx, ny, nz, 3),
        dtype=float,
    )
    h_field[...] = np.random.rand(num_samples, nt, nx, ny, nz, 3)

    h5f.close()

    return mode, pde, file_name


def test_maxwell_dataloader(synthetic_maxwell):
    mode, pde, file_name = synthetic_maxwell
    

