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
    modes = ["train", "valid", "test"]
    seed = 42
    num_samples = 16
    nt = 8
    pde.nt = nt
    nx = 32
    ny = 32
    nz = 32

    filenames = {}
    for mode in modes:
        file_name = os.path.join(tmpdir, "_".join([str(pde), mode, str(seed)]))
        file_name = file_name + "_" + str(num_samples) + ".h5"
        filenames[mode] = file_name

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

    return modes, pde, filenames


def test_maxwell_dataloader(synthetic_maxwell):
    modes, pde, filenames = synthetic_maxwell

    batch_size = 2
    time_history = 1
    time_future = 1
    time_gap = 0
    dps = DATAPIPE_REGISTRY["Maxwell3D"]
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
            breakpoint()
            for idx, (x, y) in enumerate(train_dataloader):
                breakpoint()
                assert x.shape[0] == y.shape[0] == batch_size
                assert x.shape[1] == time_history
                assert y.shape[1] == time_future
                assert x.shape[2] == y.shape[2] == 3
            # assert idx > 0
    

