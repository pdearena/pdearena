# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os.path
from subprocess import run

import xarray as xr


def generate_trajectories_shallowwater(savedir, num_samples, seed):
    import juliapkg

    juliapkg.resolve()
    file = os.path.join(os.path.dirname(__file__), "datagen.jl")
    run(
        [
            juliapkg.executable(),
            f"--project={juliapkg.project()}",
            "--startup-file=no",
            file,
            savedir,
            str(num_samples),
            str(seed),
        ]
    )


def collect_data2zarr(datapath):
    datals = os.path.join(datapath, "seed=*", "run*", "output.nc")
    data = xr.open_mfdataset(datals, concat_dim="b", combine="nested", parallel=True)
    outpath = os.path.join(os.path.dirname(datapath), f"{os.path.basename(datapath)}.zarr")
    data.to_zarr(outpath)
