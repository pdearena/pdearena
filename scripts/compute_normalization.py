# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import glob
import os

import click
import h5py
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm


def compute_normstats_weather(datapath):
    traindata = os.path.join(datapath, "train", "seed=*", "run*", "output.nc")
    data = xr.open_mfdataset(traindata, concat_dim="b", combine="nested")
    pres_mean = data.pres.mean(("b", "time")).to_numpy()
    pres_std = data.pres.std(("b", "time")).to_numpy()
    u_mean = data.u.mean(("b", "time")).to_numpy()
    u_std = data.u.std(("b", "time")).to_numpy()
    v_mean = data.v.mean(("b", "time")).to_numpy()
    v_std = data.v.std(("b", "time")).to_numpy()
    vor_mean = data.vor.mean(("b", "time")).to_numpy()
    vor_std = data.vor.std(("b", "time")).to_numpy()

    stats = {
        "pres": {
            "mean": torch.tensor(pres_mean),
            "std": torch.tensor(pres_std),
        },
        "u": {
            "mean": torch.tensor(u_mean),
            "std": torch.tensor(u_std),
        },
        "v": {
            "mean": torch.tensor(v_mean),
            "std": torch.tensor(v_std),
        },
        "vor": {
            "mean": torch.tensor(vor_mean),
            "std": torch.tensor(vor_std),
        },
    }
    return stats


def compute_normstats_maxwell(datapath):
    filelist = glob.glob(os.path.join(datapath, "*train*.h5"))

    d_field_st = {"sum": None, "sqsum": None, "count": 0}
    h_field_st = {"sum": None, "sqsum": None, "count": 0}
    for file in tqdm(filelist):
        with h5py.File(file, "r") as f:
            data = f["train"]
            d_field = data["d_field"]
            h_field = data["h_field"]
            # B T X Y Z C
            if d_field_st["sum"] is None:
                d_field_st["sum"] = np.sum(d_field, axis=(0, 1))
                d_field_st["sqsum"] = np.sum(np.asarray(d_field) ** 2, axis=(0, 1))
                d_field_st["count"] = np.prod(d_field.shape[0:2])
            else:
                d_field_st["sum"] += np.sum(d_field, axis=(0, 1))
                d_field_st["sqsum"] += np.sum(np.asarray(d_field) ** 2, axis=(0, 1))
                d_field_st["count"] += np.prod(d_field.shape[0:2])

            if h_field_st["sum"] is None:
                h_field_st["sum"] = np.sum(h_field, axis=(0, 1))
                h_field_st["sqsum"] = np.sum(np.asarray(h_field) ** 2, axis=(0, 1))
                h_field_st["count"] = np.prod(h_field.shape[0:2])
            else:
                h_field_st["sum"] += np.sum(h_field, axis=(0, 1))
                h_field_st["sqsum"] += np.sum(np.asarray(h_field) ** 2, axis=(0, 1))
                h_field_st["count"] += np.prod(h_field.shape[0:2])

    stats = {
        "d_field": {
            "mean": torch.tensor(d_field_st["sum"] / d_field_st["count"]),
            "std": torch.tensor(
                np.sqrt(d_field_st["sqsum"] / d_field_st["count"] - d_field_st["sum"] ** 2 / d_field_st["count"] ** 2)
            ),
        },
        "h_field": {
            "mean": torch.tensor(h_field_st["sum"] / h_field_st["count"]),
            "std": torch.tensor(
                np.sqrt(h_field_st["sqsum"] / h_field_st["count"] - h_field_st["sum"] ** 2 / h_field_st["count"] ** 2)
            ),
        },
    }
    return stats


@click.command()
@click.argument("datapath", type=click.Path(exists=True))
@click.option("--dataset", type=str, default="shallowwater")
def main(datapath, dataset):
    if dataset == "shallowwater":
        stats = compute_normstats_weather(datapath)
    elif dataset == "maxwell":
        stats = compute_normstats_maxwell(datapath)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
    print(stats)
    torch.save(stats, os.path.join(datapath, "normstats.pt"))
    return stats


if __name__ == "__main__":
    main()
