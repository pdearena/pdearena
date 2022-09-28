import os
import glob
import xarray as xr
import torch
import click
import h5py
import numpy as np
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
        }
    }
    return stats


@click.command()
@click.argument("datapath", type=click.Path(exists=True))
@click.option("--dataset", type=str, default="weather")
def main(datapath, dataset):
    if dataset == "weather":
        stats = compute_normstats_weather(datapath)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
    print(stats)
    torch.save(stats, os.path.join(datapath, "normstats.pt"))
    return stats


if __name__ == "__main__":
    main()
