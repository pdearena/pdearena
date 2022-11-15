import json
import os.path
import time
from datetime import datetime

import click
import torch
from omegaconf import OmegaConf

from pdearena.data.utils import PDEDataConfig
from pdearena.models.pdemodel import get_model
from pdearena.models.registry import MODEL_REGISTRY
from pdearena.utils import Timer

_PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))


def save_data(data):
    """ "Save the data file."""
    filename = os.path.join(_PROJECT_DIR, "docs", "models_fwd_time.json")
    with open(filename, "w") as f:
        data["date-created"] = str(datetime.now())
        data["gpu-name"] = torch.cuda.get_device_name()
        json.dump(data, f, sort_keys=True, indent=4)


@click.command()
@click.argument("n_warmups", type=int, default=10)
@click.argument("n_repeats", type=int, default=100)
@click.option("--sleep/--no-sleep", default=True)
@click.option("--save/--no-save", default=True)
def main(n_warmups, n_repeats, sleep, save):
    precision_megabytes = (32 / 8.0) * 1e-6
    pde = PDEDataConfig(1, 1, 14, 2)
    time_history = 4
    results = {}
    for k in MODEL_REGISTRY.keys():
        args = OmegaConf.create(
            {
                "name": k,
                "time_history": time_history,
                "time_future": 1,
                "activation": "gelu",
            }
        )
        model = get_model(args, pde).to("cuda")
        bs = 8
        input = torch.randn(bs, time_history, 3, 128, 128, device="cuda")
        for _ in range(n_warmups):
            _ = model(input)
        torch.cuda.synchronize()
        with Timer() as ft:
            for i in range(n_repeats):
                with torch.set_grad_enabled(False):
                    out = model(input)
                torch.cuda.synchronize()
        print(f"{k} forward time: {ft.dt/n_repeats:.3f}")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in model.parameters())
        model_size = total_parameters * precision_megabytes
        results[k] = {"fwd_time": ft.dt / n_repeats, "num_params": trainable_params, "model_size": model_size}
        del model
        torch.cuda.empty_cache()
        if sleep:
            time.sleep(1)

    if save:
        save_data(results)
    else:
        import pprint

        pprint.pprint(results)


if __name__ == "__main__":
    main()
