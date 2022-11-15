# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import glob
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.cli import instantiate_class

from pdedatagen.maxwell import generate_trajectories_maxwell
from pdedatagen.navier_stokes import generate_trajectories_smoke
from pdedatagen.shallowwater import generate_trajectories_shallowwater


def _safe_cpucount() -> int:
    cnt = os.cpu_count()
    if cnt is None:
        cnt = 4
    return cnt


MODE2SEED = {
    "train": 100,
    "valid": 200,
    "test": 300,
}


def main(cfg):
    seed = cfg.seed + MODE2SEED[cfg.mode]
    if "parallel" not in cfg and cfg.pdeconfig.init_args.device == "cpu":
        cfg.parallel = _safe_cpucount() // 2 + _safe_cpucount() // 4
    else:
        cfg.parallel = 1

    seed_everything(seed)
    os.makedirs(cfg.dirname, exist_ok=True)
    existing_files = glob.glob(os.path.join(cfg.dirname, f"*{cfg.mode}_seed_{cfg.seed}*.h5"))
    if cfg.overwrite:
        for file in existing_files:
            os.remove(file)
    else:
        print("Existing files:", len(existing_files))

    print(cfg)
    with open(os.path.join(cfg.dirname, f"pde_{cfg.mode}_seed_{cfg.seed}.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg.pdeconfig))

    if cfg.experiment == "smoke":
        pde = instantiate_class(tuple(), cfg.pdeconfig)
        generate_trajectories_smoke(
            pde=pde,
            mode=cfg.mode,
            num_samples=cfg.samples,
            batch_size=cfg.batchsize,
            device=torch.device(pde.device),
            dirname=cfg.dirname,
            n_parallel=cfg.parallel,
            seed=seed,
        )
    elif cfg.experiment == "smoke_cond":
        bfys = np.random.uniform(0.2, 0.5, size=cfg.samples // 32)
        for bf in bfys:
            cfg.pdeconfig.buoyancy_y = bf.item()
            pde = instantiate_class(tuple(), cfg.pdeconfig)
            generate_trajectories_smoke(
                pde=pde,
                mode=cfg.mode,
                num_samples=32,
                batch_size=cfg.batchsize,
                device=torch.device(pde.device),
                dirname=cfg.dirname,
                n_parallel=cfg.parallel,
                seed=seed,
            )
    elif cfg.experiment == "smoke_cond_eval":
        bfys = np.linspace(0.1, 0.6, num=18)
        for bf in bfys:
            cfg.pdeconfig.buoyancy_y = bf.item()
            pde = instantiate_class(tuple(), cfg.pdeconfig)
            generate_trajectories_smoke(
                pde=pde,
                mode=cfg.mode,
                num_samples=32,
                batch_size=cfg.batchsize,
                device=torch.device(pde.device),
                dirname=cfg.dirname,
                n_parallel=cfg.parallel,
                seed=seed,
            )
    elif cfg.experiment == "shallowwater":
        generate_trajectories_shallowwater(
            savedir=os.path.join(cfg.dirname, cfg.mode),
            num_samples=cfg.samples,
            seed=seed,
        )
    elif cfg.experiment == "maxwell":
        pde = instantiate_class(tuple(), cfg.pdeconfig)
        generate_trajectories_maxwell(
            pde=pde,
            mode=cfg.mode,
            num_samples=cfg.samples,
            dirname=cfg.dirname,
            n_parallel=cfg.parallel,
            seed=seed,
        )
    else:
        raise NotImplementedError()


def cli():
    # This is worth it to avoid hydra complexity
    if "--help" in sys.argv:
        print("Usage: python generate_data.py base=<config.yaml>")
        sys.exit(0)
    cfg = OmegaConf.from_cli()

    if "base" in cfg:
        basecfg = OmegaConf.load(cfg.base)
        del cfg.base
        cfg = OmegaConf.merge(basecfg, cfg)
        OmegaConf.resolve(cfg)
        main(cfg)
    else:
        raise SystemExit("Base configuration file not specified! Exiting.")


if __name__ == "__main__":
    cli()
