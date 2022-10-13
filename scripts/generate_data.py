import glob
import os
from dataclasses import dataclass

from omegaconf import OmegaConf
import numpy as np
import torch
from pytorch_lightning import seed_everything
from pdearena.pde import NavierStokes2D, PDEConfig
from pdedatagen.datagen import (
    generate_trajectories_smoke,
)
from pdearena import utils


def _safe_cpucount() -> int:
    cnt = os.cpu_count()
    if cnt is None:
        cnt = 4
    return cnt


@dataclass
class DataGenConfig:
    dirname: str
    experiment: str
    mode: str
    pdeconfig: PDEConfig
    samples: int = 2**5
    batchsize: int = 8
    overwrite: bool = True
    seed: int = 42
    parallel: int = _safe_cpucount() // 2 + _safe_cpucount() // 4

    def __post_init__(self):
        assert self.mode in ["train", "valid", "test"]


MODE2SEED = {
    "train": 100,
    "valid": 200,
    "test": 300,
}



def main(cfg):
    seed = cfg.seed + MODE2SEED[cfg.mode]
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
        pde = utils.dataclass_from_dict(NavierStokes2D, cfg.pdeconfig)
        generate_trajectories_smoke(
            pde=pde,
            mode=cfg.mode,
            num_samples=cfg.samples,
            batch_size=cfg.batchsize,
            device=torch.device(cfg.pdeconfig.device),
            dirname=cfg.dirname,
            n_parallel=cfg.parallel,
            seed=seed,
        )
    elif cfg.experiment == "smoke_cond":
        bfys = np.random.uniform(0.2, 0.5, size=cfg.samples // 32)
        for bf in bfys:
            cfg.pdeconfig.buoyancy_y = bf.item()
            pde = utils.dataclass_from_dict(NavierStokes2D, cfg.pdeconfig)
            generate_trajectories_smoke(
                pde=pde,
                mode=cfg.mode,
                num_samples=32,
                batch_size=cfg.batchsize,
                device=torch.device(cfg.pdeconfig.device),
                dirname=cfg.dirname,
                n_parallel=cfg.parallel,
                seed=seed,
            )
    elif cfg.experiment == "smoke_cond_eval":
        #bfys = np.random.uniform(0.2, 0.5, size=cfg.samples // 32)
        bfys = np.linspace(0.1, 0.6, num=18)
        for bf in bfys:
            cfg.pdeconfig.buoyancy_y = bf.item()
            pde = utils.dataclass_from_dict(NavierStokes2D, cfg.pdeconfig)
            generate_trajectories_smoke(
                pde=pde,
                mode=cfg.mode,
                num_samples=32,
                batch_size=cfg.batchsize,
                device=torch.device(cfg.pdeconfig.device),
                dirname=cfg.dirname,
                n_parallel=cfg.parallel,
                seed=seed,
            )            
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
