import logging
import math
import os

import fdtd
import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from pdearena import utils

from .pde import PDEConfig

logger = logging.getLogger(__name__)


def generate_trajectories_maxwell(
    pde: PDEConfig,
    mode: str,
    num_samples: int,
    dirname: str = "data",
    n_parallel: int = 1,
    seed: int = 42,
) -> None:
    """
    Generate data trajectories for 3D Maxwell equations
    Args:
        pde (PDEConfig): pde at hand [Maxwell3D]
        mode (str): [train, valid, test]
        num_samples (int): how many trajectories do we create

    Returns:
        None
    """

    fdtd.set_backend("numpy")
    pde_string = str(pde)
    logger.info(f"Equation: {pde_string}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Number of samples: {num_samples}")

    save_name = os.path.join(dirname, "_".join([pde_string, mode, str(seed)]))
    if mode == "train":
        save_name = save_name + "_" + str(num_samples)
    h5f = h5py.File("".join([save_name, ".h5"]), "a")
    dataset = h5f.create_group(mode)

    d_field, h_field = {}, {}

    nt, nx, ny, nz = pde.grid_size[0], pde.grid_size[1], pde.grid_size[2], pde.grid_size[3]

    d_field = dataset.create_dataset(
        "d_field",
        (num_samples, nt, nx, ny, nz, 3),
        dtype=float,
    )
    h_field = dataset.create_dataset(
        "h_field",
        (num_samples, nt, nx, ny, nz, 3),
        dtype=float,
    )

    def genfunc(idx, s):
        rng = np.random.RandomState(idx + s)
        # Initialize grid and light sources
        grid = fdtd.Grid(
            (pde.L, pde.L, pde.L),
            grid_spacing=pde.grid_spacing,
            permittivity=pde.permittivity,
            permeability=pde.permeability,
        )

        grid[0, :, :] = fdtd.PeriodicBoundary(name="xbounds")
        grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
        grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")

        outer_area = (pde.n_large - pde.n) // 2

        # light source planes in the xy plane
        for i in range(6):
            lengthx = rng.randint(2, 6)
            startx = rng.randint(0, outer_area - lengthx)
            lengthy = rng.randint(2, 6)
            starty = rng.randint(0, 16 - lengthy)
            pointz = rng.randint(0, 16)
            ampl = rng.rand() * pde.amplitude
            ps = rng.uniform(low=0.0, high=2 * math.pi)
            p = rng.randint(0, 2)
            polar = ["x", "y"]
            period = pde.wavelength / pde.sol * rng.uniform(low=0.001, high=1e3)
            grid[startx : startx + lengthx, starty : starty + lengthy, pointz] = fdtd.PlaneSource(
                period=period,
                amplitude=ampl,
                name=f"planesourcexy{i}",
                phase_shift=ps,
                polarization=polar[p],
            )

        for i in range(6):
            lengthx = rng.randint(2, 6)
            startx = rng.randint(0, 16 - lengthx)
            pointy = rng.randint(0, 16)
            lengthz = rng.randint(2, 6)
            startz = rng.randint(0, 16 - lengthz)
            ampl = rng.rand() * pde.amplitude
            ps = rng.uniform(low=0.0, high=2 * math.pi)
            p = rng.randint(0, 2)
            polar = ["x", "z"]
            period = pde.wavelength / pde.sol * rng.uniform(low=0.001, high=1e3)
            grid[startx : startx + lengthx, pointy, startz : startz + lengthz] = fdtd.PlaneSource(
                period=period,
                amplitude=ampl,
                name=f"planesourcexz{i}",
                phase_shift=ps,
                polarization=polar[p],
            )

        for i in range(6):
            pointx = rng.randint(0, 16)
            lengthy = rng.randint(2, 6)
            starty = rng.randint(0, 16 - lengthy)
            lengthz = rng.randint(2, 6)
            startz = rng.randint(0, 16 - lengthz)
            ampl = rng.rand() * pde.amplitude
            ps = rng.uniform(low=0.0, high=2 * math.pi)
            p = rng.randint(0, 2)
            polar = ["y", "z"]
            period = pde.wavelength / pde.sol * rng.uniform(low=0.001, high=1e3)
            grid[pointx, starty : starty + lengthy, startz : startz + lengthz] = fdtd.PlaneSource(
                period=period,
                amplitude=ampl,
                name=f"planesourceyz{i}",
                phase_shift=ps,
                polarization=polar[p],
            )

        d_field_, h_field_ = [], []
        grid.run(pde.skip_nt, progress_bar=False)
        for i in range(0, pde.nt):
            grid.run(pde.sample_rate, progress_bar=False)
            d_field_.append(grid.E[outer_area:-outer_area, outer_area:-outer_area, outer_area:-outer_area, :])
            h_field_.append(grid.H[outer_area:-outer_area, outer_area:-outer_area, outer_area:-outer_area, :])

        return np.array(d_field_), np.array(h_field_)

    with utils.Timer() as gentime:
        rngs = np.random.randint(np.iinfo(np.int32).max, size=num_samples)
        d_field_ls, h_field_ls = zip(
            *Parallel(n_jobs=n_parallel)(delayed(genfunc)(idx, rngs[idx]) for idx in tqdm(range(num_samples)))
        )

    logger.info(f"Took {gentime.dt:.3f} seconds")
    del rngs
    import gc

    gc.collect()

    with utils.Timer() as writetime:
        for idx in range(num_samples):
            # Saving the trajectories
            d_field[idx : (idx + 1), ...] = d_field_ls[idx]
            h_field[idx : (idx + 1), ...] = h_field_ls[idx]

    logger.info(f"Took {writetime.dt:.3f} seconds writing to disk")
    print()
    print("Data saved")
    print()
    print()
    h5f.close()
