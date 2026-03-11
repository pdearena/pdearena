# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import os

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from pdearena import utils

from .pde import PDEConfig

logger = logging.getLogger(__name__)


def _solve_wave_2d(nx, ny, nt, dx, dy, dt, c, init_u, init_v, skip_nt, sample_rate):
    """Solve 2D wave equation using leapfrog (Verlet) integration with periodic BCs.

    Args:
        nx, ny: Grid dimensions.
        nt: Total time steps (including skip_nt).
        dx, dy: Grid spacing.
        dt: Time step.
        c: Wave speed.
        init_u: Initial amplitude (nx, ny).
        init_v: Initial velocity (nx, ny).
        skip_nt: Steps to skip before recording.
        sample_rate: Save every sample_rate-th step.

    Returns:
        Trajectory array of shape (recorded_steps, nx, ny).
    """
    u_prev = init_u.copy()
    # First step via Taylor expansion: u(dt) = u(0) + v(0)*dt + 0.5*a(0)*dt^2
    lap = np.zeros_like(u_prev)
    lap += (np.roll(u_prev, 1, axis=0) + np.roll(u_prev, -1, axis=0) - 2 * u_prev) / dx**2
    lap += (np.roll(u_prev, 1, axis=1) + np.roll(u_prev, -1, axis=1) - 2 * u_prev) / dy**2
    u_curr = u_prev + init_v * dt + 0.5 * c**2 * lap * dt**2

    trajectory = []
    for step in range(nt + skip_nt):
        if step >= skip_nt and (step - skip_nt) % sample_rate == 0:
            trajectory.append(u_curr.copy())
        # Leapfrog update
        lap = np.zeros_like(u_curr)
        lap += (np.roll(u_curr, 1, axis=0) + np.roll(u_curr, -1, axis=0) - 2 * u_curr) / dx**2
        lap += (np.roll(u_curr, 1, axis=1) + np.roll(u_curr, -1, axis=1) - 2 * u_curr) / dy**2
        u_next = 2 * u_curr - u_prev + c**2 * lap * dt**2
        u_prev = u_curr
        u_curr = u_next

    return np.array(trajectory)


def _random_wave_ic(nx, ny, Lx, Ly, n_modes, rng):
    """Generate random initial conditions as superposition of Fourier modes.

    Returns:
        (u0, v0): Initial amplitude and velocity arrays.
    """
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    u0 = np.zeros((nx, ny))
    v0 = np.zeros((nx, ny))
    for _ in range(n_modes):
        kx = rng.integers(-n_modes, n_modes + 1)
        ky = rng.integers(-n_modes, n_modes + 1)
        amp = rng.standard_normal()
        phase = rng.uniform(0, 2 * np.pi)
        u0 += amp * np.sin(2 * np.pi * (kx * X / Lx + ky * Y / Ly) + phase)
    # Normalize to unit variance
    std = u0.std()
    if std > 0:
        u0 /= std
    return u0, v0


def generate_trajectories_wave(
    pde: PDEConfig,
    mode: str,
    num_samples: int,
    batch_size: int = 1,
    device=None,
    dirname: str = "data",
    n_parallel: int = 1,
    seed: int = 42,
) -> None:
    """Generate data trajectories for 2D wave equation.

    Args:
        pde (PDEConfig): Wave2D configuration.
        mode (str): [train, valid, test]
        num_samples (int): Number of trajectories to generate.
        batch_size (int): Unused, kept for API compatibility.
        device: Unused, kept for API compatibility.
        dirname (str): Output directory.
        n_parallel (int): Number of parallel jobs.
        seed (int): Random seed.
    """
    pde_string = str(pde)
    logger.info(f"Equation: {pde_string}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Number of samples: {num_samples}")

    save_name = os.path.join(dirname, "_".join([pde_string, mode, str(seed)]))
    if mode == "train":
        save_name = save_name + "_" + str(num_samples)
    h5f = h5py.File("".join([save_name, ".h5"]), "a")
    dataset = h5f.create_group(mode)

    nt, nx, ny = pde.grid_size
    h5f_u = dataset.create_dataset("u", (num_samples, nt, nx, ny), dtype=float)
    tcoord = dataset.create_dataset("t", (num_samples, nt), dtype=float)
    dt_ds = dataset.create_dataset("dt", (num_samples,), dtype=float)
    xcoord = dataset.create_dataset("x", (num_samples, nx), dtype=float)
    dx_ds = dataset.create_dataset("dx", (num_samples,), dtype=float)
    ycoord = dataset.create_dataset("y", (num_samples, ny), dtype=float)
    dy_ds = dataset.create_dataset("dy", (num_samples,), dtype=float)
    c_ds = dataset.create_dataset("c", (num_samples,), dtype=float)

    def genfunc(idx, s):
        rng = np.random.default_rng(idx + s)
        u0, v0 = _random_wave_ic(pde.nx, pde.ny, pde.Lx, pde.Ly, pde.init_modes, rng)
        traj = _solve_wave_2d(
            pde.nx, pde.ny, pde.nt, pde.dx, pde.dy, pde.dt, pde.c,
            u0, v0, pde.skip_nt, pde.sample_rate,
        )
        return traj

    with utils.Timer() as gentime:
        rngs = np.random.randint(np.iinfo(np.int32).max, size=num_samples)
        trajectories = Parallel(n_jobs=n_parallel)(
            delayed(genfunc)(idx, rngs[idx]) for idx in tqdm(range(num_samples))
        )

    logger.info(f"Took {gentime.dt:.3f} seconds")

    with utils.Timer() as writetime:
        for idx in range(num_samples):
            h5f_u[idx, ...] = trajectories[idx]
            xcoord[idx, ...] = np.linspace(0, pde.Lx, pde.nx, endpoint=False)
            dx_ds[idx] = pde.dx
            ycoord[idx, ...] = np.linspace(0, pde.Ly, pde.ny, endpoint=False)
            dy_ds[idx] = pde.dy
            tcoord[idx, ...] = np.linspace(pde.tmin, pde.tmax, nt)
            dt_ds[idx] = pde.dt * pde.sample_rate
            c_ds[idx] = pde.c

    logger.info(f"Took {writetime.dt:.3f} seconds writing to disk")

    print()
    print("Data saved")
    print()
    h5f.close()
