# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from dataclasses import dataclass


@dataclass
class PDEConfig:
    pass


@dataclass
class NavierStokes2D(PDEConfig):
    tmin: float = 0
    tmax: float = 20.0
    Lx: float = 32.0
    Ly: float = 32.0
    nt: int = 100
    nx: int = 128
    ny: int = 128
    skip_nt: int = 0
    sample_rate: int = 1
    nu: float = 0.03
    buoyancy_x: float = 0.0
    buoyancy_y: float = 0.05
    correction_strength: float = 1.0
    force_strength: float = 0.2
    force_frequency: int = 4
    n_scalar_components: int = 1  # u
    n_vector_components: int = 1  # vx, vy
    device: str = "cpu"

    def __post_init__(self):
        assert self.n_scalar_components <= 1 and self.n_vector_components <= 1

    def __repr__(self):
        return "NavierStokes2D"

    @property
    def trajlen(self):
        return int(self.nt / self.sample_rate)

    @property
    def grid_size(self):
        return (self.trajlen, self.nx, self.ny)

    @property
    def dt(self):
        return (self.tmax - self.tmin) / (self.nt)

    @property
    def dx(self):
        return self.Lx / (self.nx - 1)

    @property
    def dy(self):
        return self.Ly / (self.ny - 1)


@dataclass
class ShallowWaterWeather(PDEConfig):
    tmin: float = 0
    tmax: float = 90
    nt: int = 90
    skip_nt: int = 0
    sample_rate: int = 1
    nx: int = 64
    ny: int = 128
    Lx: float = 64.0
    Ly: float = 128.0
    n_scalar_components: int = 1  # pres
    n_vector_components: int = 1  # vx, vy

    def __post_init__(self):
        assert self.n_scalar_components <= 2 and self.n_vector_components <= 1

    def __repr__(self):
        return "ShallowWaterWeather"

    @property
    def trajlen(self):
        return int(self.nt / self.sample_rate)

    @property
    def grid_size(self):
        return (self.trajlen, self.nx, self.ny)

    @property
    def dt(self):
        return (self.tmax - self.tmin) / (self.nt)

    @property
    def dx(self):
        return self.Lx / (self.nx - 1)

    @property
    def dy(self):
        return self.Ly / (self.ny - 1)
