# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from typing import Optional

import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from pytorch_lightning.cli import instantiate_class
from torch.utils.data import DataLoader

from pdearena.data.twod.datapipes import (
    NavierStokesDatasetOpener,
    PDEEvalTimeStepData,
    RandomizedPDETrainData,
    VortWeatherDatasetOpener,
    VortWeatherDatasetOpener1Day,
    VortWeatherDatasetOpener2Day,
    WeatherDatasetOpener,
    WeatherDatasetOpener1Day,
    WeatherDatasetOpener2Day,
)


def collate_fn_cat(batch):
    # Assuming pairs
    b1 = torch.cat([b[0] for b in batch], dim=0)
    b2 = torch.cat([b[1] for b in batch], dim=0)
    return b1, b2


def collate_fn_stack(batch):
    # Assuming pairs
    b1 = torch.stack([b[0] for b in batch], dim=0)
    if len(batch[0]) > 1:
        if batch[0][1] is not None:
            b2 = torch.stack([b[1] for b in batch], dim=0)
        else:
            b2 = None
    if len(batch[0]) > 2:
        if batch[0][2] is not None:
            b3 = torch.cat([b[2] for b in batch], dim=0)
        else:
            b3 = None
    if len(batch[0]) > 3:
        if batch[0][3] is not None:
            b4 = torch.cat([b[3] for b in batch], dim=0)
        else:
            b4 = None
        return b1, b2, b3, b4

    return b1, b2, b3


def _train_filter(fname):
    return "train" in fname and "h5" in fname


def _valid_filter(fname):
    return "valid" in fname and "h5" in fname


def _test_filter(fname):
    return "test" in fname and "h5" in fname


def _weathertrain_filter(fname):
    return "train.zarr" in fname


def _weathervalid_filter(fname):
    return "valid.zarr" in fname


def _weathertest_filter(fname):
    return "test.zarr" in fname


class PDEDataModule(LightningDataModule):
    """Define the dataloading process for pde data."""

    def __init__(
        self,
        data_dir: str,
        time_history,
        time_future,
        time_gap,
        pde,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        train_limit_trajectories: int,
        valid_limit_trajectories: int,
        test_limit_trajectories: int,
        datapipe: bool = True,
        usegrid: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        if isinstance(pde, dict):
            self.pde = instantiate_class(args=tuple(), init=pde)  # TODO
        else:
            self.pde = pde
        # TODO: make this more general
        # Best to associate this with the pde object somehow?
        if "Weather" in pde["class_path"]:
            if self.pde.n_vector_components == 0:
                if self.pde.sample_rate != 1:
                    if self.pde.sample_rate == 8:
                        self.dataset_opener = VortWeatherDatasetOpener2Day
                    elif self.pde.sample_rate == 4:
                        self.dataset_opener = VortWeatherDatasetOpener1Day
                    else:
                        raise ValueError(f"Sample rate {self.pde.sample_rate} not supported")
                else:
                    self.dataset_opener = VortWeatherDatasetOpener
            else:
                if self.pde.sample_rate != 1:
                    if self.pde.sample_rate == 8:
                        self.dataset_opener = WeatherDatasetOpener2Day
                    elif self.pde.sample_rate == 4:
                        self.dataset_opener = WeatherDatasetOpener1Day
                    else:
                        raise ValueError(f"Sample rate {self.pde.sample_rate} not supported")
                else:
                    self.dataset_opener = WeatherDatasetOpener
            self.randomized_traindatapipe = RandomizedPDETrainData
            self.evaldatapipe = PDEEvalTimeStepData
            self.train_filter = _weathertrain_filter
            self.valid_filter = _weathervalid_filter
            self.test_filter = _weathertest_filter
            self.lister = lambda x: dp.iter.IterableWrapper(
                map(lambda y: os.path.join(self.data_dir, y), os.listdir(x))
            )
            self.sharder = lambda x: x
        elif len(self.pde.grid_size) == 3:
            self.dataset_opener = NavierStokesDatasetOpener
            self.randomized_traindatapipe = RandomizedPDETrainData
            self.evaldatapipe = PDEEvalTimeStepData
            self.train_filter = _train_filter
            self.valid_filter = _valid_filter
            self.test_filter = _test_filter
            self.lister = dp.iter.FileLister
            self.sharder = dp.iter.ShardingFilter
        elif len(self.pde.grid_size) == 4:
            raise NotImplementedError("3D data not supported yet")
        else:
            raise NotImplementedError(f"{self.pde}: {self.pde.grid_size}")

        self.save_hyperparameters(ignore="pde", logger=False)

    def _setup_datapipes(self):

        self.train_dp = self.randomized_traindatapipe(
            self.dataset_opener(
                self.sharder(self.lister(self.data_dir).filter(filter_fn=self.train_filter).shuffle()),
                mode="train",
                limit_trajectories=self.hparams.train_limit_trajectories,
                usegrid=self.hparams.usegrid,
            ).cycle(
                self.pde.trajlen
            ),  # We run every epoch as often as we have number of timesteps in one trajectory.
            self.pde,
            time_history=self.hparams.time_history,
            time_future=self.hparams.time_future,
            time_gap=self.hparams.time_gap,
        )

    def setup(self, stage: Optional[str] = None):
        if self.hparams.datapipe:
            self._setup_datapipes()
        else:
            self._setup_datasets()

        self.valid_dp1 = self.evaldatapipe(
            self.dataset_opener(
                self.sharder(self.lister(self.data_dir).filter(filter_fn=self.valid_filter)),
                mode="valid",
                limit_trajectories=self.hparams.valid_limit_trajectories,
                usegrid=self.hparams.usegrid,
            ),
            self.pde,
            time_history=self.hparams.time_history,
            time_future=self.hparams.time_future,
            time_gap=self.hparams.time_gap,
        )
        self.valid_dp2 = self.dataset_opener(
            self.sharder(self.lister(self.data_dir).filter(filter_fn=self.valid_filter)),
            mode="valid",
            limit_trajectories=self.hparams.valid_limit_trajectories,
            usegrid=self.hparams.usegrid,
        )

        self.test_dp = self.dataset_opener(
            self.sharder(self.lister(self.data_dir).filter(filter_fn=self.test_filter)),
            mode="test",
            limit_trajectories=self.hparams.test_limit_trajectories,
            usegrid=self.hparams.usegrid,
        )
        self.test_dp_onestep = self.evaldatapipe(
            self.dataset_opener(
                self.sharder(self.lister(self.data_dir).filter(filter_fn=self.test_filter)),
                mode="test",
                limit_trajectories=self.hparams.test_limit_trajectories,
                usegrid=self.hparams.usegrid,
            ),
            self.pde,
            time_history=self.hparams.time_history,
            time_future=self.hparams.time_future,
            time_gap=self.hparams.time_gap,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dp,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn_cat,
        )

    def val_dataloader(self):
        timestep_loader = DataLoader(
            dataset=self.valid_dp1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn_cat,
        )
        rollout_loader = DataLoader(
            dataset=self.valid_dp2,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,  # TODO: might need to reduce this
            shuffle=False,
            collate_fn=collate_fn_stack,
        )
        return [timestep_loader, rollout_loader]

    def test_dataloader(self):
        rollout_loader = DataLoader(
            dataset=self.test_dp,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn_stack,
        )
        timestep_loader = DataLoader(
            dataset=self.test_dp_onestep,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn_cat,
        )
        return [timestep_loader, rollout_loader]
