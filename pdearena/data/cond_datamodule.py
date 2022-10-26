# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from typing import List
import torch
from torch.utils.data import DataLoader
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.cli import instantiate_class

from pdearena.data.twod.datapipes import (
    RandomTimeStepPDETrainData,
    TimestepPDEEvalData,
    NavierStokesDatasetOpener,
    WeatherDatasetOpener,
)


def collate_fn_cat(batch):
    elems = range(len(batch[0]))
    return tuple(torch.cat([b[elem] for b in batch], dim=0) for elem in elems)


def collate_fn_stack(batch):
    # Assuming pairs
    b1 = torch.stack([b[0] for b in batch], dim=0)
    b2 = torch.stack([b[1] for b in batch], dim=0)
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


class CondPDEDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        pde,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        train_limit_trajectories: int,
        valid_limit_trajectories: int,
        test_limit_trajectories: int,
        eval_dts: List[int] = [1, 2, 4, 8, 16],
        datapipe: bool = True,
        usegrid: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.eval_dts = eval_dts
        if isinstance(pde, dict):
            self.pde = instantiate_class(args=tuple(), init=pde)  # TODO
        else:
            self.pde = pde
        self.save_hyperparameters(ignore="pde", logger=False)

        if "Weather" in pde["class_path"]:
            self.dataset_opener = WeatherDatasetOpener
            self.randomized_traindatapipe = RandomTimeStepPDETrainData
            self.evaldatapipe = TimestepPDEEvalData
            # self.train_filter = _weathertrain_filter
            # self.valid_filter = _weathervalid_filter
            # self.test_filter = _weathertest_filter
            self.lister = lambda x: dp.iter.IterableWrapper(
                map(lambda y: os.path.join(self.data_dir, y), os.listdir(x))
            )
            self.sharder = lambda x: x
        elif len(self.pde.grid_size) == 3:
            self.dataset_opener = NavierStokesDatasetOpener
            self.randomized_traindatapipe = RandomTimeStepPDETrainData
            self.evaldatapipe = TimestepPDEEvalData
            self.train_filter = _train_filter
            self.valid_filter = _valid_filter
            self.test_filter = _test_filter
            self.lister = dp.iter.FileLister
            self.sharder = dp.iter.ShardingFilter
        else:
            raise NotImplementedError()

    def setup(self, stage=None):
        self.train_dp = self.randomized_traindatapipe(
            self.dataset_opener(
                self.sharder(
                    self.lister(self.data_dir).filter(filter_fn=self.train_filter).shuffle()
                ),
                mode="train",
                limit_trajectories=self.hparams.train_limit_trajectories,
                usegrid=self.hparams.usegrid,
            ).cycle(
                self.pde.trajlen
            ),  # We run every epoch as often as we have number of timesteps in one trajectory.
            self.pde,
        )
        self.valid_dps = [
            self.evaldatapipe(
                self.dataset_opener(
                    self.sharder(self.lister(self.data_dir).filter(filter_fn=self.valid_filter)),
                    mode="valid",
                    limit_trajectories=self.hparams.valid_limit_trajectories,
                    usegrid=self.hparams.usegrid,
                ),
                self.pde,
                delta_t=dt,
            )
            for dt in self.eval_dts
        ]

        self.test_dp = self.dataset_opener(
            self.sharder(self.lister(self.data_dir).filter(filter_fn=self.test_filter)),
            mode="test",
            limit_trajectories=self.hparams.test_limit_trajectories,
            usegrid=self.hparams.usegrid,
        )
        self.test_dps = [
            self.evaldatapipe(
                self.dataset_opener(
                    self.sharder(self.lister(self.data_dir).filter(filter_fn=self.test_filter)),
                    mode="test",
                    limit_trajectories=self.hparams.test_limit_trajectories,
                    usegrid=self.hparams.usegrid,
                ),
                self.pde,
                delta_t=dt,
            )
            for dt in self.eval_dts
        ]

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
        timestep_loaders = [
            DataLoader(
                dataset=dp,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                collate_fn=collate_fn_cat,
            )
            for dp in self.valid_dps
        ]
        return timestep_loaders

    def test_dataloader(self):
        rollout_loader = DataLoader(
            dataset=self.test_dp,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn_stack,
        )
        timestep_loader = [
            DataLoader(
                dataset=dp,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                collate_fn=collate_fn_cat,
            )
            for dp in self.test_dps
        ]
        return [rollout_loader] + timestep_loader
