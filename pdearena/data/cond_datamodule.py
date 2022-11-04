# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import List

import torch
import torch.utils.data.datapipes.datapipe as dp
from pytorch_lightning import LightningDataModule
from pytorch_lightning.cli import instantiate_class
from torch.utils.data import DataLoader

from pdearena.data.twod.datapipes import DATAPIPE_REGISTRY


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
    """Definest the dataloading process for conditioned PDE data.

    Supports generalization experiments.

    Args:
        task (str): Name of the task.
        data_dir (str): Path to the data directory.
        pde (dict): Dictionary containing the PDE class and its arguments.
        batch_size (int): Batch size.
        pin_memory (bool): Whether to pin memory.
        num_workers (int): Number of workers.
        train_limit_trajectories (int): Number of trajectories to use for training.
        valid_limit_trajectories (int): Number of trajectories to use for validation.
        test_limit_trajectories (int): Number of trajectories to use for testing.
        eval_dts (List[int], optional): List of timesteps to use for evaluation. Defaults to [1, 2, 4, 8, 16].
        datapipe (bool, optional): Whether to use the datapipe. Defaults to True.
        usegrid (bool, optional): Whether to use the grid. Defaults to False.
    """

    def __init__(
        self,
        task: str,
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

        # if "Weather" in pde["class_path"]:
        #     self.dataset_opener = WeatherDatasetOpener
        #     self.randomized_traindatapipe = RandomTimeStepPDETrainData
        #     self.evaldatapipe = TimestepPDEEvalData
        #     # self.train_filter = _weathertrain_filter
        #     # self.valid_filter = _weathervalid_filter
        #     # self.test_filter = _weathertest_filter
        #     self.lister = lambda x: dp.iter.IterableWrapper(
        #         map(lambda y: os.path.join(self.data_dir, y), os.listdir(x))
        #     )
        #     self.sharder = lambda x: x
        # elif len(self.pde.grid_size) == 3:
        #     self.dataset_opener = NavierStokesDatasetOpener
        #     self.randomized_traindatapipe = RandomTimeStepPDETrainData
        #     self.evaldatapipe = TimestepPDEEvalData
        #     self.train_filter = _train_filter
        #     self.valid_filter = _valid_filter
        #     self.test_filter = _test_filter
        #     self.lister = dp.iter.FileLister
        #     self.sharder = dp.iter.ShardingFilter
        # else:
        #     raise NotImplementedError()

    def setup(self, stage=None):
        dps = DATAPIPE_REGISTRY[self.hparams.task]
        self.train_dp = dps["train"](
            pde=self.pde,
            data_path=self.data_dir,
            limit_trajectories=self.hparams.train_limit_trajectories,
            usegrid=self.hparams.usegrid,
        )
        self.valid_dps = [
            dps["valid"](
                pde=self.pde,
                data_path=self.data_dir,
                limit_trajectories=self.hparams.valid_limit_trajectories,
                usegrid=False,
                delta_t=dt,
            )
            for dt in self.eval_dts
        ]

        self.test_dp = dps["test"][1](
            pde=self.pde,
            data_path=self.data_dir,
            limit_trajectories=self.hparams.test_limit_trajectories,
            usegrid=self.hparams.usegrid,
            time_history=self.hparams.time_history,
            time_future=self.hparams.time_future,
            time_gap=self.hparams.time_gap,
        )
        self.test_dps = [
            dps["test"][0](
                pde=self.pde,
                data_path=self.data_dir,
                limit_trajectories=self.hparams.test_limit_trajectories,
                usegrid=False,
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
