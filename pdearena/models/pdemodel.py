# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Dict, List
from omegaconf import OmegaConf

import torch
from pdearena import utils
from pdearena.modules.loss import CustomMSELoss, ScaledLpLoss
from pdearena.modules.twod import BasicBlock, FourierBasicBlock, ResNet
from pdearena.modules.twod_oldunet import OldUnet
from pdearena.modules.twod_unet import FourierUnet, Unet, AltFourierUnet
from pdearena.modules.twod_uno import UNO
from pdearena.modules.twod_unet2015 import UNet2015
from pdearena.rollout import rollout2d
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import instantiate_class


def get_model(args, pde):
    if args.name == "Unet2015":
        model = UNet2015(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            activation=args.activation,
        )
    elif args.name == "OldUnet":
        model = OldUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            activation=args.activation,
        )
    elif args.name == "UNO":
        model = UNO(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            activation=args.activation,
        )
    elif args.name == "Unet":
        model = Unet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            activation=args.activation,
            norm=args.norm,
        )
    elif args.name == "Unet1x1":
        model = Unet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            activation=args.activation,
            norm=args.norm,
            use1x1=True,
        )

    elif args.name == "UnetAttn":
        model = Unet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            activation=args.activation,
            norm=args.norm,
            is_attn=(False, False, True, True),
            mid_attn=True,
        )
    elif args.name == "UnetMidAttn":
        model = Unet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            activation=args.activation,
            norm=args.norm,
            mid_attn=True,
        )
    elif args.name == "UnetMidAttn1x1":
        model = Unet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            activation=args.activation,
            norm=args.norm,
            mid_attn=True,
            use1x1=True,
        )

    elif args.name == "FourierUnet":
        model = FourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
        )
    elif args.name == "FourierUnetConstMode":
        model = FourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            mode_scaling=False,
        )
    elif args.name == "Fourier1Unet":
        model = FourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            n_fourier_layers=1,
        )
    elif args.name == "Fourier1UnetMidAttn":
        model = FourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            n_fourier_layers=1,
            mid_attn=True,
        )
    elif args.name == "FourierUnetMidAttnConstMode":
        model = FourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            mid_attn=True,
            mode_scaling=False,
        )

    elif args.name == "AltFourierUnet":
        model = AltFourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
        )
    elif args.name == "AltFourierUnetConstMode":
        model = AltFourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            mode_scaling=False,
        )
    elif args.name == "AltFourier1Unet":
        model = AltFourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            n_fourier_layers=1,
        )
    elif args.name == "AltFourier1UnetMidAttn":
        model = AltFourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            n_fourier_layers=1,
            mid_attn=True,
        )

    elif args.name == "FourierUnet1x1":
        model = FourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            use1x1=True,
        )
    elif args.name == "FourierUnetConstMode1x1":
        model = FourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            mode_scaling=False,
            use1x1=True,
        )
    elif args.name == "Fourier1Unet1x1":
        model = FourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            n_fourier_layers=1,
            use1x1=True,
        )
    elif args.name == "Fourier1UnetMidAttn1x1":
        model = FourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            n_fourier_layers=1,
            mid_attn=True,
            use1x1=True,
        )

    elif args.name == "FourierUnetMidAttn":
        model = FourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            mid_attn=True,
        )
    elif args.name == "FullFourierUnet":
        model = FourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            n_fourier_layers=4,
        )
    elif args.name == "FullFourierUnetMidAttn":
        model = FourierUnet(
            pde=pde,
            time_history=args.time_history,
            time_future=args.time_future,
            hidden_channels=args.hidden_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            activation=args.activation,
            norm=args.norm,
            mid_attn=True,
            n_fourier_layers=4,
        )
    else:
        raise Exception(f"Wrong model specified {args.name}")

    return model


class PDEModel(LightningModule):
    def __init__(
        self,
        name: str,
        time_history: int,
        time_future: int,
        time_gap: int,
        max_num_steps: int,
        hidden_channels: int,
        n_scalar_components: int,
        n_vector_components: int,
        modes1: int,
        modes2: int,
        activation: str,
        norm: bool,
        diffmode: bool,
        criterion: str,
        lr: float,
        unrolling: int,
        rotation: bool,
        usegrid: bool,
        pdeconfig: Dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="pdeconfig")
        self.pde = instantiate_class(args=tuple(), init=pdeconfig)
        if len(self.pde.grid_size) == 4:
            self._mode = "3D"
        elif len(self.pde.grid_size) == 3:
            self._mode = "2D"
        else:
            raise NotImplementedError(f"{self.pde}")

        self.model = get_model(self.hparams, self.pde)
        if criterion == "mse":
            self.train_criterion = CustomMSELoss()
        elif criterion == "scaledl2":
            self.train_criterion = ScaledLpLoss()
        else:
            raise NotImplementedError(f"Criterion {criterion} not implemented yet")

        self.val_criterions = {"mse": CustomMSELoss(), "scaledl2": ScaledLpLoss()}
        self.rollout_criterion = torch.nn.MSELoss(reduction="none")
        time_resolution = self.pde.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.hparams.time_history
        # Number of future points to predict
        self.max_start_time = (
            reduced_time_resolution
            - self.hparams.time_future * self.hparams.max_num_steps
            - self.hparams.time_gap
        )
        assert self.pde.skip_nt < self.max_start_time

    def forward(self, *args):
        return self.model(*args)

    def train_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.train_criterion(pred, y)
        return loss, pred, y

    def eval_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = {k: vc(pred, y) for k, vc in self.val_criterions.items()}
        return loss, pred, y

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.train_step(batch)

        if self._mode == "2D":
            scalar_loss = self.train_criterion(
                preds[:, :, 0 : self.hparams.n_scalar_components, ...],
                targets[:, :, 0 : self.hparams.n_scalar_components, ...],
            )

            if self.pde.n_vector_components > 0:
                vector_loss = self.train_criterion(
                    preds[:, :, self.hparams.n_scalar_components :, ...],
                    targets[:, :, self.hparams.n_scalar_components :, ...],
                )
            else:
                vector_loss = torch.tensor(0.0)
            self.log("train/loss", loss)
            self.log("train/scalar_loss", scalar_loss)
            self.log("train/vector_loss", vector_loss)
            return {
                "loss": loss,
                "scalar_loss": scalar_loss,
                "vector_loss": vector_loss,
            }
        elif self._mode == "3D":
            raise NotImplementedError(f"{self._mode}")

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        for key in outputs[0].keys():
            if "loss" in key:
                loss_vec = torch.stack([outputs[i][key] for i in range(len(outputs))])
                mean, std = utils.bootstrap(loss_vec, 64, 1)
                self.log(f"train/{key}_mean", mean)
                self.log(f"train/{key}_std", std)

    def compute_rolloutloss2D(self, batch: Any):

        (u, v, cond, grid) = batch

        losses = []
        for start in range(
            self.pde.skip_nt,
            self.max_start_time + 1,
            self.hparams.time_future + self.hparams.time_gap,
        ):

            end_time = start + self.hparams.time_history
            target_start_time = end_time + self.hparams.time_gap
            target_end_time = (
                target_start_time
                + self.hparams.time_future * self.hparams.max_num_steps
            )

            init_u = u[:, start:end_time, ...]
            if self.pde.n_vector_components > 0:
                init_v = v[:, start:end_time, ...]
            else:
                init_v = None

            pred_traj = rollout2d(
                self.model,
                init_u,
                init_v,
                grid,
                self.pde,
                self.hparams.time_history,
                self.hparams.max_num_steps,
            )
            targ_u = u[:, target_start_time:target_end_time, ...]
            if self.pde.n_vector_components > 0:
                targ_v = v[:, target_start_time:target_end_time, ...]
                targ_traj = torch.cat((targ_u, targ_v), dim=2)
            else:
                targ_traj = targ_u
            loss = self.rollout_criterion(pred_traj, targ_traj).mean(dim=(0, 2, 3, 4))
            losses.append(loss)
        loss_vec = torch.stack(losses, dim=0).mean(dim=0)
        return loss_vec

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            loss, preds, targets = self.eval_step(batch)
            if self._mode == "2D":
                loss["scalar_mse"] = self.val_criterions["mse"](
                    preds[:, :, 0 : self.hparams.n_scalar_components, ...],
                    targets[:, :, 0 : self.hparams.n_scalar_components, ...],
                )
                loss["vector_mse"] = self.val_criterions["mse"](
                    preds[:, :, self.hparams.n_scalar_components :, ...],
                    targets[:, :, self.hparams.n_scalar_components :, ...],
                )

                for k in loss.keys():
                    self.log(f"valid/loss/{k}", loss[k])
                return {f"{k}_loss": v for k, v in loss.items()}

            elif self._mode == "3D":
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            if self._mode == "2D":
                loss_vec = self.compute_rolloutloss2D(batch)
            elif self._mode == "3D":
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss = loss_vec.sum()
            loss_t = loss_vec.cumsum(0)
            chan_avg_loss = loss / (
                self.pde.n_scalar_components + self.pde.n_vector_components
            )
            self.log("valid/unrolled_loss", loss)
            return {
                "unrolled_loss": loss,
                "loss_timesteps": loss_t,
                "unrolled_chan_avg_loss": chan_avg_loss,
            }

    def validation_epoch_end(self, outputs: List[Any]):
        if len(outputs) > 1:
            if len(outputs[0]) > 0:
                for key in outputs[0][0].keys():
                    if "loss" in key:
                        loss_vec = torch.stack(
                            [outputs[0][i][key] for i in range(len(outputs[0]))]
                        )
                        mean, std = utils.bootstrap(loss_vec, 64, 1)
                        self.log(f"valid/{key}_mean", mean)
                        self.log(f"valid/{key}_std", std)

            if len(outputs[1]) > 0:
                unrolled_loss = torch.stack(
                    [outputs[1][i]["unrolled_loss"] for i in range(len(outputs[1]))]
                )
                loss_timesteps_B = torch.stack(
                    [outputs[1][i]["loss_timesteps"] for i in range(len(outputs[1]))]
                )
                loss_timesteps = loss_timesteps_B.mean(0)

                for i in range(self.hparams.max_num_steps):
                    self.log(f"valid/intime_{i}_loss", loss_timesteps[i])

                mean, std = utils.bootstrap(unrolled_loss, 64, 1)
                self.log("valid/unrolled_loss_mean", mean)
                self.log("valid/unrolled_loss_std", std)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            loss, preds, targets = self.eval_step(batch)
            if self._mode == "2D":
                loss["scalar_mse"] = self.val_criterions["mse"](
                    preds[:, :, 0 : self.hparams.n_scalar_components, ...],
                    targets[:, :, 0 : self.hparams.n_scalar_components, ...],
                )
                loss["vector_mse"] = self.val_criterions["mse"](
                    preds[:, :, self.hparams.n_scalar_components :, ...],
                    targets[:, :, self.hparams.n_scalar_components :, ...],
                )

                self.log("test/loss", loss)
                return {f"{k}_loss": v for k, v in loss.items()}
            elif self._mode == "3D":
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            if self._mode == "2D":
                loss_vec = self.compute_rolloutloss2D(batch)
            elif self._mode == "3D":
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss = loss_vec.sum()
            loss_t = loss_vec.cumsum(0)
            self.log("test/unrolled_loss", loss)
            # self.log("valid/normalized_unrolled_loss", loss)
            return {
                "unrolled_loss": loss,
                "loss_timesteps": loss_t,
            }

    def test_epoch_end(self, outputs: List[Any]):
        assert len(outputs) > 1
        if len(outputs[0]) > 0:
            for key in outputs[0][0].keys():
                if "loss" in key:
                    loss_vec = torch.stack(
                        [outputs[0][i][key] for i in range(len(outputs[0]))]
                    )
                    mean, std = utils.bootstrap(loss_vec, 64, 1)
                    self.log(f"test/{key}_mean", mean)
                    self.log(f"test/{key}_std", std)
        if len(outputs[1]) > 0:
            unrolled_loss = torch.stack(
                [outputs[1][i]["unrolled_loss"] for i in range(len(outputs[1]))]
            )
            loss_timesteps_B = torch.stack(
                [outputs[1][i]["loss_timesteps"] for i in range(len(outputs[1]))]
            )
            loss_timesteps = loss_timesteps_B.mean(0)
            for i in range(self.hparams.max_num_steps):
                self.log(f"test/intime_{i}_loss", loss_timesteps[i])

            mean, std = utils.bootstrap(unrolled_loss, 64, 1)
            self.log("test/unrolled_loss_mean", mean)
            self.log("test/unrolled_loss_std", std)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
