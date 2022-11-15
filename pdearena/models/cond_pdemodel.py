# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import instantiate_class

from pdearena import utils
from pdearena.data.utils import PDEDataConfig
from pdearena.models.registry import COND_MODEL_REGISTRY
from pdearena.modules.loss import CustomMSELoss, ScaledLpLoss
from pdearena.rollout import cond_rollout2d

logger = utils.get_logger(__name__)


def get_model(args, pde):
    if args.name in COND_MODEL_REGISTRY:
        _model = COND_MODEL_REGISTRY[args.name].copy()
        _model["init_args"].update(
            dict(
                n_input_scalar_components=pde.n_scalar_components,
                n_output_scalar_components=pde.n_scalar_components,
                n_input_vector_components=pde.n_vector_components,
                n_output_vector_components=pde.n_vector_components,
                time_history=1,
                time_future=1,
                activation=args.activation,
                param_conditioning=args.param_conditioning,
            )
        )
        model = instantiate_class(tuple(), _model)
    else:
        logger.warning("Model not found in registry. Using fallback. Best to add your model to the registry.")
        model = instantiate_class(tuple(), args.model)

    return model


class CondPDEModel(LightningModule):
    def __init__(
        self,
        name: str,
        max_num_steps: int,
        activation: str,
        criterion: str,
        lr: float,
        pdeconfig: PDEDataConfig,
        param_conditioning: Optional[str] = None,  # TODO
        use_scale_shift_norm: bool = False,
        model: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="pdeconfig")
        self.pde = pdeconfig  # instantiate_class(args=tuple(), init=pdeconfig)
        if (self.pde.n_spatial_dims) == 3:
            self._mode = "3D"
        elif (self.pde.n_spatial_dims) == 2:
            self._mode = "2D"
        else:
            raise NotImplementedError(f"{self.pde}")

        self.model = get_model(self.hparams, self.pde)

        if criterion == "mse":
            self.criterion = CustomMSELoss()
            self.rollout_criterion = torch.nn.MSELoss(reduction="none")
        elif criterion == "scaledl2":
            self.criterion = ScaledLpLoss()
            self.rollout_criterion = torch.nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError(f"Criterion {criterion} not implemented yet")

    def forward(self, *args):
        return self.model(*args)

    def step(self, batch):
        x, y, t, z = batch
        pred = self.model(x, t, z)
        loss = self.criterion(pred, y)
        return loss, pred, y

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)

        if self._mode == "2D":
            scalar_loss = self.criterion(
                preds[:, :, 0 : self.pde.n_scalar_components, ...],
                targets[:, :, 0 : self.pde.n_scalar_components, ...],
            )
            vector_loss = self.criterion(
                preds[:, :, self.pde.n_scalar_components :, ...],
                targets[:, :, self.pde.n_scalar_components :, ...],
            )

            self.log("train/loss", loss)
            self.log("train/scalar_loss", scalar_loss)
            self.log("train/vector_loss", vector_loss)
            return {
                "loss": loss,
                "scalar_loss": scalar_loss,
                "vector_loss": vector_loss,
            }
        else:
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
        u, v, z, grid = batch
        delta_t = 1
        time_resolution = self.pde.trajlen // delta_t
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - 1
        # Number of future points to predict
        max_start_time = reduced_time_resolution - 1 * self.hparams.max_num_steps

        losses = []
        for start in range(
            0,
            max_start_time + 1,
            1,
        ):

            end_time = start + 1
            target_start_time = start + delta_t
            target_end_time = target_start_time + delta_t * self.hparams.max_num_steps

            init_u = u[:, start:end_time, ...]
            if self.pde.n_vector_components > 0:
                init_v = v[:, start:end_time, ...]
            else:
                init_v = None

            pred_traj = cond_rollout2d(
                self.model,
                init_u,
                init_v,
                torch.ones(u.size(0)).to(device=u.device) * delta_t,
                z,
                grid,
                self.pde,
                1,
                self.hparams.max_num_steps,
            )
            targ_u = u[:, target_start_time:target_end_time:delta_t, ...]
            if self.pde.n_vector_components > 0:
                targ_v = v[:, target_start_time:target_end_time:delta_t, ...]
                targ_traj = torch.cat((targ_u, targ_v), dim=2)
            else:
                targ_traj = targ_u

            loss = self.rollout_criterion(pred_traj, targ_traj).mean(dim=(0, 2, 3, 4))
            losses.append(loss)
        loss_vec = torch.stack(losses, dim=0).mean(dim=0)
        return loss_vec

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        loss, preds, targets = self.step(batch)
        if self._mode == "2D":
            scalar_loss = self.criterion(
                preds[:, :, 0 : self.pde.n_scalar_components, ...],
                targets[:, :, 0 : self.pde.n_scalar_components, ...],
            )
            vector_loss = self.criterion(
                preds[:, :, self.pde.n_scalar_components :, ...],
                targets[:, :, self.pde.n_scalar_components :, ...],
            )

            self.log("valid/loss", loss)
            return {
                "loss": loss,
                "scalar_loss": scalar_loss,
                "vector_loss": vector_loss,
            }
        else:
            raise NotImplementedError(f"{self._mode}")

    def validation_epoch_end(self, outputs: List[Any]):
        all_vals = defaultdict(list)
        for idx in range(len(outputs)):
            if len(outputs[idx]) > 0:
                for key in outputs[idx][0].keys():
                    if "loss" in key:
                        loss_vec = torch.stack([outputs[idx][i][key] for i in range(len(outputs[idx]))])
                        mean, std = utils.bootstrap(loss_vec, 64, 1)
                        all_vals[key].append(mean)
                        self.log(f"valid/{idx}/{key}_mean", mean)
                        self.log(f"valid/{idx}/{key}_std", std)

        for key in all_vals.keys():
            mean_across_all = torch.stack(all_vals[key]).mean()
            self.log(f"valid/all/{key}_mean", mean_across_all)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx != 0:  # TODO
            start_time = time.time()
            loss, preds, targets = self.step(batch)
            end_time = time.time()
            if self._mode == "2D":
                scalar_loss = self.criterion(
                    preds[:, :, 0 : self.pde.n_scalar_components, ...],
                    targets[:, :, 0 : self.pde.n_scalar_components, ...],
                )
                vector_loss = self.criterion(
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )

                self.log("test/loss", loss)
                self.log("test/loss_time", end_time - start_time)
                return {
                    "loss": loss,
                    "scalar_loss": scalar_loss,
                    "vector_loss": vector_loss,
                }
        else:
            if self._mode == "2D":
                loss_vec = self.compute_rolloutloss2D(batch)

            # summing across "time axis"
            loss = loss_vec.sum()
            loss_t = loss_vec.cumsum(0)
            self.log("test/unrolled_loss", loss)
            # self.log("valid/normalized_unrolled_loss", loss)
            return {
                "unrolled_loss": loss,
                "loss_timesteps": loss_t,
            }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
