# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from diffusers.schedulers import DDPMScheduler
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import instantiate_class

from pdearena import utils
from pdearena.data.utils import PDEDataConfig
from pdearena.ema import ExponentialMovingAverage
from pdearena.modules.loss import CustomMSELoss, PearsonCorrelationScore, ScaledLpLoss
from pdearena.rollout import cond_rollout2d

from .registry import COND_MODEL_REGISTRY

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
                # History includes noised current time step.
                time_history=args.time_history + args.time_future,
                time_future=args.time_future,
                activation=args.activation,
                param_conditioning=args.param_conditioning,
                n_dims=pde.n_spatial_dim,
            )
        )
        model = instantiate_class(tuple(), _model)
    else:
        raise NotImplementedError(f"Model {args.name} not found in registry.")

    return model


class PDERefiner(LightningModule):
    def __init__(
        self,
        name: str,
        time_history: int,
        time_future: int,
        time_gap: int,
        max_num_steps: int,
        activation: str,
        criterion: str,
        lr: float,
        pdeconfig: PDEDataConfig,
        model: Optional[Dict] = None,
        param_conditioning: Optional[str] = None,
        padding_mode: str = "zeros",
        predict_difference: bool = False,
        difference_weight: float = 1.0,
        num_refinement_steps: int = 3,
        min_noise_std: float = 4e-7,
        ema_decay: float = 0.995,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="pdeconfig")
        self.pde = pdeconfig
        # Set padding for convolutions globally.
        if (self.pde.n_spatial_dim) == 3:
            self._mode = "3D"
            nn.Conv3d = partial(nn.Conv3d, padding_mode=self.hparams.padding_mode)
        elif (self.pde.n_spatial_dim) == 2:
            self._mode = "2D"
            nn.Conv2d = partial(nn.Conv2d, padding_mode=self.hparams.padding_mode)
        elif (self.pde.n_spatial_dim) == 1:
            self._mode = "1D"
            nn.Conv1d = partial(nn.Conv1d, padding_mode=self.hparams.padding_mode)
        else:
            raise NotImplementedError(f"{self.pde}")

        self.model = get_model(self.hparams, self.pde)
        self.train_criterion = CustomMSELoss()
        # For Diffusion models and models in general working on small errors,
        # it is better to evaluate the exponential average of the model weights
        # instead of the current weights. If an appropriate scheduler with
        # cooldown is used, the test results will be not influenced.
        self.ema = ExponentialMovingAverage(self.model, decay=self.hparams.ema_decay)
        # We use the Diffusion implementation here. Alternatively, one could
        # implement the denoising manually.
        betas = [min_noise_std ** (k / num_refinement_steps) for k in reversed(range(num_refinement_steps + 1))]
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_refinement_steps + 1,
            trained_betas=betas,
            prediction_type="v_prediction",
            clip_sample=False,
        )
        # Multiplies k before passing to frequency embedding.
        self.time_multiplier = 1000 / num_refinement_steps

        self.val_criterions = {"mse": CustomMSELoss(), "scaledl2": ScaledLpLoss()}
        self.rollout_criterions = {"mse": torch.nn.MSELoss(reduction="none"), "corr": PearsonCorrelationScore()}
        time_resolution = self.pde.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.hparams.time_history
        # Number of future points to predict
        self.max_start_time = (
            reduced_time_resolution - self.hparams.time_future * self.hparams.max_num_steps - self.hparams.time_gap
        )
        self.max_start_time = max(0, self.max_start_time)

    def forward(self, x, cond):
        return self.predict_next_solution(x, cond)

    def train_step(self, batch):
        x, y, cond = batch
        if self.hparams.predict_difference:
            # Predict difference to next step instead of next step directly.
            y = (y - x[:, -1:]) / self.hparams.difference_weight
        k = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device)
        noise_factor = self.scheduler.alphas_cumprod.to(x.device)[k]
        noise_factor = noise_factor.view(-1, *[1 for _ in range(x.ndim - 1)])
        signal_factor = 1 - noise_factor
        noise = torch.randn_like(y)
        y_noised = self.scheduler.add_noise(y, noise, k)
        x_in = torch.cat([x, y_noised], axis=1)
        pred = self.model(x_in, time=k * self.time_multiplier, z=cond)
        target = (noise_factor**0.5) * noise - (signal_factor**0.5) * y
        loss = self.train_criterion(pred, target)
        return loss, pred, target

    def eval_step(self, batch):
        x, y, cond = batch
        pred = self.predict_next_solution(x, cond)
        loss = {k: vc(pred, y) for k, vc in self.val_criterions.items()}
        return loss, pred, y

    def predict_next_solution(self, x, cond):
        y_noised = torch.randn(
            size=(x.shape[0], self.hparams.time_future, *x.shape[2:]), dtype=x.dtype, device=x.device
        )
        for k in self.scheduler.timesteps:
            time = torch.zeros(size=(x.shape[0],), dtype=x.dtype, device=x.device) + k
            x_in = torch.cat([x, y_noised], axis=1)
            pred = self.model(x_in, time=time * self.time_multiplier, z=cond)
            y_noised = self.scheduler.step(pred, k, y_noised).prev_sample
        y = y_noised
        if self.hparams.predict_difference:
            y = y * self.hparams.difference_weight + x[:, -1:]
        return y

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.train_step(batch)

        if self._mode == "1D" or self._mode == "2D":
            scalar_loss = self.train_criterion(
                preds[:, :, 0 : self.pde.n_scalar_components, ...],
                targets[:, :, 0 : self.pde.n_scalar_components, ...],
            )

            if self.pde.n_vector_components > 0:
                vector_loss = self.train_criterion(
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )
            else:
                vector_loss = torch.tensor(0.0)
            self.log("train/loss", loss)
            self.log("train/scalar_loss", scalar_loss)
            self.log("train/vector_loss", vector_loss)
            return {
                "loss": loss,
                "scalar_loss": scalar_loss.detach(),
                "vector_loss": vector_loss.detach(),
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

    def compute_rolloutloss(self, batch: Any):
        (u, v, cond, grid) = batch

        losses = {k: [] for k in self.rollout_criterions.keys()}
        for start in range(
            0,
            self.max_start_time + 1,
            self.hparams.time_future + self.hparams.time_gap,
        ):
            end_time = start + self.hparams.time_history
            target_start_time = end_time + self.hparams.time_gap
            target_end_time = target_start_time + self.hparams.time_future * self.hparams.max_num_steps

            init_u = u[:, start:end_time, ...]
            if self.pde.n_vector_components > 0:
                init_v = v[:, start:end_time, ...]
            else:
                init_v = None
            targ_u = u[:, target_start_time:target_end_time, ...]
            if self.pde.n_vector_components > 0:
                targ_v = v[:, target_start_time:target_end_time, ...]
                targ_traj = torch.cat((targ_u, targ_v), dim=2)
            else:
                targ_traj = targ_u

            pred_traj = cond_rollout2d(
                self,
                init_u,
                init_v,
                None,
                cond,
                grid,
                self.pde,
                self.hparams.time_history,
                min(targ_u.shape[1], self.hparams.max_num_steps),
            )
            for k, criterion in self.rollout_criterions.items():
                loss = criterion(pred_traj, targ_traj)
                loss = loss.mean(dim=(0,) + tuple(range(2, loss.ndim)))
                losses[k].append(loss)
        loss_vecs = {k: sum(v) / max(1, len(v)) for k, v in losses.items()}
        return loss_vecs

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            # one-step loss
            loss_mse, preds, targets = self.eval_step(batch)
            if self._mode == "1D" or self._mode == "2D":
                loss_mse["scalar_mse"] = self.val_criterions["mse"](
                    preds[:, :, 0 : self.pde.n_scalar_components, ...],
                    targets[:, :, 0 : self.pde.n_scalar_components, ...],
                )
                loss_mse["vector_mse"] = self.val_criterions["mse"](
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )

                for k in loss_mse.keys():
                    self.log(f"valid/loss/{k}", loss_mse[k])
                return {f"{k}_loss": v for k, v in loss_mse.items()}

            else:
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            # rollout loss
            if self._mode == "1D" or self._mode == "2D":
                loss_vecs = self.compute_rolloutloss(batch)
            else:
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss_mse = loss_vecs["mse"].sum()
            loss_mse_t = loss_vecs["mse"].cumsum(0)
            chan_avg_loss = loss_mse / (self.pde.n_scalar_components + self.pde.n_vector_components)
            self.log("valid/unrolled_loss", loss_mse)
            return {
                "unrolled_loss": loss_mse,
                "loss_timesteps": loss_mse_t,
                "unrolled_chan_avg_loss": chan_avg_loss,
                "corr": loss_vecs["corr"],
            }

    def validation_epoch_end(self, outputs: List[Any]):
        if len(outputs) > 1:
            if len(outputs[0]) > 0:
                for key in outputs[0][0].keys():
                    if "loss" in key:
                        loss_vec = torch.stack([outputs[0][i][key] for i in range(len(outputs[0]))])
                        mean, std = utils.bootstrap(loss_vec, 64, 1)
                        self.log(f"valid/{key}_mean", mean)
                        self.log(f"valid/{key}_std", std)

            if len(outputs[1]) > 0:
                unrolled_loss = torch.stack([outputs[1][i]["unrolled_loss"] for i in range(len(outputs[1]))])
                loss_timesteps_B = torch.stack([outputs[1][i]["loss_timesteps"] for i in range(len(outputs[1]))])
                loss_timesteps = loss_timesteps_B.mean(0)

                log_timesteps = range(0, loss_timesteps.shape[0], max(1, loss_timesteps.shape[0] // 10))

                for i in log_timesteps:
                    self.log(f"valid/intime_{i}_loss", loss_timesteps[i])

                mean, std = utils.bootstrap(unrolled_loss, 64, 1)
                self.log("valid/unrolled_loss_mean", mean)
                self.log("valid/unrolled_loss_std", std)

                # Correlation
                corr_timesteps_B = torch.stack([outputs[1][i]["corr"] for i in range(len(outputs[1]))], dim=0)
                corr_timesteps = corr_timesteps_B.mean(0)
                for threshold in [0.8, 0.9, 0.95]:
                    self.log(f"valid/time_till_corr_lower_{threshold}", (corr_timesteps > threshold).float().sum())
                for t in log_timesteps:
                    self.log(f"valid/corr_at_{t}", corr_timesteps[t])

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            loss, preds, targets = self.eval_step(batch)
            if self._mode == "1D" or self._mode == "2D":
                loss["scalar_mse"] = self.val_criterions["mse"](
                    preds[:, :, 0 : self.pde.n_scalar_components, ...],
                    targets[:, :, 0 : self.pde.n_scalar_components, ...],
                )
                loss["vector_mse"] = self.val_criterions["mse"](
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )

                self.log("test/loss", loss)
                return {f"{k}_loss": v for k, v in loss.items()}
            else:
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            if self._mode == "1D" or self._mode == "2D":
                loss_vecs = self.compute_rolloutloss(batch)
            else:
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss_mse = loss_vecs["mse"].sum()
            loss_mse_t = loss_vecs["mse"].cumsum(0)
            chan_avg_loss = loss_mse / (self.pde.n_scalar_components + self.pde.n_vector_components)
            self.log("valid/unrolled_loss", loss_mse)
            return {
                "unrolled_loss": loss_mse,
                "loss_timesteps": loss_mse_t,
                "unrolled_chan_avg_loss": chan_avg_loss,
                "corr": loss_vecs["corr"],
            }

    def test_epoch_end(self, outputs: List[Any]):
        assert len(outputs) > 1
        if len(outputs[0]) > 0:
            for key in outputs[0][0].keys():
                if "loss" in key:
                    loss_vec = torch.stack([outputs[0][i][key] for i in range(len(outputs[0]))])
                    mean, std = utils.bootstrap(loss_vec, 64, 1)
                    self.log(f"test/{key}_mean", mean)
                    self.log(f"test/{key}_std", std)
        if len(outputs[1]) > 0:
            unrolled_loss = torch.stack([outputs[1][i]["unrolled_loss"] for i in range(len(outputs[1]))])
            loss_timesteps_B = torch.stack([outputs[1][i]["loss_timesteps"] for i in range(len(outputs[1]))])
            loss_timesteps = loss_timesteps_B.mean(0)
            log_timesteps = range(0, loss_timesteps.shape[0], max(1, loss_timesteps.shape[0] // 10))
            for i in log_timesteps:
                self.log(f"test/intime_{i}_loss", loss_timesteps[i])

            mean, std = utils.bootstrap(unrolled_loss, 64, 1)
            self.log("test/unrolled_loss_mean", mean)
            self.log("test/unrolled_loss_std", std)

            # Correlation
            corr_timesteps_B = torch.stack([outputs[1][i]["corr"] for i in range(len(outputs[1]))], dim=0)
            corr_timesteps = corr_timesteps_B.mean(0)
            for threshold in [0.8, 0.9, 0.95]:
                self.log(f"tests/time_till_corr_lower_{threshold}", (corr_timesteps > threshold).float().sum())
            for t in log_timesteps:
                self.log(f"tests/corr_at_{t}", corr_timesteps[t])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    def on_fit_start(self):
        self.ema.register()

    def on_train_batch_end(self, *args, **kwargs):
        self.ema.update()

    def on_validation_start(self):
        self.apply_ema()

    def on_validation_end(self):
        self.remove_ema()

    def on_test_start(self):
        self.apply_ema()

    def on_test_end(self):
        self.remove_ema()

    def apply_ema(self):
        self.ema.apply_shadow()

    def remove_ema(self):
        self.ema.restore()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            self.ema.shadow = checkpoint["ema"]
