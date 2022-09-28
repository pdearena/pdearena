from typing import Any, Dict, Optional
import os

import torch
import numpy as np

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pdearena.rollout import rollout2d
from pdearena.visualization import plot_3d_quiver, plot_scalar, plot_2dvec
from pdearena import utils

logger = utils.get_logger(__name__)


def log_figure(fig, prefix, save_dir, trainer):
    if "None" not in save_dir:
        fig.savefig(
            os.path.join(save_dir, f"{prefix}_{trainer.global_step:03d}.png"),
            bbox_inches="tight",
        )

    try:
        trainer.logger.experiment.add_figure(f"{prefix}", fig, global_step=trainer.global_step)
    except Exception as e:
        logger.warn(f"Could not add image to logger due to {e}")


@CALLBACK_REGISTRY
class Plot2DTrajPredsCallback(Callback):
    def __init__(self, trajidx: int = 0, save_dir: str = "None") -> None:
        super().__init__()
        self.trajidx = trajidx
        self.save_dir = save_dir
        if self.save_dir != "None":
            save_dir = os.path.join(self.save_dir, str(self.trajidx))
            os.makedirs(save_dir, exist_ok=True)
            self.save_dir = save_dir

    def plot_data(self, all_pred, trainer, suffix: str):
        scalar_pred = all_pred[:, :, 0:1, ...]  # .reshape(-1, 1, *all_pred.shape[3:])
        fig, axs = plt.subplots(
            nrows=scalar_pred.size(0),
            ncols=scalar_pred.size(1),
            figsize=(5 * scalar_pred.size(1), 5 * scalar_pred.size(0)),
            squeeze=False,
        )
        for i in range(scalar_pred.size(0)):
            for j in range(scalar_pred.size(1)):
                plot_scalar(
                    axs[i, j], scalar_pred[i, j, ...].permute(1, 2, 0).detach().cpu().numpy()
                )

        fig.subplots_adjust(wspace=0, hspace=0)
        log_figure(fig, save_dir=(self.save_dir), prefix=f"scalarfield_{suffix}", trainer=trainer)
        plt.close(fig)

        vector_pred = all_pred[:, :, 1:, ...]  # .reshape(-1, 2, *all_pred.shape[3:])
        for v in range(vector_pred.size(2)):
            fig, axs = plt.subplots(
                nrows=vector_pred.size(0),
                ncols=vector_pred.size(1),
                figsize=(5 * vector_pred.size(1), 5 * vector_pred.size(0)),
                squeeze=False,
            )
            for i in range(vector_pred.size(0)):
                for j in range(vector_pred.size(1)):
                    plot_scalar(
                        axs[i, j], vector_pred[i, j, v, ...].unsqueeze(-1).detach().cpu().numpy()
                    )

            fig.subplots_adjust(wspace=0, hspace=0)
            log_figure(
                fig, save_dir=(self.save_dir), prefix=f"vector_{v}_{suffix}", trainer=trainer
            )
        # Plot vector fields as quiver plot
        if vector_pred.size(2) == 2:
            fig, axs = plt.subplots(
                nrows=vector_pred.size(0),
                ncols=vector_pred.size(1),
                figsize=(5 * vector_pred.size(1), 5 * vector_pred.size(0)),
                squeeze=False,
            )
            for i in range(vector_pred.size(0)):
                for j in range(vector_pred.size(1)):
                    plot_2dvec(axs[i, j], vector_pred[i, j].cpu().detach().numpy())

            fig.subplots_adjust(wspace=0, hspace=0)

            log_figure(
                fig, save_dir=(self.save_dir), prefix=f"vectorfield_{suffix}", trainer=trainer
            )

        plt.close(fig)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        u, v, grid = next(iter(trainer.datamodule.test_dataloader()))

        u = u[self.trajidx].unsqueeze(0).to(pl_module.device)
        v = v[self.trajidx].unsqueeze(0).to(pl_module.device)
        for start in range(
            pl_module.pde.skip_nt,
            pl_module.max_start_time + 1,
            pl_module.hparams.time_future + pl_module.hparams.time_gap,
        ):
            end_time = start + pl_module.hparams.time_history
            target_start_time = end_time + pl_module.hparams.time_gap
            target_end_time = (
                target_start_time + pl_module.hparams.time_future * pl_module.hparams.max_num_steps
            )
            targ_u = u[:, target_start_time:target_end_time, ...]
            targ_v = v[:, target_start_time:target_end_time, ...]
            # targ_u = u[
            #     :,
            #     target_start_time
            #     * pl_module.pde.n_scalar_components : target_end_time
            #     * pl_module.pde.n_scalar_components,
            # ]
            # targ_v = v[
            #     :,
            #     target_start_time
            #     * pl_module.pde.n_vector_components
            #     * 2 : target_end_time
            #     * pl_module.pde.n_vector_components
            #     * 2,
            # ]
            # add extra channel dim
            # targ_u = targ_u.reshape(
            #     targ_u.size(0), pl_module.hparams.max_num_steps, -1, *targ_u.shape[2:]
            # )
            # targ_v = targ_v.reshape(
            #     targ_v.size(0), trainer.model.hparams.max_num_steps, -1, *targ_v.shape[2:]
            # )
            # targ_v = targ_v.reshape(
            #     targ_v.size(0), -1, pl_module.hparams.max_num_steps, *targ_v.shape[2:]
            # ).permute(0, 2, 1, 3, 4)
            targ_traj = torch.cat((targ_u, targ_v), dim=2)
            self.plot_data(targ_traj, trainer, f"targ_s_{start}")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        u, v, grid = next(iter(trainer.datamodule.test_dataloader()))

        u = u[self.trajidx].unsqueeze(0).to(pl_module.device)
        v = v[self.trajidx].unsqueeze(0).to(pl_module.device)

        for start in range(
            pl_module.pde.skip_nt,
            pl_module.max_start_time + 1,
            pl_module.hparams.time_future + pl_module.hparams.time_gap,
        ):
            end_time = start + pl_module.hparams.time_history

            # init_u = u[
            #     :,
            #     start
            #     * pl_module.pde.n_scalar_components : end_time
            #     * pl_module.pde.n_scalar_components,
            # ]
            # init_v = v[
            #     :,
            #     start
            #     * pl_module.pde.n_vector_components
            #     * 2 : end_time
            #     * pl_module.pde.n_vector_components
            #     * 2,
            # ]
            init_u = u[:, start:end_time, ...]
            init_v = v[:, start:end_time, ...]
            pred_traj = rollout2d(
                pl_module.model,
                init_u,
                init_v,
                grid,
                pl_module.pde,
                pl_module.hparams.time_history,
                pl_module.hparams.max_num_steps,
            )
            self.plot_data(pred_traj, trainer, f"pred_s_{start}")
        #     _, _, data, pred = compute_unrolled_loss(
        #         pl_module.model,
        #         pl_module.criterion,
        #         u,
        #         v,
        #         start,
        #         data,
        #         pred,
        #         pl_module.pde,
        #         pl_module.hparams.time_history,
        #         pl_module.hparams.time_gap,
        #         pl_module.hparams.time_future,
        #     )
        #     # all_data.append(data)
        #     all_pred_ls.append(pred)

        # # all_data = torch.cat(all_data, dim=0)
        # all_pred = torch.cat(all_pred_ls, dim=0)

        # # all_data = all_data.reshape(all_data.size(0), pl_module.hparams.time_future, all_data.size(1) // pl_module.hparams.time_future, *all_data.shape[2:])
        # all_pred = all_pred.reshape(
        #     all_pred.size(0),
        #     pl_module.hparams.time_future,
        #     all_pred.size(1) // pl_module.hparams.time_future,
        #     *all_pred.shape[2:],
        # )
        # scalar_pred = all_pred[:, :, 0:1, ...]  # .reshape(-1, 1, *all_pred.shape[3:])
        # fig, axs = plt.subplots(
        #     nrows=scalar_pred.size(0),
        #     ncols=scalar_pred.size(1),
        #     figsize=(5 * scalar_pred.size(1), 5 * scalar_pred.size(0)),
        # )
        # for i in range(scalar_pred.size(0)):
        #     for j in range(scalar_pred.size(1)):
        #         plot_scalar(
        #             axs[i, j], scalar_pred[i, j, ...].permute(1, 2, 0).detach().cpu().numpy()
        #         )

        # fig.subplots_adjust(wspace=0, hspace=0)
        # log_figure(
        #     fig,
        #     save_dir=os.path.join(self.save_dir, str(self.trajidx)),
        #     trainer=trainer,
        #     prefix="scalarfield_preds",
        # )

        # vector_pred = all_pred[:, :, 1:, ...]  # .reshape(-1, 2, *all_pred.shape[3:])
        # for v in range(vector_pred.size(2)):
        #     fig, axs = plt.subplots(
        #         nrows=vector_pred.size(0),
        #         ncols=vector_pred.size(1),
        #         figsize=(5 * vector_pred.size(1), 5 * vector_pred.size(0)),
        #     )
        #     for i in range(vector_pred.size(0)):
        #         for j in range(vector_pred.size(1)):
        #             plot_scalar(
        #                 axs[i, j], vector_pred[i, j, v, ...].unsqueeze(-1).detach().cpu().numpy()
        #             )

        #     fig.subplots_adjust(wspace=0, hspace=0)
        #     log_figure(
        #         fig,
        #         save_dir=os.path.join(self.save_dir, str(self.trajidx)),
        #         trainer=trainer,
        #         prefix=f"vector_{v}_preds",
        #     )

        # # Plot vector fields as quiver plot
        # if vector_pred.size(2) == 2:
        #     fig, axs = plt.subplots(
        #         nrows=vector_pred.size(0),
        #         ncols=vector_pred.size(1),
        #         figsize=(5 * vector_pred.size(1), 5 * vector_pred.size(0)),
        #     )
        #     for i in range(vector_pred.size(0)):
        #         for j in range(vector_pred.size(1)):
        #             plot_2dvec(axs[i, j], vector_pred[i, j].cpu().detach().numpy())

        #     fig.subplots_adjust(wspace=0, hspace=0)

        #     log_figure(
        #         fig,
        #         save_dir=os.path.join(self.save_dir, str(self.trajidx)),
        #         trainer=trainer,
        #         prefix="vectorfield_preds",
        #     )

        # plt.close(fig)


@CALLBACK_REGISTRY
class OneStep2DPredsCallback(Callback):
    def __init__(self, save_dir: str = "None") -> None:
        super().__init__()
        self.save_dir = save_dir
        if self.save_dir != "None":
            os.makedirs(self.save_dir, exist_ok=True)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        from torch.utils.data import DataLoader
        from pdearena.data.datamodule import collate_fn_cat

        # TODO: maybe have a separate dataloader for this?
        dl = DataLoader(
            dataset=trainer.datamodule.valid_dp1,
            num_workers=0,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn_cat,
        )
        x, y = next(iter(dl))

        self.plot_data(y, trainer=trainer, suffix="targ")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        from torch.utils.data import DataLoader
        from pdearena.data.datamodule import collate_fn_cat

        # TODO: maybe have a separate dataloader for this?
        dl = DataLoader(
            dataset=trainer.datamodule.valid_dp1,
            num_workers=0,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn_cat,
        )
        x, y = next(iter(dl))
        pred = pl_module.model(x.to(pl_module.device))
        # all_pred = pred.reshape(
        #     pred.size(0),
        #     pl_module.hparams.time_future,
        #     pred.size(1) // pl_module.hparams.time_future,
        #     *pred.shape[2:],
        # )
        self.plot_data(pred, trainer, "preds")

    def plot_data(self, all_pred, trainer: "pl.Trainer", suffix: str = "") -> None:
        scalar_pred = all_pred[:, :, 0:1, ...]  # .reshape(-1, 1, *all_pred.shape[3:])
        fig, axs = plt.subplots(
            nrows=scalar_pred.size(0),
            ncols=scalar_pred.size(1),
            figsize=(5 * scalar_pred.size(1), 5 * scalar_pred.size(0)),
            squeeze=False,
        )
        for i in range(scalar_pred.size(0)):
            for j in range(scalar_pred.size(1)):
                plot_scalar(
                    axs[i, j], scalar_pred[i, j, ...].permute(1, 2, 0).detach().cpu().numpy()
                )

        fig.subplots_adjust(wspace=0, hspace=0)
        log_figure(fig, save_dir=(self.save_dir), trainer=trainer, prefix=f"scalarfield_{suffix}")

        vector_pred = all_pred[:, :, 1:, ...]  # .reshape(-1, 2, *all_pred.shape[3:])
        for v in range(vector_pred.size(2)):
            fig, axs = plt.subplots(
                nrows=vector_pred.size(0),
                ncols=vector_pred.size(1),
                figsize=(5 * vector_pred.size(1), 5 * vector_pred.size(0)),
                squeeze=False,
            )
            for i in range(vector_pred.size(0)):
                for j in range(vector_pred.size(1)):
                    plot_scalar(
                        axs[i, j], vector_pred[i, j, v, ...].unsqueeze(-1).detach().cpu().numpy()
                    )

            fig.subplots_adjust(wspace=0, hspace=0)
            log_figure(
                fig, save_dir=(self.save_dir), trainer=trainer, prefix=f"vector_{v}_{suffix}"
            )

        # Plot vector fields as quiver plot
        if vector_pred.size(2) == 2:
            fig, axs = plt.subplots(
                nrows=vector_pred.size(0),
                ncols=vector_pred.size(1),
                figsize=(5 * vector_pred.size(1), 5 * vector_pred.size(0)),
                squeeze=False,
            )
            for i in range(vector_pred.size(0)):
                for j in range(vector_pred.size(1)):
                    plot_2dvec(axs[i, j], vector_pred[i, j].cpu().detach().numpy())

            fig.subplots_adjust(wspace=0, hspace=0)

            log_figure(
                fig, save_dir=(self.save_dir), trainer=trainer, prefix=f"vectorfield_{suffix}"
            )

        plt.close(fig)
