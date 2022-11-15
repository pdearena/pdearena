import pytest
import torch

from pdearena.data.utils import PDEDataConfig
from pdearena.models.pdemodel import PDEModel


@pytest.mark.parametrize("name", ["Unet2015-64", "ResNet-128"])
@pytest.mark.parametrize("train_criterion", ["mse", "scaledl2"])
@pytest.mark.parametrize("time_history", [1, 2, 4])
@pytest.mark.parametrize("max_num_steps", [4, 5, 8])
@pytest.mark.parametrize("pdedata", [PDEDataConfig(1, 1, 14, 2), PDEDataConfig(2, 0, 20, 2)])
@pytest.mark.slow
def test_pde_model(name, train_criterion, time_history, max_num_steps, pdedata):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    time_future = 1
    litmodel = PDEModel(
        name=name,
        time_history=time_history,
        time_future=time_future,
        time_gap=0,
        max_num_steps=max_num_steps,
        activation="gelu",
        criterion=train_criterion,
        lr=1e-4,
        pdeconfig=pdedata,
    )
    litmodel = litmodel.to(device=device)
    assert litmodel is not None
    assert isinstance(litmodel.model, torch.nn.Module)
    assert litmodel.max_start_time < pdedata.trajlen

    # test one-step training
    batch = (
        torch.randn(
            8, time_history, pdedata.n_scalar_components + 2 * pdedata.n_vector_components, 64, 64, device=device
        ),
        torch.randn(
            8, time_future, pdedata.n_scalar_components + 2 * pdedata.n_vector_components, 64, 64, device=device
        ),
    )
    loss = litmodel.training_step(batch, 0)
    assert "loss" in loss.keys()

    # test one-step validation
    batch = (
        torch.randn(
            8, time_history, pdedata.n_scalar_components + 2 * pdedata.n_vector_components, 64, 64, device=device
        ),
        torch.randn(
            8, time_future, pdedata.n_scalar_components + 2 * pdedata.n_vector_components, 64, 64, device=device
        ),
    )
    loss = litmodel.validation_step(batch, 0)
    assert "mse_loss" in loss.keys()
    assert "scaledl2_loss" in loss.keys()

    # test rollout validation
    batch = (
        torch.randn(1, pdedata.trajlen, pdedata.n_scalar_components, 64, 64, device=device),
        torch.randn(1, pdedata.trajlen, pdedata.n_vector_components * 2, 64, 64, device=device),
        None,
        None,
    )
    loss = litmodel.validation_step(batch, 0, 1)
    assert "unrolled_loss" in loss.keys()
    assert "loss_timesteps" in loss.keys()
    assert loss["loss_timesteps"].size() == (max_num_steps,)
