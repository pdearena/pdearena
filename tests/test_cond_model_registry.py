import pytest
import torch
from omegaconf import OmegaConf

from pdearena.data.utils import PDEDataConfig
from pdearena.models.cond_pdemodel import get_model
from pdearena.models.registry import COND_MODEL_REGISTRY


@pytest.mark.parametrize("name", list(COND_MODEL_REGISTRY.keys()))
@pytest.mark.parametrize("param_conditioning", [None, "scalar"])
@pytest.mark.slow
def test_named_cond_models(name, param_conditioning):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pdedata = PDEDataConfig(1, 1, 56, 2)
    args = OmegaConf.create(
        {
            "name": name,
            "time_history": 4,
            "time_future": 1,
            "activation": "gelu",
            "param_conditioning": param_conditioning,
        }
    )
    model = get_model(args, pdedata).to(device)
    input = torch.randn(8, 1, 3, 64, 64, device=device)

    # scalar time
    t = torch.ones(8, dtype=torch.long, device=device)
    # scalar conditioning
    if param_conditioning == "scalar":
        z = torch.randn(8, device=device)
    else:
        z = None

    output = model(input, t, z)
    assert output.shape == (8, 1, 3, 64, 64)
