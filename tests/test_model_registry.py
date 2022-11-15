import pytest
import torch
from omegaconf import OmegaConf

from pdearena.data.utils import PDEDataConfig
from pdearena.models.pdemodel import get_model
from pdearena.models.registry import MODEL_REGISTRY


@pytest.mark.parametrize("name", list(MODEL_REGISTRY.keys()))
@pytest.mark.slow
def test_named_models(name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pdedata = PDEDataConfig(1, 1, 14, 2)
    args = OmegaConf.create(
        {
            "name": name,
            "time_history": 4,
            "time_future": 1,
            "activation": "gelu",
        }
    )
    model = get_model(args, pdedata).to(device)
    input = torch.randn(8, 4, 3, 64, 64, device=device)
    output = model(input)
    assert output.shape == (8, 1, 3, 64, 64)
