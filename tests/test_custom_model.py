import torch
from omegaconf import OmegaConf

from pdearena.data.utils import PDEDataConfig
from pdearena.models.pdemodel import get_model


def test_custom_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pdedata = PDEDataConfig(1, 1, 14, 2)
    args = OmegaConf.create(
        {
            "name": "custom_ufnet",
            "model": {
                "class_path": "pdearena.modules.twod_unet.FourierUnet",
                "init_args": {
                    "n_input_scalar_components": 1,
                    "n_input_vector_components": 1,
                    "n_output_scalar_components": 1,
                    "n_output_vector_components": 1,
                    "time_history": 4,
                    "time_future": 1,
                    "activation": "gelu",
                    "hidden_channels": 64,
                    "modes1": 8,
                    "modes2": 8,
                    "norm": True,
                    "n_blocks": 1,
                    "n_fourier_layers": 1,
                    "mid_attn": True,
                    "use1x1": True,
                },
            },
        }
    )
    model = get_model(args, pdedata).to(device)
    input = torch.randn(8, 4, 3, 64, 64, device=device)
    output = model(input)
    assert output.shape == (8, 1, 3, 64, 64)
