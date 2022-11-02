import os.path
from datetime import datetime
import json
import time
import torch
from pdearena.models.pdemodel import get_model
from pdearena.pde import NavierStokes2D
import timeit
from omegaconf import OmegaConf

_PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

class Timer(object):

    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start

MODELS = {
    "FNO-8m": {
        "name": "FourierResNet",
        "hidden_channels": 128,
        "modes1": 8,
        "modes2": 8,
        "norm": False,
    },
    "FNO-16m": {
        "name": "FourierResNet",
        "hidden_channels": 128,
        "modes1": 16,
        "modes2": 16,
        "norm": False,
    },
    "FNOs-128-32m": {
        "name": "FourierResNetSmall",
        "hidden_channels": 128,
        "modes1": 32,
        "modes2": 32,
        "norm": False,
    },
    "FNOs-128-16m": {
        "name": "FourierResNetSmall",
        "hidden_channels": 128,
        "modes1": 16,
        "modes2": 16,
        "norm": False,
    },
    "FNOs-64-32m": {
        "name": "FourierResNetSmall",
        "hidden_channels": 64,
        "modes1": 32,
        "modes2": 32,
        "norm": False,
    },    
    "FNOs-96-32m": {
        "name": "FourierResNetSmall",
        "hidden_channels": 96,
        "modes1": 32,
        "modes2": 32,
        "norm": False,
    },     
    "UNO64": {
        "name": "UNO",
        "hidden_channels": 64,
    },
    "UNO128": {
        "name": "UNO",
        "hidden_channels": 128,
    },
    "Unet2015-64": {
        "name": "Unet2015",
        "hidden_channels": 64,
    },
    "Unet2015-128": {
        "name": "Unet2015",
        "hidden_channels": 128,
    },
    "Unet2015-64-tanh": {
        "name": "Unet2015",
        "hidden_channels": 64,
        "activation": "tanh"
    },
    "Unet2015-128-tanh": {
        "name": "Unet2015",
        "hidden_channels": 128,
        "activation": "tanh",
    },
    "Unetbase64": {
        "name": "OldUnet",
        "hidden_channels": 64,
    },
    "Unetbase128": {
        "name": "OldUnet",
        "hidden_channels": 128,
    },
    "Unetmod64": {
        "name": "Unet",
        "hidden_channels": 64,
        "norm": True,
    },
    "Unetmodattn64": {
        "name": "UnetMidAttn",
        "hidden_channels": 64,
        "norm": True,
    },
    "U-FNet1-8m": {
        "name": "Fourier1Unet",
        "hidden_channels": 64,
        "modes1": 8,
        "modes2": 8,
        "norm": True,
    },
    "U-FNet1-16m": {
        "name": "Fourier1Unet",
        "hidden_channels": 64,
        "modes1": 16,
        "modes2": 16,
        "norm": True,
    },
    "U-FNet2-8m": {
        "name": "FourierUnet",
        "hidden_channels": 64,
        "modes1": 8,
        "modes2": 8,
        "norm": True,
    },
    "U-FNet2-8mc": {
        "name": "FourierUnetConstMode",
        "hidden_channels": 64,
        "modes1": 8,
        "modes2": 8,
        "norm": True,
    },    
    "U-FNet2-16m": {
        "name": "FourierUnet",
        "hidden_channels": 64,
        "modes1": 16,
        "modes2": 16,
        "norm": True,
    },
    "U-FNet2-16mc": {
        "name": "FourierUnetConstMode",
        "hidden_channels": 64,
        "modes1": 16,
        "modes2": 16,
        "norm": True,
    },
    "U-FNet2attn-16m": {
        "name": "FourierUnetMidAttn",
        "hidden_channels": 64,
        "modes1": 16,
        "modes2": 16,
        "norm": True,
    },
    "ResNet128": {
        "name": "ResNet",
        "hidden_channels": 128,
        "norm": True,

    },
    "ResNet256": {
        "name": "ResNet",
        "hidden_channels": 256,
        "norm": True,
    },
    "DilatedResNet128": {
        "name": "DilatedResNet",
        "hidden_channels": 128,
        "norm": False,
    },
    "DilatedResNet128-norm": {
        "name": "DilatedResNet",
        "hidden_channels": 128,
        "norm": True,
    },    
}

def save_data(data):
    """"Save the data file"""
    filename = os.path.join(_PROJECT_DIR, "docs", "models_fwd_time.json")
    with open(filename, "w") as f:
        data["date-created"] = str(datetime.now())
        data["gpu-name"] = torch.cuda.get_device_name()
        json.dump(data, f, sort_keys=True, indent=4)

def main():
    precision_megabytes = (32/8.0) * 1e-6
    pde = NavierStokes2D(
      tmin= 18.0,
      tmax= 102.0,
      Lx= 32.0,
      Ly= 32.0,
      nt= 56,
      nx= 128,
      ny= 128,
      skip_nt= 0,
      sample_rate= 4,
      nu= 0.01,
      buoyancy_x= 0.0,
      buoyancy_y= 0.5,
      correction_strength= 1.0,
      force_strength= 1.0,
      force_frequency= 4,
      n_scalar_components= 1,
      n_vector_components= 1,
      device= "cpu"
    )
    time_history = 4
    results = {}
    for k,v in MODELS.items():
        args = OmegaConf.create({**v, **{"time_history": time_history, "time_future": 1, "diffmode": False, "usegrid": False, "activation": "gelu",}})
        n_warmups = 5
        n_repeats = 25
        model = get_model(args, pde).to("cuda")
        bs = 8
        input = torch.randn(bs, time_history, 3, 128, 128, device='cuda')
        for _ in range(n_warmups):
            _ = model(input)
        torch.cuda.synchronize()
        with Timer() as ft:
            for i in range(n_repeats):
                with torch.set_grad_enabled(False):
                    out = model(input)
                torch.cuda.synchronize()
        print(f"{k} forward time: {ft.dt/n_repeats:.3f}")
        trainable_params = sum(
        	p.numel() for p in model.parameters() if p.requires_grad
        )
        total_parameters = sum(p.numel() for p in model.parameters())
        model_size = total_parameters * precision_megabytes
        results[k] = {"fwd_time": ft.dt/n_repeats, "num_params": trainable_params, "model_size": model_size}
        del model
        torch.cuda.empty_cache()
        time.sleep(1)

    save_data(results)        

if __name__ == "__main__":
    main()
    