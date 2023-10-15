from cliffordlayers.models.basic.threed import CliffordFourierBasicBlock3d

from pdearena import utils
from pdearena.modules.conditioned.twod_resnet import (
    FourierBasicBlock as CondFourierBasicBlock,
)
from pdearena.modules.threed import FourierBasicBlock3D
from pdearena.modules.twod_resnet import (
    BasicBlock,
    DilatedBasicBlock,
    FourierBasicBlock,
)

MODEL_REGISTRY = {
    "FNO-128-8m": {
        "class_path": "pdearena.modules.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 128,
            "norm": False,
            "num_blocks": [1, 1, 1, 1],
            "block": utils.partialclass("CustomFourierBasicBlock", FourierBasicBlock, modes1=8, modes2=8),
            "diffmode": False,
            "usegrid": False,
        },
    },
    "FNO-128-16m": {
        "class_path": "pdearena.modules.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 128,
            "norm": False,
            "num_blocks": [1, 1, 1, 1],
            "block": utils.partialclass("CustomFourierBasicBlock", FourierBasicBlock, modes1=16, modes2=16),
            "diffmode": False,
            "usegrid": False,
        },
    },
    "FNOs-128-32m": {
        "class_path": "pdearena.modules.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 128,
            "num_blocks": [1, 1],
            "block": utils.partialclass("CustomFourierBasicBlock", FourierBasicBlock, modes1=32, modes2=32),
            "norm": False,
        },
    },
    "FNOs-128-16m": {
        "class_path": "pdearena.modules.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 128,
            "num_blocks": [1, 1],
            "block": utils.partialclass("CustomFourierBasicBlock", FourierBasicBlock, modes1=16, modes2=16),
            "norm": False,
        },
    },
    "FNOs-64-32m": {
        "class_path": "pdearena.modules.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 64,
            "num_blocks": [1, 1],
            "block": utils.partialclass("CustomFourierBasicBlock", FourierBasicBlock, modes1=32, modes2=32),
            "norm": False,
        },
    },
    "FNOs-96-32m": {
        "class_path": "pdearena.modules.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 96,
            "num_blocks": [1, 1],
            "block": utils.partialclass("CustomFourierBasicBlock", FourierBasicBlock, modes1=32, modes2=32),
            "norm": False,
        },
    },
    "UNO-64": {
        "class_path": "pdearena.modules.twod_uno.UNO",
        "init_args": {
            "hidden_channels": 64,
        },
    },
    "UNO-128": {
        "class_path": "pdearena.modules.twod_uno.UNO",
        "init_args": {
            "hidden_channels": 128,
        },
    },
    "Unet2015-64": {
        "class_path": "pdearena.modules.twod_unet2015.Unet2015",
        "init_args": {
            "hidden_channels": 64,
        },
    },
    "Unet2015-128": {
        "class_path": "pdearena.modules.twod_unet2015.Unet2015",
        "init_args": {
            "hidden_channels": 128,
        },
    },
    "Unetbase-64": {
        "class_path": "pdearena.modules.twod_unetbase.Unetbase",
        "init_args": {
            "hidden_channels": 64,
        },
    },
    "Unetbase-128": {
        "class_path": "pdearena.modules.twod_unetbase.Unetbase",
        "init_args": {
            "hidden_channels": 128,
        },
    },
    "Unetmod-64": {
        "class_path": "pdearena.modules.twod_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
        },
    },
    "Unetmodattn-64": {
        "class_path": "pdearena.modules.twod_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "mid_attn": True,
        },
    },
    "Unetmod-64-1x1": {
        "class_path": "pdearena.modules.twod_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "use1x1": True,
        },
    },
    "Unetmodattn-64-1x1": {
        "class_path": "pdearena.modules.twod_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "mid_attn": True,
            "use1x1": True,
        },
    },
    "U-FNet1-8m": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 8,
            "modes2": 8,
            "norm": True,
            "n_fourier_layers": 1,
        },
    },
    "U-FNet1-16m": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 1,
        },
    },
    "U-FNet1-8m-1x1": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 8,
            "modes2": 8,
            "norm": True,
            "n_fourier_layers": 1,
            "use1x1": True,
        },
    },
    "U-FNet1-16m-1x1": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 1,
            "use1x1": True,
        },
    },
    "U-FNet2-8m": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 8,
            "modes2": 8,
            "norm": True,
            "n_fourier_layers": 2,
        },
    },
    "U-FNet2-8m-1x1": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 8,
            "modes2": 8,
            "norm": True,
            "n_fourier_layers": 2,
            "use1x1": True,
        },
    },
    "U-FNet2-8mc": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 8,
            "modes2": 8,
            "norm": True,
            "n_fourier_layers": 2,
            "mode_scaling": False,
        },
    },
    "U-FNet2-16m": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 2,
        },
    },
    "U-FNet2-16m-1x1": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 2,
            "use1x1": True,
        },
    },
    "U-FNet3-8m": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 8,
            "modes2": 8,
            "norm": True,
            "n_fourier_layers": 3,
        },
    },
    "U-FNet3-8m-1x1": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 8,
            "modes2": 8,
            "norm": True,
            "n_fourier_layers": 3,
            "use1x1": True,
        },
    },
    "U-FNet3-16m": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 3,
        },
    },
    "U-FNet3-16m-1x1": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 3,
            "use1x1": True,
        },
    },
    "U-FNet2-16mc": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 2,
            "mode_scaling": False,
        },
    },
    "U-FNet2attn-16m": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 2,
            "mid_attn": True,
        },
    },
    "U-FNet2attn-16m-1x1": {
        "class_path": "pdearena.modules.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 2,
            "mid_attn": True,
            "use1x1": True,
        },
    },
    "ResNet-128": {
        "class_path": "pdearena.modules.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 128,
            "norm": True,
            "block": BasicBlock,
            "num_blocks": [1, 1, 1, 1],
        },
    },
    "ResNet-256": {
        "class_path": "pdearena.modules.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 256,
            "norm": True,
            "block": BasicBlock,
            "num_blocks": [1, 1, 1, 1],
        },
    },
    "DilResNet-128": {
        "class_path": "pdearena.modules.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 128,
            "norm": False,
            "block": DilatedBasicBlock,
            "num_blocks": [1, 1, 1, 1],
        },
    },
    "DilResNet-128-norm": {
        "class_path": "pdearena.modules.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 128,
            "norm": True,
            "block": DilatedBasicBlock,
            "num_blocks": [1, 1, 1, 1],
        },
    },
    "MaxwellFNO3D-96-8": {
        "class_path": "pdearena.modules.threed.MaxwellResNet3D",
        "init_args": {
            "hidden_channels": 96,
            "num_blocks": [1, 1],
            "block": utils.partialclass("FourierBasicBlock3D", FourierBasicBlock3D, modes1=8, modes2=8, modes3=8),
            "diffmode": False,
        },
    },
    "MaxwellCFNO3D-32-8": {
        "class_path": "pdearena.modules.threed.CliffordMaxwellResNet3D",
        "init_args": {
            "g": [1, 1, 1],
            "hidden_channels": 32,
            "num_blocks": [1, 1],
            "block": utils.partialclass(
                "CliffordFourierBasicBlock3d", CliffordFourierBasicBlock3d, modes1=8, modes2=8, modes3=8
            ),
            "diffmode": False,
        },
    },
    "GCAFluidNet2d-32": {
        "class_path": "pdearena.modules.twod_gcaunet.GCAFluidNet2d",
        "init_args": {
            "in_channels": 1,
            "out_channels": 1,
            "hidden_channels": 32,
            "norm": True,
        },
    },
}

COND_MODEL_REGISTRY = {
    "FNO-128-16m": {
        "class_path": "pdearena.modules.conditioned.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 128,
            "norm": False,
            "num_blocks": [1, 1, 1, 1],
            "block": utils.partialclass("CustomFourierBasicBlock", CondFourierBasicBlock, modes1=16, modes2=16),
            "diffmode": False,
            "usegrid": False,
        },
    },
    "Unetmod-64": {
        "class_path": "pdearena.modules.conditioned.twod_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "use_scale_shift_norm": False,
        },
    },
    "Unetmod-64-adagn": {
        "class_path": "pdearena.modules.conditioned.twod_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "use_scale_shift_norm": True,
        },
    },
    "Unetmodattn-64": {
        "class_path": "pdearena.modules.conditioned.twod_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "mid_attn": True,
            "use_scale_shift_norm": False,
        },
    },
    "Unetmodattn-64-adagn": {
        "class_path": "pdearena.modules.conditioned.twod_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "mid_attn": True,
            "use_scale_shift_norm": True,
        },
    },
    "Unetmod-1d-64": {
        "class_path": "pdearena.modules.conditioned.oned_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "n_dims": 1,
        },
    },
    "Unetmodattn-1d-64": {
        "class_path": "pdearena.modules.conditioned.oned_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "mid_attn": True,
            "n_dims": 1,
        },
    },
    "Unetmod-1d-64-1x1": {
        "class_path": "pdearena.modules.conditioned.oned_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "use1x1": True,
            "n_dims": 1,
        },
    },
    "Unetmodattn-1d-64-1x1": {
        "class_path": "pdearena.modules.conditioned.oned_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "mid_attn": True,
            "use1x1": True,
            "n_dims": 1,
        },
    },
    "U-FNet1-16m": {
        "class_path": "pdearena.modules.conditioned.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 1,
            "use_scale_shift_norm": False,
        },
    },
    "U-FNet2-16m": {
        "class_path": "pdearena.modules.conditioned.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 2,
            "use_scale_shift_norm": False,
        },
    },
    "U-FNet1-16m-adagn": {
        "class_path": "pdearena.modules.conditioned.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 1,
            "use_scale_shift_norm": True,
        },
    },
    "U-FNet2-16m-adagn": {
        "class_path": "pdearena.modules.conditioned.twod_unet.FourierUnet",
        "init_args": {
            "hidden_channels": 64,
            "modes1": 16,
            "modes2": 16,
            "norm": True,
            "n_fourier_layers": 2,
            "use_scale_shift_norm": True,
        },
    },
}
