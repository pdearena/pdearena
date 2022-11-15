# Repository Map

``` { .bash .annotate }
📁 pdearena/
    📁 data/ # (1)
        📁 twod/ # (2)
            📁 datapipes/
                📄 common.py # (3)
                ... # (4)
    📁 models/
        📄 registry.py # (5)
        📄 pdemodel.py # (6)
        📄 cond_pdemode.py # (7)
    📁 modules/ #(8)
        📁 conditioned/
            ...
        📄 twod_resnet.py
        📄 twod_unet2015.py
        📄 twod_unetbase.py
        📄 twod_unet.py
        📄 twod_uno.py
        📄 activations.py # (9)
        📄 loss.py # (10)
        ...
    📄 utils.py
    ...
📁 pdedatagen/
    📁 configs/ #(11)
    📁 shallowwater
    📄 navier_stokes.py
    📄 pde.py # (12)
📁 scripts/
    📄 train.py # (13)
    📄 cond_train.py # (14)
    📄 generate_data.py # (15)
    📄 convertnc2zarr.py # (16)
    📄 compute_normalization.py # (17)
📁 configs/ # (18)
```

1. Everything data loading related goes here.
2. Currently we only have 2D data support. But should be easy enough to add appropriate mechanisms for 1D, 3D and beyond.
3. Common data pipe tranformations useful for building training and evaluation pipelines.
4. Dataset opening data pipes for individual datasets.
5. Model registry. Remember to register your new model here.
6. [`LightningModule`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html) to support standard PDE surrogate learning.
7. [`LightningModule`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html) to support time and parameter conditioned PDE surrogate learning.
8. All the network architectures go here.
9. Activation function registry
10. Currently supported loss functions
11. Configuration files for PDE data generation
12. Register your new PDE configurations here
13. Main training script for standard PDE surrogate training and testing
14. Main training script for conditioned PDE surrogate training and testing
15. Main script for generating data
16. Supporting script to convert `netcdf` files to `zarr` for faster data loading
17. Supporting script to compute the data normalization statistics. Add normalization methods for your data here.
18. pytorch-lightning configs to run experiments.
