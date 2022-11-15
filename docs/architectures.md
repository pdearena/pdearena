# Available Model Architectures

If you would like your architecture added, please submit a [pull request](https://github.com/microsoft/pdearena/pulls).

## Standard

| Architecture   | Module                                                                                                                              | Description                                                                                        | Based on                                                                                                                                                                                                                                        |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FNO            | `ResNet` with `FourierBasicBlock`                                                                                               | Fourier Neural Operator implementation with support for deeper architectures (8 and 4 layers)      | [:simple-github:](https://github.com/zongyi-li/fourier_neural_operator) [:simple-arxiv:](https://arxiv.org/abs/2010.08895)                                                                                                                      |
| ResNet         | `ResNet` with `BasicBlock`                                                                                                                                |         ResNet architectures using 8 residual blocks, no downscaling                                                                                           |                                                                                                                                                                                                                                                 |
| DilResNet | `ResNet` with `DilatedBasicBlock`                                                                                                                         |       ResNet where each block individually consists of 7 dilated CNN layers with dilation rates of [1, 2, 4, 8, 4, 2, 1], no downscaling                                                                                              | [:simple-arxiv:](https://arxiv.org/abs/2112.15275)                                                                                                                                                                                              |
| U-Net~2015~    | `Unet2015`                                                                                                                              | Original U-Net implementation                                                                      | [:simple-github:](https://github.com/pdebench/PDEBench/blob/main/pdebench/models/unet/unet.py)    [:simple-arxiv:](https://openreview.net/forum?id=dh_MkX0QfrK)                                                                                 |
| U-Net~base~    | `Unetbase`                                                                                                                               | Our interpretation of original U-Net implementation without bottleneck layer and using `GroupNorm` |                                                                                                                                                                                                                            |
| U-Net~mod~     | `Unet`                                                          | Modern U-Nets with Wide ResNet blocks, as used in various diffusion modeling applications          | [:simple-github:](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py) [:simple-arxiv:](https://arxiv.org/abs/2006.11239)   [:simple-arxiv:](https://arxiv.org/abs/2102.09672) |
| U-F[*]Net        | `FourierUnet` | Modern U-Nets with [*]  Fourier blocks in the downsampling path                                 | [:simple-arxiv:](https://arxiv.org/abs/2209.15616)                                                                                                                                                                                              |
| UNO            | `UNO`                                                                                                                                   | Original U-shaped Neural Operator Implementation                                                   | [:simple-github:](https://github.com/ashiq24/UNO) [:simple-arxiv:](https://arxiv.org/abs/2204.11127)                                                                                                                                            |

## Conditioned

!!! note
    Currently only scalar parameter conditioning is available.

| Architecture | Model Name      | Description |
| ------------ | --------------- | ----------- |
| FNO          | `ResNet` with `FourierBasicBlock` | `Addition` based conditioning in both spatial and spectral domain.            |
| U-Net-modern | `Unet`          | `Addition` and `AdaGN` style conditioning in the spatial domain.            |
| UF-Net       | `FourierUnet`   | `Addition` and `AdaGN` style conditioning in the spatial domain, `Addition` in the spectral domain.            |

## Model Architecture Registry Philosophy

While in principle we can make every architecture fully configurable via configuration files, we find it can affect the readability of the code quite a bit. Feel free to open issues or pull-requests for further configuration ability or any other suggestions for managing the configurability-readability tradeoffs.

## Other Projects of Interest

- [PDEBench](https://github.com/pdebench/PDEBench)
- [NVIDIA Modulus](https://developer.nvidia.com/modulus)
- [NeuralOperators.jl](https://github.com/SciML/NeuralOperators.jl)
