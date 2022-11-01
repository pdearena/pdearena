# Available Model Architectures

If you would like your architecture added, please submit a [pull request](https://github.com/microsoft/pdearena/pulls).

## Standard

| Architecture | Model Name | Description | Based on |
| ------------ | ---------- | ----------- | -------- |
| FNO          |    `FourierResNet`, `FourierResNetSmall`        |   Fourier Neural Operator Implementation with support for deeper architectures (8 and 4 layers)          |     [:simple-github:](https://github.com/zongyi-li/fourier_neural_operator) [:simple-arxiv:](https://arxiv.org/abs/2010.08895)     |
| ResNet       |    `ResNet`        |             |          |
| Dilated-ResNet |  `DilatedResNet`         |             |    [:simple-arxiv:](https://arxiv.org/abs/2112.15275)      |
| U-Net~2015~   |  `Unet2015`          |  Original U-Net Implementation           |  [:simple-github:](https://github.com/pdebench/PDEBench/blob/main/pdebench/models/unet/unet.py)    [:simple-arxiv:](https://openreview.net/forum?id=dh_MkX0QfrK)    |
| U-Net~base~   |  `OldUnet`          |  Original U-Net Implementation but with an extra layer           |  [:simple-github:]()        |
| U-Net~mod~ |  `Unet`          | Modern U-Nets with Wide ResNet blocks, as used in various diffusion modeling applications            |  [:simple-github:](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py) [:simple-arxiv:](https://arxiv.org/abs/2006.11239)   [:simple-arxiv:](https://arxiv.org/abs/2102.09672)   |
| UF-Net       |   `FourierUnet`         | Modern U-Nets with Fourier layer based downsampling            |   [:simple-arxiv:](https://arxiv.org/abs/2209.15616)       |
| UNO          |   `UNO`         |  Original U-shaped Neural Operator Implementation            | [:simple-github:](https://github.com/ashiq24/UNO) [:simple-arxiv:](https://arxiv.org/abs/2204.11127)         |


## Conditioned

| Architecture | Model Name | Description |
| ------------ | ---------- | ----------- |
| FNO          |  `FourierResNet`          |             |
| U-Net-modern |  `Unet`          |             |
| UF-Net       |  `FourierUnet`          |             |

## Model Architecture Registry Philosophy

While in principle we can make every architecture fully configurable via configuration files, we find it can affect the readability of the code quite a bit. Feel free to open issues or pull-requests for further configuration ability or any other suggestions for managing the configurability-readability tradeoffs.  