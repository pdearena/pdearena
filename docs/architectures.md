# Available Model Architectures

## Standard

| Architecture | Model Name | Description | Based on |
| ------------ | ---------- | ----------- | -------- |
| FNO          |            |             |     [:simple-github:](https://github.com/zongyi-li/fourier_neural_operator) [:simple-arxiv:](https://arxiv.org/abs/2010.08895)     |
| ResNet       |            |             |          |
| Dilated-ResNet |           |             |    [:simple-arxiv:](https://arxiv.org/abs/2112.15275)      |
| U-Net-2015   |            |  Original U-Net Implementation           |  [:simple-github:](https://github.com/pdebench/PDEBench/blob/main/pdebench/models/unet/unet.py)    [:simple-arxiv:](https://openreview.net/forum?id=dh_MkX0QfrK)    |
| U-Net-base   |            |  Original U-Net Implementation but with an extra layer           |  [:simple-github:]()        |
| U-Net-modern |            | Modern U-Nets as used in various diffusion modeling applications            |  [:simple-github:](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py) [:simple-arxiv:](https://arxiv.org/abs/2006.11239)   [:simple-arxiv:](https://arxiv.org/abs/2102.09672)   |
| UF-Net       |            | Modern U-Nets with Fourier layer based downsampling            |   [:simple-arxiv:](https://arxiv.org/abs/2209.15616)       |
| UNO          |            |              | [:simple-github:](https://github.com/ashiq24/UNO) [:simple-arxiv:](https://arxiv.org/abs/2204.11127)         |


## Conditioned

| Architecture | Model Name | Description |
| ------------ | ---------- | ----------- |
| FNO          |            |             |
| U-Net-modern |            |             |
| UF-Net       |            |             |