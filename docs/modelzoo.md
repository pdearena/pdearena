
# Model Zoo

Here we document the performance of various models currently implemented with [**PDEArena**](https://microsoft.github.io/pdearena).
The tables below provide results and useful statistics about training and inference.



| Model | Num. Params | Model Size (MB) | Forward Time | Forward+Backward Time |
| --- | --- | --- | --- | --- |
| DilatedResNet128 | 4.2 M | 16.670 | 0.118 | 0.341 |
| DilatedResNet128-norm | 4.2 M | 16.698 | 0.183 | 0.423 |
| FNO-16m | 134 M | 537.540 | 0.059 | 0.171 |
| FNO-8m | 33.7 M | 134.886 | 0.056 | 0.161 |
| FNOs-128-16m | 67.2 M | 268.840 | 0.031 | 0.089 |
| FNOs-64-32m | 67.1 M | 268.539 | 0.016 | 0.050 |
| ResNet128 | 2.4 M | 9.602 | 0.083 | 0.179 |
| ResNet256 | 9.6 M | 38.341 | 0.231 | 0.501 |
| U-FNet1-16m | 160 M | 643.639 | 0.082 | 0.196 |
| U-FNet1-8m | 148 M | 593.308 | 0.081 | 0.195 |
| U-FNet2-16m | 175 M | 700.525 | 0.083 | 0.200 |
| U-FNet2-16mc | 219 M | 876.685 | 0.084 | 0.204 |
| U-FNet2-8m | 151 M | 606.153 | 0.082 | 0.198 |
| U-FNet2-8mc | 162 M | 650.193 | 0.083 | 0.199 |
| U-FNet2attn-16m | 179 M | 717.326 | 0.085 | 0.206 |
| UNO128 | 440 M | 1761.817 | 0.158 | 0.341 |
| UNO64 | 110 M | 440.466 | 0.065 | 0.134 |
| Unet2015-128 | 124 M | 496.528 | 0.042 | 0.117 |
| Unet2015-128-tanh | 124 M | 496.528 | 0.042 | 0.118 |
| Unet2015-64 | 31.0 M | 124.172 | 0.012 | 0.037 |
| Unet2015-64-tanh | 31.0 M | 124.172 | 0.013 | 0.037 |
| Unetbase128 | 124 M | 496.588 | 0.056 | 0.134 |
| Unetbase64 | 31.1 M | 124.201 | 0.021 | 0.046 |
| Unetmod64 | 144 M | 577.055 | 0.079 | 0.185 |
| Unetmodattn64 | 148 M | 593.857 | 0.081 | 0.192 |

- **Date Created**: 2022-11-02 02:59:04.403905
- **GPU**: Tesla V100-PCIE-16GB