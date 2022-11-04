# Model Zoo

Here we document the performance of various models currently implemented with [**PDEArena**](https://microsoft.github.io/pdearena).
The tables below provide results and useful statistics about training and inference.

| Model                 | Num. Params | Model Size (MB) | Peak GPU Memory (MB) | Forward Time | Forward+Backward Time |
| --------------------- | ----------: | --------------: | -------------------: | ------------ | --------------------- |
| DilatedResNet128      |       4.2 M |            16.7 |               4849.0 | 0.118        | 0.341                 |
| DilatedResNet128-norm |       4.2 M |            16.7 |               6922.0 | 0.183        | 0.423                 |
| FNO-16m               |       134 M |           537.5 |               2949.5 | 0.059        | 0.172                 |
| FNO-8m                |      33.7 M |           134.9 |               2159.0 | 0.056        | 0.161                 |
| FNOs-128-16m          |      67.2 M |           268.8 |              1851.94 | 0.031        | 0.089                 |
| FNOs-64-32m           |      67.1 M |           268.5 |              1204.06 | 0.016        | 0.050                 |
| FNOs-96-32m           |       151 M |           604.2 |               2179.0 | 0.026        | 0.080                 |
| ResNet128             |       2.4 M |             9.6 |              4273.74 | 0.083        | 0.179                 |
| ResNet256             |       9.6 M |            38.3 |               8499.0 | 0.231        | 0.501                 |
| U-FNet1-16m           |       160 M |           643.6 |              4036.59 | 0.082        | 0.196                 |
| U-FNet1-8m            |       148 M |           593.3 |              3936.07 | 0.081        | 0.195                 |
| U-FNet2-16m           |       175 M |           700.5 |              4149.62 | 0.083        | 0.200                 |
| U-FNet2-16mc          |       219 M |           876.7 |              4496.41 | 0.084        | 0.204                 |
| U-FNet2-8m            |       151 M |           606.2 |              3960.89 | 0.082        | 0.198                 |
| U-FNet2-8mc           |       162 M |           650.2 |              4046.26 | 0.083        | 0.199                 |
| U-FNet2attn-16m       |       179 M |           717.3 |              4229.27 | 0.085        | 0.206                 |
| UNO128                |       440 M |          1761.8 |               5512.0 | 0.158        | 0.341                 |
| UNO64                 |       110 M |           440.5 |              1925.28 | 0.065        | 0.134                 |
| Unet2015-128          |       124 M |           496.5 |               3002.0 | 0.042        | 0.117                 |
| Unet2015-128-tanh     |       124 M |           496.5 |               3000.0 | 0.042        | 0.118                 |
| Unet2015-64           |      31.0 M |           124.2 |              1304.99 | 0.012        | 0.037                 |
| Unet2015-64-tanh      |      31.0 M |           124.2 |              1277.35 | 0.013        | 0.037                 |
| Unetbase128           |       124 M |           496.6 |               2999.0 | 0.056        | 0.134                 |
| Unetbase64            |      31.1 M |           124.2 |              1277.81 | 0.021        | 0.046                 |
| Unetmod64             |       144 M |           577.1 |              3898.37 | 0.079        | 0.186                 |
| Unetmodattn64         |       148 M |           593.9 |              3974.75 | 0.081        | 0.192                 |

- **Date Created**: 2022-11-02 02:59:04.403905
- **GPU**: Tesla V100-PCIE-16GB
