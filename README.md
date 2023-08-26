<p align="center">
  <img src="https://user-images.githubusercontent.com/1785175/199388258-4ca228d5-9f0b-463d-82dd-6c27015bc4ab.png" width="400px">
</p>
<h1 align="center">PDEArena</h1>

[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen)](https://microsoft.github.io/pdearena)
[![Paper](https://img.shields.io/badge/arXiv-2209.15616-blue)](https://arxiv.org/abs/2209.15616)

This repository contains code accompanying the paper [**Towards multi-spatiotemporal-scale generalized PDE modeling**](https://arxiv.org/abs/2209.15616), and as such we hope this serves as a starting point for future PDE surrogate learning research.
We have imported models from [**Clifford neural layers for PDE modeling**](https://arxiv.org/abs/2209.04934) and [**Geometric Clifford Algebra Networks**](https://arxiv.org/abs/2302.06594).

For details about usage please see [documentation](https://microsoft.github.io/pdearena).
If you have any questions or suggestions please open a [discussion](https://github.com/microsoft/pdearena/discussions). If you notice a bug, please open an [issue](https://github.com/microsoft/pdearena/issues).

## Citation

If you find this repository useful in your research, please consider citing the following papers:

Initial PDE arena, architecture zoo, Navier-Stokes and Shallow Water datasets:
```bibtex
@article{gupta2022towards,
  title={Towards Multi-spatiotemporal-scale Generalized PDE Modeling},
  author={Gupta, Jayesh K and Brandstetter, Johannes},
  journal={arXiv preprint arXiv:2209.15616},
  year={2022}
}
```

3D Clifford FNO layers, Maxwell data:
```bibtex
@article{brandstetter2022clifford,
  title={Clifford neural layers for PDE modeling},
  author={Brandstetter, Johannes and Berg, Rianne van den and Welling, Max and Gupta, Jayesh K},
  journal={arXiv preprint arXiv:2209.04934},
  year={2022}
}
```

CGAN layers, CGAN-UNet architectures:
```bibtex
@article{ruhe2023geometric,
  title={Geometric clifford algebra networks},
  author={Ruhe, David and Gupta, Jayesh K and De Keninck, Steven and Welling, Max and Brandstetter, Johannes},
  journal={arXiv preprint arXiv:2302.06594},
  year={2023}
}
```

Do remember to cite the original papers as well for individual architectures.

You can further checkout our dedicated repo [**CliffordLayers**](https://microsoft.github.io/cliffordlayers/).


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
