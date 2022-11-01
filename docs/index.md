# Introduction

Welcome to the documentation of PDEArena. 
This repository aims to provide a PDE surrogate learning framework and benchmark while following deep learning best practices.

## Installation Guide


### Using `conda`

#### Setup dependencies

```bash
# clone the repo
git clone https://github.com/microsoft/pdearena

# create and activate env
cd pdearena
conda env create --file docker/environment.yml
conda activate pdearena
```

#### Install this package

```bash
# install so the project is in PYTHONPATH
pip install -e .
```

If you also want to do data generation:

```bash
pip install -e ".[datagen]"
```

### Using `docker`

- First build docker container
```bash
cd docker
docker build -t pdearena .
```

- Next 
```bash
cd pdearena
docker run --gpus all -it --rm --user $(id -u):$(id -g) \
    -v $(pwd):/code -v /mnt/data:/data --workdir /code -e PYTHONPATH=/code \
    pdearena:latest
```
!!! note 

    - `--gpus all -it --rm --user $(id -u):$(id -g)`: enables using all GPUs and runs an interactive session with current user's UID/GUID to avoid `docker` writing files as root.
    - `-v $(pwd):/code -v /mnt/data:/data --workdir /code`: mounts current directory and data directory (i.e. the cloned git repo) to `/code` and `/data` respectively, and use the `code` directory as the current working directory.

## Downloading data from Azure

## Citation

If you find this repository useful in your work, please cite:

```bibtex
@article{gupta2022towards,
  title={Towards Multi-spatiotemporal-scale Generalized PDE Modeling},
  author={Gupta, Jayesh K and Brandstetter, Johannes},
  journal={arXiv preprint arXiv:2209.15616},
  year={2022}
}
```
Also consider starring [the github repo](https://github.com/microsoft/pdearena).

## Other Projects of Interest

- [PDEBench](https://github.com/pdebench/PDEBench)