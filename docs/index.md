# Welcome to PDEArena

[arXiv](https://arxiv.org/abs/2209.15616){ .md-button }

## Installation

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
docker run --gpus all -it -v $(pwd):/code -v /mnt/data:/data
```

## Downloading data from Azure

## Citation

## Other Projects of Interest