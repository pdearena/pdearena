# Downloading data from Huggingface

The data of PDEArena is currently hosted on Huggingface: [https://huggingface.co/pdearena](https://huggingface.co/pdearena).

Downloading the data is simple:

## Navier Stokes - 2D

Generated using [Φ~Flow~](https://github.com/tum-pbs/PhiFlow/).

### Standard dataset

SSH:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone git@hf.co:datasets/pdearena/NavierStokes-2D

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

HTTPS:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

### Conditioning dataset

SSH:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone git@hf.co:datasets/pdearena/NavierStokes-2D-conditoned

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

HTTPS:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D-conditoned

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```


## Shallow water - 2D

Generated using [SpeedyWeather.jl](https://github.com/milankl/SpeedyWeather.jl).

SSH:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone git@hf.co:datasets/pdearena/ShallowWater-2D

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

HTTPS:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/pdearena/ShallowWater-2D

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

## Maxwell - 3D

Generated using [Python 3D FDTD Simulator](https://github.com/flaport/fdtd).

SSH:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone git@hf.co:datasets/pdearena/Maxwell-3D

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

HTTPS:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/pdearena/Maxwell-3D

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

## Kuramoto-Sivashinsky - 1D

Generated using [LPSDA](https://github.com/brandstetter-johannes/LPSDA).

SSH:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone git@hf.co:datasets/pdearena/Kuramoto-Sivashinsky-1D

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

HTTPS:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/pdearena/Kuramoto-Sivashinsky-1D

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

# PDEBench

[`PDEBench`](https://github.com/pdebench/PDEBench) data can be downloaded with the included [`download_pdebenchdata.py`](<>) script:

For example to download the Incompressible Navier Stokes dataset:

```bash
export DATAVERSE_URL=https://darus.uni-stuttgart.de;
python scripts/download_pdebenchdata.py \
    --outdir /mnt/data/PDEBench/ --limit ns_incom
```
