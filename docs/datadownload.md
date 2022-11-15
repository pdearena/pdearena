# Downloading data from Azure

First make sure you have [`azcopy`](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) installed.

On Linux you can do:

```bash
wget https://aka.ms/downloadazcopy-v10-linux
tar -xvf downloadazcopy-v10-linux
# move to somewhere on your PATH
mv ./azcopy_linux_amd64_*/azcopy $HOME/.local/bin
```

## Navier Stokes - 2D

Generated using [Φ~Flow~](https://github.com/tum-pbs/PhiFlow/).

### Standard dataset

```bash
azcopy copy "https://pdearenarelease.blob.core.windows.net/datasets/NavierStokes2D_smoke" \
            "/mnt/data/" --recursive
```

### Conditioning dataset

```bash
azcopy copy "https://pdearenarelease.blob.core.windows.net/datasets/NavierStokes2D_cond_smoke_v1" \
            "/mnt/data/" --recursive
```

## Shallow water - 2D

Generated using [SpeedyWeather.jl](https://github.com/milankl/SpeedyWeather.jl).

```bash
azcopy copy "https://pdearenarelease.blob.core.windows.net/datasets/ShallowWater2D" \
            "/mnt/data/" --recursive
```

## Maxwell - 3D

Generated using [Python 3D FDTD Simulator](https://github.com/flaport/fdtd).

```bash
azcopy copy "https://pdearenarelease.blob.core.windows.net/datasets/Maxwell2D" \
            "/mnt/data/" --recursive
```

# PDEBench

[`PDEBench`](https://github.com/pdebench/PDEBench) data can be downloaded with the included [`download_pdebenchdata.py`](<>) script:

For example to download the Incompressible Navier Stokes dataset:

```bash
export DATAVERSE_URL=https://darus.uni-stuttgart.de;
python scripts/download_pdebenchdata.py \
    --outdir /mnt/data/PDEBench/ --limit ns_incom
```
