# Data Generation

!!! tip

    For multi-gpu training, make sure that the number of shards is divisible by the number of GPUs. 8 is usually a safe number.

## Navier Stokes 2D

### Standard

```bash
export seed=42;

python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \ 
    experiment=smoke mode=train samples=256 seed=$seed pdeconfig.sample_rate=4 \  
    dirname=/mnt/data/navierstokes;

python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \ 
    experiment=smoke mode=valid samples=32 seed=$seed pdeconfig.sample_rate=4 \ 
    dirname=/mnt/data/navierstokes;

python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \         
    experiment=smoke mode=test samples=32 seed=$seed pdeconfig.sample_rate=4 \
    dirname=/mnt/data/navierstokes;
```

### Conditioned

```bash
python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \ 
    experiment=smoke_cond mode=train samples=256 seed=$seed \  
    dirname=/mnt/data/navierstokes;

python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \ 
    experiment=smoke_cond mode=valid samples=32 seed=$seed \ 
    dirname=/mnt/data/navierstokes;

python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \         
    experiment=smoke_cond mode=test samples=32 seed=$seed \
    dirname=/mnt/data/navierstokes;
```

### Data normalization

The data was reasonably bounded that we didn't need any normalization.

## Shallow water 2D

```bash
export seed=42;

python scripts/generate_data.py base=pdedatagen/configs/shallowwater.yaml \ 
    experiment=shallowwater mode=train samples=256 seed=$seed \  
    dirname=/mnt/data/shallowwater;

python scripts/generate_data.py base=pdedatagen/configs/shallowwater.yaml \ 
    experiment=shallowwater mode=valid samples=32 seed=$seed \ 
    dirname=/mnt/data/shallowwater;

python scripts/generate_data.py base=pdedatagen/configs/shallowwater.yaml \         
    experiment=shallowwater mode=test samples=32 seed=$seed \
    dirname=/mnt/data/shallowwater;
```

### Data normalization

```bash
python scripts/compute_normalization.py \
    --dataset shallowwater /mnt/data/shallowwater
```

## PDEBench

[`PDEBench`](https://github.com/pdebench/PDEBench) data can be downloaded with the included [`download_pdebenchdata.py`]() script:

For example to download the Incompressible Navier Stokes dataset:
```bash
DATAVERSE_URL=https://darus.uni-stuttgart.de python scripts/download_pdebenchdata.py --outdir /mnt/data/PDEBench/ --limit ns_incom
```

### Resharding for multi-gpu experiments
Coming soon...

### Data normalization
Coming soon...

## Your PDE

Please submit a [pull request](https://github.com/microsoft/pdearena) to add a data loading pipeline for your PDE dataset.