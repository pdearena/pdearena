# Data Generation

!!! tip

    For multi-gpu training, make sure that the number of shards is divisible by the number of GPUs. 8 is usually a safe number.


## Navier Stokes 2D

### Standard

```bash
export seed=42;

python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke mode=train samples=256 seed=$seed pdeconfig.init_args.sample_rate=4 \
    dirname=/mnt/data/navierstokes;

python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke mode=valid samples=32 seed=$seed pdeconfig.init_args.sample_rate=4 \
    dirname=/mnt/data/navierstokes;

python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke mode=test samples=32 seed=$seed pdeconfig.init_args.sample_rate=4 \
    dirname=/mnt/data/navierstokes;
```

### Conditioned

```bash
export seed=42;

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

### Convert to [`zarr`](https://zarr.dev/)
We found that data loading was a lot more performant with `zarr` format rather than original [`NetCDF`](https://www.unidata.ucar.edu/software/netcdf/) format, especially with cloud storage. You can convert after data generation via:

```bash
for mode in train valid test; do
    python scripts/convertnc2zarr.py "/mnt/data/shallowwater/$mode";
done
```

### Data normalization

```bash
python scripts/compute_normalization.py \
    --dataset shallowwater /mnt/data/shallowwater
```

## Maxwell 3D

```bash
export seed=42

python scripts/generate_data.py base=pdedatagen/configs/maxwell3d.yaml \
    experiment=maxwell mode=train samples=256 seed=$seed dirname=/mnt/data/maxwell3d;

python scripts/generate_data.py base=pdedatagen/configs/maxwell3d.yaml \
    experiment=maxwell mode=valid samples=32 seed=$seed dirname=/mnt/data/maxwell3d;

python scripts/generate_data.py base=pdedatagen/configs/maxwell3d.yaml \
    experiment=maxwell mode=test samples=32 seed=$seed dirname=/mnt/data/maxwell3d;
```

### Data normalization

```bash
python scripts/compute_normalization.py \
    --dataset maxwell /mnt/data/maxwell3d
```

## PDEBench

### Generating

Follow [PDEBench's instructions](https://github.com/pdebench/PDEBench#data-generation).

### Resharding for multi-gpu experiments

Coming soon...

### Data normalization

Coming soon...

## Your PDE

Please submit a [pull request](https://github.com/microsoft/pdearena) to add a data loading pipeline for your PDE dataset.
