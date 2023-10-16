# Data Generation

!!! tip

    For multi-gpu training, make sure that the number of shards is divisible by the number of GPUs. 8 is usually a safe number.


## Navier Stokes 2D

### Standard

```bash
# This script loops over different seeds and produces 5.2k training, 1.3k valid,
# and 1.3k test trajectories of the Navier-Stokes dataset.
./pdedatagen/scripts/navierstokes_jobs.sh
```

```bash
#!/bin/bash
seeds=(197910 452145 540788 61649 337293 296323 471551 238734 \
795028 806339 144207 415274 950547 391920 891493 645944 \
431652 391355 600690 495919 874847 97373 403121 588319 \
991570 597761 453345 349940 666497 597535 61891 975433 \
856942 788627 234361 433043 153164 126188 946091 795833 \
901348 142003 515976 509478 857366 766053 792079 585498 \
772145 954313 429673 445536)
for SEED in ${seeds[*]};
do
    python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke mode=train samples=100 seed=$SEED pdeconfig.init_args.sample_rate=4 \
    dirname=pdearena_data/navierstokes/
    python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke mode=valid samples=25 seed=$SEED pdeconfig.init_args.sample_rate=4 \
    dirname=pdearena_data/navierstokes/
    python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke mode=test samples=25 seed=$SEED pdeconfig.init_args.sample_rate=4 \
    dirname=pdearena_data/navierstokes/
done
```

### Conditioned

```bash
# This script loops over different seeds and produces 6656 training, 1664 valid,
# and 1664 test trajectories of the conditioned Navier-Stokes dataset.
./pdedatagen/scripts/navierstokes_cond_jobs.sh
```

```bash
#!/bin/bash
seeds=(197910 452145 540788 61649 337293 296323 471551 238734 \
795028 806339 144207 415274 950547 391920 891493 645944 \
431652 391355 600690 495919 874847 97373 403121 588319 \
991570 597761 453345 349940 666497 597535 61891 975433 \
856942 788627 234361 433043 153164 126188 946091 795833 \
901348 142003 515976 509478 857366 766053 792079 585498 \
772145 954313 429673 445536)
for SEED in ${seeds[*]};
do
    python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke_cond mode=train samples=128 seed=$SEED \
    dirname=pdearena_data/navierstokes_cond
    python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke_cond mode=valid samples=32 seed=$SEED \
    dirname=pdearena_data/navierstokes_cond
    python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke_cond mode=test samples=32 seed=$SEED \
    dirname=pdearena_data/navierstokes_cond
done
```

### Data normalization

The Navier-Stokes data is reasonably bounded that we didn't need any normalization.

## Shallow water 2D

```bash
# This script loops over different seeds and produces 5.6k training, 1.4k valid,
# and 1.4k test trajectories of the Shallow water dataset.
./pdedatagen/scripts/navierstokes_cond_jobs.sh
```

```bash
#!/bin/bash
# This script produces 5.6k training, 1.4k valid, and 1.4k test trajectories of the Shallow water dataset.
seeds=(197910 452145 540788 61649 337293 296323 471551 238734 \
795028 806339 144207 415274 950547 391920 891493 645944 \
431652 391355 600690 495919 874847 97373 403121 588319 \
991570 597761 453345 349940 666497 597535 61891 975433 \
856942 788627 234361 433043 153164 126188 946091 795833 \
901348 142003 515976 509478 857366 766053 792079 585498 \
772145 954313 429673 445536 799432 146142 19024 438811)
for SEED in ${seeds[*]};
do
    python scripts/generate_data.py base=pdedatagen/configs/shallowwater.yaml \
    experiment=shallowwater mode=train samples=100 seed=$SEED \
    dirname=pdearena_data/shallowwater;
    python scripts/generate_data.py base=pdedatagen/configs/shallowwater.yaml \
    experiment=shallowwater mode=valid samples=25 seed=$SEED \
    dirname=pdearena_data/shallowwater;
    python scripts/generate_data.py base=pdedatagen/configs/shallowwater.yaml \
    experiment=shallowwater mode=test samples=25 seed=$SEED \
    dirname=pdearena_data/shallowwater;
done;
```

### Convert to [`zarr`](https://zarr.dev/)
We found that data loading was a lot more performant with `zarr` format rather than original [`NetCDF`](https://www.unidata.ucar.edu/software/netcdf/) format, especially with cloud storage. You can convert after data generation via:

```bash
for mode in train valid test; do
    python scripts/convertnc2zarr.py "pdearena_data/shallowwater/$mode";
done
```

### Data normalization

```bash
python scripts/compute_normalization.py \
    --dataset shallowwater pdearena_data/shallowwater
```

## Maxwell 3D

```bash
# This script loops over different seeds and produces 6.4k training,
# 1.6k valid, and 1.6k test trajectories of the Maxwell dataset.
```

```bash
#!/bin/bash
seeds=(197910 452145 540788 61649 337293 296323 471551 238734 \
795028 806339 144207 415274 950547 391920 891493 645944 \
431652 391355 600690 495919 874847 97373 403121 588319 \
991570 597761 453345 349940 666497 597535 61891 975433 \
856942 788627 234361 433043 153164 126188 946091 795833 \
901348 142003 515976 509478 857366 766053 792079 585498 \
772145 954313 429673 445536 799432 146142 19024 438811 \
190539 506225 943948 304836 854174 354248 373230 697045)
for SEED in ${seeds[*]};
do
    python scripts/generate_data.py base=pdedatagen/configs/maxwell3d.yaml \
    experiment=maxwell mode=train samples=100 seed=$SEED dirname=pdearena_data/maxwell3d/
    python scripts/generate_data.py base=pdedatagen/configs/maxwell3d.yaml \
    experiment=maxwell mode=valid samples=25 seed=$SEED dirname=pdearena_data/maxwell3d/
    python scripts/generate_data.py base=pdedatagen/configs/maxwell3d.yaml \
    experiment=maxwell mode=test samples=25 seed=$SEED dirname=pdearena_data/maxwell3d/
done
```

### Data normalization

```bash
python scripts/compute_normalization.py \
    --dataset maxwell pdearena_data/maxwell3d
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
