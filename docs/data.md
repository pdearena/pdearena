# Data Generation

!!! tip ""

    Make sure that the number of shards is divisible by the number of GPUs. 8 is usually a safe number.

## Navier Stokes 2D

```bash
export seed=42;

python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \ 
    experiment=smoke mode=train samples=256 seed=$seed \  
    dirname=/mnt/data/navierstokes;

python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \ 
    experiment=smoke mode=valid samples=32 seed=$seed \ 
    dirname=/mnt/data/navierstokes;

python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \         
    experiment=smoke mode=test samples=32 seed=$seed \
    dirname=/mnt/data/navierstokes;
```

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
