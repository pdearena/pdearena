# Training PDE surrogates

Thanks to [PyTorch Lightning](https://github.com/Lightning-AI/lightning), whether it's a single GPU experiment or multiple GPUs (even multiple nodes), setting up scalable training experiments should be fairly simple.

!!! tip

    We recommend a warmup learning rate schedule for distributed training.


## Standard PDE Surrogate Learning

```bash
python scripts/train.py -c <path/to/config>
```

For example, to run modern U-Net on Navier Stokes dataset on 4 GPUs use:

```bash
python scripts/train.py -c configs/navierstokes2d.yaml \
    --data.data_dir=/mnt/data/NavierStokes2D_smoke \
    --trainer.strategy=ddp --trainer.devices=4 \
    --trainer.max_epochs=50 \
    --data.batch_size=8 \
    --data.time_gap=0 --data.time_history=4 --data.time_future=1 \
    --model.name=Unetmod-64 \
    --model.lr=2e-4 \
    --optimizer=AdamW --optimizer.lr=2e-4 --optimizer.weight_decay=1e-5 \
    --lr_scheduler=LinearWarmupCosineAnnealingLR \
    --lr_scheduler.warmup_epochs=5 \
    --lr_scheduler.max_epochs=50 --lr_scheduler.eta_min=1e-7
```

## Conditioned PDE Surrogate Learning

```bash
python scripts/cond_train.py -c <path/to/config>
```

For example, to run modern U-Net on Navier Stokes dataset on 4 GPUs use:

```bash
python scripts/cond_train.py -c configs/cond_navierstokes2d.yaml \
    --data.data_dir=/mnt/data/NavierStokes2D_cond_smoke_v1 \
    --trainer.strategy=ddp --trainer.devices=4 \
    --trainer.max_epochs=50 \
    --data.batch_size=8 \
    --model.name=Unetmod-64 \
    --model.lr=2e-4 \
    --optimizer=AdamW --optimizer.lr=2e-4 --optimizer.weight_decay=1e-5 \
    --lr_scheduler=LinearWarmupCosineAnnealingLR \
    --lr_scheduler.warmup_epochs=5 \
    --lr_scheduler.max_epochs=50 --lr_scheduler.eta_min=1e-7
```

## Dataloading philosophy

- Use modern [`torchdata`](https://pytorch.org/data/) [iterable datapipes](https://pytorch.org/data/beta/torchdata.datapipes.iter.html#torchdata.datapipes.iter.IterDataPipe) as they scale better with cloud storage.
- Use equally sized shards for simpler scaling with PyTorch DDP.
