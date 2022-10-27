# Training PDE surrogates

Thanks to [PyTorch Lightning](https://github.com/Lightning-AI/lightning) setting up scalable training experiments should be fairly simple.

## Standard Operator Learning


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
    --model.name=Unet --model.hidden_channels=64 --model.norm=True \
    --model.lr=2e-4 \
    --optimizer=AdamW --optimizer.lr=2e-4 --optimizer.weight_decay=1e-5 \
    --lr_scheduler=LinearWarmupCosineAnnealingLR \
    --lr_scheduler.warmup_epochs=5 \
    --lr_scheduler.max_epochs=50 --lr_scheduler.eta_min=1e-7
```



## Conditioned Operator Learning

```bash
python scripts/cond_train.py -c <path/to/config>
```


## Dataloading philosophy

- Use modern `torchdata` iterable datapipes as they scale better with cloud storage.
- Use equally sized shards for simpler scaling with PyTorch DDP.