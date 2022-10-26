# Training PDE surrogates

PyTorch Lightning

## Standard Operator Learning


```bash
python scripts/train.py -c <path/to/config>
```

## Conditioned Operator Learning

```bash
python scripts/cond_train.py -c <path/to/config>
```


## Dataloading philosophy

- Use modern `torchdata` iterable datapipes as they scale better with cloud storage.