# Downloading data from Azure

Coming soon...

# PDEBench

[`PDEBench`](https://github.com/pdebench/PDEBench) data can be downloaded with the included [`download_pdebenchdata.py`]() script:

For example to download the Incompressible Navier Stokes dataset:
```bash
export DATAVERSE_URL=https://darus.uni-stuttgart.de;
python scripts/download_pdebenchdata.py \
    --outdir /mnt/data/PDEBench/ --limit ns_incom
```



