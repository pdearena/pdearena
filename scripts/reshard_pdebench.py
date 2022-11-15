import click
import h5py
import numpy as np

@click.command()
@click.argument("dirpath", type=click.Path(exists=True))
def extract(dirpath, split="train"):
    with h5py.File(filepath, 'r') as datafile:
        keys = list(datafile.keys())
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    extract()