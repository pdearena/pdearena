import click

from pdedatagen.shallowwater import collect_data2zarr


@click.command()
@click.argument("datapath", type=click.Path(exists=True))
def main(datapath):
    collect_data2zarr(datapath)


if __name__ == "__main__":
    main()
