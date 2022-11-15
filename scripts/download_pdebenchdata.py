# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import click
from easyDataverse import Dataset
from pyDataverse.api import NativeApi


@click.command()
@click.option("--dataset_id", type=str, default="doi:10.18419/darus-2986")
@click.option("--dataverse_url", type=str, default="https://darus.uni-stuttgart.de")
@click.option("--outdir", type=str)
@click.option(
    "--limit",
    type=str,
    default=None,
    help="Limit the number of files to download to only those that contain this string",
)
def main(dataset_id, dataverse_url, outdir, limit):
    dataset = Dataset()
    dataset.p_id = dataset_id
    api = NativeApi(dataverse_url)
    dv_dataset = api.get_dataset(dataset_id)
    files_ls = dv_dataset.json()["data"]["latestVersion"]["files"]
    if limit is not None:
        files = []
        for file in files_ls:
            if limit in file["dataFile"]["filename"]:
                files.append(file["dataFile"]["filename"])
    else:
        files = [file["dataFile"]["filename"] for file in files_ls]

    dataset = Dataset.from_dataverse_doi(
        doi=dataset_id,
        dataverse_url=dataverse_url,
        filenames=files,
        filedir=outdir,
    )


if __name__ == "__main__":
    main()
