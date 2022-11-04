import json
import os.path

from pytorch_lightning.utilities.model_summary.model_summary import (
    get_human_readable_count,
)

DOCS_DIR = os.path.dirname(__file__)

MODELNAMELUT = {
    "ResNet128": "ResNet128",
    "ResNet256": "ResNet256",
    "DilatedResNet-128": "DilatedResNet128",
    "DilatedResNet-128-norm": "DilatedResNet128-norm",
    "FNO-8m": r"FNO128-8~modes8~",
    "FNO-16m": r"FNO128-8~modes16~",
    "FNOs-64-32m": r"FNO64-4~modes32~",
    "FNOs-96-32m": r"FNO96-4~modes32~",
    "FNOs-128-16m": r"FNO128-4~modes16~",
    "FNOs-128-32m": r"FNO128-4~modes32~",
    "UNO64": "UNO64",
    "UNO128": "UNO128",
    "Unet2015-64": r"U-Net~2015~64",
    "Unet2015-128": r"U-Net~2015~128",
    "Unet2015-64-tanh": r"U-Net~2015~64-tanh",
    "Unetbase64": r"U-Net~base~64",
    "Unetbase128": r"U-Net~base~128",
    "Unetmod64": r"U-Net~mod~64",
    "Unetmodattn64": r"U-Net~att~64",
    "U-FNet1-8m": r"U-F1Net~modes8~",
    "U-FNet1-16m": r"U-F1Net~modes16~",
    "U-FNet2-8m": r"U-F2Net~modes8,4~",
    "U-FNet2-16m": r"U-F2Net~modes16,8~",
    "U-FNet2-8mc": r"U-F2Net~modes8,8~",
    "U-FNet2-16mc": r"U-F2Net~modes16,16~",
    "U-FNet1attn": r"U-F1Net~att,modes16~",
    "U-FNet2attn": r"U-F1Net~att,modes16,8~",
}

header = """
# Model Zoo

Here we document the performance of various models currently implemented with [**PDEArena**](https://microsoft.github.io/pdearena).
The tables below provide results and useful statistics about training and inference.

"""


def get_data_from_json(file):
    if os.path.exists(file):
        with open(file) as f:
            data = json.load(f)
        return data
    else:
        return {}


def convert_model_name(name):
    return MODELNAMELUT.get(name, name)


def get_model_zoo_table_row(name, num_params, model_size, peak_gpu_usage, fwd_time, fwd_bwd_time):
    """make modelzoo.md table row."""
    return f"| {name} | {get_human_readable_count(num_params)} | {model_size:.1f} | {peak_gpu_usage} | {fwd_time:.3f} | {fwd_bwd_time:.3f} |"


def main(outfile):
    """make modelzoo.md table."""
    fwd_time_data = None
    fwd_bwd_time_data = None
    fwd_time_data = get_data_from_json(os.path.join(DOCS_DIR, "models_fwd_time.json"))
    fwd_bwd_time_data = get_data_from_json(os.path.join(DOCS_DIR, "models_fwd_bwd_time.json"))

    date_created = fwd_time_data.pop("date-created")
    gpu = fwd_time_data.pop("gpu-name")
    models = fwd_time_data.keys() & fwd_bwd_time_data.keys()

    with open(outfile, "w") as f:
        f.write(header)
        f.write("\n\n")
        f.write(
            f"| Model | Num. Params | Model Size (MB) | Peak GPU Memory (MB) | Forward Time | Forward+Backward Time |"
        )
        f.write("\n")
        f.write("| --- | ---: | ---: | ---: | --- | --- |")
        f.write("\n")
        for model in sorted(models):
            row = get_model_zoo_table_row(
                convert_model_name(model),
                fwd_time_data[model]["num_params"],
                fwd_time_data[model]["model_size"],
                fwd_bwd_time_data[model]["peak_gpu_memory"],
                fwd_time_data[model]["fwd_time"],
                fwd_bwd_time_data[model]["fwd_bwd_time"],
            )
            f.write(row + "\n")

        f.write(f"\n- **Date Created**: {date_created}")
        f.write(f"\n- **GPU**: {gpu}")


if __name__ == "__main__":
    main("modelzoo.md")
