import json
import os.path

from pytorch_lightning.utilities.model_summary.model_summary import get_human_readable_count

DOCS_DIR = os.path.dirname(__file__)

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

def get_model_zoo_table_row(name, num_params, model_size, fwd_time, fwd_bwd_time):
    """make modelzoo.md table row"""
    return f"| {name} | {get_human_readable_count(num_params)} | {model_size:.3f} | {fwd_time:.3f} | {fwd_bwd_time:.3f} |"

def main(outfile):
    """make modelzoo.md table"""
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
        f.write(f"| Model | Num. Params | Model Size (MB) | Forward Time | Forward+Backward Time |")
        f.write("\n")
        f.write("| --- | --- | --- | --- | --- |")
        f.write("\n")
        for model in sorted(models):
            row = get_model_zoo_table_row(model, fwd_time_data[model]["num_params"], fwd_time_data[model]["model_size"], fwd_time_data[model]["fwd_time"], fwd_bwd_time_data[model]["fwd_bwd_time"])
            f.write(row + "\n")

        f.write(f"\n- **Date Created**: {date_created}")
        f.write(f"\n- **GPU**: {gpu}")


if __name__ == "__main__":
    main("modelzoo.md")