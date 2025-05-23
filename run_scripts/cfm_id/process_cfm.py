""" Convert CFM predictions into a coverage metric """
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import json
from ms_pred import common


def extract_cfm_file(spectra_file, out_dir, max_node):
    """extract_cfm_file.

    Args:
        spectra_file:
        out_dir:
        max_node:
    """
    input_name = spectra_file.stem
    meta, cfm_parsed_max = common.parse_cfm_out(spectra_file, max_merge=True)
    cfm_parsed_max_form = cfm_parsed_max.groupby("form_no_h").max().reset_index()
    cfm_parsed_max_form["inten"] /= cfm_parsed_max_form["inten"].max()
    cfm_parsed_max_form = cfm_parsed_max_form.sort_values(
        "inten", ascending=False
    ).reset_index(drop=True)
    cfm_parsed_max_form = cfm_parsed_max_form[:max_node]

    list_wrap = lambda x: x.values.tolist()
    output_tbl = {
        "ms2_inten": list_wrap(cfm_parsed_max_form["inten"]),
        "rel_inten": list_wrap(cfm_parsed_max_form["inten"]),
        "log_prob": None,
        "formula": list_wrap(cfm_parsed_max_form["form_no_h"]),
        "formula_mass_no_adduct": list_wrap(
            cfm_parsed_max_form["formula_mass_no_adduct"]
        ),
    }

    json_out = {
        "smiles": meta["SMILES"],
        "spec_name": input_name,
        "output_tbl": output_tbl,
        "cand_form": meta["Formula"],
    }

    out_file = out_dir / f"{input_name}.json"
    with open(out_file, "w") as fp:
        json.dump(json_out, fp, indent=2)


datasets = ["canopus_train_public", "nist20"]
max_nodes = [10, 20, 30, 40, 50, 100, 200]
subform_name = "magma_subform_50"
split_override = None
splits = ["split_1.tsv", "scaffold_1.tsv"]


for dataset in datasets:
    for split in splits:
        res_folder = Path(f"results/cfm_id_{dataset}/")
        cfm_output_specs = res_folder / "cfm_out"
        all_files = list(cfm_output_specs.glob("*.log"))
        split_file = f"data/spec_datasets/{dataset}/splits/{split}"
        if not Path(split_file).exists():
            print(f"Skipping {split_file} for dataset {dataset}")
            continue

        save_dir = res_folder / "inten_thresh_sweep"
        save_dir.mkdir(exist_ok=True)

        # Create full spec
        print(f"Saving inten sweep to: {save_dir}")

        pred_dir_folders = []
        for max_node in max_nodes:
            save_dir_temp = save_dir / str(max_node)
            save_dir_temp.mkdir(exist_ok=True)
            export_dir = save_dir_temp / "form_preds"
            export_dir.mkdir(exist_ok=True)

            split = pd.read_csv(split_file, sep="\t")
            test_specs = set(split[split["Fold_0"] == "test"]["spec"].values)

            to_export = [i for i in all_files if i.stem in test_specs]
            export_fn = lambda x: extract_cfm_file(x, export_dir, max_node=max_node)
            # [export_fn(i) for i in to_export]
            common.chunked_parallel(to_export, export_fn)
            pred_dir_folders.append(export_dir)

        res_files = []
        for pred_dir in pred_dir_folders:
            analysis_cmd = f"""python analysis/form_pred_eval.py \\
                --dataset {dataset} \\
                --tree-pred-folder {pred_dir} \\
                --subform-name {subform_name}
            """
            res_files.append(pred_dir.parent / "pred_eval.yaml")
            print(analysis_cmd + "\n")
            subprocess.run(analysis_cmd, shell=True)

        # Run cleanup now
        new_entries = []
        for res_file in res_files:
            new_data = yaml.safe_load(open(res_file, "r"))
            thresh = res_file.parent.stem
            new_entry = {"nm_nodes": thresh}
            new_entry.update(
                {k: v for k, v in new_data.items() if "avg" in k or "sem" in k or "std" in k}
            )
            new_entries.append(new_entry)

        df = pd.DataFrame(new_entries)
        df.to_csv(save_dir / "summary.tsv", sep="\t", index=None)
