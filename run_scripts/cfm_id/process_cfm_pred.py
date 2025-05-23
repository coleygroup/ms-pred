""" Make predictions with binned and eval """
import ast

import yaml
import pandas as pd
from pathlib import Path
import subprocess
import json
from ms_pred import common
import numpy as np


def extract_cfm_file(spectra_file, collision_energies, max_node):
    """extract_cfm_file.

    Args:
        spectra_file:
        collision_energies:
        max_node:
    """
    input_name = spectra_file.stem
    meta, cfm_parsed = common.parse_cfm_out(spectra_file, max_merge=False)

    return_list = []
    # iter over three collision energies
    for ce_val, ce_key in zip(collision_energies, ['energy0', 'energy1', 'energy2']):
        cfm_parsed_ce = cfm_parsed[cfm_parsed['energy'] == ce_key]
        assert len(cfm_parsed) > 0
        cfm_parsed_max_form = cfm_parsed_ce.groupby("form_no_h").max().reset_index()
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
            "mono_mass": list_wrap(
                cfm_parsed_max_form["formula_mass"]
            ),
        }
        json_out = {
            "smiles": meta["SMILES"],
            "spec_name": input_name,
            "output_tbl": output_tbl,
            "cand_form": meta["Formula"],
        }
        return_list.append((
            f"{input_name}_collision {ce_val}.json",
            json.dumps(json_out, indent=2)
        ))

    return return_list


def write_to_h5(out_entries, out_dir):
    h5 = common.HDF5Dataset(out_dir, mode='w')
    for entries in out_entries:
        for name, json_str in entries:
            h5.write_str(name, json_str)
    h5.close()


datasets = ["nist20"]
max_node = 100
split_override = None
datasets = ["casmi22"]
splits = ["split_1", "scaffold_1"]
splits = ["all_split"]

if __name__ == '__main__':
    for dataset in datasets:
        res_folder = Path(f"results/cfm_id_{dataset}/")
        cfm_output_specs = res_folder / "cfm_out"
        all_files = list(cfm_output_specs.glob("*.log"))

        # Create full spec
        pred_dir_folders = []
        for split in splits:
            split_file = f"data/spec_datasets/{dataset}/splits/{split}.tsv"
            label_file = f"data/spec_datasets/{dataset}/labels.tsv"
            if not Path(split_file).exists() or not Path(label_file).exists():
                print(f"Skipping {split} for {dataset} due to file not found")
                continue

            split_df = pd.read_csv(split_file, sep="\t")
            label_df = pd.read_csv(label_file, sep='\t')
            name_to_collision = dict(label_df[["spec", "collision_energies"]].values)
            save_dir = res_folder / f"{split}"
            save_dir.mkdir(exist_ok=True)

            pred_dir = save_dir / "preds/"
            pred_dir.mkdir(exist_ok=True)

            export_h5 = pred_dir / "form_preds.hdf5"
            test_specs = set(split_df[split_df["Fold_0"] == "test"]["spec"].values)

            names_to_export = [i for i in all_files if i.stem in test_specs]
            name_and_ces = []
            for name in names_to_export:
                ces = [float(i) for i in ast.literal_eval(name_to_collision[name.stem])]
                ces = sorted(ces)
                low_mid_hi_ces = (ces[0], ces[len(ces) // 2], ces[-1])

                name_and_ces.append((name, [f'{i:.0f}' for i in np.unique(low_mid_hi_ces)]))

            export_fn = lambda x: extract_cfm_file(x[0], x[1], max_node=max_node)
            write_fn = lambda x: write_to_h5(x, out_dir=export_h5)
            common.chunked_parallel(name_and_ces, export_fn, output_func=write_fn)
            pred_dir_folders.append(export_h5)

            # Convert all preds to binned

            # Convert to binned files
            out_binned = pred_dir / "binned_preds.hdf5"
            cmd = f"""python data_scripts/forms/02_form_to_binned.py \\
            --max-peaks 1000 \\
            --num-bins 15000 \\
            --upper-limit 1500 \\
            --form-folder {export_h5} \\
            --num-workers 16 \\
            --out {out_binned} """
            cmd = f"{cmd}"
            print(cmd + "\n")
            subprocess.run(cmd, shell=True)

            # Eval binned preds
            eval_cmd = f"""python analysis/spec_pred_eval.py \\
            --binned-pred-file {out_binned} \\
            --max-peaks 100 \\
            --min-inten 0 \\
            --formula-dir-name no_subform.hdf5 \\
            --dataset {dataset}  \\
            """
            print(eval_cmd)
            subprocess.run(eval_cmd, shell=True)
