""" Make splits

Make train-test-val splits by compound uniqueness.

Existing split preservation uses only 2D InChIKey matching.
"""
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Optional, Set, Tuple
import warnings

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ms_pred import common


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--data-dir", help="Data directory")
    args.add_argument(
        "--label-file",
        default="data/spec_datasets/gnps2015_debug/labels.tsv",
        help="Path to label file.",
    )
    args.add_argument(
        "--ionization",
        type=str,
        default=None,
        help="Ion that the user want to focused on (if applicable)",
    )
    args.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Percentage of validation data out of all training data.",
    )
    args.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Percentage of validation data out of all training data.",
    )
    args.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Percentage of test data out of all data.",
    )
    args.add_argument(
        "--seed", type=int, default=22, help="Random seed be reproducible"
    )
    args.add_argument("--split-name", default=None, help="split name prefix")
    args.add_argument(
        "--existing-split-file",
        default=None,
        help="Existing split file whose assignments should be preserved.",
    )
    args.add_argument(
        "--existing-label-file",
        default=None,
        help="Optional labels file corresponding to the existing split file.",
    )
    args.add_argument(
        "--split-type",
        default="inchikey",
        help="Type of split; normally split on structures",
        choices=["inchikey", "scaffold", "fingerprint"],
    )
    args.add_argument("--greedy-pack", default=False, action="store_true")
    return args.parse_args()


def get_inchikey_2d(inchikey: str) -> str:
    if pd.isna(inchikey):
        return ""
    return str(inchikey).split("-")[0]


def get_label_inchikey_2d(df: pd.DataFrame) -> np.ndarray:
    if "inchikey" in df.columns:
        inchikeys = df["inchikey"].fillna("").astype(str).values
    elif "smiles" in df.columns:
        inchikeys = [common.inchikey_from_smiles(i) for i in df["smiles"].fillna("")]
    else:
        raise ValueError("Expected either an 'inchikey' or 'smiles' column in labels.")
    return np.array([get_inchikey_2d(i) for i in inchikeys])


def get_obj_mapping(df: pd.DataFrame, split_type: str) -> Tuple[Set[str], Dict[str, str]]:
    specs = df["spec"].values
    if split_type == "inchikey":
        objs = get_label_inchikey_2d(df)
    elif split_type == "scaffold":
        smis = df["smiles"].values
        objs = common.chunked_parallel(smis, get_scaffold)
    elif split_type == "fingerprint":
        get_fp = lambda x: common.get_morgan_fp_smi(x, nbits=512)

        smis = df["smiles"].values
        fps = common.chunked_parallel(smis, get_fp)
        fps = np.vstack(fps)

        print("Computing tanimoto")
        intersect = np.einsum("ij, kj -> ik", fps, fps)
        union = fps.sum(-1)[None, :] + fps.sum(-1)[:, None] - intersect
        tani_dist = 1 - intersect / (union + 1e-22)
        print("Done with tanimoto")

        print("Running agglom clustering")
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(
            n_clusters=20,
            linkage="complete",
            affinity="precomputed",
        )
        print("Done agglom clustering")
        objs = model.fit_predict(tani_dist)
    else:
        raise NotImplementedError()

    return set(objs), dict(zip(specs, objs))


def _get_split_columns(split_df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(split_df.columns)
    if len(cols) < 2:
        raise ValueError("Existing split file must contain at least two columns.")
    return cols[0], cols[1]


def _assign_fold(
    key: str,
    fold: str,
    mapping: Dict[str, str],
    collision_type: str,
) -> str:
    priorities = {"train": 0, "val": 1, "test": 2}
    existing = mapping.get(key)
    if existing is not None and existing != fold:
        warnings.warn(
            f"Conflicting fold assignment for {collision_type} '{key}': "
            f"'{existing}' vs '{fold}'. Using higher-priority fold.",
            stacklevel=2,
        )
        fold = existing if priorities[existing] >= priorities[fold] else fold
    mapping[key] = fold
    return fold


def load_existing_assignments(
    existing_split_file: Optional[str], existing_label_file: Optional[str]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    ikey2d_to_fold: Dict[str, str] = {}
    if existing_split_file is None:
        return {}, ikey2d_to_fold

    split_df = pd.read_csv(existing_split_file, sep="\t")
    spec_col, fold_col = _get_split_columns(split_df)
    split_df = split_df[[spec_col, fold_col]].rename(columns={spec_col: "spec"})
    valid_folds = {"train", "val", "test"}
    split_df = split_df[split_df[fold_col].isin(valid_folds)]

    if existing_label_file is None:
        raise ValueError(
            "Existing label file is required when preserving assignments by 2D InChIKey."
        )

    old_labels = pd.read_csv(existing_label_file, sep="\t")
    if "spec" not in old_labels.columns:
        raise ValueError("Existing labels file must contain a 'spec' column.")

    old_labels = old_labels.copy()
    old_labels["inchikey_2d"] = get_label_inchikey_2d(old_labels)
    merged = old_labels.merge(split_df, on="spec", how="inner")
    for spec, ikey_2d, fold in merged[["spec", "inchikey_2d", fold_col]].itertuples(
        index=False
    ):
        if ikey_2d:
            _assign_fold(ikey_2d, fold, ikey2d_to_fold, "2D InChIKey")

    return {}, ikey2d_to_fold


def get_scaffold(smiles):
    """
    Given a SMILES string, returns the scaffold of the molecule using the Murcko framework.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MakeScaffoldGeneric(mol)
        return Chem.MolToInchiKey(scaffold)
    except:
        print(f"Failed on {smiles}, return none")
        return ""


def make_splits(args):
    label_path = Path(args.label_file)
    df = pd.read_csv(label_path, sep="\t")
    split_suffix = args.split_name
    seed = args.seed
    split_type = args.split_type
    greedy_pack = args.greedy_pack

    if split_suffix is None:
        split_suffix = f"split_{seed}.tsv"

    if args.ionization is not None:
        df = df[df["ionization"] == args.ionization].copy()

    obj_set, spec_to_obj = get_obj_mapping(df, split_type)
    spec_to_ikey2d = dict(zip(df["spec"].values, get_label_inchikey_2d(df)))
    _, existing_ikey_folds = load_existing_assignments(
        args.existing_split_file, args.existing_label_file
    )

    preassigned_obj_to_fold = {}
    assigned_specs = 0
    for spec, obj in spec_to_obj.items():
        matched_fold = existing_ikey_folds.get(spec_to_ikey2d.get(spec, ""))
        if matched_fold is None:
            continue

        assigned_specs += 1
        _assign_fold(obj, matched_fold, preassigned_obj_to_fold, "object")

    remaining_obj_set = obj_set.difference(preassigned_obj_to_fold)
    remaining_spec_to_obj = {
        spec: obj for spec, obj in spec_to_obj.items() if obj in remaining_obj_set
    }

    # Compute nums in terms of specs
    train_frac, val_frac, test_frac = args.train_frac, args.val_frac, args.test_frac
    num_train = int(train_frac * len(remaining_spec_to_obj))
    num_val = int(args.val_frac * len(remaining_spec_to_obj))
    num_test = min(
        len(remaining_spec_to_obj) - num_train - num_val,
        int(test_frac * len(remaining_spec_to_obj)),
    )

    if greedy_pack:
        # Pack low items
        # Get counts associated with each
        obj_to_counts = Counter(spec_to_obj.values())

        # Small to large sorted
        sorted_obj_set = sorted(list(remaining_obj_set), key=lambda x: obj_to_counts[x])
        get_num_specs = lambda x: np.sum([obj_to_counts[i] for i in x])

        train = {k for k, v in preassigned_obj_to_fold.items() if v == "train"}
        val = {k for k, v in preassigned_obj_to_fold.items() if v == "val"}
        test = {k for k, v in preassigned_obj_to_fold.items() if v == "test"}
        for obj in sorted_obj_set:
            if get_num_specs(test) < num_test:
                test.add(obj)
            elif get_num_specs(val) < num_val:
                val.add(obj)
            else:
                train.add(obj)
    else:
        train = {k for k, v in preassigned_obj_to_fold.items() if v == "train"}
        val = {k for k, v in preassigned_obj_to_fold.items() if v == "val"}
        test = {k for k, v in preassigned_obj_to_fold.items() if v == "test"}

        # Divide by compounds
        full_obj_list = list(remaining_obj_set)
        np.random.seed(seed)
        np.random.shuffle(full_obj_list)
        num_train_remaining = max(0, int((train_frac + val_frac) * len(remaining_obj_set)))
        train.update(full_obj_list[:num_train_remaining])
        test.update(full_obj_list[num_train_remaining:])

        val_fraction = args.val_frac
        val_num = int(len(full_obj_list[:num_train_remaining]) * val_fraction)
        np.random.seed(seed)
        new_train_candidates = full_obj_list[:num_train_remaining]
        if val_num > 0:
            val.update(np.random.choice(new_train_candidates, val_num, replace=False))

        # Remove val formulae inds from train formulae
        train = train.difference(val)

    fold_num = 0
    fold_name = f"Fold_{fold_num}"
    output_dir = Path(args.data_dir) / "splits"
    if not output_dir.is_dir():
        output_dir.mkdir(exist_ok=True)

    print(f"Num train total formulae: {len(train)}")
    print(f"Num val total formulae: {len(val)}")
    print(f"Num test total formulae: {len(test)}")
    print(f"Matched existing assignments for {assigned_specs} spectra")

    split_data = {"spec": [], fold_name: []}
    for _, row in df.iterrows():
        spec_fn = row["spec"]
        spec_obj = spec_to_obj[spec_fn]

        if spec_obj in train:
            fold = "train"
        elif spec_obj in test:
            fold = "test"
        elif spec_obj in val:
            fold = "val"
        else:
            fold = "exclude"
        split_data["spec"].append(spec_fn)
        split_data[fold_name].append(fold)

    assert len(split_data["spec"]) == df.shape[0]
    assert len(split_data[fold_name]) == df.shape[0]

    export_df = pd.DataFrame(split_data)
    export_df = export_df.sort_values("spec", ascending=True)
    export_df.to_csv(output_dir / split_suffix, sep="\t", index=False)


if __name__ == "__main__":
    args = get_args()
    make_splits(args)
