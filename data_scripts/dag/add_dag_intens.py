""" Add dag intensities

Given a set of predicted dags, add intensities to them from the gold standard

"""
import json
import argparse
import copy
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
import re

import ms_pred.magma.fragmentation as fragmentation
import ms_pred.common as common


# Per-process handle caches so parallel workers reuse open HDF5 files instead
# of reopening them for every entry.
_PRED_DB_CACHE: dict = {}
_TRUE_H5_CACHE: dict = {}
_PRED_H5_CACHE: dict = {}


def _get_pred_db(path) -> common.PredSpecDB:
    path = str(path)
    db = _PRED_DB_CACHE.get(path)
    if db is None:
        db = common.PredSpecDB(path)
        _PRED_DB_CACHE[path] = db
    return db


def _get_true_h5(path) -> common.HDF5Dataset:
    path = str(path)
    h5 = _TRUE_H5_CACHE.get(path)
    if h5 is None:
        h5 = common.HDF5Dataset(path)
        _TRUE_H5_CACHE[path] = h5
    return h5


def _get_pred_h5(path) -> common.HDF5Dataset:
    path = str(path)
    h5 = _PRED_H5_CACHE.get(path)
    if h5 is None:
        h5 = common.HDF5Dataset(path)
        _PRED_H5_CACHE[path] = h5
    return h5


def _parse_legacy_magma_name(name: str):
    """Parse legacy MAGMA tree keys like `spec_collision 30 eV.json`."""
    match = re.match(
        r"^(.*?)_collision\s+([0-9]+\.?[0-9]*|nan)(?:\s*eV)?(?:\.json)?$",
        name,
    )
    if match is None:
        return None
    spec_id = match.group(1)
    ce_raw = match.group(2)
    ce_label = "nan" if ce_raw == "nan" else f"{float(ce_raw):.0f}"
    return spec_id, ce_label


def _is_legacy_magma_h5(path: Path) -> bool:
    """Detect legacy MAGMA HDF5 where top-level entries are JSON datasets."""
    pred_h5 = common.HDF5Dataset(path)
    try:
        for name in pred_h5.get_all_names():
            if name == "__predspec_manifest__":
                continue
            return _parse_legacy_magma_name(name) is not None and not hasattr(pred_h5[name], "keys")
        return False
    finally:
        pred_h5.close()


def get_args():
    """get_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--pred-dag-path", action="store")
    parser.add_argument("--true-dag-path", action="store")
    parser.add_argument("--out-dag-path", action="store")
    parser.add_argument(
        "--add-raw",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--magma-output",
        action="store_true",
        default=False,
        help="If set, treat pred-dag-path as a MAGMA output PredSpecDB and add "
             "gold intensities to it, writing JSON trees consumable by GLACIER."
    )
    return parser.parse_args()


def _extract_raw_spec(true_dag_h5: common.HDF5Dataset, true_dag_name: str):
    """Pull the gold (mass, intensity) peak list from a true DAG entry."""
    if true_dag_name not in true_dag_h5:
        return None
    true_dag = json.loads(true_dag_h5.read_str(true_dag_name))
    true_tbl = true_dag.get("output_tbl")
    if not true_tbl or "mono_mass" not in true_tbl or "rel_inten" not in true_tbl:
        return None
    return [list(pair) for pair in zip(true_tbl["mono_mass"], true_tbl["rel_inten"])]


def relabel_tree(
    pred_dag_db: Path | common.PredSpecDB,
    true_dag_h5: Path,
    pred_dag_name: str,
    true_dag_name: str,
    out_dag_name: str,
    collision_energy: str,
    remark=None,
    magma_output: bool = False,
    legacy_magma_json: bool = False,
) -> Tuple[str, object]:
    """relabel_tree.

    Attach the gold spectrum (``raw_spec``) from the true DAG to a predicted /
    MAGMA-annotated DAG.

    When ``magma_output`` is set, ``pred_dag_db`` points at a MAGMA ``PredSpecDB``
    (binary ``MassSpec`` arrays). We read the ``MassSpec``, expand its integer
    fragments, and emit a JSON tree with the keys GLACIER's ``featurize_tree``
    expects (``root_canonical_smiles``, ``adduct``, ``collision_energy``,
    ``frags`` as a dict of ``{"frag": int}``, and ``raw_spec``).
    """
    true_h5 = _get_true_h5(true_dag_h5)
    raw_spec = _extract_raw_spec(true_h5, true_dag_name)
    if raw_spec is None:
        return None

    if magma_output:
        if legacy_magma_json:
            pred_h5 = _get_pred_h5(pred_dag_db)
            if pred_dag_name not in pred_h5:
                return None
            pred_dag = json.loads(pred_h5.read_str(pred_dag_name))
            if not isinstance(pred_dag, dict):
                return None
            pred_dag["raw_spec"] = raw_spec
            if collision_energy is not None:
                pred_dag["collision_energy"] = float(collision_energy)
            return out_dag_name, json.dumps(pred_dag, indent=2)

        pred_db = _get_pred_db(pred_dag_db)
        spec = pred_db.read(pred_dag_name, collision_energy, remark)
        if spec.root_canonical_smiles is None or not spec.has_frags:
            return None
        int_frags = spec.int_frags or []
        frags = {str(i): {"frag": int(frag)} for i, frag in enumerate(int_frags)}
        out_dict = {
            "root_canonical_smiles": spec.root_canonical_smiles,
            "adduct": spec.adduct,
            "collision_energy": float(spec.collision_energy),
            "frags": frags,
            "raw_spec": raw_spec,
        }
        return out_dag_name, json.dumps(out_dict, indent=2)
    else:
        pred_db = pred_dag_db if isinstance(pred_dag_db, common.PredSpecDB) else _get_pred_db(pred_dag_db)
        pred_dag = pred_db.read(pred_dag_name, collision_energy)
        assert pred_dag.root_canonical_smiles is not None
        assert pred_dag.frags is not None
        assert pred_dag.collision_energy is not None
        assert pred_dag.adduct is not None
        pred_dag.meta["raw_spec"] = raw_spec
        return out_dag_name, pred_dag


def main():
    """main."""
    args = get_args()
    pred_dag_path = Path(args.pred_dag_path)
    true_dag_path = Path(args.true_dag_path)
    out_dag_path = Path(args.out_dag_path)
    add_raw = args.add_raw

    out_dag_path.parent.mkdir(exist_ok=True)

    if args.magma_output:
        # Treat pred_dag_path as a MAGMA output PredSpecDB. Each spec is stored
        # once per collision energy, while the true DAGs are keyed by
        # "{spec}_collision {ce}". Expand the PredSpecDB per collision energy and
        # match each entry to its gold spectrum.
        true_dag_h5 = common.HDF5Dataset(true_dag_path)
        true_lookup = {}
        for true_name in true_dag_h5.get_all_names():
            spec_id = common.rm_collision_str(true_name)
            ce_label = common.get_collision_energy(true_name)
            true_lookup[(spec_id, ce_label)] = true_name
        true_dag_h5.close()

        arg_dicts = []
        is_legacy_magma = _is_legacy_magma_h5(pred_dag_path)
        if is_legacy_magma:
            pred_h5 = common.HDF5Dataset(pred_dag_path)
            for pred_name in tqdm(pred_h5.get_all_names()):
                parsed = _parse_legacy_magma_name(pred_name)
                if parsed is None:
                    continue
                spec_id, ce_label = parsed
                true_name = true_lookup.get((spec_id, ce_label))
                if true_name is None:
                    continue
                arg_dicts.append(
                    {
                        "pred_dag_db": pred_dag_path,
                        "true_dag_h5": true_dag_path,
                        "pred_dag_name": pred_name,
                        "true_dag_name": true_name,
                        "out_dag_name": f"{spec_id}_collision {ce_label}",
                        "collision_energy": ce_label,
                        "remark": None,
                        "magma_output": True,
                        "legacy_magma_json": True,
                    }
                )
            pred_h5.close()
        else:
            pred_db = common.PredSpecDB(pred_dag_path, h5_persistent=True)
            for pred_name in tqdm(pred_db.get_all_names()):
                spec_id = pred_name[5:] if pred_name.startswith("pred_") else pred_name
                ces, remarks = pred_db.get_entries(pred_name)
                for ce, remark in zip(ces, remarks):
                    ce_label = f"{float(ce):.0f}"
                    true_name = true_lookup.get((spec_id, ce_label))
                    if true_name is None:
                        continue
                    arg_dicts.append(
                        {
                            "pred_dag_db": pred_dag_path,
                            "true_dag_h5": true_dag_path,
                            "pred_dag_name": pred_name,
                            "true_dag_name": true_name,
                            "out_dag_name": f"{spec_id}_collision {ce_label}",
                            "collision_energy": ce,
                            "remark": remark,
                            "magma_output": True,
                            "legacy_magma_json": False,
                        }
                    )
            pred_db.close()
    else:
        pred_dag_h5 = common.HDF5Dataset(pred_dag_path)
        pred_dag_name_set = set(pred_dag_h5.get_all_names())
        # Do not close pred_dag_h5 here
        pred_dag_names, true_dag_names, out_dag_names, colli_engs = [], [], [], []
        true_dag_h5 = common.HDF5Dataset(true_dag_path)
        for true_dag_n in tqdm(true_dag_h5.get_all_names()):
            spec_id = common.rm_collision_str(true_dag_n)
            colli_eng = common.get_collision_energy(true_dag_n)
            pred_dag_name = 'pred_' + spec_id
            if pred_dag_name not in pred_dag_name_set:
                continue
            pred_dag_names.append(pred_dag_name)
            true_dag_names.append(true_dag_n)
            out_dag_names.append(spec_id)
            colli_engs.append(colli_eng)
                
        true_dag_h5.close()
        arg_dicts = [
            {
                "pred_dag_db": pred_dag_path,
                "true_dag_h5": true_dag_path,
                "pred_dag_name": i,
                "true_dag_name": j,
                "out_dag_name": k,
                "collision_energy": l,
                "magma_output": False,
            }
            for i, j, k, l in zip(pred_dag_names, true_dag_names, out_dag_names, colli_engs)
        ]
        pred_dag_h5.close()
    
    def write_func(outs):
        out_db = common.PredSpecDB(out_dag_path, mode='w')
        for out in outs:
            if out is None:
                continue
            out_db.write(*out)
        out_db.close()

    if len(arg_dicts) == 0:
        print(
            f"No matching entries found for pred DAGs ({pred_dag_path}) and true DAGs ({true_dag_path}). "
            f"Creating empty output at {out_dag_path}."
        )
        if args.magma_output:
            out_h5 = common.HDF5Dataset(out_dag_path, mode='w')
            out_h5.close()
        else:
            out_db = common.PredSpecDB(out_dag_path, mode='w')
            out_db.close()
        print("success!")
        return

    # Run
    wrapper_fn = lambda arg_dict: relabel_tree(**arg_dict)
    num_workers = args.num_workers
    if not args.magma_output:
        if num_workers == 0:
            outs = [wrapper_fn(i) for i in arg_dicts]
            write_func(outs)
        else:
            common.chunked_parallel(arg_dicts, wrapper_fn, output_func=write_func, max_cpu=num_workers, chunks=1000)
        print("success!")
    else:
        if args.num_workers == 0:
            outs = [wrapper_fn(i) for i in arg_dicts]
        else:
            outs = common.chunked_parallel(arg_dicts, wrapper_fn, max_cpu=args.num_workers, chunks=1000)
        # Write output to HDF5 file
        out_h5 = common.HDF5Dataset(out_dag_path, mode='w')
        out_h5.write_list_of_tuples(outs)
        out_h5.close()
        print("success!")

if __name__ == "__main__":
    main()
