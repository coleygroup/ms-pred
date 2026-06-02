from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


_WORKER_MCES_FUNC = None
_WORKER_MCES_SOLVER = None


def first_ranked_smiles(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("top_1_smiles", "top1_smiles"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value

    for key in ("ranked_smiles", "top_smiles"):
        value = entry.get(key)
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            return first if isinstance(first, str) and first else None
    return None


def load_mces_solver():
    try:
        import pulp
        from myopic_mces import MCES
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MCES requires optional dependencies 'pulp' and 'myopic_mces'. "
            "Install them before computing MCES metrics."
        ) from exc

    solvers = pulp.listSolvers(onlyAvailable=True)
    if len(solvers) == 0:
        raise RuntimeError("MCES requires an available pulp solver, but none were found.")
    return MCES, solvers[0]


def canonical_no_stereo_smiles(smiles: str) -> Optional[str]:
    from rdkit import Chem
    from ms_pred.common import chem_utils

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = Chem.MolToInchi(mol)
    mol = chem_utils.canonical_mol_from_inchi(inchi)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def compute_mces(
    true_smiles: str,
    pred_smiles: str,
    *,
    mces_func=None,
    solver=None,
    threshold: int = 15,
    time_limit: int = 600,
) -> float:
    from rdkit import Chem

    true_smi = canonical_no_stereo_smiles(true_smiles)
    pred_smi = canonical_no_stereo_smiles(pred_smiles)
    if true_smi is None:
        raise ValueError(f"Invalid true SMILES for MCES: {true_smiles}")

    true_mol = Chem.MolFromSmiles(true_smi)
    true_num_bonds = true_mol.GetNumBonds()
    pred_mol = Chem.MolFromSmiles(pred_smi) if pred_smi is not None else None
    if pred_smi is None or pred_mol is None:
        return float(2 * true_num_bonds)

    if mces_func is None or solver is None:
        mces_func, solver = load_mces_solver()

    try:
        result = mces_func(
            true_smi,
            pred_smi,
            solver=solver,
            threshold=threshold,
            always_stronger_bound=True,
            solver_options=dict(msg=0, timeLimit=time_limit),
        )
        if isinstance(result, (list, tuple)):
            result = result[1]
        return float(result)
    except Exception:
        return float(true_num_bonds + pred_mol.GetNumBonds())


def _compute_mces_pair(task: Tuple[Tuple[str, str], int, int]) -> float:
    global _WORKER_MCES_FUNC, _WORKER_MCES_SOLVER
    if _WORKER_MCES_FUNC is None or _WORKER_MCES_SOLVER is None:
        _WORKER_MCES_FUNC, _WORKER_MCES_SOLVER = load_mces_solver()
    pair, threshold, time_limit = task
    return compute_mces(
        pair[0],
        pair[1],
        mces_func=_WORKER_MCES_FUNC,
        solver=_WORKER_MCES_SOLVER,
        threshold=threshold,
        time_limit=time_limit,
    )


def top1_mces_pairs_from_individuals(
    individuals: Sequence[Dict[str, Any]],
    *,
    force: bool = False,
) -> Tuple[List[Tuple[int, str, str]], int, int]:
    pairs = []
    missing = 0
    existing = 0
    for entry_idx, entry in enumerate(individuals):
        if not force and entry.get("top_1_mces") is not None:
            existing += 1
            continue

        true_smiles = (
            entry.get("true_smiles")
            or entry.get("smiles")
            or entry.get("query_smiles")
        )
        pred_smiles = first_ranked_smiles(entry)
        if not true_smiles or not pred_smiles:
            missing += 1
            continue
        pairs.append((entry_idx, str(true_smiles), str(pred_smiles)))
    return pairs, missing, existing


def compute_top1_mces_for_individuals(
    individuals: Sequence[Dict[str, Any]],
    *,
    mces_func=None,
    solver=None,
    num_workers: int = 1,
    threshold: int = 15,
    time_limit: int = 600,
    force: bool = False,
) -> Dict[str, Any]:
    if num_workers < 1:
        raise ValueError("num_workers must be positive.")
    if num_workers > 1 and (mces_func is not None or solver is not None):
        raise ValueError("Custom MCES functions are only supported with num_workers=1.")

    pairs, missing, existing = top1_mces_pairs_from_individuals(
        individuals,
        force=force,
    )

    if not pairs:
        values = []
    elif num_workers == 1:
        if mces_func is None or solver is None:
            mces_func, solver = load_mces_solver()
        values = [
            compute_mces(
                true_smiles,
                pred_smiles,
                mces_func=mces_func,
                solver=solver,
                threshold=threshold,
                time_limit=time_limit,
            )
            for _, true_smiles, pred_smiles in pairs
        ]
    else:
        unique_pairs = list(
            dict.fromkeys((true_smi, pred_smi) for _, true_smi, pred_smi in pairs)
        )
        from joblib import Parallel, delayed

        unique_values = Parallel(n_jobs=num_workers)(
            delayed(_compute_mces_pair)((pair, threshold, time_limit))
            for pair in unique_pairs
        )
        value_map = dict(zip(unique_pairs, unique_values))
        values = [value_map[(true_smi, pred_smi)] for _, true_smi, pred_smi in pairs]

    for (entry_idx, _, _), value in zip(pairs, values):
        individuals[entry_idx]["top_1_mces"] = float(value)

    samples = [
        float(entry["top_1_mces"])
        for entry in individuals
        if entry.get("top_1_mces") is not None
    ]
    mean = float(np.mean(samples)) if samples else float("nan")
    median = float(np.median(samples)) if samples else float("nan")
    return {
        "computed": len(values),
        "existing": existing,
        "missing": missing,
        "n": len(samples),
        "mean": mean,
        "median": median,
    }
