from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.stats import bootstrap
import yaml


DEFAULT_CONFIDENCE_LEVEL = 0.999
DEFAULT_N_RESAMPLES = 20000


def load_yaml_results(path: str) -> Dict[str, Any]:
    path_obj = Path(path)
    if path_obj.suffix not in {".yaml", ".yml"}:
        raise ValueError(
            f"Unsupported results format: {path_obj.suffix}; expected a YAML rerank_eval file."
        )
    with path_obj.open("r") as f:
        return yaml.safe_load(f)


def _as_float_array(values: Iterable[float], metric_name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError(f"No samples available for {metric_name}.")
    return array


def bootstrap_mean(
    values: Iterable[float],
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    random_state: Optional[int] = 1,
    metric_name: str = "metric",
) -> Dict[str, float]:
    samples = _as_float_array(values, metric_name)
    res = bootstrap(
        (samples,),
        np.mean,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    return {
        "mean": float(np.mean(samples)),
        "ci_low": float(res.confidence_interval.low),
        "ci_high": float(res.confidence_interval.high),
        "n": int(samples.size),
    }


def hit_samples_from_individuals(
    individuals: Sequence[Dict[str, Any]],
    k: int,
    metrics: Optional[Dict[str, float]] = None,
) -> List[float]:
    key = f"top_{k}"
    samples = [float(entry[key]) for entry in individuals if key in entry]
    if samples:
        return samples

    if metrics is None:
        raise ValueError(f"Missing per-entry {key} samples and aggregate metrics.")

    metric_key = f"hit_rate_at_{k}"
    if metric_key not in metrics:
        raise ValueError(f"Missing {metric_key} in aggregate metrics.")

    num_hits = int(round(float(metrics[metric_key]) * len(individuals)))
    return [1.0] * num_hits + [0.0] * (len(individuals) - num_hits)


def similarity_samples_from_individuals(individuals: Sequence[Dict[str, Any]]) -> List[float]:
    return [
        1.0 - float(entry["true_dist"])
        for entry in individuals
        if "true_dist" in entry and entry["true_dist"] is not None
    ]


def numeric_samples_from_individuals(
    individuals: Sequence[Dict[str, Any]],
    key: str,
) -> List[float]:
    return [
        float(entry[key])
        for entry in individuals
        if key in entry and entry[key] is not None
    ]


def summarize_rerank_bootstrap(
    data: Dict[str, Any],
    *,
    dist_fn: str,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    random_state: Optional[int] = 1,
) -> Dict[str, Dict[str, float]]:
    individuals = data.get("individuals", [])
    metrics = data.get("metrics", {})
    if not individuals:
        raise ValueError("No individuals found in rerank_eval results.")

    summaries = {}
    sim = similarity_samples_from_individuals(individuals)
    if sim:
        summaries[f"{dist_fn}_similarity"] = bootstrap_mean(
            sim,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
            metric_name=f"{dist_fn} similarity",
        )

    for k in (1, 5, 20):
        samples = hit_samples_from_individuals(individuals, k, metrics)
        summaries[f"top_{k}"] = bootstrap_mean(
            samples,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
            metric_name=f"top {k}",
        )

    mces_samples = numeric_samples_from_individuals(individuals, "top_1_mces")
    if mces_samples:
        summaries["top_1_mces"] = bootstrap_mean(
            mces_samples,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
            metric_name="top-1 MCES",
        )

    return summaries
