import argparse
from pathlib import Path

import yaml

from ms_pred.common.mces_utils import compute_top1_mces_for_individuals


def get_args():
    parser = argparse.ArgumentParser(
        description="Add top-1 MCES metrics to a rerank_eval YAML file."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Input rerank_eval YAML containing individuals with true_smiles and top_1_smiles.",
    )
    parser.add_argument(
        "--outfile",
        default=None,
        help="Output YAML path. Defaults to updating --path in place.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for MCES calculation.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=15,
        help="MCES threshold passed to myopic_mces.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="Per-pair solver time limit in seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute entries that already contain top_1_mces.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive.")
    if args.threshold <= 0:
        raise ValueError("--threshold must be positive.")
    if args.time_limit <= 0:
        raise ValueError("--time-limit must be positive.")

    path = Path(args.path)
    outfile = Path(args.outfile) if args.outfile is not None else path

    with path.open("r") as fp:
        data = yaml.safe_load(fp)

    individuals = data.get("individuals", [])
    if not individuals:
        raise ValueError(f"No individuals found in {path}.")

    summary = compute_top1_mces_for_individuals(
        individuals,
        num_workers=args.num_workers,
        threshold=args.threshold,
        time_limit=args.time_limit,
        force=args.force,
    )

    metrics = data.setdefault("metrics", {})
    metrics["top_1_mces"] = summary["mean"]
    metrics["mces@1"] = summary["mean"]
    metrics["median_top_1_mces"] = summary["median"]
    metrics["median_mces@1"] = summary["median"]
    data["avg_top_1_mces"] = summary["mean"]
    data["median_top_1_mces"] = summary["median"]
    data["top_1_mces_count"] = summary["n"]
    data["top_1_mces_missing"] = summary["missing"]

    with outfile.open("w") as fp:
        yaml.dump(data, fp, indent=2)

    print(f"Wrote top-1 MCES metrics to {outfile}")
    print(
        "Top-1 MCES mean: "
        f"{summary['mean']:.4f} "
        f"(median {summary['median']:.4f}, n={summary['n']}, "
        f"computed={summary['computed']}, existing={summary['existing']}, "
        f"missing={summary['missing']})"
    )


if __name__ == "__main__":
    main()
