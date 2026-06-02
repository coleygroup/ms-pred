import argparse
from pathlib import Path

import numpy as np
import yaml

from ms_pred.common.bootstrap_utils import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_N_RESAMPLES,
    load_yaml_results,
    summarize_rerank_bootstrap,
)


def _print_summary(label: str, summary: dict, *, precision: int, percent: bool = False):
    scale = 100 if percent else 1
    fmt = f"{{:.{precision}f}}"
    print(f"{label} Estimated mean: {fmt.format(summary['mean'] * scale)}")
    print(
        f"{label} {DEFAULT_CONFIDENCE_LEVEL * 100:.1f}% Confidence interval: ("
        f"{fmt.format(summary['ci_low'] * scale)}, {fmt.format(summary['ci_high'] * scale)})"
    )


def main():
    np.random.seed(1)

    parser = argparse.ArgumentParser(
        description="Compute bootstrap confidence intervals for retrieval metrics."
    )
    parser.add_argument(
        "--path",
        default="results/joint_train_msg/split_rnd1/retrieval_msg_test_mass_256_pre_finetune/rerank_eval_cos.yaml",
        help="Path to rerank_eval YAML output (defaults to cosine run).",
    )
    parser.add_argument(
        "--outfile",
        default=None,
        help="Output YAML path. Defaults to <input stem>_bootstrap.yaml.",
    )
    parser.add_argument(
        "--dist-fn",
        choices=["entropy", "cosine"],
        default="entropy",
        help="Which distance metric to analyze (defaults to entropy).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for reported metrics (defaults to 4).",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=DEFAULT_N_RESAMPLES,
        help=f"Number of bootstrap resamples (defaults to {DEFAULT_N_RESAMPLES}).",
    )
    args = parser.parse_args()

    if args.precision < 0:
        raise ValueError("--precision must be non-negative.")
    if args.n_resamples <= 0:
        raise ValueError("--n-resamples must be positive.")

    data = load_yaml_results(args.path)
    summaries = summarize_rerank_bootstrap(
        data,
        dist_fn=args.dist_fn,
        n_resamples=args.n_resamples,
    )

    input_path = Path(args.path)
    outfile = (
        Path(args.outfile)
        if args.outfile is not None
        else input_path.with_name(f"{input_path.stem}_bootstrap.yaml")
    )
    output = {
        "source": str(input_path),
        "dist_fn": args.dist_fn,
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "n_resamples": args.n_resamples,
        "random_state": 1,
        "summaries": summaries,
    }
    with outfile.open("w") as fp:
        yaml.dump(output, fp, indent=2)

    sim_key = f"{args.dist_fn}_similarity"
    if sim_key in summaries:
        _print_summary(
            f"{args.dist_fn.title()} similarity distribution",
            summaries[sim_key],
            precision=args.precision,
        )

    for k in (1, 5, 20):
        _print_summary(
            f"Top {k} distribution",
            summaries[f"top_{k}"],
            precision=args.precision,
            percent=True,
        )

    if "top_1_mces" in summaries:
        _print_summary(
            "Top 1 MCES distribution",
            summaries["top_1_mces"],
            precision=args.precision,
        )
    print(f"Wrote bootstrap summaries to {outfile}")


if __name__ == "__main__":
    main()
