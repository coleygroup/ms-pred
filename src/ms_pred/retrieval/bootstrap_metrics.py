import argparse
from pathlib import Path

from scipy.stats import bootstrap
import numpy as np
import yaml


def load_results(path: str):
    path_obj = Path(path)
    if path_obj.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Unsupported results format: {path_obj.suffix}; expected a YAML rerank_eval file.")
    with path_obj.open("r") as f:
        return yaml.safe_load(f)


def main():
    np.random.seed(1)

    parser = argparse.ArgumentParser(description="Compute bootstrap confidence intervals for retrieval metrics.")
    parser.add_argument(
        "--path",
        default="results/joint_train_msg/split_rnd1/retrieval_msg_test_mass_256_pre_finetune/rerank_eval_cos.yaml",
        help="Path to rerank_eval YAML output (defaults to cosine run).",
    )
    parser.add_argument("--dist-fn", choices=["entropy", "cosine"], default="entropy", help="Which distance metric to analyze (defaults to entropy).")
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for reported metrics (defaults to 4).",
    )
    args = parser.parse_args()

    if args.precision < 0:
        raise ValueError("--precision must be non-negative.")

    data = load_results(args.path)
    # print(data)

    sim = []

    for entry in data['individuals']:
        if "true_dist" in entry:
            sim.append(1 - entry['true_dist'])
        
    if args.dist_fn == "entropy":
        res = bootstrap(
            (sim,),  # Must be a tuple
            np.mean,  # Statistic function
            confidence_level=0.999,
            n_resamples=20000,
        )
        fmt = f"{{:.{args.precision}f}}"
        print(f"Entropy distribution Estimated proportion: {fmt.format(np.mean(sim))}")
        print(
            "Entropy distribution 99.9% Confidence interval: ("
            f"{fmt.format(res.confidence_interval.low)}, {fmt.format(res.confidence_interval.high)})"
        )

    if args.dist_fn == "cosine":
        res = bootstrap(
            (sim,),  # Must be a tuple
            np.mean,  # Statistic function
            confidence_level=0.999,
            n_resamples=20000,
        )
        fmt = f"{{:.{args.precision}f}}"
        print(f"Cosine distribution Estimated proportion: {fmt.format(np.mean(sim))}")
        print(
            "Cosine distribution 99.9% Confidence interval: ("
            f"{fmt.format(res.confidence_interval.low)}, {fmt.format(res.confidence_interval.high)})"
        )

    metrics = data['metrics']
    print(metrics)
    # Metrics are means of a Bernoulli distribution, so reconstruct the Bernoulli samples for bootstrapping.
    top1 = int(metrics['hit_rate_at_1'] * len(data['individuals'])) * [1] + int((1 - metrics['hit_rate_at_1']) * len(data['individuals'])) * [0]
    top5 = int(metrics['hit_rate_at_5'] * len(data['individuals'])) * [1] + int((1 - metrics['hit_rate_at_5']) * len(data['individuals'])) * [0]
    top20 = int(metrics['hit_rate_at_20'] * len(data['individuals'])) * [1] + int((1 - metrics['hit_rate_at_20']) * len(data['individuals'])) * [0]

    res = bootstrap(
        (top1,),  # Must be a tuple
        np.mean,  # Statistic function
        confidence_level=0.999,
        n_resamples=20000,
    )
    fmt = f"{{:.{args.precision}f}}"
    print(f"Top 1 distribution Estimated proportion: {fmt.format(np.mean(top1)*100)}")
    print(
        "Top 1 distribution 99.9% Confidence interval: ("
        f"{fmt.format(res.confidence_interval.low * 100)}, {fmt.format(res.confidence_interval.high * 100)})"
    )

    res = bootstrap(
        (top5,),  # Must be a tuple
        np.mean,  # Statistic function
        confidence_level=0.999,
        n_resamples=20000,
    )
    print(f"Top 5 distribution Estimated proportion: {fmt.format(np.mean(top5)*100)}")
    print(
        "Top 5 distribution 99.9% Confidence interval: ("
        f"{fmt.format(res.confidence_interval.low * 100)}, {fmt.format(res.confidence_interval.high * 100)})"
    )

    res = bootstrap(
        (top20,),  # Must be a tuple
        np.mean,  # Statistic function
        confidence_level=0.999,
        n_resamples=20000,
    )
    print(f"Top 20 distribution Estimated proportion: {fmt.format(np.mean(top20)*100)}")
    print(
        "Top 20 distribution 99.9% Confidence interval: ("
        f"{fmt.format(res.confidence_interval.low * 100)}, {fmt.format(res.confidence_interval.high * 100)})"
    )


if __name__ == "__main__":
    main()

