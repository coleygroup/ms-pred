""" base_hyperopt.py

Abstract away common hyperopt functionality

"""
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import Callable

import pytorch_lightning as pl

import ray
from ray import tune
from ray.air.config import RunConfig
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers.async_hyperband import ASHAScheduler

import ms_pred.common as common


def add_hyperopt_args(parser):
    # Tune args
    ha = parser.add_argument_group("Hyperopt Args")
    ha.add_argument("--cpus-per-trial", default=1, type=int)
    ha.add_argument("--gpus-per-trial", default=1, type=float)
    ha.add_argument("--num-h-samples", default=50, type=int)
    ha.add_argument("--grace-period", default=60 * 15, type=int)
    ha.add_argument("--max-concurrent", default=10, type=int)
    ha.add_argument("--tune-checkpoint", default=None)

    # Overwrite default savedir
    time_name = datetime.now().strftime("%Y_%m_%d")
    save_default = f"results/{time_name}_hyperopt/"
    parser.set_defaults(save_dir=save_default)


def run_hyperopt(
    kwargs: dict,
    score_function: Callable,
    param_space_function: Callable,
    initial_points: list,
):
    """run_hyperopt.

    Args:
        kwargs: All dictionary args for hyperopt and train
        score_function: Trainable function that sets up model train
        param_space_function: Function to suggest new params
        initial_points: List of initial params to try
    """
    ray.init("local")

    # Fix base_args based upon tune args
    kwargs["gpu"] = kwargs.get("gpus_per_trial", 0) > 0
    # max_t = args.max_epochs

    if kwargs["debug"]:
        kwargs["num_h_samples"] = 10
        kwargs["max_epochs"] = 5

    save_dir = kwargs["save_dir"]
    common.setup_logger(
        save_dir, log_name="hyperopt.log", debug=kwargs.get("debug", False)
    )
    pl.seed_everything(kwargs.get("seed"))

    # Define score function
    trainable = tune.with_parameters(
        score_function, base_args=kwargs, orig_dir=Path().resolve()
    )

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    metric = "val_loss"

    # Include cpus and gpus per trial
    trainable = tune.with_resources(
        trainable,
        resources=tune.PlacementGroupFactory(
            [
                {
                    "CPU": kwargs.get("cpus_per_trial"),
                    "GPU": kwargs.get("gpus_per_trial"),
                },
                {
                    "CPU": kwargs.get("num_workers"),
                },
            ],
            strategy="PACK",
        ),
    )

    search_algo = OptunaSearch(
        metric=metric,
        mode="min",
        points_to_evaluate=initial_points,
        space=param_space_function,
    )
    search_algo = ConcurrencyLimiter(
        search_algo, max_concurrent=kwargs["max_concurrent"]
    )

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            mode="min",
            metric=metric,
            search_alg=search_algo,
            scheduler=ASHAScheduler(
                max_t=24 * 60 * 60,  # max_t,
                time_attr="time_total_s",
                grace_period=kwargs.get("grace_period"),
                reduction_factor=2,
            ),
            num_samples=kwargs.get("num_h_samples"),
        ),
        run_config=RunConfig(name=None, local_dir=kwargs["save_dir"]),
    )

    if kwargs.get("tune_checkpoint") is not None:
        ckpt = str(Path(kwargs["tune_checkpoint"]).resolve())
        tuner = tuner.restore(path=ckpt, restart_errored=True)

    results = tuner.fit()
    best_trial = results.get_best_result()
    output = {"score": best_trial.metrics[metric], "config": best_trial.config}
    out_str = yaml.dump(output, indent=2)
    logging.info(out_str)
    with open(Path(save_dir) / "best_trial.yaml", "w") as f:
        f.write(out_str)

    # Output full res table
    results.get_dataframe().to_csv(
        Path(save_dir) / "full_res_tbl.tsv", sep="\t", index=None
    )
