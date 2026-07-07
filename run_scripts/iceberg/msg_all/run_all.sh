#!/usr/bin/env bash

######################
# Full MassSpecGym ICEBERG CE-imputation pipeline.
#
# 1. Train ICEBERG on the MSG subset with collision energy labels
# 2. Use this model to impute intensities for those entries with missing collision energies
# 3. Create the complete dataset with imputed collision energy labels
# 4. Train ICEBERG on the completed dataset
#
# Assumes data/spec_datasets/msg_all already exists with labels.tsv,
# spec_files.hdf5, retrieval candidate tables, magma outputs, and
# subformulae/no_subform.hdf5. msg_known_ce is derived by filtering msg_all
# labels.tsv to rows with known collision energies.
######################

done_dir="results/msg_all_pipeline_done"
mkdir -p "$done_dir"

run_stage() {
  stage_name="$1"
  shift
  marker="$done_dir/$stage_name.done"
  if [ -f "$marker" ]; then
    echo "Skipping $stage_name; found $marker"
    return 0
  fi
  echo "Running $stage_name"
  "$@"
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "$stage_name failed with exit code $status"
    exit "$status"
  fi
  touch "$marker"
}

run_stage 01_train_msg_known_ce bash run_scripts/iceberg/msg_all/01_run_dag_gen_train_msg_known_ce.sh
run_stage 02_predict_msg_known_ce bash run_scripts/iceberg/msg_all/02_run_dag_gen_predict_msg_known_ce.sh
run_stage 03_train_inten_msg_known_ce bash run_scripts/iceberg/msg_all/03_train_dag_inten_msg_known_ce.sh
run_stage 04_impute_missing_collision_energies python run_scripts/iceberg/msg_all/04_impute_missing_collision_energies.py
run_stage 05_build_msg_all_iceberg_dataset python run_scripts/iceberg/msg_all/05_build_msg_all_iceberg_dataset.py
run_stage 06_assign_subformulae_msg_all_iceberg bash run_scripts/iceberg/msg_all/06_assign_subformulae_msg_all_iceberg.sh
run_stage 07_train_msg_all_iceberg bash run_scripts/iceberg/msg_all/07_run_dag_gen_train_msg_all_iceberg.sh
run_stage 08_predict_msg_all_iceberg bash run_scripts/iceberg/msg_all/08_run_dag_gen_predict_msg_all_iceberg.sh
run_stage 09_train_inten_msg_all_iceberg bash run_scripts/iceberg/msg_all/09_train_dag_inten_msg_all_iceberg.sh
# Optional full MassSpecGym retrieval evaluation.
#run_stage 10_retrieval_msg_all_iceberg_formula python run_scripts/iceberg/msg_all/10_run_retrieval_msg_all_iceberg.py --split test_formula
#run_stage 10_retrieval_msg_all_iceberg_mass python run_scripts/iceberg/msg_all/10_run_retrieval_msg_all_iceberg.py --split test_mass
