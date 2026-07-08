#!/usr/bin/env bash

######################
# MassSpecGym simulation-challenge ICEBERG pipeline.
#
# Assumes data/spec_datasets/msg_simulation already exists with labels.tsv,
# spec_files.hdf5, retrieval candidate tables, magma outputs, and
# subformulae/no_subform.hdf5. Unlike msg_all, this setting uses spectra with
# known collision energies and does not run collision-energy imputation.
######################

bash run_scripts/iceberg/msg_simulation/01_run_dag_gen_train_msg_simulation.sh
bash run_scripts/iceberg/msg_simulation/02_run_dag_gen_predict_msg_simulation.sh
bash run_scripts/iceberg/msg_simulation/03_train_dag_inten_msg_simulation.sh

# Retrieval runs. Uncomment these if you want retrieval evaluation.
#python run_scripts/iceberg/msg_simulation/04_run_retrieval_msg_simulation.py --split test_formula
#python run_scripts/iceberg/msg_simulation/04_run_retrieval_msg_simulation.py --split test_mass
