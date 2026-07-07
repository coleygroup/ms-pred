#!/usr/bin/env bash

workers=${workers:-32}
subform_file="data/spec_datasets/msg_all_iceberg/subformulae/no_subform.hdf5"

if [ -f "$subform_file" ]; then
  echo "$subform_file exists; skipping"
  exit 0
fi

if [ -L data/spec_datasets/msg_all_iceberg/subformulae ]; then
  rm data/spec_datasets/msg_all_iceberg/subformulae
fi
mkdir -p data/spec_datasets/msg_all_iceberg/subformulae

python data_scripts/forms/01_assign_subformulae.py \
  --data-dir data/spec_datasets/msg_all_iceberg \
  --labels-file data/spec_datasets/msg_all_iceberg/labels.tsv \
  --use-all \
  --output-dir-name no_subform.hdf5 \
  --collision-source labels \
  --num-workers "$workers"
