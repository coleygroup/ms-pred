#!/usr/bin/env bash

dataset=${1:-${dataset:-nist23}} # nist20, nist23, msg_all, msg_simulate
max_peaks=${max_peaks:-50}
ppm_diff=${ppm_diff:-20}
workers=${workers:-32}

data_dir="data/spec_datasets/$dataset"
labels="$data_dir/labels.tsv"
spec_files="$data_dir/spec_files.hdf5"
subform_file="$data_dir/subformulae/no_subform.hdf5"

python3 src/ms_pred/magma/run_magma.py  \
--spectra-dir "$spec_files"  \
--output-dir "$data_dir/magma_outputs"  \
--spec-labels "$labels" \
--max-peaks $max_peaks \
--ppm-diff $ppm_diff \
--workers $workers

if [ -f "$subform_file" ]; then
  echo "no_subform.hdf5 exists for $dataset, skipping"
else
  mkdir -p "$data_dir/subformulae"
  python data_scripts/forms/01_assign_subformulae.py \
  --data-dir "$data_dir" \
  --labels-file "$labels" \
  --use-all \
  --output-dir-name no_subform.hdf5 \
  --num-workers $workers
fi
