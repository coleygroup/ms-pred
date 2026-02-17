#!/bin/bash

# Example paths (update these as needed)
PRED_MAGMA_H5="data/spec_datasets/nist20/magma_outputs/magma_tree.hdf5"
TRUE_DAG_H5="data/spec_datasets/nist20/subformulae/no_subform.hdf5"
OUT_MAGMA_H5="data/spec_datasets/nist20/magma_outputs/magma_tree_with_inten.hdf5"

python data_scripts/dag/add_dag_intens.py \
  --pred-dag-folder "$PRED_MAGMA_H5" \
  --true-dag-folder "$TRUE_DAG_H5" \
  --out-dag-folder "$OUT_MAGMA_H5" \
  --num-workers 32 \
  --magma-output \
  --add-raw