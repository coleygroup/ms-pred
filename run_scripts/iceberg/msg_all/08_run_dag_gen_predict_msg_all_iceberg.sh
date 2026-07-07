python launcher_scripts/run_from_config.py configs/iceberg/msg_all/dag_gen_predict_train_msg_all_iceberg_decoy0.yaml
decoy0_status=$?
if [ "$decoy0_status" -ne 0 ]; then
  exit "$decoy0_status"
fi

python data_scripts/dag/add_dag_intens.py \
  --pred-dag-path results/iceberg_msg_all_iceberg/split_rnd1/preds_train_100/tree_preds.hdf5 \
  --true-dag-path data/spec_datasets/msg_all_iceberg/subformulae/no_subform.hdf5 \
  --out-dag-path results/iceberg_msg_all_iceberg/split_rnd1/preds_train_100_inten.hdf5 \
  --num-workers 32 &
inten_pid=$!

python launcher_scripts/run_from_config.py configs/iceberg/msg_all/dag_gen_predict_train_msg_all_iceberg_decoy10.yaml
decoy10_status=$?

wait "$inten_pid"
inten_status=$?

if [ "$decoy10_status" -ne 0 ]; then
  exit "$decoy10_status"
fi
if [ "$inten_status" -ne 0 ]; then
  exit "$inten_status"
fi
