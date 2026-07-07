python launcher_scripts/run_from_config.py configs/iceberg/msg_all/dag_gen_predict_train_msg_known_ce.yaml

# Assign intensities to prediction for next training run
python data_scripts/dag/add_dag_intens.py \
	--pred-dag-path  results/iceberg_msg_known_ce/split_rnd1/preds_train_100/tree_preds.hdf5 \
	--true-dag-path data/spec_datasets/msg_known_ce/subformulae/no_subform.hdf5 \
	--out-dag-path results/iceberg_msg_known_ce/split_rnd1/preds_train_100_inten.hdf5  \
	--num-workers 32
