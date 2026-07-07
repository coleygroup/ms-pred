python launcher_scripts/run_from_config.py configs/iceberg/nist23/dag_gen_predict_train_nist23.yaml

# Assign intensities to prediction for next training run

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-path results/iceberg_nist23/split_1_rnd1/preds_train_100/tree_preds.hdf5 \
	--true-dag-path data/spec_datasets/nist23/subformulae/no_subform.hdf5 \
	--out-dag-path results/iceberg_nist23/split_1_rnd1/preds_train_100_inten.hdf5  \
	--num-workers 32

#python data_scripts/dag/add_dag_intens.py \
#	--pred-dag-path results/iceberg_nist23/split_1_rnd2/preds_train_100/tree_preds.hdf5 \
#	--true-dag-path data/spec_datasets/nist23/subformulae/no_subform.hdf5 \
#	--out-dag-path results/iceberg_nist23/split_1_rnd2/preds_train_100_inten.hdf5  \
#	--num-workers 32
#
#python data_scripts/dag/add_dag_intens.py \
#	--pred-dag-path results/iceberg_nist23/split_1_rnd3/preds_train_100/tree_preds.hdf5 \
#	--true-dag-path data/spec_datasets/nist23/subformulae/no_subform.hdf5 \
#	--out-dag-path results/iceberg_nist23/split_1_rnd3/preds_train_100_inten.hdf5  \
#	--num-workers 32

#python data_scripts/dag/add_dag_intens.py \
#	--pred-dag-path results/iceberg_nist23/scaffold_1_rnd1/preds_train_100/tree_preds.hdf5 \
#	--true-dag-path data/spec_datasets/nist23/subformulae/no_subform.hdf5 \
#	--out-dag-path results/iceberg_nist23/scaffold_1_rnd1/preds_train_100_inten.hdf5  \
#	--num-workers 32

#python data_scripts/dag/add_dag_intens.py \
#	--pred-dag-path results/iceberg_nist23/scaffold_1_rnd2/preds_train_100/tree_preds.hdf5 \
#	--true-dag-path data/spec_datasets/nist23/subformulae/no_subform.hdf5 \
#	--out-dag-path results/iceberg_nist23/scaffold_1_rnd2/preds_train_100_inten.hdf5  \
#	--num-workers 32
#
#python data_scripts/dag/add_dag_intens.py \
#	--pred-dag-path results/iceberg_nist23/scaffold_1_rnd3/preds_train_100/tree_preds.hdf5 \
#	--true-dag-path data/spec_datasets/nist23/subformulae/no_subform.hdf5 \
#	--out-dag-path results/iceberg_nist23/scaffold_1_rnd3/preds_train_100_inten.hdf5  \
#	--num-workers 32
