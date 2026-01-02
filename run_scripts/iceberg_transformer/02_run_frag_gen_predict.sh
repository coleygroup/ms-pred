python /home/rxwangtw/ms-pred-dev/configs/iceberg_transformer/frag_predict_nist20.yaml

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-path  results/frag_only_nist20/split_1_rnd1/preds_train_100/frag_preds.hdf5 \
	--true-dag-path data/spec_datasets/nist20/subformulae/no_subform.hdf5 \
	--out-dag-path results/frag_only_nist20/split_1_rnd1/preds_train_100_inten.hdf5  \
	--num-workers 32

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-path  results/frag_only_nist20/split_1_rnd2/preds_train_100/tree_preds.hdf5 \
	--true-dag-path data/spec_datasets/nist20/subformulae/no_subform.hdf5 \
	--out-dag-path results/frag_only_nist20/split_1_rnd2/preds_train_100_inten.hdf5  \
	--num-workers 32

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-path  results/frag_only_nist20/split_1_rnd3/preds_train_100/tree_preds.hdf5 \
	--true-dag-path data/spec_datasets/nist20/subformulae/no_subform.hdf5 \
	--out-dag-path results/frag_only_nist20/split_1_rnd3/preds_train_100_inten.hdf5  \
	--num-workers 32

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-path  results/frag_only_nist20/scaffold_1_rnd1/preds_train_100/tree_preds.hdf5 \
	--true-dag-path data/spec_datasets/nist20/subformulae/no_subform.hdf5 \
	--out-dag-path results/frag_only_nist20/scaffold_1_rnd1/preds_train_100_inten.hdf5  \
	--num-workers 32

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-path  results/frag_only_nist20/scaffold_1_rnd2/preds_train_100/tree_preds.hdf5 \
	--true-dag-path data/spec_datasets/nist20/subformulae/no_subform.hdf5 \
	--out-dag-path results/frag_only_nist20/scaffold_1_rnd2/preds_train_100_inten.hdf5  \
	--num-workers 32

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-path  results/frag_only_nist20/scaffold_1_rnd3/preds_train_100/tree_preds.hdf5 \
	--true-dag-path data/spec_datasets/nist20/subformulae/no_subform.hdf5 \
	--out-dag-path results/frag_only_nist20/scaffold_1_rnd3/preds_train_100_inten.hdf5  \
	--num-workers 32