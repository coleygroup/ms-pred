. data_scripts/dag/run_magma.sh
. run_scripts/iceberg_transformer/add_inten.sh
. run_scripts/iceberg_transformer/01_train_joint.sh
python run_scripts/iceberg_transformer/02_predict_inten.py
python run_scripts/iceberg_transformer/03_run_retrieval.py