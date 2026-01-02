. data_scripts/dag/run_magma.sh
. run_scripts/iceberg_transformer/01_run_frag_gen_train.sh
. run_scripts/iceberg_transformer/02_run_frag_gen_predict.sh
. run_scripts/iceberg_transformer/03_train_inten.sh
python run_scripts/iceberg_transformer/04_predict_inten.py
python run_scripts/iceberg_transformer/05_run_retrieval.py