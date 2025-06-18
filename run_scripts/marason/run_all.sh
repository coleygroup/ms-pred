. data_scripts/dag/run_magma.sh
. run_scripts/marason/01_run_marason_gen_train.sh
python run_scripts/marason/02_sweep_gen_thresh.py  # comment this if you want speedup and don't care about evaluations
. run_scripts/marason/03_run_marason_gen_predict.sh
. run_scripts/marason/04_train_marason_inten.sh
python run_scripts/iceberg/05_predict_marason_inten.py  # comment this if you want speedup and don't care about evaluations
. run_scripts/marason/06_run_retrieval.sh
