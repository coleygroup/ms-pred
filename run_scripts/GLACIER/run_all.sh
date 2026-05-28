. data_scripts/dag/run_magma.sh
. run_scripts/GLACIER/add_inten.sh
. run_scripts/GLACIER/01_train_joint.sh
python run_scripts/GLACIER/02_predict_inten.py
python run_scripts/GLACIER/03_run_retrieval.py