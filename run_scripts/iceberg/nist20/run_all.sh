######################
# The following script will only run for split_1_rnd1 (random split, seed=1), which is suitable if you want to train
# your own ICEBERG for applications.
######################
# If you want to replicate the reported result with random + scaffold splits and 3 random seeds, please uncomment
# all entries in the following files
# * ``configs/iceberg/nist20/*.yaml``
# * ``02_sweep_gen_thresh.py``
# * ``05_predict_dag_inten.py``
# * ``06_run_retrieval.py``
. data_scripts/dag/run_magma.sh
. run_scripts/iceberg/nist20/01_run_dag_gen_train.sh
#python run_scripts/iceberg/nist20/02_sweep_gen_thresh.py  # ICEBERG-gen evaluation
. run_scripts/iceberg/nist20/03_run_dag_gen_predict.sh
. run_scripts/iceberg/nist20/04_train_dag_inten.sh
#python run_scripts/iceberg/nist20/05_predict_dag_inten.py  # ICEBERG-inten evaluation
#python run_scripts/iceberg/nist20/06_run_retrieval.py  # retrieval evaluation
#python run_scripts/iceberg/nist20/07_retrieval_conf.py  # retrieval confidence analysis
