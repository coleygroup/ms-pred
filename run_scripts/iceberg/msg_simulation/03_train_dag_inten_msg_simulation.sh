python launcher_scripts/run_from_config.py configs/iceberg/msg_simulation/dag_inten_train_msg_simulation.yaml
inten_status=$?
if [ "$inten_status" -ne 0 ]; then
  exit "$inten_status"
fi

python launcher_scripts/run_from_config.py configs/iceberg/msg_simulation/dag_inten_contr_finetune_msg_simulation.yaml
