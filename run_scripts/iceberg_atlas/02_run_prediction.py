from ms_pred import common
from pathlib import Path

vis_devices = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 x 2080Ti
max_parallel_per_gpu = 1
task_dir = "data/retrieval/pubchem/atlas/formula"
task_dir = Path(task_dir)
save_dir = "data/retrieval/pubchem/atlas/spectra"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
# all pytorch c++ warnings are suppressed
script_pattern = """TORCH_CPP_LOG_LEVEL=ERROR python src/ms_pred/dag_pred/predict_smis.py \\
    --batch-size 32 \\
    --sparse-out \\
    --sparse-k 100 \\
    --max-nodes 100 \\
    --gen-checkpoint ~/ms-models/iceberg_results_20250816/dag_nist20/split_1_rnd1/version_0/best.ckpt \\
    --inten-checkpoint ~/ms-models/iceberg_results_20250816/dag_inten_nist20/split_1_rnd1/version_1/best.ckpt \\
    --save-dir {out_dir} \\
    --out-name {out_name} \\
    --dataset-labels {in_tsv} \\
    --num-cpu-workers 32 \\
    --num-gpu-workers 2 \\
    --gpu \\
    --adduct-shift \\
    && touch {out_dir}/success || touch {out_dir}/fail
"""

scripts_to_run = []
for in_tsv in task_dir.iterdir():
    out_dir = save_dir / in_tsv.stem
    out_name = f'preds.hdf5'
    scripts_to_run.append(script_pattern.format(in_tsv=in_tsv, out_dir=out_dir, out_name=out_name))

common.subprocess_parallel(scripts_to_run, max_parallel_per_gpu, gpus=vis_devices)
