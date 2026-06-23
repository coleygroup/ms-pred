#!/bin/bash
# engaging configs: L40S (mit_preemptable)
#SBATCH -n 10
#SBATCH -t 0-24:00:00
#SBATCH -J iceberg_atlas
#SBATCH --output=logs/iceberg_atlas_l40s_%A_%a.log

#SBATCH -p mit_preemptable
#SBATCH --mem-per-cpu=4000
#SBATCH -G l40s:1
#SBATCH --requeue

#SBATCH --array=0-240

MIN_ID=0
MAX_ID=3000

## engaging configs: 2080Ti (sched_mit_ccoley)
##SBATCH -n 6
##SBATCH -t 0-24:00:00
##SBATCH -J iceberg_atlas
##SBATCH --output=logs/iceberg_atlas_2080ti_%A_%a.log
#
##SBATCH -p sched_mit_ccoley
##SBATCH --mem-per-cpu=8000
##SBATCH -G 1
##SBATCH --requeue
#
##SBATCH --array=0-240
#
#MIN_ID=3001
#MAX_ID=4567

# usage: sbatch run_scripts/iceberg_atlas/02_run_prediction_slurm.sh

set -euo pipefail

# --------------------------
# Runtime setup
# --------------------------
mkdir -p logs

# Temporarily relax nounset for system profile scripts (they may read unset vars like LC_ALL)
set +u
source /etc/profile
source /home/runzhong/.bashrc
conda activate ms-gen
set -u

# --------------------------
# Paths
# --------------------------
FORMULA_DIR="data/retrieval/pubchem/atlas/formula"
OUT_ROOT="data/retrieval/pubchem/atlas/spectra"

REMOTE_HOST="molgpu03.mit.edu"
REMOTE_BASE="/home/runzhong/ms-pred/data/retrieval/pubchem/atlas/h_minus_spectra"

GEN_CKPT="$HOME/ms-models/iceberg_results_20250816/dag_nist20/split_1_rnd1/version_0/best.ckpt"
INTEN_CKPT="$HOME/ms-models/iceberg_results_20250816/dag_inten_nist20/split_1_rnd1/version_1/best.ckpt"

# --------------------------
# Slurm array info
# --------------------------
NTASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
TASKID="${SLURM_ARRAY_TASK_ID:-0}"

# --------------------------
# GPU-aware batch size
# --------------------------
get_batch_size() {
  # Default (safe) if we cannot detect
  local default_bs=32

  if ! command -v nvidia-smi &>/dev/null; then
    echo "$default_bs"
    return 0
  fi

  # Get GPU names (one per line). If multiple, check all.
  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sort -u)

  if ((${#gpu_names[@]} == 0)); then
    echo "$default_bs"
    return 0
  fi

  # If *any* visible GPU is a 2080 Ti => use 32
  local name
  for name in "${gpu_names[@]}"; do
    if [[ "$name" == *"2080 Ti"* ]] || [[ "$name" == *"RTX 2080 Ti"* ]]; then
      echo 32
      return 0
    fi
  done

  # If *any* visible GPU matches the fast list => 128
  for name in "${gpu_names[@]}"; do
    if [[ "$name" == *"L40S"* ]] || [[ "$name" == *"A100"* ]] || [[ "$name" == *"H100"* ]] || [[ "$name" == *"H200"* ]]; then
      echo 128
      return 0
    fi
  done

  # Otherwise keep the default
  echo "$default_bs"
}

BATCH_SIZE="$(get_batch_size)"
echo "Detected GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ';' - || echo 'unknown')"
echo "Using batch size: ${BATCH_SIZE}"

# --------------------------
# Enumerate all pubchem TSVs
# --------------------------
mapfile -t IDS < <(
  find "$FORMULA_DIR" -maxdepth 1 -type f -name 'pubchem_*.tsv' -printf '%f\n' \
    | sed -n 's/^pubchem_\([0-9]\+\)\.tsv$/\1/p' \
    | awk -v lo="$MIN_ID" -v hi="$MAX_ID" '$1>=lo && $1<=hi' \
    | sort -n
)

TOTAL="${#IDS[@]}"
if [[ "$TOTAL" -eq 0 ]]; then
  echo "No pubchem_*.tsv files found in $FORMULA_DIR"
  exit 0
fi

# --------------------------
# Chunking logic (serial per array task)
# --------------------------
CHUNK=$(( (TOTAL + NTASKS - 1) / NTASKS ))
START=$(( TASKID * CHUNK ))
END=$(( START + CHUNK - 1 ))
if [[ "$END" -ge "$TOTAL" ]]; then END=$(( TOTAL - 1 )); fi

if [[ "$START" -ge "$TOTAL" ]]; then
  echo "Array task ${TASKID}: no assigned work."
  exit 0
fi

echo "Array task ${TASKID}/${NTASKS}: processing indices [$START..$END] of $TOTAL"

# --------------------------
# Worker
# --------------------------
run_one() {
  local number_id="$1"

  local in_tsv="${FORMULA_DIR}/pubchem_${number_id}.tsv"
  local out_dir="${OUT_ROOT}/pubchem_${number_id}"
  local out_name="preds.hdf5"
  local out_file="${out_dir}/${out_name}"

  if [[ -f "${out_dir}/success" ]]; then
    echo "[pubchem_${number_id}] already completed"
    return 0
  fi

  mkdir -p "$out_dir"

  echo "[pubchem_${number_id}] running prediction"
  if python src/ms_pred/iceberg/predict_smis.py \
      --batch-size "$BATCH_SIZE" \
      --sparse-out \
      --sparse-k 100 \
      --max-nodes 100 \
      --gen-checkpoint "$GEN_CKPT" \
      --inten-checkpoint "$INTEN_CKPT" \
      --save-dir "$out_dir" \
      --out-name "$out_name" \
      --dataset-labels "$in_tsv" \
      --num-cpu-workers 0 \
      --num-gpu-workers 2 \
      --gpu \
      --adduct-shift
  then
    touch "${out_dir}/success"
  else
    touch "${out_dir}/fail"
    echo "[pubchem_${number_id}] prediction failed"

    local remote_dir="${REMOTE_BASE}/pubchem_${number_id}"
    ssh -o BatchMode=yes "${REMOTE_HOST}" "mkdir -p '${remote_dir}'"
    for f in "${out_dir}/joint_pred.log" "${out_dir}/fail"; do
      if [[ -f "$f" ]]; then
        scp -o BatchMode=yes "$f" "${REMOTE_HOST}:'${remote_dir}/'"
      fi
    done
    return 1
  fi

  # --------------------------
  # Copy to molgpu03 + cleanup
  # --------------------------
  local remote_dir="${REMOTE_BASE}/pubchem_${number_id}"

  ssh -o BatchMode=yes "${REMOTE_HOST}" "mkdir -p '${remote_dir}'"
  scp -o BatchMode=yes \
      "${out_file}" \
      "${out_dir}/success" \
      "${out_dir}/joint_pred.log" \
      "${REMOTE_HOST}:'${remote_dir}/'"

  rm -f "${out_file}"
  echo "[pubchem_${number_id}] transferred and cleaned up"
}

# --------------------------
# Serial execution
# --------------------------
for (( idx=START; idx<=END; idx++ )); do
  number_id="${IDS[$idx]}"
  # Continue to next ID even if one fails
  if ! run_one "$number_id"; then
    echo "[pubchem_${number_id}] continuing to next job."
  fi
done
echo "Array task ${TASKID}: done"
