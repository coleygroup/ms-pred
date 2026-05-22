#!/usr/bin/env bash
# Run GPU-marked tests on an Engaging GPU node via srun.
# Usage: bash tests/run_gpu.sh [extra pytest args]
#
# Defaults follow the lab convention (see ~/.cursor/skills/engaging-cluster/SKILL.md):
# request multiple partitions at once, modest memory for a sanity job.
# Override with env vars, e.g.
#   GPU_PARTITION=pi_ccoley GPU_TIME=00:30:00 bash tests/run_gpu.sh -v

set -euo pipefail

PARTITION="${GPU_PARTITION:-ou_cheme_gpu,mit_normal_gpu,pi_ccoley,mit_preemptable}"
TIME="${GPU_TIME:-00:10:00}"
GRES="${GPU_GRES:-gpu:1}"
MEM="${GPU_MEM:-16G}"
CPUS="${GPU_CPUS:-2}"

srun -p "$PARTITION" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    pytest -m gpu -ra "$@"
