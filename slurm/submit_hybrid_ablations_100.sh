#!/bin/bash

# Submit the full 100-image hybrid ablation suite as separate SLURM jobs.
#
# Usage:
#   bash slurm/submit_hybrid_ablations_100.sh
#
# Optional overrides:
#   REPO_ROOT=/home/<id>/cv_project/cutler-multiscale
#   DATA_ROOT=/home/<id>/data
#   TOTAL_IMAGES=100
#   VARIANTS="baseline hp90 hp80 topk8 tightcrop"

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${HOME}/data}"
TOTAL_IMAGES="${TOTAL_IMAGES:-100}"
VARIANTS_STRING="${VARIANTS:-baseline hp90 hp80 topk8 tightcrop}"

read -r -a VARIANTS <<< "${VARIANTS_STRING}"

echo "=== Submitting hybrid ablation suite ==="
echo "  REPO_ROOT    : ${REPO_ROOT}"
echo "  DATA_ROOT    : ${DATA_ROOT}"
echo "  TOTAL_IMAGES : ${TOTAL_IMAGES}"
echo "  VARIANTS     : ${VARIANTS[*]}"
echo

for variant in "${VARIANTS[@]}"; do
    job_id=$(
        sbatch --parsable \
            --job-name="hyab_${variant}_${TOTAL_IMAGES}" \
            --export=ALL,REPO_ROOT="${REPO_ROOT}",DATA_ROOT="${DATA_ROOT}",TOTAL_IMAGES="${TOTAL_IMAGES}",VARIANT="${variant}" \
            "${REPO_ROOT}/slurm/run_hybrid_ablation_100.sh"
    )
    echo "${variant} -> ${job_id}"
done

echo
echo "Monitor with:"
echo "  squeue -u ${USER}"
