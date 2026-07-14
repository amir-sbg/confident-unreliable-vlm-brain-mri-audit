#!/bin/bash
#SBATCH --job-name=vlmaudit_infer
#SBATCH --partition=qTRDGPUH         # GSU TReNDS GPU partition
#SBATCH --account=psy53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-8                  # <-- edit: 0 to (n_models - 1)
#SBATCH --chdir=/data/users2/asabbaghziarani1/vlm-brainmri-audit
#SBATCH --output=slurm/logs/infer_array_%A_%a.out
#SBATCH --error=slurm/logs/infer_array_%A_%a.err

# ── Array inference: one model per task ───────────────────────────────────────
# Submit:  sbatch slurm/run_inference_array.slurm
# Aggregation depends on this job completing:
#   INFER_JOB=<job_id>
#   sbatch --dependency=afterok:$INFER_JOB slurm/aggregate_figures.slurm

set -euo pipefail

PROJECT_DIR="/data/users2/asabbaghziarani1/vlm-brainmri-audit"  # hardcoded: SLURM runs a spool copy, BASH_SOURCE is unreliable
cd "$PROJECT_DIR"
mkdir -p logs slurm/logs raw_responses

# conda is not on PATH in a non-interactive SLURM shell; source it by absolute path.
source /home/users/asabbaghziarani1/miniconda3/etc/profile.d/conda.sh
conda activate vlmaudit
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_DIR}/cache/hf"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Parse tests must already have passed (run run_all.slurm Stage 3 first, or:)
python -m pytest tests/test_parse.py -q --tb=short || {
    echo "Parser tests failed! Aborting."; exit 1
}

python -m src.inference.run_inference \
    --models            config/models.yaml \
    --tasks             config/tasks.yaml \
    --prompts           config/prompts.yaml \
    --manifest          data/exam_set.csv \
    --output-dir        raw_responses \
    --slurm-array-index "${SLURM_ARRAY_TASK_ID}" \
    --phrasing          neutral terse clinician \
    --format            MC OE \
    --batch-size        8 \
    --log-file          "slurm/logs/infer_${SLURM_ARRAY_TASK_ID}.log" \
    2>&1

echo "Array task ${SLURM_ARRAY_TASK_ID} complete: $(date)"
