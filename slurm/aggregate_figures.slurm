#!/bin/bash
#SBATCH --job-name=vlmaudit_agg
#SBATCH --partition=qTRDGPUH         # GPU partition (this stage is CPU-only; swap to a CPU partition if you have one)
#SBATCH --account=psy53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/data/users2/asabbaghziarani1/vlm-brainmri-audit
#SBATCH --output=slurm/logs/aggregate_%j.out
#SBATCH --error=slurm/logs/aggregate_%j.err

# Dependent job: submit with --dependency=afterok:<inference_job_id>
# sbatch --dependency=afterok:$INFER_JOB slurm/aggregate_figures.slurm

set -euo pipefail
PROJECT_DIR="/data/users2/asabbaghziarani1/vlm-brainmri-audit"  # hardcoded: SLURM runs a spool copy, BASH_SOURCE is unreliable
cd "$PROJECT_DIR"
mkdir -p logs slurm/logs

# conda is not on PATH in a non-interactive SLURM shell; source it by absolute path.
source /home/users/asabbaghziarani1/miniconda3/etc/profile.d/conda.sh
conda activate vlmaudit
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "Aggregation start: $(date)"

python -m src.analysis.aggregate \
    --responses-dir raw_responses \
    --manifest      data/exam_set.csv \
    --tasks         config/tasks.yaml \
    --output-dir    results \
    2>&1 | tee slurm/logs/aggregate.log

python -m src.analysis.make_figures \
    --results-dir results \
    --output-dir  figures \
    --manifest    data/exam_set.csv \
    2>&1 | tee slurm/logs/make_figures.log

echo "Aggregation complete: $(date)"
