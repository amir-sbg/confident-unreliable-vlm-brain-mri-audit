#!/bin/bash
#SBATCH --job-name=vlmaudit_medgemma
#SBATCH --partition=qTRDGPUH
#SBATCH -w arctrddgxa001
#SBATCH --account=psy53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                 # medgemma-4b fits one GPU in bf16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --chdir=/data/users2/asabbaghziarani1/vlm-brainmri-audit
#SBATCH --output=slurm/logs/medgemma_%j.out
#SBATCH --error=slurm/logs/medgemma_%j.err

# ── MedGemma (E4: general-vs-medical) — single gated model, resumable ─────────────
# Runs ONLY google/medgemma-4b-it (the other models are already in raw_responses), then
# re-aggregates official results + the paper tables so the general-vs-medical comparison appears.
#
# PREREQS (one time, by the human):
#   1. Accept the license at  https://huggingface.co/google/medgemma-4b-it  (logged in).
#   2. Create a READ token at  https://huggingface.co/settings/tokens
#   3. Submit with the token in the environment, e.g.:
#        HF_TOKEN=hf_xxx sbatch slurm/run_medgemma.slurm
#      (or `export HF_TOKEN=hf_xxx` before sbatch). Submit from a LOGIN node.
#
# Idempotent/resumable: if it hits the 24h wall, just re-submit — it continues where it stopped.

set -euo pipefail
PROJECT_DIR="/data/users2/asabbaghziarani1/vlm-brainmri-audit"
cd "$PROJECT_DIR"
mkdir -p logs slurm/logs raw_responses

source /home/users/asabbaghziarani1/miniconda3/etc/profile.d/conda.sh
conda activate vlmaudit
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_DIR}/cache/hf"
export TRANSFORMERS_CACHE="${HF_HOME}"

# Fail fast with a clear message if the gated-model token is missing.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "FATAL: HF_TOKEN not set. Accept the medgemma license + create a token, then:"
    echo "       HF_TOKEN=hf_xxx sbatch slurm/run_medgemma.slurm"
    exit 1
fi
# transformers/huggingface_hub read these:
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

MODEL="google/medgemma-4b-it"
echo "====== MedGemma run (E4) ======"; echo "Start: $(date)"; echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

COMMON=(--models config/models.yaml --tasks config/tasks.yaml --prompts config/prompts.yaml
        --manifest data/exam_set.csv --output-dir raw_responses --model "$MODEL"
        --phrasing neutral --batch-size 1)

# Stage 1: MC first (the headline accuracy/calibration comparison lands even if OE runs long).
echo "--- MedGemma neutral MC ---"
python -m src.inference.run_inference "${COMMON[@]}" --format MC \
    --log-file "slurm/logs/medgemma_mc.log" 2>&1

# Stage 2: add OE (resumes; MC rows are skipped via the JSONL dedup) for hallucination metrics.
echo "--- MedGemma neutral MC+OE ---"
python -m src.inference.run_inference "${COMMON[@]}" --format MC OE \
    --log-file "slurm/logs/medgemma_oe.log" 2>&1

# Re-aggregate: official results + paper tables (MedGemma now appears in the model comparison).
echo "--- Re-aggregate ---"
python -m src.analysis.aggregate --responses-dir raw_responses --manifest data/exam_set.csv \
    --tasks config/tasks.yaml --output-dir results 2>&1 | tee slurm/logs/medgemma_aggregate.log
python -m src.analysis.paper_aggregate --responses-dir raw_responses --manifest data/exam_set.csv \
    --tasks config/tasks.yaml --output-dir results/paper 2>&1 | tee slurm/logs/medgemma_paper.log
python -m src.analysis.make_figures --results-dir results --output-dir figures \
    --manifest data/exam_set.csv 2>&1 | tee slurm/logs/medgemma_figures.log

echo "====== MedGemma run complete: $(date) ======"
echo "Check: results/paper/safety_summary.csv should now include MedGemma-4B (tier B)."
