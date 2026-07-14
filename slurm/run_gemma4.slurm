#!/bin/bash
#SBATCH --job-name=vlmaudit_gemma4
#SBATCH --partition=qTRDGPUH
#SBATCH -w arctrddgxa003
#SBATCH --account=psy53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4                  # 4-GPU data-parallel: one shard process per GPU (~4x throughput)
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --chdir=/data/users2/asabbaghziarani1/vlm-brainmri-audit
#SBATCH --output=slurm/logs/gemma4_%j.out
#SBATCH --error=slurm/logs/gemma4_%j.err

# ── Gemma-4-12B-it (Google's newest flagship) — FLAGSHIP/RECENCY audit ────────────────────────────
# NOTE ON ROLE: this is NOT the MedGemma base control (that is gemma-3-4b-it / slurm/run_gemma3.slurm).
# This audits the latest general VLM (gemma4_unified, any-to-any, ~24 GB bf16) on the same exam set.
#
# RUN THE SMOKE TEST FIRST:  sbatch slurm/smoke_gemma4.slurm  → confirm it prints "VERDICT: PASS"
# before launching this. Gemma-4 is a new architecture; the smoke test verifies our hf_runner's
# image-in-content path works with its processor.
#
# 4-GPU DATA PARALLELISM: 4 processes, one pinned per GPU, each a stable-hash shard
# (md5(image_id|prompt_id) % 4) of the neutral prompts → own raw_responses/...shard{k}of4.jsonl
# (no append races); aggregation globs *.jsonl so all shards roll up under google/gemma-4-12B-it.
#
# TIMING (honest): 12B (~24 GB) is ~2.5-3x slower than the 4B. Stage 1 (MC, 17,618 items) ≈ ~9 h on
# 4 GPUs and lands the headline; Stage 2 (+OE) will likely exceed one 24 h wall — it is RESUMABLE, so
# just re-submit to continue. Aggregation runs after EACH stage so the MC headline is saved regardless.
#
# Gemma-4 is ungated (gated:false on HF): HF_TOKEN is OPTIONAL (passed if present to avoid rate limits).

set -euo pipefail
PROJECT_DIR="/data/users2/asabbaghziarani1/vlm-brainmri-audit"
cd "$PROJECT_DIR"
mkdir -p logs slurm/logs raw_responses

source /home/users/asabbaghziarani1/miniconda3/etc/profile.d/conda.sh
conda activate vlmaudit
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_DIR}/cache/hf"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_MODE=disabled            # 4 concurrent shards; CSVs are the deliverable
export TOKENIZERS_PARALLELISM=false   # avoid fork-after-parallelism warnings across 4 procs
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

MODEL="google/gemma-4-12B-it"
NSHARDS=4
echo "====== Gemma-4-12B-it run (flagship/recency audit) ======"; echo "Start: $(date)"; echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Prefetch the ~24 GB snapshot ONCE before fanning out (warm cache for all 4 workers; instant if the
# smoke test already downloaded it).
echo "--- Prefetch model snapshot ---"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
p = snapshot_download("google/gemma-4-12B-it", token=tok)
print("prefetch ok:", p)
PY

COMMON=(--models config/models.yaml --tasks config/tasks.yaml --prompts config/prompts.yaml
        --manifest data/exam_set.csv --output-dir raw_responses
        --include-optional --model "$MODEL" --phrasing neutral --batch-size 1 --num-shards "$NSHARDS")

# Launch one process per GPU; fail the job if ANY shard exits nonzero (set -e doesn't catch bg procs).
run_sharded () {
    local fmt="$1" tag="$2"
    local pids=() rc=0
    for k in $(seq 0 $((NSHARDS-1))); do
        CUDA_VISIBLE_DEVICES=$k python -m src.inference.run_inference "${COMMON[@]}" \
            --format $fmt --shard-index "$k" \
            --log-file "slurm/logs/gemma4_${tag}_shard${k}.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    return $rc
}

# Re-aggregate official results + paper tables + figures. Called after Stage 1 so the MC headline is on
# disk even if Stage 2's OE pass is cut off by the wall; again after Stage 2 for OE hallucination metrics.
aggregate_all () {
    local tag="$1"
    python -m src.analysis.aggregate --responses-dir raw_responses --manifest data/exam_set.csv \
        --tasks config/tasks.yaml --output-dir results 2>&1 | tee "slurm/logs/gemma4_aggregate_${tag}.log"
    python -m src.analysis.paper_aggregate --responses-dir raw_responses --manifest data/exam_set.csv \
        --tasks config/tasks.yaml --output-dir results/paper 2>&1 | tee "slurm/logs/gemma4_paper_${tag}.log"
    python -m src.analysis.make_figures --results-dir results --output-dir figures \
        --manifest data/exam_set.csv 2>&1 | tee "slurm/logs/gemma4_figures_${tag}.log"
    python -m src.analysis.paper_figures 2>&1 | tee "slurm/logs/gemma4_paperfigs_${tag}.log" || true
}

echo "--- Stage 1: neutral MC (4-GPU sharded; ~9 h) ---"
run_sharded "MC" "mc"
echo "--- Aggregate after Stage 1 (headline lands now) ---"
aggregate_all "mc"

echo "--- Stage 2: neutral MC+OE (resumes; MC rows skipped via per-shard dedup; may exceed the wall) ---"
run_sharded "MC OE" "oe"
echo "--- Aggregate after Stage 2 (adds OE hallucination metrics) ---"
aggregate_all "final"

echo "====== Gemma-4-12B-it run complete: $(date) ======"
echo "Check: results/paper/safety_summary.csv should now list Gemma-4-12B (tier A)."
echo "Reminder: this is the flagship audit, NOT the MedGemma base control (that is run_gemma3.slurm)."
