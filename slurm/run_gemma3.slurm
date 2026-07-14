#!/bin/bash
#SBATCH --job-name=vlmaudit_gemma3
#SBATCH --partition=qTRDGPUH
#SBATCH -w arctrddgxa003
#SBATCH --account=psy53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4                  # 4-GPU data-parallel: one shard process per GPU (~4x throughput)
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --chdir=/data/users2/asabbaghziarani1/vlm-brainmri-audit
#SBATCH --output=slurm/logs/gemma3_%j.out
#SBATCH --error=slurm/logs/gemma3_%j.err

# ── Gemma-3-4B (base model of MedGemma-4B) — controlled base-vs-medical comparison ────────────────
# MedGemma-4B is a medical fine-tune of google/gemma-3-4b-it. Running the base isolates the causal
# effect of medical fine-tuning (same architecture, size, pretraining) instead of comparing MedGemma
# against unrelated Qwen/InternVL backbones.
#
# 4-GPU DATA PARALLELISM: 4 processes, one pinned per GPU via CUDA_VISIBLE_DEVICES, each handling a
# stable-hash shard (md5(image_id|prompt_id) % 4) of the neutral prompts. Each shard writes its OWN
# raw_responses/google__gemma-3-4b-it.shard{k}of4.jsonl (no append races). Aggregation globs *.jsonl,
# so all four shards roll up under model id google/gemma-3-4b-it automatically.
#
# SCOPE (deadline-aware): neutral phrasing only; Stage 1 = MC (headline accuracy/calibration), then
# Stage 2 = MC+OE (adds OE hallucination metrics; MC rows are skipped via per-shard resume dedup).
#
# PREREQS (human, one time):
#   1. Accept the license at  https://huggingface.co/google/gemma-3-4b-it  (logged in).
#      NOTE: this is a SEPARATE license acceptance from medgemma — accept it even if medgemma worked.
#   2. Create/READ token at  https://huggingface.co/settings/tokens
#   3. Submit from a LOGIN node with the token in the environment:
#        HF_TOKEN=hf_xxx sbatch slurm/run_gemma3.slurm
#
# Resumable: if it hits the wall or a shard dies, just re-submit — each shard continues where it left off.

set -euo pipefail
PROJECT_DIR="/data/users2/asabbaghziarani1/vlm-brainmri-audit"
cd "$PROJECT_DIR"
mkdir -p logs slurm/logs raw_responses

source /home/users/asabbaghziarani1/miniconda3/etc/profile.d/conda.sh
conda activate vlmaudit
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_DIR}/cache/hf"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_MODE=disabled            # 4 concurrent shards; CSVs are the deliverable, keep wandb out of the way
export TOKENIZERS_PARALLELISM=false   # avoid fork-after-parallelism warnings/deadlocks across 4 procs

# Fail fast with a clear message if the gated-model token is missing.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "FATAL: HF_TOKEN not set. Accept the gemma-3-4b-it license + create a token, then:"
    echo "       HF_TOKEN=hf_xxx sbatch slurm/run_gemma3.slurm"
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

MODEL="google/gemma-3-4b-it"
NSHARDS=4
echo "====== Gemma-3-4B run (base-vs-medical control) ======"; echo "Start: $(date)"; echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Prefetch the model ONCE before fanning out, so 4 workers read a warm cache instead of racing the download.
echo "--- Prefetch model snapshot ---"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
p = snapshot_download("google/gemma-3-4b-it", token=tok)
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
            --log-file "slurm/logs/gemma3_${tag}_shard${k}.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    return $rc
}

# Re-aggregate official results + paper tables + figures. Called after Stage 1 so the headline
# (MC-based: coverage, accuracy, balanced acc, ECE, conf-on-wrong) is on disk even if Stage 2's OE
# pass is later cut off by the wall clock. Called again after Stage 2 to add OE hallucination metrics.
aggregate_all () {
    local tag="$1"
    python -m src.analysis.aggregate --responses-dir raw_responses --manifest data/exam_set.csv \
        --tasks config/tasks.yaml --output-dir results 2>&1 | tee "slurm/logs/gemma3_aggregate_${tag}.log"
    python -m src.analysis.paper_aggregate --responses-dir raw_responses --manifest data/exam_set.csv \
        --tasks config/tasks.yaml --output-dir results/paper 2>&1 | tee "slurm/logs/gemma3_paper_${tag}.log"
    python -m src.analysis.make_figures --results-dir results --output-dir figures \
        --manifest data/exam_set.csv 2>&1 | tee "slurm/logs/gemma3_figures_${tag}.log"
    python -m src.analysis.paper_figures 2>&1 | tee "slurm/logs/gemma3_paperfigs_${tag}.log" || true
}

echo "--- Stage 1: neutral MC (4-GPU sharded) ---"
run_sharded "MC" "mc"
echo "--- Aggregate after Stage 1 (headline lands now) ---"
aggregate_all "mc"

echo "--- Stage 2: neutral MC+OE (resumes; MC rows skipped via per-shard dedup) ---"
run_sharded "MC OE" "oe"
echo "--- Aggregate after Stage 2 (adds OE hallucination metrics) ---"
aggregate_all "final"

echo "====== Gemma-3-4B run complete: $(date) ======"
echo "Check: results/paper/safety_summary.csv should now list Gemma-3-4B (tier A) alongside MedGemma-4B."
echo "The base-vs-medical delta (ECE, conf-on-wrong, balanced acc) is the controlled C3 result."
