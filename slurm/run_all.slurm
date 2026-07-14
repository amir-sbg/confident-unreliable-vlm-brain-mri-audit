#!/bin/bash
#SBATCH --job-name=vlmaudit_all
#SBATCH --partition=qTRDGPUH
#SBATCH -w arctrddgxa003
#SBATCH --account=psy53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8             # Stage 4 fans out one model per GPU (parallel inference)
#SBATCH --time=48:00:00
#SBATCH --chdir=/data/users2/asabbaghziarani1/vlm-brainmri-audit
#SBATCH --output=slurm/logs/run_all_%j.out
#SBATCH --error=slurm/logs/run_all_%j.err
# Optional: pin to a specific GPU node (uncomment if your scheduler needs it)

# ── Idempotent, resumable pipeline ────────────────────────────────────────────
# Each stage checks for its output before running, so resubmitting resumes cleanly.
# Inference uses JSONL append with key-based dedup for crash recovery.

set -euo pipefail

# Hardcoded (robust): SLURM runs a spool copy of this script, so ${BASH_SOURCE[0]}
# does NOT point at slurm/ — deriving the path from it lands in an unwritable spool dir.
PROJECT_DIR="/data/users2/asabbaghziarani1/vlm-brainmri-audit"
cd "$PROJECT_DIR"
mkdir -p logs slurm/logs results results/ablations figures raw_responses data

# ── Environment ───────────────────────────────────────────────────────────────
# conda is not on PATH in a non-interactive SLURM shell; source it by absolute path.
source /home/users/asabbaghziarani1/miniconda3/etc/profile.d/conda.sh
conda activate vlmaudit

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_DIR}/cache/hf"
export TRANSFORMERS_CACHE="${HF_HOME}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# FlashInfer's sampler JIT-compiles a CUDA kernel needing nvcc/CUDA_HOME (absent on these
# GPU nodes — runtime only, no toolkit), which crashed engine init with "Could not find nvcc".
# We decode greedily, so the native Torch sampler is equivalent. (Also set in vllm_runner.py.)
# Fallback if a different FlashInfer JIT ever surfaces: `module load cuda/12.1` before running.
export VLLM_USE_FLASHINFER_SAMPLER=0

# ── Hugging Face token for GATED models ───────────────────────────────────────
# meta-llama/Llama-3.2-11B-Vision-Instruct and google/medgemma-4b-it are GATED:
# they require accepting each model's license on huggingface.co (under the account
# whose token you use) AND a token here. Without it, those two models are SKIPPED
# (non-fatal — their workers log the 401 and exit; the other models still run).
# To include them, accept both licenses on HF, then uncomment:
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
if [ -z "${HF_TOKEN:-}" ]; then
    echo "NOTE: HF_TOKEN not set — gated models (Llama-3.2-11B-Vision, medgemma-4b-it) will be skipped."
fi

# ── Weights & Biases (real-time monitoring) ───────────────────────────────────
# Run `wandb login` once before submitting (or export WANDB_API_KEY).
# To turn off entirely:  export WANDB_DISABLED=true
export WANDB_PROJECT="fpsa_res"              # project name
export WANDB_RUN_GROUP="fpsa_res"            # parallel inference workers group here
export WANDB_RESULTS_NAME="fpsa_res_results"  # the headline run (results + figures)
export WANDB_RESULTS_RUN_ID="fpsa_res_results"
# WANDB_DIR is named "wandb" and sits on PYTHONPATH, so a bare `import wandb` can resolve
# to this empty dir as a namespace package (wandb.init missing) — which also masks a missing
# install. Check for the real attribute, not just importability, before deciding to install.
export WANDB_DIR="${PROJECT_DIR}/wandb"
mkdir -p "$WANDB_DIR"
python -c "import wandb, sys; sys.exit(0 if hasattr(wandb, 'init') else 1)" 2>/dev/null \
    || pip install -q "wandb>=0.16"

# API keys (set via SLURM environment or secrets manager; empty = skip API models)
# export OPENAI_API_KEY=""
# export ANTHROPIC_API_KEY=""
# export GOOGLE_API_KEY=""

echo "====== VLM Brain-MRI Audit Pipeline ======"
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Stage 0: Verify datasets ──────────────────────────────────────────────────
echo ""
echo "--- Stage 0: Dataset check ---"
python -m src.data.download --check-only --dataset brats ixi oasis 2>&1 | tee slurm/logs/stage0_check.log

# ── Stage 1: Slice extraction ─────────────────────────────────────────────────
if [ ! -f "data/slice_provenance.jsonl" ]; then
    echo ""
    echo "--- Stage 1: Slice extraction ---"
    python -m src.data.slice_extract \
        --config config/datasets.yaml \
        --dataset brats ixi oasis negative_controls \
        --output-dir data/slices \
        --provenance-out data/slice_provenance.jsonl \
        2>&1 | tee slurm/logs/stage1_extract.log
else
    echo "--- Stage 1: Slice provenance exists, skipping ---"
fi

# ── Stage 2: Build manifest ───────────────────────────────────────────────────
if [ ! -f "data/exam_set.csv" ]; then
    echo ""
    echo "--- Stage 2: Build exam_set.csv ---"
    python -m src.data.build_manifest \
        --provenance data/slice_provenance.jsonl \
        --config config/datasets.yaml \
        --output data/exam_set.csv \
        2>&1 | tee slurm/logs/stage2_manifest.log
else
    echo "--- Stage 2: exam_set.csv exists, skipping ---"
fi

# Verify manifest
N_IMAGES=$(python -c "import pandas as pd; print(len(pd.read_csv('data/exam_set.csv')))")
echo "Manifest: $N_IMAGES images"
if [ "$N_IMAGES" -lt 100 ]; then
    echo "WARNING: Fewer than 100 images in exam_set.csv — check data availability."
fi

# ── Stage 3: Parser tests ─────────────────────────────────────────────────────
echo ""
echo "--- Stage 3: Parser unit tests ---"
python -m pytest tests/test_parse.py -v --tb=short 2>&1 | tee slurm/logs/stage3_tests.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "FATAL: Parser tests failed. Fix before scaling."
    exit 1
fi

# ── Stage 4: Inference — all non-optional models, ONE PER GPU in parallel ─────
# Each model writes its own raw_responses/<model>.jsonl, so concurrent workers
# never collide and a rerun resumes any worker that died. Tier A + B are both
# covered here (--list-models returns every non-optional, non-ablation model).
echo ""
echo "--- Stage 4: Parallel inference (neutral phrasing, MC+OE) ---"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
echo "Detected $NGPU GPU(s)"

mapfile -t MODELS < <(python -m src.inference.run_inference --models config/models.yaml --list-models)
if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "FATAL: no models returned by --list-models"; exit 1
fi
echo "Models (${#MODELS[@]}): ${MODELS[*]}"

declare -a PIDS=()
idx=0
for m in "${MODELS[@]}"; do
    gpu=$(( idx % NGPU ))
    label=$(echo "$m" | tr '/' '_')
    echo "  [GPU $gpu] launching $m"
    CUDA_VISIBLE_DEVICES=$gpu python -m src.inference.run_inference \
        --models     config/models.yaml \
        --tasks      config/tasks.yaml \
        --prompts    config/prompts.yaml \
        --manifest   data/exam_set.csv \
        --output-dir raw_responses \
        --model      "$m" \
        --phrasing   neutral \
        --format     MC OE \
        --batch-size 8 \
        --log-file   "slurm/logs/infer_${label}.log" \
        > "slurm/logs/infer_${label}.out" 2>&1 &
    PIDS+=($!)
    idx=$(( idx + 1 ))
    # Throttle: keep at most NGPU workers running at once.
    while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do sleep 10; done
done

# Wait for all workers; tolerate individual failures (rerun resumes them).
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "WARNING: inference worker (pid $pid) exited non-zero — rerun resumes it"
done
echo "Stage 4 parallel inference complete."

# ── Stage 5: Ablation A1 — terse + clinician phrasings on one model ───────────
echo ""
echo "--- Stage 5: Ablation A1 (phrasings) ---"
ABLATION_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
CUDA_VISIBLE_DEVICES=0 python -m src.inference.run_inference \
    --models     config/models.yaml \
    --tasks      config/tasks.yaml \
    --prompts    config/prompts.yaml \
    --manifest   data/exam_set.csv \
    --output-dir raw_responses \
    --model      "$ABLATION_MODEL" \
    --phrasing   terse clinician \
    --format     MC OE \
    --batch-size 8 \
    2>&1 | tee slurm/logs/stage5_ablation_A1.log \
    || echo "WARNING: Stage 5 (ablation A1) failed — non-fatal, continuing to aggregate/figures"

# ── Stage 7: Optional Tier C API models ──────────────────────────────────────
if [ -n "${OPENAI_API_KEY:-}" ] || [ -n "${ANTHROPIC_API_KEY:-}" ] || [ -n "${GOOGLE_API_KEY:-}" ]; then
    echo ""
    echo "--- Stage 7: Inference (Tier C API models) ---"
    python -m src.inference.run_inference \
        --models          config/models.yaml \
        --tasks           config/tasks.yaml \
        --prompts         config/prompts.yaml \
        --manifest        data/exam_set.csv \
        --output-dir      raw_responses \
        --include-optional \
        --phrasing        neutral \
        --format          MC OE \
        2>&1 | tee slurm/logs/stage7_inference_C.log
else
    echo "--- Stage 7: No API keys found; skipping Tier C ---"
fi

# ── Stage 8: Aggregate + metrics ─────────────────────────────────────────────
echo ""
echo "--- Stage 8: Aggregate results ---"
python -m src.analysis.aggregate \
    --responses-dir raw_responses \
    --manifest      data/exam_set.csv \
    --tasks         config/tasks.yaml \
    --output-dir    results \
    2>&1 | tee slurm/logs/stage8_aggregate.log

# ── Stage 9: Figures ─────────────────────────────────────────────────────────
echo ""
echo "--- Stage 9: Generate figures ---"
python -m src.analysis.make_figures \
    --results-dir results \
    --output-dir  figures \
    --manifest    data/exam_set.csv \
    2>&1 | tee slurm/logs/stage9_figures.log \
    || echo "WARNING: Stage 9 (figures) failed — non-fatal, results CSVs are already written"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "====== Pipeline complete ======"
echo "End: $(date)"
echo ""
echo "Key outputs:"
for f in results/main_accuracy.csv results/calibration.csv results/hallucination.csv \
          results/parser_audit_sample.csv figures/accuracy_by_task.png \
          figures/reliability.png figures/failure_gallery.png; do
    if [ -f "$f" ]; then
        echo "  ✓ $f ($(wc -l < "$f") lines)"
    else
        echo "  ✗ $f MISSING"
    fi
done
