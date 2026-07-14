#!/bin/bash
#SBATCH --job-name=vlmaudit_gemma4_smoke
#SBATCH --partition=qTRDGPUH
#SBATCH -w arctrddgxa002
#SBATCH --account=psy53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --chdir=/data/users2/asabbaghziarani1/vlm-brainmri-audit
#SBATCH --output=slurm/logs/gemma4_smoke_%j.out
#SBATCH --error=slurm/logs/gemma4_smoke_%j.err

# ── SMOKE TEST: can our hf_runner load + run google/gemma-4-12B-it (new gemma4_unified arch)? ──────
# Gemma-4 is a NEW architecture (gemma4_unified, any-to-any). Before committing to the ~9 h full run,
# load the 12B model on ONE GPU and push 6 real (image,prompt) MC rows through the EXACT code path the
# full run uses (HFRunner.load + generate with the image embedded in the chat message content). Then
# print the raw responses so we can see whether outputs are non-empty / sensible / parseable.
#
# Writes to a THROWAWAY dir (smoke_out/) so it never contaminates the real raw_responses/ aggregation.
# gemma-4 is ungated, so HF_TOKEN is OPTIONAL here (passed if present to avoid anon rate limits).

set -euo pipefail
PROJECT_DIR="/data/users2/asabbaghziarani1/vlm-brainmri-audit"
cd "$PROJECT_DIR"
mkdir -p logs slurm/logs smoke_out

source /home/users/asabbaghziarani1/miniconda3/etc/profile.d/conda.sh
conda activate vlmaudit
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_DIR}/cache/hf"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

MODEL="google/gemma-4-12B-it"
echo "====== Gemma-4-12B-it SMOKE ======"; echo "Start: $(date)"; echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo "--- Run 6 neutral-MC prompts through the real hf_runner path ---"
python -m src.inference.run_inference \
    --models config/models.yaml --tasks config/tasks.yaml --prompts config/prompts.yaml \
    --manifest data/exam_set.csv --output-dir smoke_out \
    --include-optional --model "$MODEL" --phrasing neutral --format MC \
    --limit 6 --batch-size 1 --log-file slurm/logs/gemma4_smoke.log 2>&1

echo "--- Inspect outputs (non-empty? sensible?) ---"
python - <<'PY'
import json, glob, sys
fs = glob.glob("smoke_out/*gemma-4-12B-it*.jsonl")
if not fs:
    print("FAIL: no smoke output file written — load/generate likely crashed (see log above)"); sys.exit(1)
rows = [json.loads(l) for l in open(fs[0])]
ne = sum(1 for r in rows if (r.get("response") or "").strip())
print(f"rows={len(rows)}  non_empty={ne}")
for r in rows:
    t = " ".join((r.get("response") or "").split())
    print(f"\n[{r.get('task')}] {r.get('gen_seconds')}s  response: {t[:200]}")
print("\nVERDICT:", "PASS — processor/chat-template OK; safe to launch the full run"
      if ne >= 4 else
      "FAIL/INVESTIGATE — empty or broken outputs; do NOT launch the full run until fixed")
PY
echo "====== Smoke complete: $(date) ======"
echo "If PASS:  sbatch slurm/run_gemma4.slurm    (add HF_TOKEN=hf_xxx to be safe; model is ungated)"
echo "Cleanup:  rm -rf smoke_out/"
