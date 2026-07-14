"""
Main inference script: manifest × prompts × models → raw_responses/{model}.jsonl

Features:
  - Resume: skips already-written (model, image_id, prompt_id) tuples
  - Model selection via --model or SLURM array index
  - Ablation flags: --phrasing, --format, --few-shot, --slice-fracs
  - Optional batching for vLLM models
  - Logs decoding config for reproducibility

Usage:
    python -m src.inference.run_inference \
        --models config/models.yaml \
        --tasks config/tasks.yaml \
        --prompts config/prompts.yaml \
        --manifest data/exam_set.csv \
        --output-dir raw_responses \
        [--model Qwen/Qwen2.5-VL-7B-Instruct] \
        [--phrasing neutral] \
        [--format MC OE] \
        [--slurm-array-index 0] \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.inference.base import make_runner
from src.prompts.render import render_all
from src.utils import (
    append_jsonl, load_done_keys, load_yaml, setup_logging, get_logger,
    init_wandb, wandb_log, wandb_summary, finish_wandb,
)

log = get_logger(__name__)

RESUME_KEY_FIELDS = ("model", "image_id", "prompt_id")


def selected_models(models_cfg: dict, args: argparse.Namespace) -> list[dict]:
    """Return the list of model configs to run, respecting CLI flags and SLURM array index."""
    all_models = models_cfg.get("models", [])

    # Filter by optional flag
    if not getattr(args, "include_optional", False):
        all_models = [m for m in all_models if not m.get("optional", False)]

    # Filter ablation_only unless explicitly requested
    if not getattr(args, "ablation_only", False):
        all_models = [m for m in all_models if not m.get("ablation_only", False)]

    # Explicit model filter
    if getattr(args, "model", None):
        all_models = [m for m in all_models if m["id"] == args.model]
        if not all_models:
            log.error(f"Model not found in registry: {args.model}")
            sys.exit(1)
        return all_models

    # SLURM array index selects one model
    array_idx = getattr(args, "slurm_array_index", None)
    if array_idx is not None:
        if array_idx >= len(all_models):
            log.error(f"Array index {array_idx} out of range (have {len(all_models)} models)")
            sys.exit(1)
        return [all_models[array_idx]]

    return all_models


def iter_prompt_rows(
    manifest: pd.DataFrame,
    prompt_rows: list[dict],
    formats: list[str],
    phrasings: list[str],
) -> list[dict]:
    """Filter and return prompt rows matching the requested formats/phrasings."""
    filtered = []
    for row in prompt_rows:
        if row["format"] not in formats:
            continue
        if row["phrasing"] not in phrasings:
            continue
        filtered.append(row)
    return filtered


def run_model(
    model_cfg: dict,
    prompt_rows: list[dict],
    output_dir: Path,
    dry_run: bool = False,
    batch_size: int = 1,
    out_suffix: str = "",
) -> None:
    """Run inference for one model over all prompt rows. Resumes if output file exists.

    out_suffix lets data-parallel shards write to distinct files
    (e.g. ``.shard0of4``) so concurrent processes never append to the same JSONL.
    """
    model_id    = model_cfg["id"]
    model_label = model_id.replace("/", "__")
    out_path    = output_dir / f"{model_label}{out_suffix}.jsonl"

    # Resume: load already-completed keys
    done_keys = load_done_keys(out_path, RESUME_KEY_FIELDS)
    log.info(f"{model_id}: {len(done_keys)} already done, {len(prompt_rows)} total prompts")

    pending = [
        r for r in prompt_rows
        if (model_id, r["image_id"], r["prompt_id"]) not in done_keys
    ]
    log.info(f"{model_id}: {len(pending)} prompts to run")

    if not pending:
        log.info(f"{model_id}: nothing to do, skipping")
        return

    if dry_run:
        log.info(f"[DRY RUN] Would run {len(pending)} items for {model_id}")
        return

    # Skip optional models if their API key is absent
    if model_cfg.get("optional") and model_cfg.get("backend") == "api":
        if not _api_key_present(model_cfg.get("provider", "")):
            log.info(f"Skipping optional API model {model_id} (no API key)")
            return

    runner = make_runner(model_cfg)
    try:
        runner.load()
    except Exception as e:
        log.error(f"Failed to load {model_id}: {e}")
        return

    system_prompt = pending[0]["system_prompt"] if pending else ""

    # Per-worker wandb run (own run, grouped under WANDB_RUN_GROUP) for live progress.
    init_wandb(
        role="worker",
        name=f"fpsa_res_{_wandb_model_tag(model_id)}{out_suffix.replace('.', '_')}",  # +_shard0of4 when sharded
        config={
            "model": model_id,
            "backend": model_cfg.get("backend"),
            "tier": model_cfg.get("tier"),
            "n_pending": len(pending),
            "n_total_prompts": len(prompt_rows),
        },
    )

    # vLLM supports efficient batching
    use_batch = model_cfg.get("backend") == "vllm" and batch_size > 1
    chunks = _batch_list(pending, batch_size) if use_batch else [[r] for r in pending]

    n_pending = len(pending)
    total_done = 0
    t_start = time.perf_counter()
    for chunk in chunks:
        if use_batch:
            results = _run_batch(runner, chunk, model_cfg)
        else:
            results = [_run_single(runner, chunk[0], model_cfg)]

        for row, result in zip(chunk, results):
            record = _make_record(model_id, row, result)
            append_jsonl(out_path, record)
            total_done += 1

        if total_done % 25 == 0:
            elapsed = max(1e-6, time.perf_counter() - t_start)
            rate = total_done / elapsed
            eta_min = (n_pending - total_done) / rate / 60.0 if rate > 0 else 0.0
            log.info(f"  {model_id}: {total_done}/{n_pending} done ({rate:.1f}/s, ETA {eta_min:.1f}m)")
            wandb_log({
                "done": total_done,
                "frac_complete": total_done / max(1, n_pending),
                "items_per_sec": rate,
                "eta_min": eta_min,
            })

    log.info(f"{model_id}: completed {total_done} items -> {out_path}")
    wandb_summary({"completed_items": total_done, "n_pending": n_pending, "model": model_id})
    finish_wandb()


def _run_single(runner, row: dict, model_cfg: dict) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        result = runner.generate(
            image_path=row["image_path"],
            prompt=row["prompt_text"],
            system=row["system_prompt"],
            max_tokens=model_cfg.get("max_tokens"),
        )
    except Exception as e:
        log.error(f"Generate error: {e}")
        result = {"text": "", "raw": str(e), "gen_seconds": time.perf_counter() - t0}
    return result


def _run_batch(runner, rows: list[dict], model_cfg: dict) -> list[dict[str, Any]]:
    items = [(r["image_path"], r["prompt_text"], r["system_prompt"]) for r in rows]
    try:
        return runner.generate_batch(items, max_tokens=model_cfg.get("max_tokens"))
    except AttributeError:
        # Runner doesn't implement generate_batch; fall back to sequential
        return [_run_single(runner, r, model_cfg) for r in rows]
    except Exception as e:
        log.error(f"Batch generate error: {e}")
        return [{"text": "", "raw": str(e), "gen_seconds": 0.0}] * len(rows)


def _make_record(model_id: str, row: dict, result: dict) -> dict:
    return {
        "model":          model_id,
        "image_id":       row["image_id"],
        "prompt_id":      row["prompt_id"],
        "task":           row["task"],
        "phrasing":       row["phrasing"],
        "format":         row["format"],
        "mc_options":     row.get("mc_options", []),
        "mc_option_order": row.get("mc_option_order", []),
        "mc_correct_letter": row.get("mc_correct_letter", ""),
        "image_path":     row.get("image_path", ""),
        "response":       result.get("text", ""),
        "gen_seconds":    round(result.get("gen_seconds", 0.0), 3),
        "timestamp_utc":  _utc_iso(),
    }


def _utc_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _api_key_present(provider: str) -> bool:
    key_map = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    return bool(os.environ.get(key_map.get(provider, ""), ""))


def _batch_list(lst: list, n: int) -> list[list]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def _wandb_model_tag(model_id: str) -> str:
    """Readable run-name suffix from a model id, e.g.
    'Qwen/Qwen2.5-VL-7B-Instruct' -> 'qwen2.5-vl-7b'."""
    tag = model_id.split("/")[-1].lower()
    for suf in ("-instruct", "-it", "-chat", "-hf"):
        if tag.endswith(suf):
            tag = tag[: -len(suf)]
    tag = "".join(c if (c.isalnum() or c in ".-_") else "-" for c in tag)
    return tag.strip("-") or "model"


def enrich_prompt_rows(prompt_rows: list[dict], manifest: pd.DataFrame) -> list[dict]:
    """Add image_path from manifest into each prompt row."""
    path_map = dict(zip(manifest["image_id"], manifest["path"]))
    for row in prompt_rows:
        row["image_path"] = path_map.get(row["image_id"], "")
    return [r for r in prompt_rows if r["image_path"]]


def main():
    parser = argparse.ArgumentParser(description="Run VLM inference over exam_set.csv.")
    parser.add_argument("--models",   default="config/models.yaml")
    parser.add_argument("--tasks",    default="config/tasks.yaml")
    parser.add_argument("--prompts",  default="config/prompts.yaml")
    parser.add_argument("--manifest", default="data/exam_set.csv")
    parser.add_argument("--output-dir", default="raw_responses")
    parser.add_argument("--model",    default=None, help="Run only this model id.")
    parser.add_argument("--phrasing", nargs="+", default=["neutral"],
                        choices=["neutral", "terse", "clinician"],
                        help="Which prompt phrasings to run (ablation A1).")
    parser.add_argument("--format",   nargs="+", default=["MC", "OE"],
                        choices=["MC", "OE"])
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Data-parallel sharding: split prompts into N disjoint shards by a "
                             "stable hash of (image_id|prompt_id). Launch one process per GPU, "
                             "each with a different --shard-index, to get ~Nx throughput.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which shard in [0, num_shards) this process handles.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap prompt rows (after sharding) to this many. For smoke-testing a "
                             "new model on a handful of items before a full run.")
    parser.add_argument("--slurm-array-index", type=int, default=None,
                        help="SLURM array task index selects one model.")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="vLLM batch size (images per forward pass).")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print what would run, but don't call the model.")
    parser.add_argument("--list-models", action="store_true",
                        help="Print the selected model ids (one per line) and exit. "
                             "Used by the SLURM launcher to fan out one model per GPU.")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    # --list-models: enumerate selected models (respects --include-optional/--ablation-only) and exit.
    if args.list_models:
        models_cfg = load_yaml(args.models)
        for m in selected_models(models_cfg, args):
            print(m["id"])
        return

    log_file = args.log_file or f"logs/inference_{args.model or 'all'}.log"
    setup_logging(log_file=log_file)

    models_cfg  = load_yaml(args.models)
    tasks_cfg   = load_yaml(args.tasks)
    prompts_cfg = load_yaml(args.prompts)
    manifest    = pd.read_csv(args.manifest)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Manifest: {len(manifest)} rows")

    # Render all prompts (fast, no GPU)
    log.info("Rendering prompt matrix...")
    all_prompt_rows = render_all(prompts_cfg, tasks_cfg, manifest, phrasings=["neutral", "terse", "clinician"])
    all_prompt_rows = enrich_prompt_rows(all_prompt_rows, manifest)
    # Filter to requested phrasings and formats
    filtered_rows = iter_prompt_rows(manifest, all_prompt_rows, args.format, args.phrasing)
    log.info(f"Filtered prompt rows: {len(filtered_rows)}")

    # Data-parallel sharding: keep only the rows that belong to this shard. We hash a STABLE
    # key (md5 of image_id|prompt_id) rather than Python's hash(), which is salted per process
    # (PYTHONHASHSEED) and would make shards overlap/miss across the 4 GPU workers. Hashing the
    # prompt identity (not a list index) also keeps each prompt's shard fixed regardless of which
    # formats are present, so a later MC+OE stage lands in the same shard file as the MC stage and
    # resume-dedup works.
    out_suffix = ""
    if args.num_shards > 1:
        import hashlib
        def _shard_of(r) -> int:
            key = f"{r['image_id']}|{r['prompt_id']}".encode()
            return int(hashlib.md5(key).hexdigest(), 16) % args.num_shards
        before = len(filtered_rows)
        filtered_rows = [r for r in filtered_rows if _shard_of(r) == args.shard_index]
        out_suffix = f".shard{args.shard_index}of{args.num_shards}"
        log.info(f"Shard {args.shard_index}/{args.num_shards}: {len(filtered_rows)}/{before} rows")

    if args.limit is not None and args.limit > 0:
        filtered_rows = filtered_rows[:args.limit]
        log.info(f"--limit: truncated to {len(filtered_rows)} rows (smoke test)")

    # Select models
    models_to_run = selected_models(models_cfg, args)
    log.info(f"Models to run: {[m['id'] for m in models_to_run]}")

    for model_cfg in models_to_run:
        log.info(f"=== {model_cfg['id']} ===")
        run_model(
            model_cfg=model_cfg,
            prompt_rows=filtered_rows,
            output_dir=output_dir,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            out_suffix=out_suffix,
        )

    log.info("Inference complete.")


if __name__ == "__main__":
    main()
