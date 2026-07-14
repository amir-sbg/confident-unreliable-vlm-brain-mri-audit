"""
Defensive Weights & Biases wrapper.

Design goals:
  - NEVER crash the pipeline because of wandb (missing install, no auth, network).
    Every public function swallows exceptions and degrades to a no-op.
  - One results run named `fpsa_res` (override via WANDB_RESULTS_NAME) carries the
    final tables, figures, and headline metrics. aggregate.py and make_figures.py
    attach to the SAME run via a fixed id + resume="allow".
  - Parallel inference workers each open their own run (job_type="inference"),
    grouped under WANDB_RUN_GROUP (default "fpsa_res") so they show up together.

Env vars:
  WANDB_PROJECT          project name           (default "fpsa_res")
  WANDB_RUN_GROUP        group for all runs     (default "fpsa_res")
  WANDB_RESULTS_NAME     name of the results run(default "fpsa_res_results")
  WANDB_RESULTS_RUN_ID   fixed id for resume    (default "fpsa_res_results")
  WANDB_DISABLED=true    turn everything off
"""
from __future__ import annotations

import os
from pathlib import Path

from .logging_utils import get_logger

log = get_logger(__name__)

_run = None  # module-level handle to the active run (per process)


def _enabled() -> bool:
    return os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")


def init_wandb(role: str, name: str | None = None, config: dict | None = None):
    """Start a wandb run. role: 'results' (shared, resumable) | 'worker' (own run).
    Returns the run object or None. Never raises."""
    global _run
    if not _enabled():
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed; skipping experiment logging (pip install wandb).")
        return None

    project = os.environ.get("WANDB_PROJECT", "fpsa_res")
    group = os.environ.get("WANDB_RUN_GROUP", "fpsa_res")
    kwargs: dict = dict(project=project, group=group, config=config or {})

    if role == "results":
        kwargs["name"] = name or os.environ.get("WANDB_RESULTS_NAME", "fpsa_res_results")
        kwargs["id"] = os.environ.get("WANDB_RESULTS_RUN_ID", "fpsa_res_results")
        kwargs["resume"] = "allow"
        kwargs["job_type"] = "results"
    else:
        kwargs["name"] = name or "worker"
        kwargs["job_type"] = "inference"

    try:
        _run = wandb.init(**kwargs)
        log.info(f"wandb: started '{kwargs['name']}' (project={project}, group={group}, role={role})")
        return _run
    except Exception as e:
        log.warning(f"wandb.init failed ({e}); continuing without logging.")
        _run = None
        return None


def wandb_log(data: dict, step: int | None = None) -> None:
    if _run is None:
        return
    try:
        import wandb
        wandb.log(data, step=step)
    except Exception:
        pass


def wandb_summary(data: dict) -> None:
    if _run is None:
        return
    try:
        for k, v in data.items():
            _run.summary[k] = v
    except Exception:
        pass


def log_csv_table(path: str | Path, key: str) -> None:
    """Log a CSV file as a wandb.Table."""
    if _run is None:
        return
    try:
        import wandb
        import pandas as pd
        p = Path(path)
        if not p.exists():
            return
        df = pd.read_csv(p)
        wandb.log({key: wandb.Table(dataframe=df)})
    except Exception as e:
        log.warning(f"wandb table log failed for {path}: {e}")


def log_image(path: str | Path, key: str, caption: str | None = None) -> None:
    if _run is None:
        return
    try:
        import wandb
        p = Path(path)
        if not p.exists():
            return
        wandb.log({key: wandb.Image(str(p), caption=caption or Path(path).name)})
    except Exception:
        pass


def finish_wandb() -> None:
    global _run
    if _run is None:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass
    _run = None
