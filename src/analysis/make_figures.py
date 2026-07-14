"""
Generate all publication-ready figures from results/*.csv.

Reads only CSV files (+ manifest for failure gallery image paths) — no GPU.

Outputs:
  figures/accuracy_by_task.png      Figure 1
  figures/reliability.png           Figure 2
  figures/failure_gallery.png       Figure 3
  figures/ablation_A1_phrasing.png
  figures/ablation_A2_format.png
  figures/ablation_A6_scale.png
  figures/ablation_A7_slice_pos.png

Usage:
    python -m src.analysis.make_figures \
        --results-dir results \
        --output-dir figures \
        [--manifest data/exam_set.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for cluster
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import setup_logging, get_logger

log = get_logger(__name__)

# ── Style ──────────────────────────────────────────────────────────────────────

TIER_COLORS = {"A": "#4E79A7", "B": "#F28E2B", "C": "#59A14F"}
MODEL_COLORS = plt.cm.tab10.colors
FONT_SIZE = 11
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

TASK_LABELS = {
    "T1-MOD": "Modality", "T2-PLANE": "Plane", "T3-ISBRAIN": "Brain?",
    "T4-TUMOR": "Tumor Det.", "T5-LAT": "Laterality",
    "T6-VQA": "VQA", "T7-ABSTAIN": "Abstain",
}

MODEL_TIER_MAP: dict[str, str] = {}   # filled from main_accuracy.csv if 'tier' column present

# Per-task random-chance baseline = 1 / |answer space| (drawn per task group, not one global line).
RANDOM_BASELINE = {
    "T1-MOD": 1/7, "T2-PLANE": 1/3, "T3-ISBRAIN": 1/2, "T4-TUMOR": 1/2, "T5-LAT": 1/4,
}


def _model_color(model: str, i: int) -> str:
    tier = MODEL_TIER_MAP.get(model, "A")
    return TIER_COLORS.get(tier, MODEL_COLORS[i % len(MODEL_COLORS)])


def _short_model(model_id: str) -> str:
    """Shorten model ID for legend labels."""
    parts = model_id.split("/")
    return parts[-1].replace("-Instruct", "").replace("-it", "")[:22]


# ── Figure 1: Accuracy by Task ────────────────────────────────────────────────

def plot_accuracy_by_task(
    accuracy_csv: Path,
    output_path: Path,
) -> None:
    """Grouped bar chart: accuracy by task, colored by model tier."""
    df = pd.read_csv(accuracy_csv)
    if df.empty:
        log.warning("main_accuracy.csv is empty; skipping Figure 1.")
        return

    # Main run: neutral phrasing, MC format (or best available)
    df_mc = df[df["format"] == "MC"] if "format" in df.columns else df
    # Use only tasks that have a defined chance baseline (drops empty T6/T7 columns), and only
    # models that actually produced gradeable answers (drops degenerate ~0-coverage models).
    df_mc = df_mc[df_mc["task"].isin(RANDOM_BASELINE)]
    answered_col = "acc_n" if "acc_n" in df_mc.columns else None
    if answered_col:
        # Drop degenerate models (e.g. InternVL2.5-2B answers ~1 item): require a real answered
        # count on at least one task. Threshold well above the degenerate model's max (=1).
        usable = df_mc.groupby("model")[answered_col].max()
        keep = set(usable[usable > 100].index)
        df_mc = df_mc[df_mc["model"].isin(keep)]

    tasks  = sorted(df_mc["task"].unique(), key=lambda t: list(TASK_LABELS.keys()).index(t)
                    if t in TASK_LABELS else 99)
    models = sorted(df_mc["model"].unique())

    # Update tier map
    if "tier" in df_mc.columns:
        for _, row in df_mc.iterrows():
            MODEL_TIER_MAP[row["model"]] = row.get("tier", "A")

    n_tasks  = len(tasks)
    n_models = len(models)
    bar_w    = 0.8 / n_models
    xs       = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 1.5), 5))
    for i, model in enumerate(models):
        mdf = df_mc[df_mc["model"] == model]
        accs, ci_los, ci_his = [], [], []
        for task in tasks:
            row = mdf[mdf["task"] == task]
            if row.empty:
                accs.append(0)
                ci_los.append(0)
                ci_his.append(0)
            else:
                r = row.iloc[0]
                accs.append(r.get("acc_accuracy", float("nan")))
                ci_los.append(r.get("acc_accuracy", 0) - r.get("acc_ci_lo", r.get("acc_accuracy", 0)))
                ci_his.append(r.get("acc_ci_hi", r.get("acc_accuracy", 0)) - r.get("acc_accuracy", 0))

        offset = (i - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(
            xs + offset, accs, bar_w * 0.9,
            label=_short_model(model),
            color=_model_color(model, i),
            alpha=0.85,
        )
        ax.errorbar(
            xs + offset, accs,
            yerr=[ci_los, ci_his],
            fmt="none", color="black", capsize=3, linewidth=1,
        )

    # Per-task random-chance baseline: short horizontal segment spanning each task's bar group.
    for j, task in enumerate(tasks):
        base = RANDOM_BASELINE.get(task)
        if base is None:
            continue
        ax.plot([j - 0.45, j + 0.45], [base, base], ls="--", lw=1.0, color="gray",
                label="Random chance (per task)" if j == 0 else None)

    ax.set_xticks(xs)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (MC, neutral)")
    ax.set_title("VLM Brain-MRI Audit — Accuracy by Task (vs. per-task chance)")
    ax.grid(axis="y", alpha=0.3)
    # Legend: only the models actually plotted + the chance line (no empty tier patches).
    ax.legend(loc="upper right", ncol=2, framealpha=0.75)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    log.info(f"Figure 1 saved: {output_path}")


# ── Figure 2: Reliability Diagram ─────────────────────────────────────────────

def plot_reliability_diagram(
    calibration_csv: Path,
    graded_csv: Path,
    output_path: Path,
    n_models: int = 3,
) -> None:
    """Reliability (calibration) diagram for up to n_models models."""
    from src.scoring.metrics import compute_reliability_diagram_data

    if not graded_csv.exists():
        log.warning(f"{graded_csv} not found; skipping Figure 2.")
        return

    graded_df = pd.read_csv(graded_csv)
    cal_df    = pd.read_csv(calibration_csv) if calibration_csv.exists() else pd.DataFrame()

    # Select top-N models by ECE (best calibrated get shown).
    # calibration.csv writes the column as "ece" (build_calibration_csv), not "ece_ece".
    if not cal_df.empty and "ece" in cal_df.columns:
        top_models = cal_df.nsmallest(n_models, "ece")["model"].tolist()
    else:
        top_models = graded_df["model"].unique()[:n_models].tolist()

    bin_data = compute_reliability_diagram_data(graded_df, models=top_models)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")

    for i, model in enumerate(top_models):
        bins = bin_data.get(model, [])
        if not bins:
            continue
        confs = [b["conf"] for b in bins]
        accs  = [b["acc"]  for b in bins]
        color = _model_color(model, i)
        ax.plot(confs, accs, "o-", color=color, lw=1.5, ms=5, label=_short_model(model))

        # ECE annotation
        ece_val = None
        if not cal_df.empty:
            row = cal_df[cal_df["model"] == model]
            if not row.empty:
                ece_val = row.iloc[0].get("ece")
        if ece_val is not None:
            ax.annotate(f"ECE={ece_val:.3f}", xy=(confs[-1], accs[-1]),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=8, color=color)

    ax.set_xlabel("Mean confidence (bin)")
    ax.set_ylabel("Accuracy (bin)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability Diagram — VLM Confidence Calibration")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    log.info(f"Figure 2 saved: {output_path}")


# ── Figure 3: Failure Gallery ─────────────────────────────────────────────────

def plot_failure_gallery(
    graded_csv: Path,
    manifest_path: Path,
    output_path: Path,
    n_panels: int = 6,
    seed: int = 42,
) -> None:
    """
    Grid of n_panels failure cases: MRI slice + model's confident wrong statement + GT.
    Selects confidently_wrong=True cases across diverse tasks/models.
    """
    if not graded_csv.exists():
        log.warning(f"{graded_csv} not found; skipping Figure 3.")
        return

    graded_df = pd.read_csv(graded_csv)
    manifest  = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()

    # Filter to confidently wrong OE cases with non-empty response
    fails = graded_df[
        graded_df.get("confidently_wrong", pd.Series(dtype=bool)).fillna(False) &
        (graded_df.get("format", pd.Series(dtype=str)) == "OE") &
        graded_df.get("response", pd.Series(dtype=str)).notna()
    ]

    if fails.empty:
        log.warning("No confidently-wrong OE cases found; using random incorrect instead.")
        fails = graded_df[~graded_df.get("correct", pd.Series(dtype=bool)).fillna(True)]

    if fails.empty:
        log.warning("No failure cases; skipping Figure 3.")
        return

    # Sample diverse across tasks
    fails = fails.copy()
    rng = np.random.default_rng(seed)
    tasks = fails["task"].unique()
    sampled = []
    per_task = max(1, n_panels // len(tasks))
    for task in tasks:
        sub = fails[fails["task"] == task]
        n = min(per_task, len(sub))
        idx = rng.choice(len(sub), size=n, replace=False)
        sampled.extend(sub.iloc[idx].to_dict(orient="records"))

    sampled = sampled[:n_panels]

    # Build image path map
    if not manifest.empty:
        path_map = dict(zip(manifest["image_id"], manifest["path"]))
    else:
        path_map = graded_df.set_index("image_id")["image_path"].to_dict() if "image_path" in graded_df.columns else {}

    import textwrap
    n_cols = min(3, len(sampled)) or 1
    n_rows = int(np.ceil(len(sampled) / n_cols))
    # Taller cells + a reserved caption strip under each panel so quotes are never clipped.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.6 * n_rows))
    axes = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]

    for ax_idx, (rec, ax) in enumerate(zip(sampled, axes)):
        image_id = rec.get("image_id", "")
        img_path = path_map.get(image_id, "")

        if img_path and Path(img_path).exists():
            img = Image.open(img_path).convert("L")
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(0, 1, 0.32, 1.0))
        else:
            ax.text(0.5, 0.66, "image not found", ha="center", va="center",
                    color="#888", transform=ax.transAxes, fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        model_short = _short_model(str(rec.get("model", "")))
        task_label  = TASK_LABELS.get(rec.get("task", ""), rec.get("task", ""))
        gt, pred, conf = rec.get("gt_label", "?"), rec.get("pred_answer", "?"), rec.get("confidence", "?")
        ax.set_title(f"[{task_label}] {model_short}", fontsize=9, color="crimson", pad=3)
        ax.text(0.5, 0.30, f"GT: {gt}   |   Pred: {pred}   |   Conf: {conf}%",
                transform=ax.transAxes, fontsize=8, color="crimson", ha="center", va="top")
        # Full model statement, word-wrapped in the reserved strip (no truncation/clipping).
        stmt = " ".join(str(rec.get("response", "")).split())
        wrapped = "\n".join(textwrap.wrap(stmt, width=58)[:5]) or "(empty response)"
        ax.text(0.02, 0.24, f'“{wrapped}”', transform=ax.transAxes, fontsize=7,
                color="#222", ha="left", va="top")

    for ax in axes[len(sampled):]:
        ax.set_visible(False)

    fig.suptitle("Failure Gallery: Confident Wrong Responses", fontsize=13, y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure 3 saved: {output_path}")


# ── Ablation figures ──────────────────────────────────────────────────────────

def plot_ablation(
    csv_path: Path,
    x_col: str,
    group_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    if not csv_path.exists():
        log.warning(f"Ablation CSV not found: {csv_path}; skipping.")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = df[group_col].unique()
    for i, grp in enumerate(sorted(groups)):
        sub = df[df[group_col] == grp].sort_values(x_col)
        label = _short_model(str(grp)) if group_col == "model" else str(grp)
        ax.plot(sub[x_col].astype(str), sub[y_col], "o-", label=label,
                color=MODEL_COLORS[i % len(MODEL_COLORS)])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    log.info(f"Ablation figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate all result figures.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir",  default="figures")
    parser.add_argument("--manifest",    default="data/exam_set.csv")
    args = parser.parse_args()

    setup_logging(log_file="logs/make_figures.log")
    R = Path(args.results_dir)
    F = Path(args.output_dir)
    F.mkdir(parents=True, exist_ok=True)

    plot_accuracy_by_task(R / "main_accuracy.csv",  F / "accuracy_by_task.png")
    plot_reliability_diagram(
        R / "calibration.csv", R / "graded_all.csv", F / "reliability.png"
    )
    plot_failure_gallery(
        R / "graded_all.csv", Path(args.manifest), F / "failure_gallery.png"
    )

    abl = R / "ablations"
    plot_ablation(abl / "ablation_A1_phrasing.csv",  "phrasing", "model", "accuracy",
                  "A1: Accuracy vs Prompt Phrasing", "Phrasing", F / "ablation_A1_phrasing.png")
    plot_ablation(abl / "ablation_A2_format.csv",    "format",   "model", "accuracy",
                  "A2: Accuracy by Answer Format (MC vs OE)", "Format", F / "ablation_A2_format.png")
    plot_ablation(abl / "ablation_A6_scale.csv",     "params_b", "task",  "accuracy",
                  "A6: Accuracy vs Model Scale", "Parameters (B)", F / "ablation_A6_scale.png")
    plot_ablation(abl / "ablation_A7_slice_pos.csv", "slice_frac", "model", "accuracy",
                  "A7: Accuracy vs Slice Position", "Slice Fraction", F / "ablation_A7_slice_pos.png")

    # ── Log figures to the shared `fpsa_res` wandb run ────────────────────────
    from src.utils import init_wandb, log_image, finish_wandb
    init_wandb(role="results")   # name -> WANDB_RESULTS_NAME (fpsa_res_results)
    log_image(F / "accuracy_by_task.png", "figures/accuracy_by_task")
    log_image(F / "reliability.png",      "figures/reliability")
    log_image(F / "failure_gallery.png",  "figures/failure_gallery")
    for p in sorted(F.glob("ablation_*.png")):
        log_image(p, f"figures/{p.stem}")
    finish_wandb()

    log.info("All figures complete.")


if __name__ == "__main__":
    main()
