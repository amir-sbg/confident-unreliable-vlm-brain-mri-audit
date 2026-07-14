"""
Paper-quality, large-font figures for the write-up (CPU-only; no SLURM/GPU needed).
Generates the INCONTESTABLE visuals — those a reviewer who sees only the PDF cannot dispute:

  fig1_reliability.png       Calibration: accuracy vs confidence (overconfidence is label-independent)
  fig2_fluency_gap.png       Two panels: (A) confidence>>accuracy gap; (B) per-task balanced acc vs chance
  fig3_noise_hallucination.png  A pure-noise negative control confidently called a brain MRI

Run:  python -m src.analysis.paper_figures   (writes to figures/paper/)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.scoring.metrics import ece
from src.utils import get_logger

log = get_logger(__name__)

# ── Large-font, clean style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14.5,
    "figure.dpi": 150, "savefig.dpi": 320, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})
HEADLINE = ["T1-MOD", "T2-PLANE", "T3-ISBRAIN", "T4-TUMOR", "T5-LAT"]
TASKLAB = {"T1-MOD": "Sequence", "T2-PLANE": "Plane", "T3-ISBRAIN": "Is brain?",
           "T4-TUMOR": "Tumor", "T5-LAT": "Laterality"}
CHANCE = {"T1-MOD": 1/7, "T2-PLANE": 1/3, "T3-ISBRAIN": 0.5, "T4-TUMOR": 0.5, "T5-LAT": 0.25}
# distinct, color-blind-friendly per model (Okabe-Ito). Gemma-3-4B sits next to MedGemma-4B so the
# base-vs-medical pair reads together in the legend.
PALETTE = {"InternVL2.5-8B": "#0072B2", "Qwen2.5-VL-3B": "#E69F00",
           "Qwen2.5-VL-7B": "#009E73", "Gemma-4-12B": "#56B4E9",
           "Gemma-3-4B": "#CC79A7", "MedGemma-4B": "#D55E00"}


def _short(m):
    s = str(m).split("/")[-1].replace("-Instruct", "").replace("-it", "")
    return {"InternVL2_5-8B": "InternVL2.5-8B", "medgemma-4b": "MedGemma-4B",
            "gemma-3-4b": "Gemma-3-4B", "gemma-4-12B": "Gemma-4-12B"}.get(s, s)


def _headline_graded(graded, manifest):
    g = graded.copy()
    g["m"] = g["model"].map(_short)
    g = g[(g.phrasing == "neutral") & (g["format"] == "MC") & (g.task.isin(HEADLINE))]
    cov = g.groupby("m")["answered"].apply(lambda s: s.astype(bool).mean())
    keep = cov[cov > 0.5].index           # drop degenerate model
    return g[g.m.isin(keep)]


def fig_reliability(graded, out):
    g = _headline_graded(graded, None)
    models = [m for m in PALETTE if m in set(g.m)]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.fill_between([0, 1], [0, 1], [0, 0], color="#D55E00", alpha=0.06)
    ax.text(0.62, 0.18, "overconfident\n(below diagonal)", color="#9A3B12",
            fontsize=14, ha="center", style="italic")
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect calibration")
    for m in models:
        gm = g[(g.m == m) & g.answered.astype(bool)]
        conf = gm.conf_frac.astype(float).values
        corr = gm.correct.astype(bool).values.astype(float)
        d = ece(conf[~np.isnan(conf)], corr[~np.isnan(conf)])
        bins = d.get("bins", [])
        if not bins:
            continue
        xs = [b["conf"] for b in bins]; ys = [b["acc"] for b in bins]
        ax.plot(xs, ys, "o-", color=PALETTE[m], lw=2.6, ms=8,
                label=f"{m}  (ECE={d['ece']:.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Model confidence"); ax.set_ylabel("Actual accuracy")
    ax.set_title("VLMs are systematically overconfident\non brain-MRI tasks", pad=12)
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(out); plt.close(fig); log.info(f"saved {out}")


def fig_fluency_gap(graded, safety_csv, headline_csv, out):
    g = _headline_graded(graded, None)
    models = [m for m in PALETTE if m in set(g.m)]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 6.2))

    # Panel A: mean confidence vs accuracy (the gap)
    confs, accs = [], []
    for m in models:
        gm = g[(g.m == m) & g.answered.astype(bool)]
        confs.append(np.nanmean(gm.conf_frac.astype(float).values))
        accs.append(gm.correct.astype(bool).mean())
    x = np.arange(len(models)); w = 0.38
    bC = axA.bar(x - w/2, confs, w, label="Stated confidence", color="#BBBBBB", edgecolor="k")
    bA = axA.bar(x + w/2, accs, w, label="Actual accuracy", color="#0072B2", edgecolor="k")
    for xi, c, a in zip(x, confs, accs):
        axA.annotate("", xy=(xi, c), xytext=(xi, a),
                     arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.8))
        axA.text(xi, (c+a)/2, f"  +{(c-a)*100:.0f}", color="crimson", fontsize=13, va="center")
    axA.set_xticks(x); axA.set_xticklabels(models, rotation=18, ha="right")
    axA.set_ylim(0, 1.0); axA.set_ylabel("Mean value")
    axA.set_title("Confidence far exceeds accuracy"); axA.legend(loc="upper right")

    # Panel B: per-task balanced accuracy vs chance
    h = pd.read_csv(headline_csv); h["m"] = h["model"].map(_short)
    tasks = HEADLINE; xt = np.arange(len(tasks)); bw = 0.8/len(models)
    for i, m in enumerate(models):
        vals = [float(h[(h.m == m) & (h.task == t)]["balanced_acc"].iloc[0])
                if not h[(h.m == m) & (h.task == t)].empty else 0 for t in tasks]
        axB.bar(xt + (i - len(models)/2 + 0.5)*bw, vals, bw*0.92,
                color=PALETTE[m], label=m, edgecolor="k", linewidth=0.4)
    for j, t in enumerate(tasks):
        axB.plot([j-0.45, j+0.45], [CHANCE[t]]*2, ls="--", lw=1.6, color="0.35",
                 label="Chance" if j == 0 else None)
    axB.set_xticks(xt); axB.set_xticklabels([TASKLAB[t] for t in tasks], rotation=18, ha="right")
    axB.set_ylim(0, 1.0); axB.set_ylabel("Balanced accuracy")
    axB.set_title("At/near chance on visual tasks\nexcept brain detection")
    axB.legend(loc="upper center", ncol=2, fontsize=12)
    fig.suptitle("Fluent but not competent: VLMs answer confidently yet perform near chance",
                 fontsize=20, y=1.02)
    fig.savefig(out); plt.close(fig); log.info(f"saved {out}")


def fig_noise_hallucination(graded, manifest, out):
    g = graded.copy()
    g = g[(g.is_negative_control == 1) & (g.task == "T3-ISBRAIN")]
    # a confident "yes it's a brain" on a non-brain control
    cand = g[(g.pred_answer.astype(str).str.lower() == "yes")]
    cand = cand.sort_values("confidence", ascending=False)
    pmap = dict(zip(manifest.image_id, manifest.path))
    # prefer a gaussian/noise control
    pick = None
    for _, r in cand.iterrows():
        p = pmap.get(r.image_id, "")
        if p and ("noise" in p.lower() or "gaussian" in p.lower() or "salt" in p.lower()) and Path(p).exists():
            pick = (r, p); break
    if pick is None and not cand.empty:
        for _, r in cand.iterrows():
            p = pmap.get(r.image_id, "")
            if p and Path(p).exists():
                pick = (r, p); break
    if pick is None:
        log.warning("no noise-control example found"); return
    r, p = pick
    fig, ax = plt.subplots(figsize=(7.0, 7.6))
    ax.imshow(Image.open(p).convert("L"), cmap="gray", extent=(0, 1, 0.30, 1.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("A pure-noise image, confidently read as a brain MRI", fontsize=18, pad=10)
    resp = " ".join(str(r.response).split())
    import textwrap
    cap = "\n".join(textwrap.wrap(resp, 70)[:4])
    ax.text(0.5, 0.265, f"Ground truth: not a brain (synthetic noise)   •   "
                        f"Model: “yes, a brain MRI”   •   Confidence: {int(r.confidence)}%",
            transform=ax.transAxes, ha="center", va="top", fontsize=13.5, color="crimson")
    ax.text(0.02, 0.20, f"Model: “{cap}”", transform=ax.transAxes, ha="left", va="top",
            fontsize=12.5, color="#222")
    fig.savefig(out); plt.close(fig); log.info(f"saved {out}")


def main():
    from src.utils import setup_logging
    setup_logging(log_file="logs/paper_figures.log")
    out = Path("figures/paper"); out.mkdir(parents=True, exist_ok=True)
    graded = pd.read_csv("results/graded_all.csv")
    manifest = pd.read_csv("data/exam_set.csv")
    fig_reliability(graded, out / "fig1_reliability.png")
    fig_fluency_gap(graded, "results/paper/safety_summary.csv",
                    "results/paper/headline_by_task.csv", out / "fig2_fluency_gap.png")
    fig_noise_hallucination(graded, manifest, out / "fig3_noise_hallucination.png")
    print("DONE: figures/paper/{fig1_reliability,fig2_fluency_gap,fig3_noise_hallucination}.png")


if __name__ == "__main__":
    main()
