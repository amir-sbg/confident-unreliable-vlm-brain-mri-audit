"""
Aggregate raw_responses/*.jsonl -> graded rows -> results/*.csv

Produces:
  results/main_accuracy.csv
  results/calibration.csv
  results/hallucination.csv
  results/ablations/*.csv
  results/parser_audit_sample.csv   (50-item hand-check sample)

Usage:
    python -m src.analysis.aggregate \
        --responses-dir raw_responses \
        --manifest data/exam_set.csv \
        --tasks config/tasks.yaml \
        --output-dir results
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring.parse import parse_batch
from src.scoring.grade import grade_batch
from src.scoring.metrics import compute_model_task_metrics, compute_reliability_diagram_data
from src.utils import load_yaml, read_jsonl, write_csv_atomic, setup_logging, get_logger

log = get_logger(__name__)


def load_all_responses(responses_dir: Path) -> pd.DataFrame:
    """Load all model JSONL files into a single DataFrame."""
    records = []
    for jsonl_path in sorted(responses_dir.glob("*.jsonl")):
        recs = read_jsonl(jsonl_path)
        log.info(f"  {jsonl_path.name}: {len(recs)} records")
        records.extend(recs)
    if not records:
        log.error(f"No response files found in {responses_dir}")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    log.info(f"Total response records: {len(df)}")
    return df


def build_graded_df(
    responses_df: pd.DataFrame,
    manifest: pd.DataFrame,
    tasks_cfg: dict,
) -> pd.DataFrame:
    """Parse and grade all responses, joining with manifest GT labels."""
    log.info("Parsing responses...")
    parsed = parse_batch(responses_df.to_dict(orient="records"), tasks_cfg)
    log.info(f"  Parsed {len(parsed)} records")
    log.info("Grading responses...")
    graded = grade_batch(parsed, manifest)
    log.info(f"  Graded {len(graded)} records")
    df = pd.DataFrame(graded)
    return df


def write_parser_audit_sample(
    graded_df: pd.DataFrame,
    output_path: Path,
    n: int = 50,
    seed: int = 42,
) -> None:
    """Write a random sample of graded rows for manual parser agreement check."""
    sample = graded_df.sample(min(n, len(graded_df)), random_state=seed)
    cols = [
        "model", "image_id", "task", "format", "phrasing",
        "response", "pred_answer", "gt_label", "confidence",
        "abstained", "unparseable", "correct", "hallucination",
        "asserted_findings",
    ]
    available = [c for c in cols if c in sample.columns]
    write_csv_atomic(output_path, sample[available].to_dict(orient="records"), fieldnames=available)
    log.info(f"Parser audit sample: {len(sample)} rows -> {output_path}")


def build_main_accuracy_csv(
    graded_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Table 2: accuracy & macro-F1 per task per model with CIs."""
    metrics_rows = compute_model_task_metrics(graded_df)
    if not metrics_rows:
        log.warning("No metrics computed.")
        return
    df_out = pd.DataFrame(metrics_rows)
    # Sort by model, task for readability
    df_out = df_out.sort_values(["model", "task", "format"]).reset_index(drop=True)
    write_csv_atomic(output_path, df_out.to_dict(orient="records"))
    log.info(f"main_accuracy.csv: {len(df_out)} rows -> {output_path}")


def build_calibration_csv(
    graded_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """calibration.csv: ECE and Brier per model (aggregated across tasks)."""
    from src.scoring.metrics import ece, brier_score, overconfidence_metrics
    import numpy as np

    rows = []
    for model, grp in graded_df.groupby("model"):
        answered = grp["answered"].values.astype(bool)
        conf = grp["conf_frac"].values.astype(float)
        correct = grp["correct"].values.astype(bool)
        gradeable = (grp["gradeable"].values.astype(bool)
                     if "gradeable" in grp.columns else np.ones(len(grp), bool))
        scored = answered & gradeable
        valid = scored & ~np.isnan(conf)

        ece_d   = ece(conf[valid], correct[valid])
        brier_d = brier_score(conf[valid], correct[valid])
        oc_d    = overconfidence_metrics(conf, correct, scored)

        rows.append({
            "model": model,
            "ece":    ece_d.get("ece"),
            "ece_ci_lo": ece_d.get("ci_lo"),
            "ece_ci_hi": ece_d.get("ci_hi"),
            "brier": brier_d.get("brier"),
            "brier_ci_lo": brier_d.get("ci_lo"),
            "brier_ci_hi": brier_d.get("ci_hi"),
            **oc_d,
        })

    df_out = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    write_csv_atomic(output_path, df_out.to_dict(orient="records"))
    log.info(f"calibration.csv: {len(df_out)} rows -> {output_path}")


def build_hallucination_csv(
    graded_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """hallucination.csv: hallucination rate, confidently-wrong, abstention."""
    from src.scoring.metrics import hallucination_rate, abstention_appropriateness
    import numpy as np

    rows = []
    for model, grp in graded_df.groupby("model"):
        is_nc = grp["is_negative_control"].values.astype(bool)
        oe_mask = (grp["format"] == "OE").values
        halluc = grp["hallucination"].values.astype(bool)
        abstained = grp["abstained"].values.astype(bool)
        conf_wrong = grp["confidently_wrong"].values.astype(bool)
        answered = grp["answered"].values.astype(bool)
        should_abs = is_nc | (grp["task"] == "T7-ABSTAIN").values

        hall_d = hallucination_rate(halluc, oe_mask)
        abs_d  = abstention_appropriateness(abstained, should_abs)
        cw_rate = float(conf_wrong[answered].mean()) if answered.sum() > 0 else float("nan")

        rows.append({
            "model": model,
            **{f"hall_{k}": v for k, v in hall_d.items()},
            **{f"abs_{k}": v for k, v in abs_d.items()},
            "confidently_wrong_rate": cw_rate,
            "n_total": int(len(grp)),
        })

    df_out = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    write_csv_atomic(output_path, df_out.to_dict(orient="records"))
    log.info(f"hallucination.csv: {len(df_out)} rows -> {output_path}")


def build_exam_set_table(manifest: pd.DataFrame, output_path: Path) -> None:
    """Table 1: exam-set composition."""
    summary = {
        "total_images": len(manifest),
        "total_subjects": manifest["subject_id"].nunique(),
    }
    rows = []
    for dataset, grp in manifest.groupby("dataset"):
        rows.append({
            "dataset": dataset,
            "n_images": len(grp),
            "n_subjects": grp["subject_id"].nunique(),
            "modalities": ",".join(sorted(grp["label_modality"].dropna().unique())),
            "tumor_yes": int((grp["label_tumor_present"] == "yes").sum()),
            "tumor_no": int((grp["label_tumor_present"] == "no").sum()),
            "planes": ",".join(sorted(grp["label_plane"].dropna().unique())),
        })
    write_csv_atomic(output_path, rows)
    log.info(f"exam_set_table.csv -> {output_path}")


def build_ablation_csvs(graded_df: pd.DataFrame, ablations_dir: Path) -> None:
    """
    Produce per-ablation CSV files.
    A1: phrasing variance
    A2: MC vs OE delta
    A4: Instruct vs Thinking (if reasoning models present)
    A6: scale sweep
    A7: slice position
    """
    ablations_dir.mkdir(parents=True, exist_ok=True)

    def _acc(g):
        """Accuracy + n over answered AND gradeable rows (excludes no-GT items)."""
        answered = g["answered"].values.astype(bool)
        gradeable = (g["gradeable"].values.astype(bool)
                     if "gradeable" in g.columns else np.ones(len(g), bool))
        scored = answered & gradeable
        correct = g["correct"].values.astype(bool)
        acc = float(correct[scored].mean()) if scored.sum() > 0 else float("nan")
        return acc, int(scored.sum())

    # A1: phrasing
    if "phrasing" in graded_df.columns:
        a1 = []
        for (model, task, phrasing), grp in graded_df.groupby(["model", "task", "phrasing"]):
            acc, n = _acc(grp)
            a1.append({"model": model, "task": task, "phrasing": phrasing, "accuracy": acc, "n": n})
        write_csv_atomic(ablations_dir / "ablation_A1_phrasing.csv", a1)

    # A2: MC vs OE
    a2 = []
    for (model, task), grp in graded_df.groupby(["model", "task"]):
        for fmt in ["MC", "OE"]:
            sub = grp[grp["format"] == fmt]
            if sub.empty:
                continue
            acc, n = _acc(sub)
            a2.append({"model": model, "task": task, "format": fmt, "accuracy": acc, "n": n})
    write_csv_atomic(ablations_dir / "ablation_A2_format.csv", a2)

    # A6: scale sweep within family (params_b mapped from the model registry in main())
    if "params_b" in graded_df.columns and graded_df["params_b"].notna().any():
        a6 = []
        sub_df = graded_df[graded_df["params_b"].notna()]
        for (model, params_b, task), grp in sub_df.groupby(["model", "params_b", "task"]):
            acc, n = _acc(grp)
            a6.append({"model": model, "params_b": params_b, "task": task, "accuracy": acc, "n": n})
        write_csv_atomic(ablations_dir / "ablation_A6_scale.csv", a6)

    # A7: slice position
    if "slice_frac" in graded_df.columns:
        a7 = []
        for (model, task, frac), grp in graded_df.groupby(["model", "task", "slice_frac"]):
            acc, n = _acc(grp)
            a7.append({"model": model, "task": task, "slice_frac": frac, "accuracy": acc, "n": n})
        write_csv_atomic(ablations_dir / "ablation_A7_slice_pos.csv", a7)

    log.info(f"Ablation CSVs written to {ablations_dir}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate raw responses to results CSVs.")
    parser.add_argument("--responses-dir", default="raw_responses")
    parser.add_argument("--manifest", default="data/exam_set.csv")
    parser.add_argument("--tasks", default="config/tasks.yaml")
    parser.add_argument("--models", default="config/models.yaml",
                        help="Model registry, used to map params_b for the A6 scale ablation.")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    setup_logging(log_file="logs/aggregate.log")
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "ablations").mkdir(exist_ok=True)

    tasks_cfg = load_yaml(args.tasks)
    manifest  = pd.read_csv(args.manifest)
    log.info(f"Manifest: {len(manifest)} images")

    # Load + parse + grade
    responses_df = load_all_responses(Path(args.responses_dir))
    if responses_df.empty:
        log.error("No responses to aggregate.")
        return

    graded_df = build_graded_df(responses_df, manifest, tasks_cfg)

    # Enrich graded_df with manifest columns (for ablations)
    manifest_cols = ["image_id", "slice_frac", "is_negative_control", "dataset",
                     "label_modality", "label_plane", "label_tumor_present"]
    graded_df = graded_df.merge(
        manifest[[c for c in manifest_cols if c in manifest.columns]],
        on="image_id",
        how="left",
        suffixes=("", "_manifest"),
    )

    # Map params_b per model (from the registry) so the A6 scale ablation can run.
    try:
        models_cfg = load_yaml(args.models)
        params_map = {m["id"]: m.get("params_b") for m in models_cfg.get("models", [])}
        graded_df["params_b"] = graded_df["model"].map(params_map)
    except Exception as e:
        log.warning(f"Could not map params_b from {args.models}: {e}")

    # Write outputs
    build_exam_set_table(manifest, results_dir / "exam_set_table.csv")
    build_main_accuracy_csv(graded_df, results_dir / "main_accuracy.csv")
    build_calibration_csv(graded_df, results_dir / "calibration.csv")
    build_hallucination_csv(graded_df, results_dir / "hallucination.csv")
    write_parser_audit_sample(graded_df, results_dir / "parser_audit_sample.csv")
    build_ablation_csvs(graded_df, results_dir / "ablations")

    # Save full graded CSV for downstream use (make_figures)
    graded_path = results_dir / "graded_all.csv"
    graded_df.to_csv(graded_path, index=False)
    log.info(f"Full graded CSV: {graded_path}")

    # ── Log results to the shared `fpsa_res` wandb run ────────────────────────
    from src.utils import init_wandb, log_csv_table, wandb_summary, finish_wandb
    init_wandb(role="results")   # name -> WANDB_RESULTS_NAME (fpsa_res_results)
    for fname, key in [
        ("exam_set_table.csv",  "tables/exam_set"),
        ("main_accuracy.csv",   "tables/main_accuracy"),
        ("calibration.csv",     "tables/calibration"),
        ("hallucination.csv",   "tables/hallucination"),
    ]:
        log_csv_table(results_dir / fname, key)
    try:
        acc = pd.read_csv(results_dir / "main_accuracy.csv")
        if not acc.empty and "acc_accuracy" in acc.columns:
            wandb_summary({"acc/overall_mean": float(acc["acc_accuracy"].mean())})
            for model, grp in acc.groupby("model"):
                wandb_summary({f"acc/by_model/{model}": float(grp["acc_accuracy"].mean())})
    except Exception as e:
        log.warning(f"wandb summary metrics skipped: {e}")
    finish_wandb()

    log.info("Aggregation complete.")


if __name__ == "__main__":
    main()
