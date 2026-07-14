"""
Build a manual parser-audit sheet (Codex R6): a stratified sample of graded responses with the
raw model text + the parser's extracted fields, plus BLANK human-label columns to fill in. A human
(ideally one medically aware) fills `human_*`, then `compute_parser_agreement.py`-style logic (or a
quick pandas groupby) reports parser-vs-human agreement for pred_answer / confidence / abstention.

Usage:
  python -m src.analysis.parser_audit --graded results/graded_all.csv --n 100 --out results/parser_audit_manual.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import write_csv_atomic, setup_logging, get_logger

log = get_logger(__name__)

HUMAN_COLS = ["human_pred_answer", "human_confidence", "human_abstained", "human_asserted_findings",
              "parser_agrees_answer", "parser_agrees_conf", "parser_agrees_abstain", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", default="results/graded_all.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/parser_audit_manual.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logging(log_file="logs/parser_audit.log")
    g = pd.read_csv(args.graded)

    # Stratify across task × format so MC and OE, and all tasks, are represented.
    strata = g.groupby(["task", "format"], group_keys=False)
    per = max(1, args.n // max(1, strata.ngroups))
    sample = strata.apply(lambda d: d.sample(min(per, len(d)), random_state=args.seed))
    if len(sample) > args.n:
        sample = sample.sample(args.n, random_state=args.seed)

    cols = [c for c in ["model", "image_id", "task", "format", "phrasing", "response",
                        "pred_answer", "confidence", "abstained", "unparseable",
                        "asserted_findings", "gt_label", "correct"] if c in sample.columns]
    sheet = sample[cols].copy()
    for c in HUMAN_COLS:
        sheet[c] = ""   # to be filled by the human reviewer

    write_csv_atomic(args.out, sheet.to_dict("records"), fieldnames=cols + HUMAN_COLS)
    log.info(f"Parser-audit sheet: {len(sheet)} rows across {strata.ngroups} task×format strata -> {args.out}")
    print(f"DONE parser audit: {len(sheet)} rows -> {args.out}")
    print("Fill human_* columns, then agreement = mean(parser_agrees_*).")


if __name__ == "__main__":
    main()
