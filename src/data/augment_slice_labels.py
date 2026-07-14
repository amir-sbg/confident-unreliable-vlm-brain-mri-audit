"""
Augment an existing exam_set.csv with slice-level tumor/laterality columns (fixes R1) WITHOUT
re-extracting PNGs or re-running inference. Loads each BraTS slice's segmentation mask and derives
slice_tumor_present / slice_tumor_area_px / slice_tumor_area_frac / slice_laterality.

Usage:
    python -m src.data.augment_slice_labels \
        --manifest data/exam_set.csv \
        --provenance data/slice_provenance.jsonl \
        --area-threshold 25

Writes the augmented manifest back in place and keeps a one-time subject-level backup
(exam_set.subject_level.bak.csv). Idempotent: re-running re-derives the slice columns.

Going forward, `build_manifest.py` should call `slice_labels.add_slice_labels_to_manifest`
directly so fresh runs emit these columns automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.slice_labels import add_slice_labels_to_manifest
from src.utils import read_jsonl, setup_logging, get_logger

log = get_logger(__name__)

SLICE_COLS = ["slice_tumor_present", "slice_tumor_area_px", "slice_tumor_area_frac", "slice_laterality"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/exam_set.csv")
    ap.add_argument("--provenance", default="data/slice_provenance.jsonl")
    ap.add_argument("--area-threshold", type=int, default=25)
    args = ap.parse_args()

    setup_logging(log_file="logs/augment_slice_labels.log")
    manifest_path = Path(args.manifest)
    df = pd.read_csv(manifest_path)
    prov = read_jsonl(args.provenance)

    # Drop any prior slice columns so re-runs are clean, then recompute.
    df = df.drop(columns=[c for c in SLICE_COLS if c in df.columns], errors="ignore")

    backup = manifest_path.with_name("exam_set.subject_level.bak.csv")
    if not backup.exists():
        pd.read_csv(manifest_path).to_csv(backup, index=False)
        log.info(f"Backed up subject-level manifest -> {backup}")

    log.info(f"Computing slice-level labels (area_threshold={args.area_threshold}px)...")
    out = add_slice_labels_to_manifest(df, prov, area_threshold=args.area_threshold)
    out.to_csv(manifest_path, index=False)

    brats = out[out["dataset"] == "brats"]
    vc = brats["slice_tumor_present"].value_counts().to_dict()
    log.info(f"Augmented {manifest_path}: {len(out)} rows, +{len(SLICE_COLS)} slice cols.")
    log.info(f"BraTS slice_tumor_present: {vc}")
    print(f"DONE: BraTS slice_tumor_present={vc}")


if __name__ == "__main__":
    main()
