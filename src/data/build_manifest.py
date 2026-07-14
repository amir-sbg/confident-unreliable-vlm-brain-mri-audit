"""
Build exam_set.csv by merging slice provenance with derived ground-truth labels.

Ground truth derivation:
  - modality:        from provenance (IXI filename, BraTS channel name)
  - plane:           from provenance (set at extraction time)
  - tumor_present:   BraTS -> yes (seg mask has non-zero voxels); IXI/OASIS -> no
  - laterality:      BraTS -> centroid x vs mid-sagittal; healthy -> none
  - demo_{sex,age}:  IXI .xls / OASIS metadata CSV

Usage:
    python -m src.data.build_manifest \
        --provenance data/slice_provenance.jsonl \
        --config config/datasets.yaml \
        --output data/exam_set.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from src.utils import load_yaml, setup_logging, get_logger

log = get_logger(__name__)

# Columns in the final exam_set.csv
EXAM_SET_COLUMNS = [
    "image_id", "path", "dataset", "subject_id",
    "modality", "plane", "slice_frac",
    "label_modality", "label_plane", "label_tumor_present", "label_laterality",
    "label_lobe",
    "demo_sex", "demo_age",
    "is_negative_control", "resolution",
]

LATERALITY_THRESHOLD = 0.60   # fraction of tumor mass in one hemisphere -> lateralized


def derive_laterality_from_mask(seg_path: Path, subject_id: str) -> str:
    """
    Compute laterality by comparing tumor voxel mass in each hemisphere.
    Assumes RAS+ orientation (x > mid -> right hemisphere).
    Returns: 'left' | 'right' | 'bilateral' | 'none'
    """
    try:
        seg_img = nib.load(str(seg_path))
        seg_img = nib.as_closest_canonical(seg_img)
        seg_data = seg_img.get_fdata(dtype=np.float32)
    except Exception as e:
        log.warning(f"Cannot load seg mask for {subject_id}: {e}")
        return "unknown"

    tumor_voxels = seg_data > 0
    if not tumor_voxels.any():
        return "none"

    mid_x = seg_data.shape[0] / 2.0
    xs = np.where(tumor_voxels)[0]
    n_right = (xs > mid_x).sum()     # RAS: x > mid -> right
    n_left  = (xs <= mid_x).sum()
    total   = len(xs)

    frac_right = n_right / total
    frac_left  = n_left  / total

    if frac_right >= LATERALITY_THRESHOLD:
        return "right"
    if frac_left  >= LATERALITY_THRESHOLD:
        return "left"
    return "bilateral"


def load_ixi_demographics(meta_path: Path) -> dict[str, dict]:
    """Load IXI demographics from the .xls file. Returns {IXI_id: {sex, age}}."""
    if not meta_path.exists():
        log.warning(f"IXI metadata not found at {meta_path}")
        return {}
    try:
        df = pd.read_excel(meta_path)
        # IXI standard columns: IXI_ID, SEX_ID (1=male, 2=female), AGE
        df.columns = [c.strip().upper() for c in df.columns]
        demo: dict[str, dict] = {}
        for _, row in df.iterrows():
            ixi_id = str(int(row.get("IXI_ID", 0))).zfill(3)
            sex_code = row.get("SEX_ID", row.get("SEX", None))
            sex = "M" if sex_code == 1 else ("F" if sex_code == 2 else None)
            age = float(row["AGE"]) if "AGE" in df.columns and not pd.isna(row["AGE"]) else None
            demo[f"IXI{ixi_id}"] = {"sex": sex, "age": age}
        return demo
    except Exception as e:
        log.warning(f"Failed to parse IXI metadata: {e}")
        return {}


def load_oasis_demographics(meta_path: Path) -> dict[str, dict]:
    """Load OASIS-1 demographics CSV. Returns {subject_id: {sex, age}}."""
    if not meta_path.exists():
        log.warning(f"OASIS metadata not found at {meta_path}")
        return {}
    try:
        df = pd.read_csv(meta_path)
        df.columns = [c.strip().upper() for c in df.columns]
        demo: dict[str, dict] = {}
        id_col = "ID" if "ID" in df.columns else df.columns[0]
        for _, row in df.iterrows():
            sid = str(row[id_col]).strip()
            sex = str(row.get("M/F", row.get("SEX", ""))).strip()
            age = float(row["AGE"]) if "AGE" in df.columns and not pd.isna(row["AGE"]) else None
            demo[sid] = {"sex": sex, "age": age}
        return demo
    except Exception as e:
        log.warning(f"Failed to parse OASIS metadata: {e}")
        return {}


def find_seg_mask(vol_path: Path) -> Path | None:
    """Find the BraTS segmentation mask corresponding to a volume path."""
    parent = vol_path.parent
    name = vol_path.name

    # Try standard BraTS naming: replace modality suffix with seg
    for suffix in ["t1n", "t1c", "t2w", "t2f", "t1", "t1ce", "t2", "flair"]:
        if suffix in name.lower():
            seg_name = re.sub(
                rf"[-_]{suffix}[-_]?", "-seg-",
                name, flags=re.IGNORECASE
            ).rstrip("-") + ".gz" if not name.endswith(".gz") else ""
            break

    # Simpler: look for *seg*.nii.gz in same directory
    segs = list(parent.glob("*seg*.nii.gz")) + list(parent.glob("*seg*.nii"))
    if segs:
        return segs[0]
    return None


def build_manifest(
    provenance_jsonl: Path,
    config: dict,
    output_csv: Path,
    add_slice_labels: bool = True,
    slice_area_threshold: int = 25,
) -> pd.DataFrame:
    """
    Core build logic. Reads provenance, derives GT labels, writes exam_set.csv.
    When `add_slice_labels` is set, also emits slice-level tumor/laterality columns derived from
    each BraTS slice's segmentation mask (fixes R1 — subject-level labels mislabel tumor-free slices).
    """
    from src.utils.io import read_jsonl

    records = read_jsonl(provenance_jsonl)
    if not records:
        log.error(f"No provenance records found in {provenance_jsonl}")
        return pd.DataFrame()

    log.info(f"Processing {len(records)} provenance records")

    # Load demographics
    ixi_demo: dict = {}
    oasis_demo: dict = {}
    ixi_meta = Path(config["datasets"]["ixi"].get("metadata_file", "data/ixi/IXI.xls"))
    oasis_meta = Path(config["datasets"]["oasis"].get("metadata_file", "data/oasis/oasis_cross-sectional.csv"))
    if ixi_meta.exists():
        ixi_demo = load_ixi_demographics(ixi_meta)
    if oasis_meta.exists():
        oasis_demo = load_oasis_demographics(oasis_meta)

    # Cache laterality per BraTS subject (expensive: loads NIfTI masks)
    laterality_cache: dict[str, str] = {}

    rows: list[dict] = []
    for rec in records:
        dataset   = rec.get("dataset", "")
        subject_id = rec.get("subject_id", "")
        modality  = rec.get("modality", "")
        plane     = rec.get("plane", "")
        is_nc     = rec.get("is_negative_control", False)

        # ── Ground truth labels ───────────────────────────────────────────────
        label_modality = modality  # already derived at extraction
        label_plane    = plane

        if is_nc:
            label_tumor_present = "no"
            label_laterality    = "none"
            demo_sex = demo_age = None
        elif dataset == "brats":
            label_tumor_present = "yes"
            if subject_id not in laterality_cache:
                vol_path = Path(rec.get("volume_path", ""))
                seg_path = find_seg_mask(vol_path)
                if seg_path:
                    laterality_cache[subject_id] = derive_laterality_from_mask(seg_path, subject_id)
                else:
                    log.warning(f"No seg mask found for {subject_id}")
                    laterality_cache[subject_id] = "unknown"
            label_laterality = laterality_cache[subject_id]
            demo_sex = demo_age = None
        elif dataset == "ixi":
            label_tumor_present = "no"
            label_laterality    = "none"
            demo_info = ixi_demo.get(subject_id, {})
            demo_sex  = demo_info.get("sex")
            demo_age  = demo_info.get("age")
        elif dataset == "oasis":
            label_tumor_present = "no"
            label_laterality    = "none"
            demo_info = oasis_demo.get(subject_id, {})
            demo_sex  = demo_info.get("sex")
            demo_age  = demo_info.get("age")
        else:
            label_tumor_present = rec.get("label_tumor_present", "unknown")
            label_laterality    = rec.get("label_laterality", "unknown")
            demo_sex = demo_age = None

        # lobe is optional / exploratory
        label_lobe = rec.get("label_lobe", None)

        rows.append({
            "image_id":             rec.get("image_id"),
            "path":                 rec.get("path"),
            "dataset":              dataset,
            "subject_id":           subject_id,
            "modality":             modality,
            "plane":                plane,
            "slice_frac":           rec.get("slice_frac"),
            "label_modality":       label_modality,
            "label_plane":          label_plane,
            "label_tumor_present":  label_tumor_present,
            "label_laterality":     label_laterality,
            "label_lobe":           label_lobe,
            "demo_sex":             demo_sex,
            "demo_age":             demo_age,
            "is_negative_control":  int(bool(is_nc)),
            "resolution":           rec.get("original_slice_resolution", rec.get("resolution", "512x512")),
        })

    df = pd.DataFrame(rows, columns=EXAM_SET_COLUMNS)
    df = df.dropna(subset=["image_id", "path"])
    df = df.drop_duplicates(subset=["image_id"])

    # Slice-level tumor/laterality labels (R1): derived from each slice's segmentation mask so a
    # tumor-free slice of a tumor subject is graded correctly. Backward-compatible (extra columns).
    if add_slice_labels:
        from src.data.slice_labels import add_slice_labels_to_manifest
        df = add_slice_labels_to_manifest(df, records, area_threshold=slice_area_threshold)
        log.info(f"  Slice labels added (area_threshold={slice_area_threshold}px):\n"
                 f"{df[df['dataset']=='brats']['slice_tumor_present'].value_counts().to_string()}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info(f"exam_set.csv: {len(df)} rows -> {output_csv}")

    # Summary stats
    log.info(f"  Dataset counts:\n{df['dataset'].value_counts().to_string()}")
    log.info(f"  Tumor present:\n{df['label_tumor_present'].value_counts().to_string()}")
    log.info(f"  Modality:\n{df['label_modality'].value_counts().to_string()}")
    log.info(f"  Plane:\n{df['label_plane'].value_counts().to_string()}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build exam_set.csv from slice provenance.")
    parser.add_argument("--provenance", default="data/slice_provenance.jsonl")
    parser.add_argument("--config", default="config/datasets.yaml")
    parser.add_argument("--output", default="data/exam_set.csv")
    parser.add_argument("--no-slice-labels", action="store_true",
                        help="Skip slice-level tumor/laterality labels (R1).")
    parser.add_argument("--slice-area-threshold", type=int, default=25,
                        help="Min mask area (px) for a slice to count as visible tumor.")
    args = parser.parse_args()

    setup_logging(log_file="logs/build_manifest.log")
    config = load_yaml(args.config)
    build_manifest(
        provenance_jsonl=Path(args.provenance),
        config=config,
        output_csv=Path(args.output),
        add_slice_labels=not args.no_slice_labels,
        slice_area_threshold=args.slice_area_threshold,
    )


if __name__ == "__main__":
    main()
