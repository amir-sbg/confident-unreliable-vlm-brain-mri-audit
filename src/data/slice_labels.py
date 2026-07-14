"""
Slice-level tumor/laterality labels for BraTS (fixes the subject-level labeling bug).

Background (see CC Work/ai/risk-register.md R1, decision-log): the original manifest labeled
EVERY slice of a tumor subject as `tumor_present=yes` because the *subject* has a tumor. But most
individual 2D slices through a tumor brain show no tumor, so T4/T5 were mis-graded — a model
correctly answering "no visible tumor" on a tumor-free slice was scored wrong.

This module derives labels from the **segmentation-mask slice that corresponds to the exact image
slice** (same volume, plane, and depth index), using the identical reorientation + indexing as
`slice_extract.py`, so the seg slice aligns pixel-for-pixel with the extracted PNG.

Fields returned per slice:
  slice_tumor_present     "yes" | "no"        (any tumor voxel in this slice's mask)
  slice_tumor_area_px     int                 (tumor voxel count in this slice)
  slice_tumor_area_frac   float               (tumor voxels / slice voxels)
  slice_laterality        "left"|"right"|"bilateral"|"none"  (in-slice; see below)

Laterality semantics by plane (RAS+: x>mid → right):
  - axial  (data[:,:,idx]) / coronal (data[:,idx,:]): in-plane axis-0 is L/R → centroid vs mid.
  - sagittal (data[idx,:,:]): no in-plane L/R axis; the whole slice sits at one x, so if tumor is
    present the laterality is the side of that sagittal plane (idx vs mid_x).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from src.data.slice_extract import reorient_to_ras, extract_slice
from src.data.build_manifest import find_seg_mask, LATERALITY_THRESHOLD
from src.utils import get_logger

log = get_logger(__name__)

# Small cache so we load each subject's seg volume once, not once per slice.
_SEG_CACHE: dict[str, np.ndarray | None] = {}


def _load_seg_ras(volume_path: str) -> np.ndarray | None:
    """Find + load the segmentation mask for a BraTS modality volume, reoriented to RAS."""
    if volume_path in _SEG_CACHE:
        return _SEG_CACHE[volume_path]
    seg_path = find_seg_mask(Path(volume_path))
    if seg_path is None:
        _SEG_CACHE[volume_path] = None
        return None
    try:
        seg = reorient_to_ras(nib.load(str(seg_path)))
        data = seg.get_fdata(dtype=np.float32)
    except Exception as e:
        log.warning(f"Failed to load seg {seg_path}: {e}")
        data = None
    _SEG_CACHE[volume_path] = data
    return data


def _slice_laterality(seg_slice: np.ndarray, plane: str, full_shape, slice_idx: int) -> str:
    """Laterality of tumor within a single slice (see module docstring)."""
    tumor = seg_slice > 0
    if not tumor.any():
        return "none"
    if plane == "sagittal":
        # whole slice is at one x position; side = which half of the volume's x-axis it sits in
        mid_x = full_shape[0] / 2.0
        return "right" if slice_idx > mid_x else "left"
    # axial/coronal: in-plane axis-0 is the L/R (x) axis
    xs = np.where(tumor)[0]
    mid = seg_slice.shape[0] / 2.0
    frac_right = (xs > mid).sum() / len(xs)
    frac_left = (xs <= mid).sum() / len(xs)
    if frac_right >= LATERALITY_THRESHOLD:
        return "right"
    if frac_left >= LATERALITY_THRESHOLD:
        return "left"
    return "bilateral"


def _slice_index(shape, plane: str, frac: float) -> int:
    """Same index formula as slice_extract.extract_slice."""
    axis = {"axial": 2, "sagittal": 0, "coronal": 1}[plane]
    return int(np.clip(int(round(frac * shape[axis])), 0, shape[axis] - 1))


def compute_slice_labels(volume_path: str, plane: str, slice_frac: float) -> dict[str, Any]:
    """
    Derive slice-level labels for one BraTS image slice from its segmentation mask.
    Returns the four `slice_*` fields. If the seg mask is unavailable, returns 'unknown'.
    """
    seg = _load_seg_ras(volume_path)
    if seg is None:
        return {"slice_tumor_present": "unknown", "slice_tumor_area_px": -1,
                "slice_tumor_area_frac": float("nan"), "slice_laterality": "unknown"}

    seg_slice = extract_slice(seg, plane, slice_frac)   # identical indexing to the PNG slice
    area_px = int((seg_slice > 0).sum())
    present = "yes" if area_px > 0 else "no"
    frac = area_px / float(seg_slice.size) if seg_slice.size else 0.0
    lat = _slice_laterality(seg_slice, plane, seg.shape, _slice_index(seg.shape, plane, slice_frac)) \
        if present == "yes" else "none"
    return {"slice_tumor_present": present, "slice_tumor_area_px": area_px,
            "slice_tumor_area_frac": round(frac, 6), "slice_laterality": lat}


def _apply_area_threshold(labels: dict, area_threshold: int) -> dict:
    """A slice counts as 'visible tumor' only if its mask area >= area_threshold px.
    Sub-threshold slivers are treated as no-visible-tumor (laterality 'none')."""
    if labels.get("slice_tumor_present") == "yes" and 0 <= labels.get("slice_tumor_area_px", 0) < area_threshold:
        labels = {**labels, "slice_tumor_present": "no", "slice_laterality": "none"}
    return labels


def add_slice_labels_to_manifest(manifest_df, provenance_records: list[dict], area_threshold: int = 25):
    """
    Add slice-level label columns to a manifest DataFrame using provenance (which carries
    volume_path/plane/slice_frac). BraTS rows get mask-derived slice labels (thresholded at
    `area_threshold` px); non-BraTS rows get tumor 'no'/'none' by construction. Backward
    compatible: callers that don't have these columns are unaffected.
    """
    import pandas as pd  # local import

    prov_by_id = {r["image_id"]: r for r in provenance_records}
    rows = []
    for _, row in manifest_df.iterrows():
        iid = row["image_id"]
        ds = row.get("dataset", "")
        if ds == "brats":
            p = prov_by_id.get(iid, {})
            labels = compute_slice_labels(p.get("volume_path", ""), p.get("plane", row.get("plane")),
                                          float(p.get("slice_frac", row.get("slice_frac", 0.5))))
            labels = _apply_area_threshold(labels, area_threshold)
        else:
            # healthy / negative-control slices carry no tumor by construction
            labels = {"slice_tumor_present": "no", "slice_tumor_area_px": 0,
                      "slice_tumor_area_frac": 0.0, "slice_laterality": "none"}
        rows.append({**labels, "image_id": iid})
    lab_df = pd.DataFrame(rows)
    return manifest_df.merge(lab_df, on="image_id", how="left")
