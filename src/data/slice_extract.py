"""
NIfTI volume -> 2D PNG slices with provenance.

Usage:
    python -m src.data.slice_extract \
        --config config/datasets.yaml \
        --dataset brats \
        --output-dir data/slices \
        [--slice-fracs 0.4 0.5 0.6]  # override for ablation A7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import NamedTuple

import nibabel as nib
import numpy as np
from PIL import Image

from src.utils import load_yaml, setup_logging, get_logger

log = get_logger(__name__)


class SliceSpec(NamedTuple):
    subject_id: str
    volume_path: Path
    modality: str
    dataset: str
    plane: str          # axial | sagittal | coronal
    slice_frac: float


def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient volume to RAS+ canonical orientation for deterministic plane semantics."""
    return nib.as_closest_canonical(img)


def intensity_window_normalize(data: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Robust percentile clip then scale to uint8 [0, 255]."""
    lo = np.percentile(data, p_low)
    hi = np.percentile(data, p_high)
    if hi == lo:
        return np.zeros_like(data, dtype=np.uint8)
    clipped = np.clip(data, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    return (normalized * 255).astype(np.uint8)


def extract_slice(data_3d: np.ndarray, plane: str, frac: float) -> np.ndarray:
    """
    Extract a 2D slice from a 3D volume along the specified plane.
    Plane semantics assume RAS+ orientation:
      axial    -> z axis (dim 2), anterior-posterior x left-right
      sagittal -> x axis (dim 0), inferior-superior x anterior-posterior
      coronal  -> y axis (dim 1), inferior-superior x left-right
    """
    if plane == "axial":
        idx = int(round(frac * data_3d.shape[2]))
        idx = np.clip(idx, 0, data_3d.shape[2] - 1)
        slc = data_3d[:, :, idx]
    elif plane == "sagittal":
        idx = int(round(frac * data_3d.shape[0]))
        idx = np.clip(idx, 0, data_3d.shape[0] - 1)
        slc = data_3d[idx, :, :]
    elif plane == "coronal":
        idx = int(round(frac * data_3d.shape[1]))
        idx = np.clip(idx, 0, data_3d.shape[1] - 1)
        slc = data_3d[:, idx, :]
    else:
        raise ValueError(f"Unknown plane: {plane}")
    return slc


def letterbox_resize(slc: np.ndarray, target: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Resize 2D array to target with letterboxing (black padding) to preserve aspect ratio."""
    h, w = slc.shape
    th, tw = target
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    img = Image.fromarray(slc, mode="L")
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (tw, th), 0)
    paste_x = (tw - nw) // 2
    paste_y = (th - nh) // 2
    canvas.paste(img, (paste_x, paste_y))
    return np.array(canvas)


def make_image_id(subject_id: str, dataset: str, modality: str, plane: str, slice_frac: float) -> str:
    """Deterministic image ID encoding provenance."""
    frac_str = f"{slice_frac:.2f}".replace(".", "p")
    return f"{dataset}__{subject_id}__{modality}__{plane}__{frac_str}"


def make_filename(image_id: str) -> str:
    return f"{image_id}.png"


def save_slice_with_provenance(
    slc_uint8: np.ndarray,
    output_dir: Path,
    image_id: str,
    provenance: dict,
) -> Path:
    """Save PNG and embed provenance as a JSON text chunk (via PIL pnginfo)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / make_filename(image_id)
    img = Image.fromarray(slc_uint8, mode="L")
    from PIL.PngImagePlugin import PngInfo
    meta = PngInfo()
    meta.add_text("provenance", json.dumps(provenance))
    img.save(out_path, pnginfo=meta)
    return out_path


def get_volume_resolution(img: nib.Nifti1Image) -> tuple[float, float, float]:
    """Return voxel sizes in mm."""
    zooms = img.header.get_zooms()
    return float(zooms[0]), float(zooms[1]), float(zooms[2])


def extract_slices_from_volume(
    spec: SliceSpec,
    output_dir: Path,
    target_size: tuple[int, int] = (512, 512),
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> list[dict]:
    """
    Extract all requested slices from one volume.
    Returns list of provenance dicts (one per slice).
    """
    log.info(f"Loading {spec.volume_path}")
    try:
        img = nib.load(str(spec.volume_path))
    except Exception as e:
        log.error(f"Failed to load {spec.volume_path}: {e}")
        return []

    img = reorient_to_ras(img)
    data = img.get_fdata(dtype=np.float32)
    res = get_volume_resolution(img)

    data_uint8 = intensity_window_normalize(data, p_low, p_high)

    image_id = make_image_id(
        spec.subject_id, spec.dataset, spec.modality, spec.plane, spec.slice_frac
    )

    slc_raw = extract_slice(data_uint8, spec.plane, spec.slice_frac)
    slc_resized = letterbox_resize(slc_raw, target_size)

    # Original resolution before letterbox (for ablation A3)
    orig_h, orig_w = slc_raw.shape
    orig_res_str = f"{orig_w}x{orig_h}"

    provenance = {
        "image_id": image_id,
        "subject_id": spec.subject_id,
        "dataset": spec.dataset,
        "modality": spec.modality,
        "plane": spec.plane,
        "slice_frac": spec.slice_frac,
        "volume_path": str(spec.volume_path),
        "volume_sha256_prefix": _sha256_prefix(spec.volume_path),
        "original_shape": list(data.shape),
        "voxel_size_mm": list(res),
        "output_size": list(target_size),
        "original_slice_resolution": orig_res_str,
        "intensity_clip": [p_low, p_high],
    }

    out_path = save_slice_with_provenance(slc_resized, output_dir, image_id, provenance)
    provenance["path"] = str(out_path)
    log.debug(f"Saved {out_path}")
    return [provenance]


def _sha256_prefix(path: Path, n_bytes: int = 65536) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            h.update(f.read(n_bytes))
    except OSError:
        return "unknown"
    return h.hexdigest()[:16]


def discover_volumes(dataset_cfg: dict, dataset_name: str) -> list[tuple[str, Path, str]]:
    """
    Discover (subject_id, volume_path, modality) tuples for a dataset.
    Returns list sorted for determinism.
    """
    root = Path(dataset_cfg["root"])
    if not root.exists():
        log.warning(f"Dataset root does not exist: {root}")
        return []

    modalities = dataset_cfg.get("modalities", [])
    results: list[tuple[str, Path, str]] = []

    if dataset_name == "brats":
        results = _discover_brats(root, dataset_cfg)
    elif dataset_name == "ixi":
        results = _discover_ixi(root, dataset_cfg)
    elif dataset_name == "oasis":
        results = _discover_oasis(root, dataset_cfg)
    else:
        log.warning(f"No discovery logic for dataset: {dataset_name}")

    return sorted(results, key=lambda x: (x[0], x[2]))


def _discover_brats(root: Path, cfg: dict) -> list[tuple[str, Path, str]]:
    results = []
    modality_map = {
        "t1n": "T1", "t1c": "T1ce", "t2w": "T2", "t2f": "FLAIR",
        # legacy BraTS 2023 suffixes
        "t1": "T1", "t1ce": "T1ce", "t2": "T2", "flair": "FLAIR",
    }
    for nii_path in sorted(root.rglob("*.nii.gz")):
        name = nii_path.stem.replace(".nii", "").lower()
        if "seg" in name:
            continue
        for suffix, modality in modality_map.items():
            if name.endswith(suffix) or f"-{suffix}" in name or f"_{suffix}" in name:
                subject_id = _brats_subject_id(nii_path)
                results.append((subject_id, nii_path, modality))
                break
    return results


def _brats_subject_id(path: Path) -> str:
    for part in reversed(path.parts):
        if re.match(r"BraTS.*\d{5}", part, re.IGNORECASE):
            return part
    return path.parent.name


def _discover_ixi(root: Path, cfg: dict) -> list[tuple[str, Path, str]]:
    results = []
    modality_map = {"T1": "T1", "T2": "T2", "PD": "PD", "MRA": "MRA", "DTI": "DWI"}
    for nii_path in sorted(root.rglob("*.nii.gz")):
        name = nii_path.stem.replace(".nii", "")
        for key, modality in modality_map.items():
            if f"-{key}" in name:
                subject_id = name.split("-")[0]
                results.append((subject_id, nii_path, modality))
                break
    return results


def _discover_oasis(root: Path, cfg: dict) -> list[tuple[str, Path, str]]:
    results = []
    for nii_path in sorted(root.rglob("*.nii.gz")):
        name = nii_path.stem.replace(".nii", "")
        subject_id = name.split("_")[0] if "_" in name else name
        results.append((subject_id, nii_path, "T1"))
    return results


def run_extraction(
    dataset_name: str,
    config: dict,
    output_dir: Path,
    planes: list[str],
    slice_fracs: list[float],
    n_subjects: int | None,
    seed: int,
) -> list[dict]:
    """
    Main extraction entry point for one dataset.
    Returns flat list of provenance dicts.
    """
    from src.utils.seeds import subsample_indices

    ds_cfg = config["datasets"][dataset_name]
    volumes = discover_volumes(ds_cfg, dataset_name)

    if not volumes:
        log.warning(f"No volumes found for {dataset_name} at {ds_cfg['root']}")
        return []

    # Subsample subjects (not volumes) for balance
    subjects = sorted({v[0] for v in volumes})
    n_sub = n_subjects or ds_cfg.get("n_subjects", len(subjects))
    indices = subsample_indices(len(subjects), min(n_sub, len(subjects)), seed)
    selected_subjects = {subjects[i] for i in indices}
    selected_volumes = [v for v in volumes if v[0] in selected_subjects]
    log.info(f"{dataset_name}: {len(selected_subjects)} subjects, {len(selected_volumes)} volumes")

    global_cfg = config.get("global", {})
    target_size = tuple(global_cfg.get("output_image_size", [512, 512]))
    p_low, p_high = global_cfg.get("intensity_clip_percentile", [1, 99])

    all_provenance: list[dict] = []
    for subject_id, vol_path, modality in selected_volumes:
        for plane in planes:
            for frac in slice_fracs:
                spec = SliceSpec(
                    subject_id=subject_id,
                    volume_path=vol_path,
                    modality=modality,
                    dataset=dataset_name,
                    plane=plane,
                    slice_frac=frac,
                )
                provs = extract_slices_from_volume(
                    spec, output_dir, target_size=target_size, p_low=p_low, p_high=p_high
                )
                all_provenance.extend(provs)

    log.info(f"{dataset_name}: extracted {len(all_provenance)} slices")
    return all_provenance


def generate_negative_controls(config: dict, output_dir: Path) -> list[dict]:
    """Generate negative control images (noise, corrupted, chest X-ray placeholders)."""
    from src.utils.seeds import make_rng

    nc_cfg = config["datasets"]["negative_controls"]
    output_dir = output_dir / "negative_controls"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = make_rng(nc_cfg.get("subsample_seed", 45))
    provenance_list: list[dict] = []

    for subcat, params in nc_cfg.get("subcategories", {}).items():
        if subcat == "chest_xray":
            src = Path(params.get("source", "data/chest_xray_sample"))
            if src.exists():
                imgs = sorted(src.glob("*.png")) + sorted(src.glob("*.jpg"))
                for i, img_path in enumerate(imgs[: params.get("n", 30)]):
                    image_id = f"negctrl__{subcat}__item{i:04d}"
                    out_path = output_dir / f"{image_id}.png"
                    img = Image.open(img_path).convert("L").resize((512, 512))
                    img.save(out_path)
                    provenance_list.append(_nc_prov(image_id, out_path, subcat))
            else:
                log.warning(f"Chest X-ray sample not found at {src}; skipping.")
            continue

        n = params.get("n", 20)
        for i in range(n):
            image_id = f"negctrl__{subcat}__item{i:04d}"
            out_path = output_dir / f"{image_id}.png"
            arr = _generate_noise_image(subcat, params, rng, i)
            Image.fromarray(arr, mode="L").save(out_path)
            provenance_list.append(_nc_prov(image_id, out_path, subcat))

    log.info(f"Generated {len(provenance_list)} negative controls")
    return provenance_list


def _generate_noise_image(subcat: str, params: dict, rng: np.random.Generator, idx: int) -> np.ndarray:
    size = (512, 512)
    if subcat == "gaussian_noise":
        mean = params.get("mean", 0.5) * 255
        std = params.get("std", 0.3) * 255
        arr = rng.normal(mean, std, size).clip(0, 255).astype(np.uint8)
    elif subcat == "salt_pepper":
        arr = np.ones(size, dtype=np.uint8) * 128
        density = params.get("density", 0.3)
        mask = rng.random(size) < density
        arr[mask] = rng.integers(0, 2, mask.sum()) * 255
    elif subcat == "corrupted_mri":
        arr = rng.integers(0, 256, size, dtype=np.uint8)
        from scipy.ndimage import gaussian_filter
        arr = gaussian_filter(arr.astype(float), sigma=rng.integers(5, 20)).astype(np.uint8)
    else:
        arr = np.zeros(size, dtype=np.uint8)
    return arr


def _nc_prov(image_id: str, out_path: Path, subcat: str) -> dict:
    return {
        "image_id": image_id,
        "path": str(out_path),
        "dataset": "negative_controls",
        "subject_id": image_id,
        "modality": "unknown",
        "plane": "unknown",
        "slice_frac": 0.5,
        "is_negative_control": True,
        "negative_control_type": subcat,
        "label_tumor_present": "no",
        "label_modality": "unknown",
        "label_plane": "unknown",
        "label_laterality": "none",
        "resolution": "512x512",
    }


def main():
    parser = argparse.ArgumentParser(description="Extract 2D PNG slices from NIfTI volumes.")
    parser.add_argument("--config", default="config/datasets.yaml")
    parser.add_argument("--dataset", nargs="+", default=["brats", "ixi", "oasis", "negative_controls"])
    parser.add_argument("--output-dir", default="data/slices")
    parser.add_argument("--slice-fracs", nargs="+", type=float, default=None,
                        help="Override slice fractions (e.g. 0.2 0.8 for ablation A7)")
    parser.add_argument("--provenance-out", default="data/slice_provenance.jsonl")
    args = parser.parse_args()

    setup_logging(log_file="logs/slice_extract.log")
    config = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    all_prov: list[dict] = []

    for ds_name in args.dataset:
        if ds_name == "negative_controls":
            prov = generate_negative_controls(config, output_dir)
        else:
            ds_cfg = config["datasets"][ds_name]
            if args.slice_fracs:
                fracs = args.slice_fracs
            else:
                fracs = ds_cfg.get("slice_fracs_main", [0.4, 0.5, 0.6])
            planes = ds_cfg.get("planes", ["axial", "sagittal", "coronal"])
            prov = run_extraction(
                dataset_name=ds_name,
                config=config,
                output_dir=output_dir / ds_name,
                planes=planes,
                slice_fracs=fracs,
                n_subjects=ds_cfg.get("n_subjects"),
                seed=ds_cfg.get("subsample_seed", 42),
            )
        all_prov.extend(prov)

    from src.utils.io import append_jsonl
    prov_path = Path(args.provenance_out)
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.unlink(missing_ok=True)
    for rec in all_prov:
        append_jsonl(prov_path, rec)

    log.info(f"Total slices extracted: {len(all_prov)}")
    log.info(f"Provenance written to {prov_path}")


if __name__ == "__main__":
    main()
