"""
Dataset download and verification helpers.

BraTS 2024: requires Synapse account + synapseclient. Start this download manually.
IXI:        direct HTTP download from brain-development.org.
OASIS:      requires account; links provided for manual download.
Negative controls: generated in-pipeline; chest X-ray sample downloaded here.

Usage:
    python -m src.data.download --dataset ixi --output-dir data/
    python -m src.data.download --dataset brats --check-only   # verify existing download
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from src.utils import setup_logging, get_logger

log = get_logger(__name__)

# Official IXI host (canonical tarballs linked from https://brain-development.org/ixi-dataset/).
# NOTE: this Imperial College server intermittently returns HTTP 403 for datacenter/cluster
# egress IPs (verified blocking this cluster 2026-06-17). When that happens, use the HF mirror.
IXI_OFFICIAL_BASE = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI"
IXI_URLS = {
    "T1":  f"{IXI_OFFICIAL_BASE}/IXI-T1.tar",
    "T2":  f"{IXI_OFFICIAL_BASE}/IXI-T2.tar",
    "PD":  f"{IXI_OFFICIAL_BASE}/IXI-PD.tar",
    "MRA": f"{IXI_OFFICIAL_BASE}/IXI-MRA.tar",
}
IXI_METADATA_URL = f"{IXI_OFFICIAL_BASE}/IXI.xls"

# Hugging Face mirror of the official IXI tarballs (reachable when the official host blocks
# the cluster IP). Same archives, same internal IXI###-Site-####-{mod}.nii.gz naming.
# Covers T1/T2/PD only (no MRA/DTI). Verified downloadable (HTTP 206, application/x-tar) 2026-06-17.
IXI_HF_MIRROR_BASE = "https://huggingface.co/datasets/Santhosh1884/IXI-Datasets/resolve/main"
IXI_HF_URLS = {
    "T1": f"{IXI_HF_MIRROR_BASE}/IXI-T1.tar",
    "T2": f"{IXI_HF_MIRROR_BASE}/IXI-T2.tar",
    "PD": f"{IXI_HF_MIRROR_BASE}/IXI-PD.tar",
}

_BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# NIH CXR-14 sample (small publicly accessible subset for chest X-ray negative controls)
# Using a very small publicly-known subset pointer
NIH_CXR_SAMPLE_INFO = """
Chest X-ray negative controls:
  Download a small sample from NIH ChestX-ray14 (public domain):
  https://nihcc.app.box.com/v/ChestXray-NIHCC
  or use the Kaggle dataset: https://www.kaggle.com/datasets/nih-chest-xrays/data
  Place PNG/JPG files in: data/chest_xray_sample/
"""


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        print(f"\r  {pct:5.1f}% ({downloaded // 1_000_000}MB / {total_size // 1_000_000}MB)", end="", flush=True)


def _curl_download(url: str, dest: Path, referer: str | None = None) -> bool:
    """Resumable download via curl: follows redirects (-L), resumes (-C -),
    fails on HTTP error (-f), browser UA. Returns True on success."""
    cmd = ["curl", "-L", "-f", "-C", "-", "-A", _BROWSER_UA,
           "--retry", "3", "--retry-delay", "5",
           "--connect-timeout", "30", "-o", str(dest), url]
    if referer:
        cmd[1:1] = ["-e", referer]
    rc = subprocess.call(cmd)
    if rc != 0:
        log.warning(f"curl exited {rc} for {url}")
    return rc == 0


def _extract_ixi_tar(tar_path: Path, mod_dir: Path) -> None:
    """Extract one IXI modality tarball into mod_dir (idempotent)."""
    if mod_dir.exists() and any(mod_dir.glob("*.nii.gz")):
        log.info(f"IXI already extracted at {mod_dir}")
        return
    mod_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Extracting {tar_path} -> {mod_dir}")
    try:
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(mod_dir)
        log.info(f"Extracted {len(list(mod_dir.glob('*.nii.gz')))} files to {mod_dir}")
    except Exception as e:
        log.error(f"Extraction failed for {tar_path}: {e}")


def download_ixi(
    output_dir: Path,
    modalities: list[str] | None = None,
    prefer_mirror: bool = True,
) -> None:
    """Download IXI tar archives (HF mirror first, official host as fallback) and extract.

    The HF mirror covers T1/T2/PD; MRA is only on the official host (which may 403 a cluster IP).
    """
    output_dir = output_dir / "ixi"
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities = modalities or ["T1", "T2", "PD", "MRA"]

    # Grab the demographics spreadsheet (small) for build_manifest.
    xls_path = output_dir / "IXI.xls"
    if not xls_path.exists():
        if _curl_download(IXI_METADATA_URL, xls_path,
                          referer="https://brain-development.org/ixi-dataset/"):
            log.info(f"IXI.xls -> {xls_path}")
        else:
            log.warning("Could not fetch IXI.xls (demographics); subgroup labels will be empty.")

    for mod in modalities:
        tar_path = output_dir / f"IXI-{mod}.tar"
        if tar_path.exists() and tar_path.stat().st_size > 10_000_000:
            log.info(f"IXI-{mod} archive already present ({tar_path.stat().st_size // 1_000_000}MB)")
        else:
            # Build ordered source list: HF mirror first (reachable), then official host.
            sources: list[tuple[str, str, str | None]] = []
            if prefer_mirror and mod in IXI_HF_URLS:
                sources.append(("HF mirror", IXI_HF_URLS[mod], None))
            if mod in IXI_URLS:
                sources.append(("official", IXI_URLS[mod], "https://brain-development.org/ixi-dataset/"))
            if not sources:
                log.warning(f"No source for IXI modality: {mod}")
                continue

            ok = False
            for name, url, ref in sources:
                log.info(f"Downloading IXI-{mod} from {name}: {url}")
                if _curl_download(url, tar_path, referer=ref):
                    log.info(f"IXI-{mod} downloaded ({tar_path.stat().st_size // 1_000_000}MB)")
                    ok = True
                    break
                log.warning(f"{name} source failed for IXI-{mod}")
            if not ok:
                log.error(
                    f"All sources failed for IXI-{mod}. "
                    f"{'MRA is only on the official host (which may block cluster IPs).' if mod == 'MRA' else ''}"
                )
                continue

        _extract_ixi_tar(tar_path, output_dir / mod)


def check_brats(brats_dir: Path) -> bool:
    """Verify BraTS directory has expected structure."""
    if not brats_dir.exists():
        log.error(f"BraTS directory not found: {brats_dir}")
        log.info("Download BraTS 2024 from Synapse: https://www.synapse.org/brats2024")
        log.info("  pip install synapseclient")
        log.info("  synapse get -r syn51514105 --downloadLocation data/brats2024")
        return False

    niis = list(brats_dir.rglob("*.nii.gz"))
    log.info(f"BraTS: found {len(niis)} NIfTI files in {brats_dir}")
    segs = [f for f in niis if "seg" in f.name.lower()]
    log.info(f"  Segmentation masks found: {len(segs)}")
    return len(niis) > 0


def check_ixi(ixi_dir: Path) -> bool:
    """Verify IXI directory has extracted files."""
    if not ixi_dir.exists():
        log.warning(f"IXI directory not found: {ixi_dir}")
        return False
    niis = list(ixi_dir.rglob("*.nii.gz"))
    log.info(f"IXI: found {len(niis)} NIfTI files in {ixi_dir}")
    return len(niis) > 0


def check_oasis(oasis_dir: Path) -> bool:
    if not oasis_dir.exists():
        log.warning(f"OASIS directory not found: {oasis_dir}")
        log.info("Register and download from: https://www.oasis-brains.org/")
        return False
    niis = list(oasis_dir.rglob("*.nii.gz")) + list(oasis_dir.rglob("*.nii"))
    log.info(f"OASIS: found {len(niis)} NIfTI files")
    return len(niis) > 0


def verify_file_sha256(path: Path, expected_sha: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    return actual == expected_sha


def main():
    parser = argparse.ArgumentParser(description="Download and verify datasets.")
    parser.add_argument("--dataset", nargs="+",
                        choices=["brats", "ixi", "oasis", "all"],
                        default=["ixi"])
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check existence; do not download.")
    parser.add_argument("--ixi-modalities", nargs="+", default=["T1", "T2", "PD", "MRA"])
    args = parser.parse_args()

    setup_logging(log_file="logs/download.log")
    output_dir = Path(args.output_dir)
    datasets = args.dataset
    if "all" in datasets:
        datasets = ["brats", "ixi", "oasis"]

    for ds in datasets:
        log.info(f"=== {ds.upper()} ===")
        if ds == "brats":
            brats_dir = output_dir / "brats2024"
            if not check_brats(brats_dir):
                fallback = output_dir / "brats2023"
                if fallback.exists():
                    log.info(f"Using BraTS 2023 fallback at {fallback}")
                    check_brats(fallback)
        elif ds == "ixi":
            ixi_dir = output_dir / "ixi"
            if not args.check_only:
                download_ixi(output_dir, args.ixi_modalities)
            check_ixi(ixi_dir)
        elif ds == "oasis":
            oasis_dir = output_dir / "oasis"
            if not check_oasis(oasis_dir):
                log.info("OASIS requires manual registration. See: https://www.oasis-brains.org/")

    # Check chest X-ray sample
    cxr_dir = output_dir / "chest_xray_sample"
    if not cxr_dir.exists() or not any(cxr_dir.iterdir()):
        log.info(NIH_CXR_SAMPLE_INFO)
    else:
        cxr_count = len(list(cxr_dir.glob("*.png")) + list(cxr_dir.glob("*.jpg")))
        log.info(f"Chest X-ray sample: {cxr_count} images in {cxr_dir}")


if __name__ == "__main__":
    main()
