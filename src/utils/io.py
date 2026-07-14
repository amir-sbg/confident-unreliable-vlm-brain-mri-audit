"""
I/O utilities: JSONL append/read, CSV helpers, atomic file ops.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON record as a line to a JSONL file (creates if absent)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all lines from a JSONL file; skip blank/malformed lines."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                import warnings
                warnings.warn(f"Skipping malformed JSON at {path}:{i+1}: {e}")
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Lazily iterate JSONL records."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_done_keys(path: str | Path, key_fields: tuple[str, ...]) -> set[tuple]:
    """
    Read a JSONL file and return the set of completed key tuples.
    Used by run_inference.py to support resumable jobs.
    """
    done: set[tuple] = set()
    for rec in iter_jsonl(path):
        key = tuple(rec.get(f) for f in key_fields)
        done.add(key)
    return done


def write_csv_atomic(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write a CSV atomically (write to temp, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.touch()
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def ensure_dir(*paths: str | Path) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file with helpful error context."""
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
