"""
Deterministic seed management. All randomness in the pipeline goes through here.
"""
from __future__ import annotations

import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Return a seeded NumPy default_rng for a specific operation."""
    return np.random.default_rng(seed)


def subsample_indices(n_total: int, n_sample: int, seed: int) -> np.ndarray:
    """Return sorted indices for a reproducible subsample without replacement."""
    rng = make_rng(seed)
    if n_sample >= n_total:
        return np.arange(n_total)
    return np.sort(rng.choice(n_total, size=n_sample, replace=False))


def shuffle_mc_options(options: list[str], item_seed: int) -> tuple[list[str], list[int]]:
    """
    Shuffle MC options deterministically for a given item.
    Returns (shuffled_options, original_indices) so the shuffle is reproducible and logged.
    """
    rng = make_rng(item_seed)
    indices = list(range(len(options)))
    rng.shuffle(indices)
    shuffled = [options[i] for i in indices]
    return shuffled, indices
