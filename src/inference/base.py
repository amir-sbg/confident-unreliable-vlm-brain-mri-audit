"""
Abstract base class for VLM runners.
All runners must implement load() and generate().
"""
from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


def encode_image_b64(image_path: str | Path) -> str:
    """Encode a PNG image to a base64 string for API calls."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class VLMRunner(ABC):
    """Abstract VLM runner. Instantiated once per model; generate() called per item."""

    def __init__(self, model_cfg: dict):
        self.model_cfg = model_cfg
        self.model_id  = model_cfg["id"]
        self.max_tokens = model_cfg.get("max_tokens", 512)
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights / initialize API client. Called once before inference."""
        ...

    @abstractmethod
    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        system: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate a response for one (image, prompt) pair.
        Returns {"text": str, "raw": Any, "gen_seconds": float}.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.model_id})"


def make_runner(model_cfg: dict) -> VLMRunner:
    """Factory: return the appropriate runner based on model_cfg['backend']."""
    backend = model_cfg.get("backend", "vllm")
    if backend == "vllm":
        from src.inference.vllm_runner import VLLMRunner
        return VLLMRunner(model_cfg)
    elif backend == "hf":
        from src.inference.hf_runner import HFRunner
        return HFRunner(model_cfg)
    elif backend == "api":
        from src.inference.api_runner import APIRunner
        return APIRunner(model_cfg)
    else:
        raise ValueError(f"Unknown backend: {backend}")
