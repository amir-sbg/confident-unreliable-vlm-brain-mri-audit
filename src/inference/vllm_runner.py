"""
vLLM-based runner for open VLMs.
Uses vLLM's LLM engine with chat-template-aware vision input.

Before running: verify the installed vLLM version supports the chosen model architecture.
If unsupported, set backend: hf in models.yaml to use hf_runner.py instead.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# FlashInfer's sampler JIT-compiles a CUDA kernel at engine startup (during vLLM's
# memory-profiling dummy sampler run), which needs `nvcc`/CUDA_HOME — absent on compute
# nodes that ship only the GPU runtime/driver. We decode greedily (temperature=0), so the
# native Torch sampler is output-identical. Disable the FlashInfer sampler to skip the JIT.
# Must be set before vLLM's engine starts; this module is imported before LLM() is built,
# and spawned EngineCore subprocesses inherit os.environ.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from src.inference.base import VLMRunner, encode_image_b64
from src.utils import get_logger

log = get_logger(__name__)

# vLLM's `dtype` only accepts auto/half/float16/bfloat16/float/float32.
# FP8 is a weight-quantization setting (`quantization="fp8"`), not a compute dtype.
_DTYPE_ALIASES = {
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "fp16": "float16", "float16": "float16", "half": "half",
    "fp32": "float32", "float32": "float32", "float": "float",
    "auto": "auto",
}
_FP8_KEYS = {"fp8", "float8", "fp8_e4m3", "fp8_e5m2"}


def _resolve_vllm_dtype_quant(dtype_cfg: str) -> tuple[str, str | None]:
    """Map a config `dtype` to vLLM (dtype, quantization).

    Returns (compute_dtype, quantization). For fp8 the compute dtype stays bf16
    and fp8 is passed as the quantization method.
    """
    d = (dtype_cfg or "bfloat16").lower()
    if d in _FP8_KEYS:
        return "bfloat16", "fp8"
    return _DTYPE_ALIASES.get(d, "bfloat16"), None


class VLLMRunner(VLMRunner):
    """Runs a VLM via vLLM's offline inference engine."""

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None

    def load(self) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install vllm")

        dtype_str, quantization = _resolve_vllm_dtype_quant(self.model_cfg.get("dtype", "bfloat16"))
        log.info(f"Loading {self.model_id} via vLLM (dtype={dtype_str}, quantization={quantization})")
        llm_kwargs = dict(
            model=self.model_id,
            dtype=dtype_str,
            max_model_len=self.model_cfg.get("max_model_len", 8192),
            gpu_memory_utilization=self.model_cfg.get("gpu_memory_utilization", 0.85),
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
        )
        if quantization:
            llm_kwargs["quantization"] = quantization
        self.llm = LLM(**llm_kwargs)
        # Deterministic decoding
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        self._loaded = True
        log.info(f"Model loaded: {self.model_id}")

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        system: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Call load() before generate()")

        from vllm import SamplingParams

        sp = self.sampling_params
        if max_tokens is not None:
            sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        image_path = str(image_path)
        messages = self._build_messages(image_path, prompt, system)

        t0 = time.perf_counter()
        try:
            outputs = self.llm.chat(messages=[messages], sampling_params=sp)
            text = outputs[0].outputs[0].text if outputs else ""
        except Exception as e:
            log.error(f"vLLM generate error for {image_path}: {e}")
            text = ""
        gen_seconds = time.perf_counter() - t0

        return {"text": text, "raw": None, "gen_seconds": gen_seconds}

    def _build_messages(self, image_path: str, prompt: str, system: str) -> list[dict]:
        """Construct OpenAI-compatible messages with image for vLLM chat()."""
        b64 = encode_image_b64(image_path)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        })
        return messages

    def generate_batch(
        self,
        items: list[tuple[str | Path, str, str]],  # (image_path, prompt, system)
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Batch inference for efficiency. Items are (image_path, prompt, system) tuples.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before generate_batch()")

        from vllm import SamplingParams

        sp = self.sampling_params
        if max_tokens is not None:
            sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        all_messages = [
            self._build_messages(str(img), pmt, sys_)
            for img, pmt, sys_ in items
        ]

        t0 = time.perf_counter()
        try:
            outputs = self.llm.chat(messages=all_messages, sampling_params=sp)
        except Exception as e:
            log.error(f"vLLM batch generate error: {e}")
            return [{"text": "", "raw": None, "gen_seconds": 0.0}] * len(items)
        gen_seconds = time.perf_counter() - t0

        results = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            results.append({"text": text, "raw": None, "gen_seconds": gen_seconds / len(items)})
        return results
