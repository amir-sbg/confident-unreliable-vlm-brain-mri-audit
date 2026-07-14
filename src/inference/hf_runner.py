"""
HuggingFace Transformers runner — fallback when vLLM doesn't support a model.
Processes one image at a time (no native batching in the ABC interface).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.inference.base import VLMRunner
from src.utils import get_logger

# Token for gated repos (e.g. google/medgemma-*). transformers also reads these
# env vars automatically, but we pass it explicitly to be robust.
_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

log = get_logger(__name__)


class HFRunner(VLMRunner):
    """Runs a vision-language model via HuggingFace Transformers."""

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.model = None
        self.processor = None
        self.device = None

    def load(self) -> None:
        from transformers import AutoProcessor, BitsAndBytesConfig

        # Prefer AutoModelForImageTextToText (correct for Gemma-3 / MedGemma and
        # other current VLM architectures); fall back to AutoModelForVision2Seq
        # for older architectures or older transformers versions.
        model_classes = []
        try:
            from transformers import AutoModelForImageTextToText
            model_classes.append(AutoModelForImageTextToText)
        except ImportError:
            pass
        try:
            from transformers import AutoModelForVision2Seq
            model_classes.append(AutoModelForVision2Seq)
        except ImportError:
            pass
        if not model_classes:
            raise ImportError(
                "transformers exposes neither AutoModelForImageTextToText nor "
                "AutoModelForVision2Seq; upgrade transformers."
            )

        log.info(f"Loading {self.model_id} via HuggingFace Transformers")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_str = self.model_cfg.get("dtype", "bf16")
        torch_dtype = torch.bfloat16 if dtype_str in ("bf16", "bfloat16") else torch.float16

        # 4-bit quantization if requested (saves memory for large models on single GPU)
        quant_cfg = None
        if dtype_str == "fp8" or self.model_cfg.get("quantize_4bit", False):
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=_HF_TOKEN,
        )

        # Place the whole model on the SINGLE visible GPU. device_map="auto" silently offloads layers
        # to CPU when the visible GPU lacks free memory (e.g. a co-tenant on a shared node) — we saw
        # this cause ~17 s/item (~30x slowdown) on one contended GPU while sibling shards ran at
        # ~0.5 s/item. Forcing {"": 0} keeps the model fully on-GPU (fast) or fails LOUDLY with OOM
        # instead of degrading silently. All our hf-backend models (4B/12B) fit one 40-80GB GPU; each
        # data-parallel shard pins its GPU via CUDA_VISIBLE_DEVICES, so device 0 == that shard's GPU.
        device_map = {"": 0} if torch.cuda.is_available() else None

        last_err = None
        for cls in model_classes:
            try:
                self.model = cls.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    quantization_config=quant_cfg,
                    token=_HF_TOKEN,
                )
                self.model.eval()
                self._loaded = True
                log.info(f"Loaded {self.model_id} with {cls.__name__} on {self.device}")
                return
            except Exception as e:
                last_err = e
                log.warning(f"{cls.__name__} could not load {self.model_id}: {e}")

        log.error(f"Failed to load {self.model_id} with any model class: {last_err}")
        raise RuntimeError(f"Failed to load {self.model_id}: {last_err}")

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        system: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Call load() before generate()")

        max_tokens = max_tokens or self.max_tokens
        image = Image.open(image_path).convert("RGB")

        # Modern VLM convention (Gemma3/MedGemma, Qwen2-VL, …): embed the image INSIDE the message
        # content and let apply_chat_template insert the image token. Do NOT also pass images=,
        # which both double-counts the arg ("multiple values for 'images'") and, via the old
        # processor(text=,images=) fallback, leaves the prompt with 0 image tokens.
        content_msgs = []
        if system:
            content_msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
        content_msgs.append({"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]})

        t0 = time.perf_counter()
        inputs = None
        # Strategy A: image-in-content, no images= kwarg (correct for Gemma3Processor).
        try:
            inputs = self.processor.apply_chat_template(
                content_msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
        except Exception as e_a:
            log.debug(f"chat-template (image-in-content) failed: {e_a}")
            # Strategy B: {"type":"image"} placeholder + images= kwarg (older processors).
            try:
                ph_msgs = []
                if system:
                    ph_msgs.append({"role": "system", "content": system})
                ph_msgs.append({"role": "user",
                                "content": [{"type": "image"}, {"type": "text", "text": prompt}]})
                templated = self.processor.apply_chat_template(
                    ph_msgs, add_generation_prompt=True, tokenize=False)
                inputs = self.processor(text=templated, images=image, return_tensors="pt")
            except Exception as e_b:
                log.error(f"Both processor strategies failed for {self.model_id}: A={e_a}; B={e_b}")
                return {"text": "", "raw": f"processor_error: {e_b}",
                        "gen_seconds": time.perf_counter() - t0}
        inputs = inputs.to(self.model.device)

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            # Decode only the generated tokens (strip prompt)
            n_input = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            generated = output_ids[:, n_input:]
            text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        except Exception as e:
            log.error(f"HF generate error for {image_path}: {e}")
            text = ""

        gen_seconds = time.perf_counter() - t0
        return {"text": text, "raw": None, "gen_seconds": gen_seconds}
