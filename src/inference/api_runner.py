"""
API runner for frontier VLMs: OpenAI, Google Gemini, Anthropic Claude.
Skipped automatically if API keys are absent.
Includes retry/backoff and cache by (model, image_id, prompt_id).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from src.inference.base import VLMRunner, encode_image_b64
from src.utils import get_logger

log = get_logger(__name__)

_RETRY_DELAYS = [1, 2, 5, 10, 30]   # seconds between retries


class APIRunner(VLMRunner):
    """Wrapper for cloud API models (OpenAI / Google / Anthropic)."""

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.provider = model_cfg.get("provider", "openai")
        self.client = None
        self._cache_dir = Path("raw_responses/.api_cache")

    def load(self) -> None:
        """Initialize the appropriate API client."""
        if self.provider == "openai":
            self._load_openai()
        elif self.provider == "google":
            self._load_google()
        elif self.provider == "anthropic":
            self._load_anthropic()
        else:
            raise ValueError(f"Unknown API provider: {self.provider}")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = True

    def _load_openai(self) -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set; skipping OpenAI model.")
        try:
            import openai
            self.client = openai.OpenAI(api_key=key)
            log.info(f"OpenAI client initialized for {self.model_id}")
        except ImportError:
            raise ImportError("openai package not installed: pip install openai")

    def _load_google(self) -> None:
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError("GOOGLE_API_KEY not set; skipping Google model.")
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            self.client = genai.GenerativeModel(self.model_id)
            log.info(f"Google Gemini client initialized for {self.model_id}")
        except ImportError:
            raise ImportError("google-generativeai not installed: pip install google-generativeai")

    def _load_anthropic(self) -> None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set; skipping Anthropic model.")
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=key)
            log.info(f"Anthropic client initialized for {self.model_id}")
        except ImportError:
            raise ImportError("anthropic not installed: pip install anthropic")

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
        cache_key = self._cache_key(str(image_path), prompt)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                result = self._call_api(image_path, prompt, system, max_tokens)
                self._save_cache(cache_key, result)
                return result
            except Exception as e:
                log.warning(f"API call attempt {attempt+1} failed for {self.model_id}: {e}")
                if attempt == len(_RETRY_DELAYS):
                    log.error(f"All retries exhausted for {self.model_id}")
                    return {"text": "", "raw": str(e), "gen_seconds": 0.0}

        return {"text": "", "raw": None, "gen_seconds": 0.0}

    def _call_api(self, image_path, prompt: str, system: str, max_tokens: int) -> dict:
        if self.provider == "openai":
            return self._call_openai(image_path, prompt, system, max_tokens)
        elif self.provider == "google":
            return self._call_google(image_path, prompt, system, max_tokens)
        elif self.provider == "anthropic":
            return self._call_anthropic(image_path, prompt, system, max_tokens)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, image_path, prompt: str, system: str, max_tokens: int) -> dict:
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
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return {
            "text": response.choices[0].message.content,
            "raw": response.model_dump(),
            "gen_seconds": time.perf_counter() - t0,
        }

    def _call_google(self, image_path, prompt: str, system: str, max_tokens: int) -> dict:
        from PIL import Image as PILImage
        import google.generativeai as genai
        img = PILImage.open(image_path).convert("RGB")
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        t0 = time.perf_counter()
        response = self.client.generate_content(
            [full_prompt, img],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.0,
            ),
        )
        return {
            "text": response.text,
            "raw": str(response),
            "gen_seconds": time.perf_counter() - t0,
        }

    def _call_anthropic(self, image_path, prompt: str, system: str, max_tokens: int) -> dict:
        b64 = encode_image_b64(image_path)
        t0 = time.perf_counter()
        kwargs: dict = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        # A safety refusal returns stop_reason="refusal" with an empty content array;
        # also skip non-text blocks (e.g. thinking) so we don't index into the wrong type.
        text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text = block.text
                break
        return {
            "text": text,
            "raw": response.model_dump(),
            "gen_seconds": time.perf_counter() - t0,
        }

    def _cache_key(self, image_path: str, prompt: str) -> str:
        h = hashlib.sha256(f"{self.model_id}|{image_path}|{prompt}".encode()).hexdigest()[:24]
        return h

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _save_cache(self, key: str, result: dict) -> None:
        try:
            p = self._cache_path(key)
            safe_result = {k: v for k, v in result.items() if k != "raw"}
            with open(p, "w") as f:
                json.dump(safe_result, f)
        except Exception:
            pass
