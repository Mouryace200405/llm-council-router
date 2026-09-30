import base64
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from huggingface_hub import InferenceClient
from openai import OpenAI

from config.pipeline_config import (
    EXPERT_MODELS,
    INFERENCE_BACKEND,
    OLLAMA_BASE_URL,
    get_inference_api_key,
    get_inference_base_url,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    text: str
    model_name: str
    latency_ms: float
    tokens_used: Optional[int] = None
    metadata: dict = None


class ModelAdapter(ABC):
    def __init__(self, expert_key: str):
        cfg = EXPERT_MODELS[expert_key]
        self.key = expert_key
        self.model_id = cfg["model_id"]
        self.description = cfg["description"]

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        ...


class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for OpenAI-compatible APIs (Ollama, HF Router, Together, etc.)."""

    def __init__(self, expert_key: str):
        super().__init__(expert_key)
        base_url = get_inference_base_url()
        api_key = get_inference_api_key()
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info(
            "OpenAICompatibleAdapter[%s]: model=%s base_url=%s",
            expert_key, self.model_id, base_url,
        )

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start = time.time()
        messages = [{"role": "user", "content": prompt}]
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **kwargs,
        )
        latency = (time.time() - start) * 1000
        choice = resp.choices[0]
        return ModelResponse(
            text=choice.message.content or "",
            model_name=self.model_id,
            latency_ms=latency,
            tokens_used=(
                choice.get_stats().usage.total_tokens
                if hasattr(choice, "get_stats") else None
            ),
        )


class DummyAdapter(ModelAdapter):
    """Lightweight dummy adapter for testing without network calls."""

    def __init__(self, expert_key: str):
        self.key = expert_key
        self.model_id = f"dummy_{expert_key}"
        self.description = f"Dummy {expert_key}"

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start = time.time()
        time.sleep(0.05)
        latency = (time.time() - start) * 1000
        return ModelResponse(
            text=f"[{self.key.upper()} simulation] Processed: {prompt[:80]}...",
            model_name=self.model_id,
            latency_ms=latency,
        )


class OllamaAdapter(ModelAdapter):
    """Adapter using native Ollama API — supports images for vision models."""

    def __init__(self, expert_key: str):
        super().__init__(expert_key)
        base_url = OLLAMA_BASE_URL.replace("/v1", "")
        self._api_url = f"{base_url}/api/chat"
        logger.info(
            "OllamaAdapter[%s]: model=%s api_url=%s",
            expert_key, self.model_id, self._api_url,
        )

    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> ModelResponse:
        start = time.time()
        options = {"num_predict": kwargs.get("max_tokens", 2048)}
        # When images are present, reduce GPU layers to leave VRAM for vision encoder
        if images:
            options["num_gpu"] = kwargs.get("num_gpu", 10)
        payload: Dict = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": options,
        }
        if images:
            payload["messages"][0]["images"] = images

        resp = requests.post(self._api_url, json=payload, timeout=600)
        if not resp.ok:
            logger.error(
                "OllamaAdapter[%s] HTTP %d: %s | model=%s prompt_len=%d images=%s",
                self.key, resp.status_code, resp.text[:500],
                self.model_id, len(prompt),
                len(images) if images else 0,
            )
        resp.raise_for_status()
        data = resp.json()
        latency = (time.time() - start) * 1000
        text = data.get("message", {}).get("content", "")
        return ModelResponse(
            text=text,
            model_name=self.model_id,
            latency_ms=latency,
        )


def build_adapter(expert_key: str, dummy: bool = False) -> ModelAdapter:
    if dummy:
        return DummyAdapter(expert_key)
    # rag and reasoning use the same model as multimodal (qwen3.5:9b)
    if expert_key in ("multimodal", "rag", "reasoning") and INFERENCE_BACKEND.lower() == "ollama":
        return OllamaAdapter(expert_key)
    return OpenAICompatibleAdapter(expert_key)


CodingAdapter = lambda dummy=False: build_adapter("coding", dummy)
MultimodalAdapter = lambda dummy=False: build_adapter("multimodal", dummy)
RagAdapter = lambda dummy=False: build_adapter("rag", dummy)
ReasoningAdapter = lambda dummy=False: build_adapter("reasoning", dummy)
GeneralAdapter = lambda dummy=False: build_adapter("general", dummy)
